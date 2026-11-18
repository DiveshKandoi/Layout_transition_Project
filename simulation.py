"""Discrete-event simulation that is tied to the optimisation model.

It reads the same inputs as optimization.py (standard minutes, shared
operators, process efficiency, line operators), so the two models describe the
same factory.  It does two jobs:

1. run_simulation / run_replications
   Replay one month of the planned product mix through the process layout and
   through the hybrid layout (n dedicated lines + the reduced shared stitching
   crew) and report throughput, flow time, WIP and shared-stitching utilisation.

2. line_capacity_check / calibrated_params
   Feed a saturated dedicated line through finite buffers and measure what it
   really produces in one month, then compare with the capacity the MILP
   assumes (cap_line).  calibrated_params() returns the inputs with cap_line
   replaced by the simulated value.

Line consistency
    When the model inputs come from line_balancing (optimization.apply_line_design)
    they carry 'station_loads' and L_line = number of stations.  The simulation
    builds exactly that line (one worker per station, finite buffers) and takes
    L_line operators out of shared stitching in the hybrid layout, so the MILP,
    the line balance and the simulation describe the same line.  Without a
    design it falls back to an RPW benchmark line at CYCLE_TIME (0.45 min), which
    is NOT the budget-driven design; check_line_consistency() reports the gap.
    Machine counts do not limit the simulated line: the design gives every
    station its own machine of each type it needs, so there is no machine
    contention to simulate.  The six cutting and six finishing servers are
    simulation resources, not validated workforce numbers.

Simulation time is 26 x 8 x 60 = 12,480 paid minutes/month. Shared workers
deliver 6,240 productive standard minutes at 50% efficiency. Dedicated-line
capacity separately uses 75% machine uptime.
"""
import random

import numpy as np
import pandas as pd
import simpy

from line_balancing import TASKS, rpw_heuristic
from optimization import (PRODUCTS, default_params, PAID_MINUTES, PRODUCTIVE_MINUTES_PER_WORKER, BASE_WORKERS)

MONTH_MIN = PAID_MINUTES                 # 12,480 paid minutes/month
LINE_UPTIME = 0.75
CYCLE_TIME = 0.45                        # fallback benchmark cycle time (RPW line)


def station_loads(cycle_time=CYCLE_TIME):
    """Minutes of work at each station of the balanced dedicated line."""
    stations, _ = rpw_heuristic(cycle_time)
    return [sum(TASKS[t]['time'] for t in s) for s in stations]


def check_line_consistency(params):
    """Problems that make the simulated line differ from the MILP's line.

    Returns a list of plain-language messages (empty list = consistent).
    """
    cfg = default_params(params['demand'] if params and 'demand' in params else None)
    if params:
        cfg.update(params)
    problems = []
    loads = cfg.get('station_loads')
    if not loads:
        problems.append(
            "No line design supplied: the simulation uses a benchmark RPW line "
            f"({len(station_loads())} stations at {CYCLE_TIME} min) while the "
            f"MILP assumes {cfg['L_line']} operators and {cfg['cap_line']:,} "
            "units/month per line.")
        return problems
    if len(loads) != cfg['L_line']:
        problems.append(f"{len(loads)} simulated stations but L_line = {cfg['L_line']}.")
    theoretical = MONTH_MIN / max(x / LINE_UPTIME for x in loads)
    if abs(theoretical - cfg['cap_line']) > 1.0:
        problems.append(f"Bottleneck {max(loads):.3f} min gives {theoretical:,.0f} "
                        f"units/month but the MILP uses cap_line = {cfg['cap_line']:,}.")
    return problems


def _sample(rng, mean, cv):
    return max(0.01, rng.normalvariate(mean, cv * mean))


# --------------------------------------------------------------------------
# A balanced line made of stations in series with finite buffers
# --------------------------------------------------------------------------
def _build_line(env, loads, buffer, cv, rng, on_done):
    """Start one worker per station. Returns the line's input store.

    A station that finishes a garment waits (is blocked) while the next buffer
    is full, and waits (is starved) while its own input is empty, so variability
    costs throughput, as on a real line.  on_done(item) fires when a garment
    leaves the last station.
    """
    n = len(loads)
    stores = [simpy.Store(env, capacity=buffer) for _ in range(n)]

    def worker(i):
        while True:
            item = yield stores[i].get()
            yield env.timeout(_sample(rng, loads[i], cv))
            if i < n - 1:
                yield stores[i + 1].put(item)      # blocks if next buffer is full
            else:
                on_done(item)

    for i in range(n):
        env.process(worker(i))
    return stores[0]


# --------------------------------------------------------------------------
# 1. Layout comparison
# --------------------------------------------------------------------------
def run_simulation(layout_type="Process", params=None, month_demand=None,
                   n_lines=1, warmup=1000, seed=42, buffer=3, cv=0.1,
                   cutting_servers=6, finishing_servers=6, transfer_operators=None, shared_hires=0):
    """One simulated month of the planned product mix.

    layout_type   'Process' or 'Hybrid'
    month_demand  {product: units this month}; default = the model's month 30
    n_lines       dedicated lines in the hybrid layout (each takes L_line
                  operators from shared stitching only if labour_coupling is on;
                  otherwise the line operators are hired and the crew is unchanged)
    cutting_servers / finishing_servers  about 6 servers each corresponds to
                  45,000 standard minutes at ~80 % efficiency
    """
    cfg = default_params(params['demand'] if params and 'demand' in params else None)
    if params:
        cfg.update(params)
    if month_demand is None:
        month_demand = {p: cfg['demand'][p][30] for p in PRODUCTS}
    sam = cfg['std_times']
    eff = cfg['process_efficiency']

    if layout_type == "Hybrid":
        line_staff = cfg['L_line'] * n_lines
        transferred = min(cfg['ops_shared_stitch'], line_staff) if transfer_operators is None else int(transfer_operators)
        if transferred < 0 or transferred > min(cfg['ops_shared_stitch'], line_staff):
            raise ValueError("transfer_operators must be between zero and the available line staffing.")
        shared_ops = cfg['ops_shared_stitch'] - transferred + int(shared_hires)
        hired_line = line_staff - transferred
    elif layout_type == "Process":
        transferred, hired_line = 0, 0
        shared_ops = cfg['ops_shared_stitch']
    else:
        raise ValueError("layout_type must be 'Process' or 'Hybrid'")
    if shared_ops < 1:
        raise ValueError("No shared stitching operators remain; hire shared replacements or transfer fewer workers.")

    total = sum(month_demand.values())
    rate = total / MONTH_MIN                          # arrivals per effective minute
    shares = [month_demand[p] / total for p in PRODUCTS]

    rng = random.Random(seed)
    env = simpy.Environment()
    cutting = simpy.Resource(env, capacity=cutting_servers)
    finishing = simpy.Resource(env, capacity=finishing_servers)
    shared = simpy.Resource(env, capacity=int(shared_ops))

    flow = []
    stats = {'arrivals': 0, 'busy': 0.0, 'wip': 0, 'area': 0.0,
             'last': float(warmup), 'rr': 0}
    loads = cfg.get('station_loads') or station_loads()   # designed line if provided
    loads = [x / LINE_UPTIME for x in loads]
    if cfg.get('station_loads') and len(loads) != cfg['L_line']:
        raise ValueError(f"Line design has {len(loads)} stations but L_line is "
                         f"{cfg['L_line']}; the simulation and the MILP would differ.")
    lines = []

    def update_wip(delta):
        now = env.now
        if now > warmup:
            stats['area'] += stats['wip'] * (now - max(stats['last'], warmup))
            stats['last'] = now
        stats['wip'] += delta

    def garment(p):
        arrive = env.now
        update_wip(+1)
        if arrive > warmup:
            stats['arrivals'] += 1
        with cutting.request() as r:
            yield r
            yield env.timeout(_sample(rng, sam[p]['Cutting'] / eff, cv))
        if layout_type == "Hybrid" and p == 'A' and n_lines > 0:
            done = env.event()
            line = lines[stats['rr'] % n_lines]             # round robin
            stats['rr'] += 1
            yield line.put(done)
            yield done
        else:
            with shared.request() as r:
                yield r
                t = _sample(rng, sam[p]['Stitching'] / eff, cv)
                yield env.timeout(t)
                if env.now > warmup:
                    stats['busy'] += t
        with finishing.request() as r:
            yield r
            yield env.timeout(_sample(rng, sam[p]['Finishing'] / eff, cv))
        update_wip(-1)
        if env.now > warmup:
            flow.append(env.now - arrive)

    if layout_type == "Hybrid":
        lines = [_build_line(env, loads, buffer, cv, rng, lambda ev: ev.succeed())
                 for _ in range(n_lines)]

    def source():
        while True:
            yield env.timeout(rng.expovariate(rate))
            env.process(garment(rng.choices(PRODUCTS, weights=shares)[0]))

    env.process(source())
    env.run(until=MONTH_MIN)
    update_wip(0)

    window = MONTH_MIN - warmup
    done_n = len(flow)
    return {
        'Layout': layout_type,
        'Lines': n_lines if layout_type == "Hybrid" else 0,
        'Shared_Ops': int(shared_ops),
        'Transferred_Operators': int(transferred),
        'Hired_Line_Operators': int(hired_line),
        'Shared_Hires': int(shared_hires),
        'Line_Stations': len(loads),
        'Offered': stats['arrivals'],
        'Completed': done_n,
        'Throughput': done_n / window,
        'Avg_Flow_Time': float(np.mean(flow)) if done_n else 0.0,
        'WIP': stats['area'] / window,
        'Shared_Util': stats['busy'] / (shared_ops * window),
    }


def run_replications(layout_type, n=10, seed=42, **kwargs):
    """Mean of n independent runs, plus the standard deviation of flow time."""
    runs = [run_simulation(layout_type, seed=seed + k, **kwargs) for k in range(n)]
    df = pd.DataFrame(runs)
    out = df.drop(columns='Layout').mean().to_dict()
    out['Layout'] = layout_type
    out['Flow_Time_SD'] = float(df['Avg_Flow_Time'].std(ddof=1)) if n > 1 else 0.0
    out['Replications'] = n
    return out


# --------------------------------------------------------------------------
# 2. Does the simulated line deliver the capacity the MILP assumes?
# --------------------------------------------------------------------------
def line_capacity_check(params=None, buffer=3, cv=0.1, reps=10, seed=7,
                        warmup=1000, cycle_time=CYCLE_TIME):
    """Saturated line: measure monthly output with finite buffers.

    Returns the theoretical capacity (MONTH_MIN / bottleneck load), the
    simulated capacity, their ratio (line efficiency from variability), and
    the cap_line used by the MILP.
    """
    cap_milp = (params or {}).get('cap_line', default_params()['cap_line'])
    loads = (params or {}).get('station_loads') or station_loads(cycle_time)
    theoretical = MONTH_MIN / max(x / LINE_UPTIME for x in loads)
    sim_loads = [x / LINE_UPTIME for x in loads]
    results = []
    for k in range(reps):
        rng = random.Random(seed + k)
        env = simpy.Environment()
        count = {'n': 0}

        def on_done(_):
            if env.now > warmup:
                count['n'] += 1

        inbox = _build_line(env, sim_loads, buffer, cv, rng, on_done)

        def feeder():
            while True:
                yield inbox.put(object())          # always a garment waiting

        env.process(feeder())
        env.run(until=MONTH_MIN)
        results.append(count['n'] / (MONTH_MIN - warmup) * MONTH_MIN)
    sim = float(np.mean(results))
    return {'stations': len(loads), 'bottleneck_min': max(loads),
            'theoretical': theoretical, 'simulated': sim,
            'simulated_sd': float(np.std(results, ddof=1)) if reps > 1 else 0.0,
            'ratio': sim / theoretical, 'milp_cap_line': cap_milp,
            'gap_vs_milp': sim - cap_milp}


def calibrated_params(params=None, **kwargs):
    """Model inputs with cap_line replaced by the simulated line capacity."""
    check = line_capacity_check(params, **kwargs)
    new = dict(params or {})
    new['cap_line'] = int(round(check['simulated']))
    return new, check


if __name__ == "__main__":
    chk = line_capacity_check()
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in chk.items()})
    for lay, n in (("Process", 0), ("Hybrid", 1)):
        r = run_replications(lay, n=5, n_lines=n)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
