"""Assembly-line balancing for the dedicated T-shirt stitching line.

Two ways of using the module
    A. Benchmark (unchanged):  fix the cycle time, minimise the stations
         rpw_heuristic()              Ranked Positional Weight heuristic
         exact_station_assignment()   exact MILP minimising the number of stations
    B. Budget-driven design (new):  fix the capital budget of ONE line, choose
       the station assignment AND the machine at every operation so that the
       CYCLE TIME is minimised.  The result feeds the 36-month layout MILP.
         design_line()                budget -> cycle time, stations, install cost
         design_frontier()            the same for a range of budgets
         minimum_budget()             cheapest line that can run at all

The outputs of design_line() map one-to-one onto the layout-model inputs
    cycle_time  -> cap_line      = days x hours x 60 x uptime / cycle_time
    stations    -> L_line        (one operator per station)
    install_cost-> cost_install  (machines + optional station / line costs)
    monthly overhead -> cost_run (electricity + maintenance + consumables of
                       the machines the design selected, per installed line)
see optimization.apply_line_design().

Investment (one-off, cost_install) and monthly overhead (recurring, cost_run)
are kept separate.  Overhead is always computed from the machine counts that
the design selects; nothing here uses a fixed monthly overhead figure.

Works with PuLP 2.x, 3.x and 4.x.  PuLP 4.x no longer bundles the CBC binary
and creates variables through model.add_variable, so install a solver with
pip install highspy
"""
import math
import time

import pulp

# --------------------------------------------------------------------------
# Line data: supplied 14-operation T-shirt sequence
# --------------------------------------------------------------------------
TASKS = {
    1:  {'time': 0.29, 'preds': [],     'op': 'shoulder_sewing'},
    2:  {'time': 0.08, 'preds': [],     'op': 'collar_inseam'},
    3:  {'time': 0.34, 'preds': [1, 2], 'op': 'collar_sewing'},
    4:  {'time': 0.25, 'preds': [3],    'op': 'neck_label_sewing'},
    5:  {'time': 0.35, 'preds': [4],   'op': 'neck_binding_sewing'},
    6:  {'time': 0.08, 'preds': [5],   'op': 'binding_control'},
    7:  {'time': 0.36, 'preds': [6],   'op': 'sleeve_sewing'},
    8:  {'time': 0.09, 'preds': [],    'op': 'care_label_preparing'},
    9:  {'time': 0.43, 'preds': [7, 8], 'op': 'side_seam'},
    10: {'time': 0.42, 'preds': [9],   'op': 'sleeve_hem_cover_seam'},
    11: {'time': 0.13, 'preds': [10], 'op': 'sleeve_hem_reinforcement'},
    12: {'time': 0.33, 'preds': [11], 'op': 'hem_cover_seam'},
    13: {'time': 0.06, 'preds': [12], 'op': 'hem_reinforcement'},
    14: {'time': 0.25, 'preds': [13], 'op': 'flag_label_sewing'},
}

# Installed cost = purchase price + assumed setup/commissioning.
# No speed premium is assumed because no comparative speed data was supplied.
MACHINES = {
    '4-thread overlock': {'cost': 35_000, 'purchase_cost': 32_000,
                          'setup_cost': 3_000, 'speed': 1.0,
                          'power_kw': 0.4, 'consumables_monthly': 250},
    'Lock-stitch': {'cost': 24_840, 'purchase_cost': 21_840,
                    'setup_cost': 3_000, 'speed': 1.0,
                    'power_kw': 0.4, 'consumables_monthly': 250},
    'Chain-stitch': {'cost': 65_000, 'purchase_cost': 62_000,
                     'setup_cost': 3_000, 'speed': 1.0,
                     'power_kw': 0.4, 'consumables_monthly': 250},
    'Blade cover-stitch': {'cost': 48_650, 'purchase_cost': 45_650,
                           'setup_cost': 3_000, 'speed': 1.0,
                           'power_kw': 0.4, 'consumables_monthly': 250},
    # Manual control is a workstation, not a powered machine in machine_info.txt.
    'Manual': {'cost': 0, 'purchase_cost': 0, 'setup_cost': 0,
               'speed': 1.0, 'power_kw': 0.0, 'consumables_monthly': 0},
}

OPERATION_MACHINES = {
    'shoulder_sewing': ['4-thread overlock'],
    'collar_inseam': ['Lock-stitch'],
    'collar_sewing': ['4-thread overlock'],
    'neck_label_sewing': ['Lock-stitch'],
    'neck_binding_sewing': ['Chain-stitch'],
    'binding_control': ['Manual'],
    'sleeve_sewing': ['4-thread overlock'],
    'care_label_preparing': ['Lock-stitch'],
    'side_seam': ['4-thread overlock'],
    'sleeve_hem_cover_seam': ['Blade cover-stitch'],
    'sleeve_hem_reinforcement': ['Lock-stitch'],
    'hem_cover_seam': ['Blade cover-stitch'],
    'hem_reinforcement': ['Lock-stitch'],
    'flag_label_sewing': ['Lock-stitch'],
}

OVERHEAD_ASSUMPTIONS = {
    'operating_hours_month': 26 * 8,  # 208 scheduled machine hours/month
    'electricity_tariff_per_kwh': 8.0,
    'maintenance_rate_annual': 0.02,
}


def get_solver(time_limit=None, msg=False, gap_rel=None):
    """Return an available PuLP solver (HiGHS first, then CBC)."""
    available = pulp.listSolvers(onlyAvailable=True)
    gap = {} if gap_rel is None else {'gapRel': gap_rel}
    if 'HiGHS' in available:
        return pulp.HiGHS(msg=msg, timeLimit=time_limit, **gap)
    if 'HiGHS_CMD' in available:
        return pulp.HiGHS_CMD(msg=msg, timeLimit=time_limit, **gap)
    if 'PULP_CBC_CMD' in available:
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, **gap)
    raise RuntimeError(
        "No MILP solver found. Install one with:  pip install highspy")


def new_var(model, name, low=None, up=None, cat='Continuous'):
    """Create a variable on any PuLP version (4.x needs model.add_variable)."""
    if hasattr(model, 'add_variable'):
        return model.add_variable(name, lowBound=low, upBound=up, cat=cat)
    return pulp.LpVariable(name, lowBound=low, upBound=up, cat=cat)


def solve_model(model, solver):
    """Solve and return (is_optimal, status_code) on any PuLP version.

    PuLP 4.x returns an LpSolveStats object and has no model.status;
    PuLP 2.x/3.x return an int and set model.status.  Code 1 means optimal.
    """
    res = model.solve(solver)
    code = getattr(res, 'status', None)
    if code is None:
        code = getattr(model, 'status', res)
    code = int(getattr(code, 'value', code))
    return code == 1, code


def _check_cycle_time(cycle_time):
    longest = max(d['time'] for d in TASKS.values())
    if cycle_time < longest:
        raise ValueError(
            f"Cycle time {cycle_time} is shorter than the longest task "
            f"({longest}); no feasible line exists.")


def get_successors(task, _cache=None):
    """All direct and indirect successors of a task."""
    if _cache is None:
        _cache = set()
    for t, data in TASKS.items():
        if task in data['preds'] and t not in _cache:
            _cache.add(t)
            get_successors(t, _cache)
    return _cache


def rpw_heuristic(cycle_time=0.45):
    """Ranked Positional Weight heuristic (station-oriented).

    Returns (stations, balance_efficiency) where stations is a list of task
    lists.  After every assignment the eligible task list is re-scanned, so
    the highest-ranked eligible task is always picked next.
    """
    _check_cycle_time(cycle_time)
    eps = 1e-9

    pw = {i: d['time'] + sum(TASKS[s]['time'] for s in get_successors(i))
          for i, d in TASKS.items()}
    ranked = sorted(TASKS, key=lambda i: (-pw[i], i))

    stations, current, load, assigned = [], [], 0.0, set()
    while len(assigned) < len(TASKS):
        pick = None
        for task in ranked:                      # highest PW first
            if (task not in assigned
                    and all(p in assigned for p in TASKS[task]['preds'])
                    and load + TASKS[task]['time'] <= cycle_time + eps):
                pick = task
                break
        if pick is None:                         # nothing fits: close station
            stations.append(current)
            current, load = [], 0.0
        else:
            current.append(pick)
            load += TASKS[pick]['time']
            assigned.add(pick)
    if current:
        stations.append(current)

    total = sum(d['time'] for d in TASKS.values())
    return stations, total / (len(stations) * cycle_time)


def exact_station_assignment(cycle_time=0.45, time_limit=None):
    """Exact MILP minimising the number of stations (works in PuLP 2-4).

    Returns (n_stations, balance_efficiency, assignment) where assignment maps
    station number -> list of tasks.
    """
    _check_cycle_time(cycle_time)
    tasks = list(TASKS)
    # An upper bound from the heuristic keeps the model small.
    ub = len(rpw_heuristic(cycle_time)[0])
    S = list(range(1, ub + 1))

    prob = pulp.LpProblem("Line_Balancing", pulp.LpMinimize)
    x = {(i, s): new_var(prob, f"x_{i}_{s}", cat='Binary')
         for i in tasks for s in S}
    y = {s: new_var(prob, f"y_{s}", cat='Binary') for s in S}

    prob += pulp.lpSum(y[s] for s in S)

    for i in tasks:                                     # each task once
        prob += pulp.lpSum(x[i, s] for s in S) == 1
    for s in S:                                         # cycle time
        prob += (pulp.lpSum(TASKS[i]['time'] * x[i, s] for i in tasks)
                 - cycle_time * y[s]) <= 0
    for i in tasks:                                     # precedence
        for j in TASKS[i]['preds']:
            prob += (pulp.lpSum(s * x[j, s] for s in S)
                     - pulp.lpSum(s * x[i, s] for s in S)) <= 0
    for s in S[:-1]:                                    # use stations in order
        prob += (y[s + 1] - y[s]) <= 0

    ok, code = solve_model(prob, get_solver(time_limit))
    if not ok:
        raise RuntimeError(f"Line-balancing solve failed (status code {code})")

    n = int(round(sum(y[s].value() for s in S)))
    assignment = {s: [i for i in tasks if round(x[i, s].value()) == 1]
                  for s in S if y[s].value() > 0.5}
    total = sum(d['time'] for d in TASKS.values())
    return n, total / (n * cycle_time), assignment


def line_capacity(cycle_time, days=26, hours=8, uptime=0.75):
    """Units per month from the bottleneck cycle time."""
    return days * hours * 60 * uptime / cycle_time


# Reference machine mix from machine_info.txt (11 powered machines + the
# manual workstation).  Used for reporting and as a regression test only; the
# design itself never reads it.
REFERENCE_MIX = {'4-thread overlock': 4, 'Lock-stitch': 4, 'Chain-stitch': 1,
                 'Blade cover-stitch': 2, 'Manual': 1}

# Totals printed in the source file machine_info.txt.  They do NOT agree with
# the itemised prices (4x32,000 + 4x21,840 + 62,000 + 2x45,650 = 368,660).
# They are kept here only so the discrepancy can be reported, never used.
SOURCE_REPORTED = {'purchase_cost': 371_660, 'installed_cost': 404_660}


def reference_investment(machines=None):
    """Itemised investment of the reference mix and the source discrepancy."""
    machines = MACHINES if machines is None else machines
    purchase = sum(n * machines[m]['purchase_cost'] for m, n in REFERENCE_MIX.items())
    setup = sum(n * machines[m]['setup_cost'] for m, n in REFERENCE_MIX.items())
    installed = sum(n * machines[m]['cost'] for m, n in REFERENCE_MIX.items())
    return {'purchase_cost': purchase, 'setup_cost': setup,
            'installed_cost': installed,
            'source_purchase_cost': SOURCE_REPORTED['purchase_cost'],
            'source_installed_cost': SOURCE_REPORTED['installed_cost'],
            'purchase_discrepancy': SOURCE_REPORTED['purchase_cost'] - purchase,
            'installed_discrepancy': SOURCE_REPORTED['installed_cost'] - installed}


# --------------------------------------------------------------------------
# Dynamic monthly overhead based on the machine mix selected by the design
# --------------------------------------------------------------------------
def calculate_monthly_overhead(machine_counts, machines=None, operating_hours=None,
                               electricity_tariff=None, maintenance_rate=None):
    """Return monthly electricity, maintenance and consumables for a machine mix.

    Electricity = count x kW x operating hours x Rs/kWh.
    Maintenance = annual maintenance rate x machine purchase value / 12.
    Consumables = the source estimate of Rs 250 per powered sewing machine.
    """
    machines = MACHINES if machines is None else machines
    hours = OVERHEAD_ASSUMPTIONS['operating_hours_month'] if operating_hours is None else float(operating_hours)
    tariff = OVERHEAD_ASSUMPTIONS['electricity_tariff_per_kwh'] if electricity_tariff is None else float(electricity_tariff)
    rate = OVERHEAD_ASSUMPTIONS['maintenance_rate_annual'] if maintenance_rate is None else float(maintenance_rate)
    if min(hours, tariff, rate) < 0:
        raise ValueError("Overhead assumptions must be non-negative.")

    electricity = purchase_value = installed_value = consumables = 0.0
    powered_count = 0
    kw_used = set()
    counts = {}
    for machine, raw_count in machine_counts.items():
        if machine not in machines:
            raise KeyError(f"Unknown machine type: {machine}")
        count = int(raw_count)
        if count < 0 or count != raw_count:
            raise ValueError("Machine counts must be non-negative integers.")
        counts[machine] = count
        data = machines[machine]
        kw = float(data.get('power_kw', 0.0))
        electricity += count * kw * hours * tariff
        purchase_value += count * float(data.get('purchase_cost', data['cost']))
        installed_value += count * float(data['cost'])
        consumables += count * float(data.get('consumables_monthly', 0.0))
        if kw > 0:
            powered_count += count
            kw_used.add(kw)

    maintenance = purchase_value * rate / 12.0
    per_type = {m: {'count': c,
                    'electricity': c * float(machines[m].get('power_kw', 0.0)) * hours * tariff,
                    'maintenance': c * float(machines[m].get('purchase_cost', machines[m]['cost'])) * rate / 12.0,
                    'consumables': c * float(machines[m].get('consumables_monthly', 0.0))}
                for m, c in counts.items()}
    return {
        'per_type': per_type,
        'machine_counts': counts,
        'powered_machine_count': powered_count,
        'purchase_value': purchase_value,
        'installed_machine_value': installed_value,
        'electricity_monthly': electricity,
        'maintenance_monthly': maintenance,
        'consumables_monthly': consumables,
        'total_monthly_overhead': electricity + maintenance + consumables,
        'assumptions': {
            'operating_hours_month': hours,
            'electricity_tariff_per_kwh': tariff,
            'power_kw_per_powered_machine': (kw_used.pop() if len(kw_used) == 1 else None),
            'maintenance_rate_annual': rate,
            'maintenance_basis': 'purchase cost',
        },
    }


def calculate_overhead_per_unit(monthly_overhead, monthly_output):
    """Allocate monthly machine overhead over the units actually produced.

    Returns None when the output is zero (or negative): there is no unit to
    carry the overhead, and a made-up number would be misleading.  Pass actual
    production for the real figure; passing theoretical capacity gives only a
    full-capacity estimate and must be labelled as such.
    """
    if monthly_output is None or monthly_output <= 0:
        return None
    return float(monthly_overhead) / float(monthly_output)


# ==========================================================================
# B. Budget-driven line design
# ==========================================================================
def _prepare(machines=None, tasks=None, operation_machines=None):
    """Resolve the inputs and build {task: [(machine, minutes), ...]}."""
    machines = machines or MACHINES
    tasks = tasks or TASKS
    op_map = operation_machines or OPERATION_MACHINES
    for m, d in machines.items():
        if d['cost'] < 0 or d['speed'] <= 0:
            raise ValueError(f"Machine {m!r}: cost must be >= 0 and speed > 0.")
    options = {}
    for i, d in tasks.items():
        eligible = [m for m in op_map.get(d['op'], []) if m in machines]
        if not eligible:
            raise ValueError(
                f"Task {i} ({d['op']}) has no eligible machine in the machine list.")
        options[i] = [(m, d['time'] / machines[m]['speed']) for m in eligible]
    return machines, tasks, options


def cycle_time_floor(options):
    """Smallest cycle time any budget could reach (slowest task on its fastest machine)."""
    return max(min(t for _, t in opts) for opts in options.values())


def _summarise(stations, machines, station_cost, line_cost):
    """stations: list of [(task, machine, minutes), ...] -> design dict."""
    loads = [sum(t for _, _, t in st) for st in stations]
    station_machines = [sorted({m for _, m, _ in st}) for st in stations]
    machine_cost = sum(machines[m]['cost'] for ms in station_machines for m in ms)
    counts = {}
    for ms in station_machines:
        for m in ms:
            counts[m] = counts.get(m, 0) + 1
    cycle = max(loads)
    return {
        'cycle_time': cycle,
        'stations': len(stations),
        'install_cost': machine_cost + station_cost * len(stations) + line_cost,
        'cost_breakdown': {'machines': machine_cost,
                           'stations': station_cost * len(stations),
                           'line_fixed': line_cost},
        'assignment': {k + 1: [i for i, _, _ in st] for k, st in enumerate(stations)},
        'task_machine': {i: m for st in stations for i, m, _ in st},
        'station_machines': {k + 1: ms for k, ms in enumerate(station_machines)},
        'station_loads': loads,
        'machine_counts': counts,
        'balance_efficiency': sum(loads) / (len(stations) * cycle),
    }


# --------------------------------------------------------------------------
# Exact method 1: dynamic programming over "tasks finished so far" sets
# --------------------------------------------------------------------------
# A line is a sequence of stations; after station k the finished tasks form a
# set closed under predecessors (an order ideal).  Each station is therefore an
# edge between two such sets, carrying a (load, cost) pair for every machine
# choice.  For a fixed cycle time C the cheapest line is a shortest path, and
# the optimal C is one of the edge loads, so a bisection finds it exactly.
# For an almost-serial line like the T-shirt line this takes milliseconds.
class _TooLarge(Exception):
    """Precedence graph too wide for the DP; the MILP is used instead."""


def _build_edges(options, tasks, machines, station_cost,
                 max_ideals=800, max_combos=60_000):
    from itertools import product
    ids = sorted(tasks)
    bit = {i: 1 << k for k, i in enumerate(ids)}
    pred_mask = {i: sum(bit[p] for p in tasks[i]['preds']) for i in ids}
    seen, frontier = {0}, [0]
    while frontier:
        nxt = []
        for a in frontier:
            for i in ids:
                if not a & bit[i] and (pred_mask[i] & a) == pred_mask[i]:
                    b = a | bit[i]
                    if b not in seen:
                        seen.add(b)
                        nxt.append(b)
                        if len(seen) > max_ideals:
                            raise _TooLarge
        frontier = nxt
    ideals = sorted(seen, key=lambda m: (bin(m).count('1'), m))
    edges = {}                                   # a -> [(b, options), ...]
    for a in ideals:
        for b in ideals:
            if b == a or (b & a) != a:
                continue
            batch = [i for i in ids if (b ^ a) & bit[i]]
            n_combo = 1
            for i in batch:
                n_combo *= len(options[i])
            if n_combo > max_combos:
                raise _TooLarge
            rows = []
            for combo in product(*[options[i] for i in batch]):
                machine_set = {m for m, _ in combo}
                rows.append((sum(t for _, t in combo),
                             sum(machines[m]['cost'] for m in machine_set) + station_cost,
                             tuple((i, m, t) for i, (m, t) in zip(batch, combo))))
            rows.sort(key=lambda r: (r[0], r[1]))
            pareto, best = [], float('inf')       # load up, cost strictly down
            for r in rows:
                if r[1] < best - 1e-9:
                    pareto.append(r)
                    best = r[1]
            edges.setdefault(a, []).append((b, pareto))
    return ideals, edges, (1 << len(ids)) - 1


def _dp_paths(ideals, edges, full, n_max, cycle):
    """Cheapest cost to finish with exactly k stations, all loads <= cycle."""
    from bisect import bisect_right
    INF = float('inf')
    dist = [{a: INF for a in ideals} for _ in range(n_max + 1)]
    parent = [{} for _ in range(n_max + 1)]
    dist[0][0] = 0.0
    for k in range(1, n_max + 1):
        for a in ideals:
            base = dist[k - 1][a]
            if base == INF:
                continue
            for b, pareto in edges.get(a, ()):
                pos = bisect_right([r[0] for r in pareto], cycle + 1e-9) - 1
                if pos < 0:
                    continue
                row = pareto[pos]               # cheapest option that fits
                if base + row[1] < dist[k][b] - 1e-9:
                    dist[k][b] = base + row[1]
                    parent[k][b] = (a, row)
    return dist, parent


def _dp_design(options, tasks, machines, n_max, budget, station_cost, line_cost,
               tiebreak):
    ideals, edges, full = _build_edges(options, tasks, machines, station_cost)
    candidates = sorted({round(r[0], 9) for lst in edges.values()
                         for _, pareto in lst for r in pareto})

    def best_for(cycle):
        dist, parent = _dp_paths(ideals, edges, full, n_max, cycle)
        ok = [(k, dist[k][full] + line_cost) for k in range(1, n_max + 1)
              if dist[k][full] + line_cost <= budget + 1e-6]
        return ok, parent

    ok, _ = best_for(candidates[-1])
    if not ok:
        return None, 'Infeasible'
    lo, hi = 0, len(candidates) - 1             # smallest feasible candidate
    while lo < hi:
        mid = (lo + hi) // 2
        if best_for(candidates[mid])[0]:
            hi = mid
        else:
            lo = mid + 1
    cycle = candidates[lo]
    ok, parent = best_for(cycle)
    key = (lambda x: (x[0], x[1])) if tiebreak == 'stations' else (lambda x: (x[1], x[0]))
    k, _ = min(ok, key=key)
    stations, node = [], full
    for step in range(k, 0, -1):
        a, row = parent[step][node]
        stations.append(list(row[2]))
        node = a
    stations.reverse()
    return _summarise(stations, machines, station_cost, line_cost), 'Optimal'


def _dp_min_cost(options, tasks, machines, n_max, station_cost, line_cost):
    ideals, edges, full = _build_edges(options, tasks, machines, station_cost)
    slowest = max(r[0] for lst in edges.values() for _, pareto in lst for r in pareto)
    dist, _ = _dp_paths(ideals, edges, full, n_max, slowest)
    return min(dist[k][full] for k in range(1, n_max + 1)) + line_cost


# --------------------------------------------------------------------------
# Exact method 2: MILP (any precedence graph; slower)
# --------------------------------------------------------------------------
def _balance_model(options, tasks, machines, n_stations, budget, station_cost,
                   line_cost, objective, cycle_cap=None, stations_cap=None,
                   cost_cap=None):
    """MILP: assign tasks to stations and pick a machine for every task.

    objective: 'cycle' | 'stations' | 'cost'
    Variables  w[i,m,s]  task i on machine type m at station s
               u[m,s]    station s holds a machine of type m
               y[s]      station s is used           C  cycle time
    """
    prob = pulp.LpProblem("Budget_Line_Balancing", pulp.LpMinimize)
    nv = lambda name, **kw: new_var(prob, name, **kw)
    S = list(range(1, n_stations + 1))
    types = list(machines)
    mid = {m: k for k, m in enumerate(types)}

    w = {(i, m, s): nv(f"w_{i}_{mid[m]}_{s}", cat='Binary')
         for i in options for m, _ in options[i] for s in S}
    u = {(m, s): nv(f"u_{mid[m]}_{s}", cat='Binary') for m in types for s in S}
    y = {s: nv(f"y_{s}", cat='Binary') for s in S}
    C = nv("C", low=0)

    cost = (pulp.lpSum(machines[m]['cost'] * u[m, s] for m in types for s in S)
            + station_cost * pulp.lpSum(y[s] for s in S) + line_cost)
    n_used = pulp.lpSum(y[s] for s in S)

    if objective == 'cycle':
        prob += pulp.lpSum([C])
    elif objective == 'stations':
        prob += n_used
    elif objective == 'cost':
        prob += cost
    else:
        raise ValueError(objective)

    for i in options:                                       # each task once
        prob += pulp.lpSum(w[i, m, s] for m, _ in options[i] for s in S) == 1
    for (i, m, s), var in w.items():                        # machine present
        prob += var - u[m, s] <= 0
    for s in S:
        load = pulp.lpSum(t * w[i, m, s] for i in options for m, t in options[i])
        prob += load - C <= 0                               # cycle time
        for m in types:
            prob += u[m, s] - y[s] <= 0
        prob += (pulp.lpSum(w[i, m, s] for i in options for m, _ in options[i])
                 - len(options) * y[s]) <= 0                # tasks only at used stations
    for s in S[:-1]:
        prob += y[s + 1] - y[s] <= 0                        # use stations in order
    for i, d in tasks.items():                              # precedence
        for j in d['preds']:
            prob += (pulp.lpSum(s * w[j, m, s] for m, _ in options[j] for s in S)
                     - pulp.lpSum(s * w[i, m, s] for m, _ in options[i] for s in S)) <= 0
    if budget is not None:
        prob += cost - budget <= 0
    if cycle_cap is not None:
        prob += C - cycle_cap <= 0
    if stations_cap is not None:
        prob += n_used - stations_cap <= 0
    if cost_cap is not None:
        prob += cost - cost_cap <= 0
    return prob, dict(w=w, u=u, y=y, C=C, S=S)


def _solve_stage(options, tasks, machines, n_stations, budget, station_cost,
                 line_cost, objective, time_limit, **caps):
    prob, v = _balance_model(options, tasks, machines, n_stations, budget,
                             station_cost, line_cost, objective, **caps)
    ok, _ = solve_model(prob, get_solver(time_limit, gap_rel=1e-7))
    return v if ok else None


def _read_design(v, options, machines, station_cost, line_cost):
    stations = []
    for s in v['S']:
        placed = [(i, m, t) for i in options for m, t in options[i]
                  if round(v['w'][i, m, s].value() or 0) == 1]
        if placed:
            stations.append(placed)
    return _summarise(stations, machines, station_cost, line_cost)


def _milp_design(options, tasks, machines, n_max, budget, station_cost, line_cost,
                 tiebreak, time_limit):
    """Lexicographic MILP: (1) minimum cycle time, (2)+(3) tie-break."""
    args = (options, tasks, machines, n_max, budget, station_cost, line_cost)
    v1 = _solve_stage(*args, 'cycle', time_limit)
    if v1 is None:
        cheapest = _solve_stage(options, tasks, machines, n_max, None, station_cost,
                                line_cost, 'cost', time_limit)
        if cheapest is None:
            return None, 'Failed'
        design = _read_design(cheapest, options, machines, station_cost, line_cost)
        return (None, 'Infeasible') if design['install_cost'] > budget else (None, 'Failed')
    c_star = v1['C'].value() + 1e-6
    first, second = ('stations', 'cost') if tiebreak == 'stations' else ('cost', 'stations')
    v2 = _solve_stage(*args, first, time_limit, cycle_cap=c_star)
    if v2 is None:
        return _read_design(v1, options, machines, station_cost, line_cost), 'Optimal'
    caps = {'cycle_cap': c_star}
    if first == 'stations':
        caps['stations_cap'] = round(sum(v2['y'][s].value() for s in v2['S']))
    else:
        caps['cost_cap'] = _read_design(v2, options, machines, station_cost,
                                        line_cost)['install_cost'] + 1e-6
    v3 = _solve_stage(*args, second, time_limit, **caps)
    chosen = v3 if v3 is not None else v2
    return _read_design(chosen, options, machines, station_cost, line_cost), 'Optimal'


def build_machines(rows, power_kw=0.4, consumables_monthly=250.0):
    """Full machine records from (name, purchase, setup, speed) rows.

    The overhead calculation needs purchase cost, power and consumables as well
    as cost and speed.  Machines whose default power is zero (the manual
    station) stay unpowered whatever power_kw is.
    """
    out = {}
    for name, purchase, setup, speed in rows:
        powered = MACHINES[name].get('power_kw', 0.0) > 0
        out[name] = {'cost': purchase + setup, 'purchase_cost': purchase,
                     'setup_cost': setup, 'speed': speed,
                     'power_kw': power_kw if powered else 0.0,
                     'consumables_monthly': consumables_monthly if powered else 0.0}
    return out


def check_design(design, machines=None, tasks=None, operation_machines=None):
    """Internal-consistency check of a design; returns a list of problems.

    Every task must sit at exactly one station on a machine that its operation
    allows, the station must hold that machine, machine counts must equal the
    number of (station, type) pairs, and station loads must respect the cycle
    time.  An empty list means consistent.
    """
    machines = MACHINES if machines is None else machines
    tasks = TASKS if tasks is None else tasks
    op_map = OPERATION_MACHINES if operation_machines is None else operation_machines
    problems = []
    seen = [t for ts in design['assignment'].values() for t in ts]
    if sorted(seen) != sorted(tasks):
        problems.append("tasks are not each assigned exactly once")
    pos = {t: s for s, ts in design['assignment'].items() for t in ts}
    for i, d in tasks.items():
        m = design['task_machine'].get(i)
        if m not in op_map.get(d['op'], []):
            problems.append(f"task {i} uses ineligible machine {m!r}")
        elif m not in design['station_machines'].get(pos.get(i), []):
            problems.append(f"task {i}: station {pos.get(i)} does not hold {m!r}")
        for p in d['preds']:
            if pos.get(p, 0) > pos.get(i, 0):
                problems.append(f"precedence {p}->{i} violated")
    counts = {}
    for ms in design['station_machines'].values():
        for m in ms:
            counts[m] = counts.get(m, 0) + 1
    if counts != design['machine_counts']:
        problems.append("machine counts differ from the station machine lists")
    if max(design['station_loads']) > design['cycle_time'] + 1e-6:
        problems.append("a station load exceeds the cycle time")
    cost = sum(machines[m]['cost'] * n for m, n in counts.items())
    if abs(cost - design['cost_breakdown']['machines']) > 1e-6:
        problems.append("machine cost does not match machine counts")
    return problems


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def minimum_budget(machines=None, station_cost=0.0, line_cost=0.0, tasks=None,
                   operation_machines=None, time_limit=20):
    """Cheapest line that can run at all (any cycle time, any station count)."""
    machines, tasks, options = _prepare(machines, tasks, operation_machines)
    try:
        return _dp_min_cost(options, tasks, machines, len(tasks), station_cost, line_cost)
    except _TooLarge:
        v = _solve_stage(options, tasks, machines, len(tasks), None, station_cost,
                         line_cost, 'cost', time_limit)
        return None if v is None else _read_design(
            v, options, machines, station_cost, line_cost)['install_cost']


def design_line(budget, machines=None, max_stations=None, station_cost=0.0,
                line_cost=0.0, tiebreak='stations', method='auto',
                days=26, hours=8, uptime=0.75, time_limit=30,
                tasks=None, operation_machines=None, operating_hours=None,
                electricity_tariff=None, maintenance_rate=None):
    """Minimise the cycle time of ONE dedicated line under a capital budget.

    budget        maximum capital cost of the line (Rs)
    machines      {type: {'cost': Rs, 'speed': x}} (default MACHINES)
    max_stations  upper bound on stations / operators (default: number of tasks)
    station_cost  extra cost per used station (table, trolley, lighting ...)
    line_cost     fixed cost of the line (installation, power, layout ...)
    tiebreak      among designs with the optimal cycle time pick the one with
                  the fewest 'stations' (default, saves labour) or the lowest 'cost'
    operating_hours, electricity_tariff, maintenance_rate
                  overhead assumptions (defaults: OVERHEAD_ASSUMPTIONS)
    method        'auto' (DP when the precedence graph allows, else MILP),
                  'dp' (exact, milliseconds) or 'milp' (exact, any graph, slower)

    Returns a dict.  status is 'Optimal', 'Infeasible' (budget too small; see
    min_budget_needed) or 'Failed' (MILP did not finish).  On success it holds
    cycle_time, stations (= L_line), install_cost (= cost_install) and cap_line
    plus the station assignment, machines and station loads.
    """
    started = time.perf_counter()
    machines, tasks, options = _prepare(machines, tasks, operation_machines)
    n_max = min(int(max_stations or len(tasks)), len(tasks))
    floor = cycle_time_floor(options)
    if tiebreak not in ('stations', 'cost'):
        raise ValueError("tiebreak must be 'stations' or 'cost'")
    if method not in ('auto', 'dp', 'milp'):
        raise ValueError("method must be 'auto', 'dp' or 'milp'")

    used, design, status = method, None, 'Failed'
    if method in ('auto', 'dp'):
        try:
            design, status = _dp_design(options, tasks, machines, n_max, budget,
                                        station_cost, line_cost, tiebreak)
            used = 'dp'
        except _TooLarge:
            if method == 'dp':
                raise ValueError("Precedence graph too wide for the DP; use method='milp'.")
            used = 'milp'
    if used == 'milp':
        design, status = _milp_design(options, tasks, machines, n_max, budget,
                                      station_cost, line_cost, tiebreak, time_limit)

    info = {'status': status, 'method': used, 'budget': budget,
            'cycle_time_floor': floor, 'solve_seconds': time.perf_counter() - started}
    if design is None:
        if status == 'Infeasible':
            info['min_budget_needed'] = minimum_budget(
                machines, station_cost, line_cost, tasks, operation_machines, time_limit)
        return info
    design.update(info)
    design['cap_line'] = math.floor(
        line_capacity(design['cycle_time'], days, hours, uptime) + 1e-9)
    design['budget_used_pct'] = 100 * design['install_cost'] / budget if budget else 0.0
    design['at_cycle_floor'] = design['cycle_time'] <= floor + 1e-6
    # Monthly overhead of ONE installed line, from the machines it contains.
    overhead = calculate_monthly_overhead(
        design['machine_counts'], machines=machines,
        operating_hours=operating_hours, electricity_tariff=electricity_tariff,
        maintenance_rate=maintenance_rate)
    design['monthly_overhead'] = overhead
    # FULL-CAPACITY estimate only (line running flat out).  The overhead per
    # unit at actual production is computed from the MILP output.
    design['overhead_per_unit_at_capacity'] = calculate_overhead_per_unit(
        overhead['total_monthly_overhead'], design['cap_line'])
    design['machine_check'] = check_design(design, machines, tasks, operation_machines)
    return design


def design_frontier(budgets, **kwargs):
    """Design the line for several budgets (diminishing returns of more capital)."""
    rows = []
    for b in budgets:
        d = design_line(b, **kwargs)
        row = {'budget': b, 'status': d['status']}
        if d['status'] == 'Optimal':
            row.update(cycle_time=d['cycle_time'], stations=d['stations'],
                       install_cost=d['install_cost'], cap_line=d['cap_line'],
                       monthly_overhead=d['monthly_overhead']['total_monthly_overhead'],
                       overhead_per_unit_at_capacity=d['overhead_per_unit_at_capacity'],
                       machine_counts=d['machine_counts'],
                       at_cycle_floor=d['at_cycle_floor'])
        rows.append(row)
    return rows


if __name__ == "__main__":
    ct = 0.45
    total = sum(d['time'] for d in TASKS.values())
    print(f"Work content {total:.2f} min, theoretical minimum stations "
          f"{math.ceil(total / ct)}")
    st, eff = rpw_heuristic(ct)
    print(f"RPW   : {len(st)} stations, efficiency {eff:.2%}")
    n, eff, asg = exact_station_assignment(ct)
    print(f"Exact : {n} stations, efficiency {eff:.2%}")
    print(f"Line capacity at cycle {ct}: {line_capacity(ct):,.0f} units/month")

    print("\nBudget-driven design (sourced machine data)")
    for budget in (700_000, 1_000_000, 1_500_000):
        d = design_line(budget)
        if d['status'] != 'Optimal':
            print(f"  budget Rs {budget:,}: {d['status']}, need Rs {d.get('min_budget_needed') or 0:,.0f}")
        else:
            print(f"  budget Rs {budget:,}: cycle {d['cycle_time']:.3f} min, "
                  f"{d['stations']} stations, cost Rs {d['install_cost']:,.0f}, "
                  f"{d['cap_line']:,} units/month ({d['solve_seconds']:.2f}s)")
            oh = d['monthly_overhead']
            print(f"      machines {d['machine_counts']}")
            print(f"      overhead/month Rs {oh['total_monthly_overhead']:,.2f} "
                  f"(electricity {oh['electricity_monthly']:,.2f}, maintenance "
                  f"{oh['maintenance_monthly']:,.2f}, consumables "
                  f"{oh['consumables_monthly']:,.2f}); "
                  f"Rs {d['overhead_per_unit_at_capacity']:.3f}/unit at FULL capacity")
