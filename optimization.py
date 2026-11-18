"""Layout-transition MILP: process layout -> dedicated T-shirt line.

Works with PuLP 2.x, 3.x and 4.x.  PuLP 4.x no longer bundles CBC and creates
variables through model.add_variable, so install a solver once:

    pip install pulp highspy

Public functions (same names as before)
    generate_demand, solve_milp, calc_break_even, solve_rolling_horizon
New helpers
    default_params, solve_rolling_horizon_detail, calc_break_even_detail
    apply_line_design   feed a balanced-line design (cycle time, stations,
                        install cost, machine overhead) into cap_line, L_line,
                        cost_install and cost_run
    overhead_schedule   month-by-month machine overhead and overhead per unit
                        at ACTUAL dedicated production, from a solved plan

Cost structure of one dedicated line (kept separate, never double counted)
    cost_install  one-off, paid when a line is installed (machines + setup)
    cost_run      recurring, per INSTALLED line per month: machine overhead
                  (electricity + maintenance + consumables) computed from the
                  line design's machine counts + cost_run_other (default 0)
    cost_line_labour  recurring payroll per installed line = operators x the
                  stitching wage (derived from `wage`; pass a number to override)
    cost_remove   one-off, paid when a line is removed
The objective term (cost_run + cost_line_labour) * N[t] charges each of the N[t]
installed lines exactly once per month, so machine overhead scales with the
number of lines and appears in no other term.
"""
import math

import numpy as np
import pulp

PRODUCTS = ['A', 'B', 'C', 'D']
DEPTS = ['Cutting', 'Stitching', 'Finishing']
HORIZON = 36
POLICIES = ("Optimized", "Always-Process", "Dedicated-From-Start")

WAGES = {'Cutting': 22_411,        # fabric cutter, skilled
         'Stitching': 20_371,      # single-operation sewing operator, semi-skilled
         'Finishing': 18_456}      # trimming / packing helper, unskilled (Rs per month)
WORKING_DAYS = 26
HOURS_PER_DAY = 8
PAID_MINUTES = WORKING_DAYS * HOURS_PER_DAY * 60
WORKER_EFFICIENCY = 0.50
PRODUCTIVE_MINUTES_PER_WORKER = PAID_MINUTES * WORKER_EFFICIENCY
BASE_WORKERS = {'Cutting': 6, 'Stitching': 24, 'Finishing': 6}


# --------------------------------------------------------------------------
# PuLP compatibility helpers
# --------------------------------------------------------------------------
def get_solver(time_limit=None, msg=False):
    """Return an available solver (HiGHS first, then CBC)."""
    available = pulp.listSolvers(onlyAvailable=True)
    if 'HiGHS' in available:
        return pulp.HiGHS(msg=msg, timeLimit=time_limit)
    if 'HiGHS_CMD' in available:
        return pulp.HiGHS_CMD(msg=msg, timeLimit=time_limit)
    if 'PULP_CBC_CMD' in available:
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    raise RuntimeError("No MILP solver found. Install one with:  "
                       "pip install highspy")


def new_var(model, name, low=None, up=None, cat='Continuous'):
    """Create a variable on any PuLP version."""
    if hasattr(model, 'add_variable'):
        return model.add_variable(name, lowBound=low, upBound=up, cat=cat)
    return pulp.LpVariable(name, lowBound=low, upBound=up, cat=cat)


def solve_model(model, solver):
    """Solve and return (is_optimal, status_code) on any PuLP version."""
    res = model.solve(solver)
    code = getattr(res, 'status', None)           # PuLP 4.x: LpSolveStats
    if code is None:
        code = getattr(model, 'status', res)      # PuLP 2.x / 3.x: int
    code = int(getattr(code, 'value', code))
    return code == 1, code


def val(var):
    """Numeric value of a variable (0.0 if the solver left it unset)."""
    v = var.value()
    return 0.0 if v is None else float(v)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def generate_demand(base_A=2000, max_A=20000, volatility=0.0, seed=None):
    """Demand trajectory for 36 months. Product A ramps linearly.

    volatility > 0 adds N(0, volatility * trend) noise to product A only.
    """
    rng = np.random.default_rng(seed)
    T = range(1, HORIZON + 1)
    demand = {p: {} for p in PRODUCTS}
    for t in T:
        trend = base_A + (max_A - base_A) / (HORIZON - 1) * (t - 1)
        noise = rng.normal(0, volatility * trend) if volatility > 0 else 0.0
        demand['A'][t] = max(0.0, trend + noise)
        demand['B'][t] = 1000.0
        demand['C'][t] = 800.0
        demand['D'][t] = 1200.0
    return demand


def default_params(demand=None):
    """Authoritative factory baseline and model inputs."""
    productive = PRODUCTIVE_MINUTES_PER_WORKER
    return {
        'demand': demand if demand is not None else generate_demand(),
        'wage': dict(WAGES), 'paid_minutes': PAID_MINUTES,
        'worker_efficiency': WORKER_EFFICIENCY,
        'productive_minutes_per_worker': productive,
        'base_workers': dict(BASE_WORKERS),
        'cost_nonlabour': {'A': 131, 'B': 223, 'C': 245, 'D': 213},
        'cost_nonlabour_ded_A': 121,
        'cap_shared': {d: BASE_WORKERS[d] * productive for d in DEPTS},
        'cap_line': 20800, 'cost_install': 1_200_000, 'cost_remove': 100_000,
        'cost_run': 10_000, 'cost_run_other': 0, 'line_overhead': None,
        'cost_line_labour': None,
        'cost_exp': {'Cutting': 50_000, 'Stitching': 0, 'Finishing': 40_000},
        'cost_exp_run': None,
        'ops_per_module': {'Cutting': 3, 'Stitching': 0, 'Finishing': 3},
        'cost_exp_upkeep': {'Cutting': 2500, 'Stitching': 0, 'Finishing': 1500},
        'cost_term': {'Cutting': 10000, 'Stitching': 0, 'Finishing': 10000},
        'cap_add': {'Cutting': 15000, 'Stitching': 0, 'Finishing': 15000},
        'M_max': {'Cutting': 5, 'Stitching': 0, 'Finishing': 5},
        'cost_hold': {p: 5 for p in PRODUCTS},
        'cost_short': {p: 500 for p in PRODUCTS},
        'N_max': 3, 'B_t': 10_000_000,
        'std_times': {
            'A': {'Cutting': .90, 'Stitching': 3.46, 'Finishing': 1.64},
            'B': {'Cutting': 1.27, 'Stitching': 7.43, 'Finishing': 2.51},
            'C': {'Cutting': 1.84, 'Stitching': 12.56, 'Finishing': 3.32},
            'D': {'Cutting': 1.54, 'Stitching': 10.93, 'Finishing': 3.33}},
        'allow_expansion': False, 'labour_coupling': True, 'L_line': 11,
        'ops_shared_stitch': 24, 'process_efficiency': .50,
        'ded_uses_shared': True, 'expansion_requires_A_shortage': True,
        'cost_layoff': 0,
    }


def apply_line_design(params, design, days=26, hours=8, uptime=0.75):
    """Override the dedicated-line inputs with a balanced-line design.

    `design` is the dict returned by line_balancing.design_line().  The three
    strategic inputs are replaced before the 36-month MILP is built:

        cap_line     = days x hours x 60 x uptime / cycle_time   (whole units)
        cost_install = actual cost of the machines (and station / line costs)
        L_line       = number of stations (one operator per station)

    The station loads are also stored so the simulation uses the same line.
    Returns a new params dict; the input is not modified.
    """
    if design.get('status') != 'Optimal':
        raise ValueError("Cannot apply a line design with status "
                         f"{design.get('status')!r}.")
    out = dict(params)
    out['cap_line'] = math.floor(
        days * hours * 60 * uptime / design['cycle_time'] + 1e-9)
    out['cost_install'] = design['install_cost']
    out['L_line'] = int(design['stations'])
    out['station_loads'] = list(design['station_loads'])
    # Recurring machine overhead of one installed line, from its machine counts.
    # It replaces the manual cost_run; cost_run_other (default 0) is added on top.
    oh = design.get('monthly_overhead')
    if oh is not None:
        other = float(out.get('cost_run_other') or 0.0)
        out['cost_run'] = oh['total_monthly_overhead'] + other
        out['line_overhead'] = {
            'electricity': oh['electricity_monthly'],
            'maintenance': oh['maintenance_monthly'],
            'consumables': oh['consumables_monthly'],
            'machine_total': oh['total_monthly_overhead'],
            'other': other,
            'machine_counts': dict(oh['machine_counts']),
            'powered_machine_count': oh['powered_machine_count'],
            'purchase_value': oh['purchase_value'],
            'installed_machine_value': oh['installed_machine_value'],
        }
    out['line_design'] = {k: design[k] for k in
                          ('cycle_time', 'stations', 'install_cost', 'budget',
                           'balance_efficiency', 'machine_counts') if k in design}
    return out


def line_design_warnings(params):
    cfg = _cfg(params)
    notes = []
    if cfg['L_line'] * cfg['N_max'] > cfg['ops_shared_stitch']:
        notes.append(
            f"At most {cfg['ops_shared_stitch']} existing stitching operators can be "
            "transferred. Additional dedicated-line operators must be hired.")
    if cfg.get('line_overhead') is None:
        notes.append("Monthly machine overhead is a manual figure because no budget-driven line design is applied.")
    if cfg['cost_install'] > cfg['B_t']:
        notes.append("One line costs more than the cumulative capital budget.")
    return notes


def derive_labour_costs(cfg):
    """Derive module payroll; labour is charged as monthly workforce payroll."""
    workers = {d: int(math.ceil(cfg['cap_add'][d] /
                 cfg['productive_minutes_per_worker'])) if cfg['cap_add'][d] > 0 else 0
               for d in DEPTS}
    running = {d: workers[d] * cfg['wage'][d] + cfg['cost_exp_upkeep'][d]
               for d in DEPTS}
    return {'rate_per_minute': {d: cfg['wage'][d] / cfg['paid_minutes'] for d in DEPTS},
            'labour_per_garment': {p: 0.0 for p in PRODUCTS},
            'cost_proc': dict(cfg['cost_nonlabour']),
            'cost_ded_A': cfg['cost_nonlabour_ded_A'],
            'cost_line_labour': 0.0, 'cost_exp_run': running,
            'expansion_workers': workers}


def _cfg(params):
    """Fill defaults and derive worker-based capacity and payroll."""
    params = dict(params or {})
    cfg = default_params(params.get('demand'))
    cfg.update({k: v for k, v in params.items() if v is not None})
    cfg['productive_minutes_per_worker'] = cfg['paid_minutes'] * cfg['worker_efficiency']
    cfg['cap_shared'] = {
        'Cutting': cfg['base_workers']['Cutting'] * cfg['productive_minutes_per_worker'],
        'Stitching': cfg['ops_shared_stitch'] * cfg['productive_minutes_per_worker'],
        'Finishing': cfg['base_workers']['Finishing'] * cfg['productive_minutes_per_worker']}
    derived = derive_labour_costs(cfg)
    cfg['cost_proc'] = dict(cfg['cost_nonlabour'])
    cfg['cost_ded_A'] = cfg['cost_nonlabour_ded_A']
    # Retained only as a zero-valued compatibility field. Actual operator payroll
    # is represented by incremental hire variables; transfers stay in baseline payroll.
    cfg['cost_line_labour'] = 0.0
    cfg['cost_exp_run'] = derived['cost_exp_run']
    cfg['expansion_workers'] = derived['expansion_workers']
    for key in ('M_max', 'cap_add', 'cost_exp', 'cost_term', 'ops_per_module'):
        cfg[key]['Stitching'] = 0
    cfg['cost_exp_run']['Stitching'] = 0
    return cfg


# Model construction (shared by the full solve and the rolling horizon)
# --------------------------------------------------------------------------
def build_model(cfg, months, demand, policy="Optimized",
                N0=0, M0=None, Inv0=None, spent0=0.0,
                HLine0=0, HShared0=0):
    """Build monthly MILP with explicit staffing, transfers and worker capacity."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    P, D = PRODUCTS, DEPTS
    M0, Inv0 = M0 or {d: 0 for d in D}, Inv0 or {p: 0.0 for p in P}
    first, last = months[0], months[-1]
    is_final = last == HORIZON
    model = pulp.LpProblem(f"Layout_Transition_{policy}", pulp.LpMinimize)
    nv = lambda name, **kw: new_var(model, name, **kw)

    N = {t: nv(f"N_{t}", low=0, up=cfg['N_max'], cat='Integer') for t in months}
    Inst = {t: nv(f"Install_{t}", low=0, cat='Integer') for t in months}
    Rem = {t: nv(f"Remove_{t}", low=0, cat='Integer') for t in months}
    Exp = {(d,t): nv(f"Expand_{d}_{t}", low=0, up=cfg['M_max'][d], cat='Integer')
           for d in D for t in months}
    M = {(d,t): nv(f"M_{d}_{t}", low=0, up=cfg['M_max'][d], cat='Integer')
         for d in D for t in months}
    Qp = {(p,t): nv(f"Qproc_{p}_{t}", low=0) for p in P for t in months}
    Qd = {(p,t): nv(f"Qded_{p}_{t}", low=0) for p in P for t in months}
    Inv = {(p,t): nv(f"Inv_{p}_{t}", low=0) for p in P for t in months}
    Sh = {(p,t): nv(f"Short_{p}_{t}", low=0) for p in P for t in months}
    T = {t: nv(f"Transfer_{t}", low=0, up=cfg['ops_shared_stitch'], cat='Integer')
         for t in months}
    HL = {t: nv(f"HiredLine_{t}", low=0, cat='Integer') for t in months}
    HireL = {t: nv(f"HireLine_{t}", low=0, cat='Integer') for t in months}
    LayoffL = {t: nv(f"LayoffLine_{t}", low=0, cat='Integer') for t in months}
    HS = {t: nv(f"HiredShared_{t}", low=0, cat='Integer') for t in months}
    HireS = {t: nv(f"HireShared_{t}", low=0, cat='Integer') for t in months}
    Aexp = {t: nv(f"AExpansionTrigger_{t}", cat='Binary') for t in months}

    base_payroll = sum(cfg['base_workers'][d] * cfg['wage'][d] for d in D)
    obj = 0
    for t in months:
        obj += pulp.lpSum(cfg['cost_proc'][p] * Qp[p,t] for p in P)
        obj += cfg['cost_ded_A'] * Qd['A',t]
        obj += cfg['cost_install'] * Inst[t] + cfg['cost_remove'] * Rem[t]
        obj += cfg['cost_run'] * N[t]
        obj += cfg['wage']['Stitching'] * (HL[t] + HS[t])
        obj += cfg['cost_layoff'] * LayoffL[t]
        obj += pulp.lpSum(cfg['cost_hold'][p] * Inv[p,t] +
                          cfg['cost_short'][p] * Sh[p,t] for p in P)
        obj += pulp.lpSum(cfg['cost_exp'][d] * Exp[d,t] +
                          cfg['cost_exp_run'][d] * M[d,t] for d in D)
        obj += base_payroll
    if is_final:
        obj += cfg['cost_remove'] * N[last]
        obj += pulp.lpSum(cfg['cost_term'][d] * M[d,last] for d in D)
    model += obj

    for t in months:
        prev_N = N0 if t == first else N[t-1]
        prev_HL = HLine0 if t == first else HL[t-1]
        prev_HS = HShared0 if t == first else HS[t-1]
        model += N[t] - prev_N - Inst[t] + Rem[t] == 0
        model += HL[t] - prev_HL - HireL[t] + LayoffL[t] == 0
        model += HS[t] - prev_HS - HireS[t] == 0
        model += T[t] + HL[t] == cfg['L_line'] * N[t]
        model += T[t] <= cfg['ops_shared_stitch']
        model += LayoffL[t] <= prev_HL + HireL[t]
        if policy == "Always-Process":
            model += N[t] == 0
            model += T[t] == 0
            model += HL[t] == 0
        elif policy == "Dedicated-From-Start":
            required = min(cfg['N_max'], max(1, math.ceil(
                demand['A'][t] / cfg['cap_line'])))
            model += N[t] == required

        for d in D:
            prev_M = M0[d] if t == first else M[d,t-1]
            model += M[d,t] - prev_M - Exp[d,t] == 0
            if not cfg['allow_expansion'] or d == 'Stitching':
                model += Exp[d,t] == 0

        for p in P:
            prev_inv = Inv0[p] if t == first else Inv[p,t-1]
            model += Inv[p,t] - prev_inv - Qp[p,t] - Qd[p,t] + demand[p][t] - Sh[p,t] == 0
            if p != 'A':
                model += Qd[p,t] == 0
        model += Qd['A',t] <= cfg['cap_line'] * N[t]

        # A shortage is the trigger for expansion; cost minimisation decides
        # whether expansion is worth buying. B/C/D shortages cannot trigger it.
        model += Sh['A',t] <= demand['A'][t] * Aexp[t]
        model += Sh['A',t] >= 0.001 * Aexp[t]
        for d in ('Cutting','Finishing'):
            model += Exp[d,t] <= cfg['M_max'][d] * Aexp[t]

        for d in D:
            work = pulp.lpSum(cfg['std_times'][p][d] * Qp[p,t] for p in P)
            cap = cfg['cap_shared'][d] + cfg['cap_add'][d] * M[d,t]
            if d == 'Stitching':
                cap = (cfg['ops_shared_stitch'] - T[t] + HS[t]) * cfg['productive_minutes_per_worker']
            elif cfg['ded_uses_shared']:
                work += cfg['std_times']['A'][d] * Qd['A',t]
            model += work <= cap

        spent = spent0 + pulp.lpSum(
            cfg['cost_install'] * Inst[tau] +
            pulp.lpSum(cfg['cost_exp'][d] * Exp[d,tau] for d in D)
            for tau in range(first,t+1))
        model += spent <= cfg['B_t']

    if is_final:
        for p in P:
            model += Inv[p,last] == 0
    vars_ = dict(N=N, Inst=Inst, Rem=Rem, Exp=Exp, M=M, Qp=Qp, Qd=Qd,
                 Inv=Inv, Sh=Sh, Transfer=T, HiredLine=HL, HireLine=HireL,
                 LayoffLine=LayoffL, HiredShared=HS, HireShared=HireS,
                 AExpansionTrigger=Aexp)
    return model, vars_


def _breakdown(cfg, v, months, is_final):
    """Cost components, including baseline and incremental workforce payroll."""
    keys = ['shared_production','dedicated_production','installation','removal',
            'line_running','line_hired_payroll','shared_hiring_payroll',
            'baseline_payroll','inventory_holding','shortage','expansion',
            'expansion_running','layoff','terminal']
    b = dict.fromkeys(keys, 0.0)
    base_payroll = sum(cfg['base_workers'][d] * cfg['wage'][d] for d in DEPTS)
    for t in months:
        b['shared_production'] += sum(cfg['cost_proc'][p] * val(v['Qp'][p,t]) for p in PRODUCTS)
        b['dedicated_production'] += cfg['cost_ded_A'] * val(v['Qd']['A',t])
        b['installation'] += cfg['cost_install'] * val(v['Inst'][t])
        b['removal'] += cfg['cost_remove'] * val(v['Rem'][t])
        b['line_running'] += cfg['cost_run'] * val(v['N'][t])
        b['line_hired_payroll'] += cfg['wage']['Stitching'] * val(v['HiredLine'][t])
        b['shared_hiring_payroll'] += cfg['wage']['Stitching'] * val(v['HiredShared'][t])
        b['baseline_payroll'] += base_payroll
        b['inventory_holding'] += sum(cfg['cost_hold'][p] * val(v['Inv'][p,t]) for p in PRODUCTS)
        b['shortage'] += sum(cfg['cost_short'][p] * val(v['Sh'][p,t]) for p in PRODUCTS)
        b['expansion'] += sum(cfg['cost_exp'][d] * val(v['Exp'][d,t]) for d in DEPTS)
        b['expansion_running'] += sum(cfg['cost_exp_run'][d] * val(v['M'][d,t]) for d in DEPTS)
        b['layoff'] += cfg['cost_layoff'] * val(v['LayoffLine'][t])
    if is_final:
        t = months[-1]
        b['terminal'] = cfg['cost_remove'] * val(v['N'][t]) + sum(
            cfg['cost_term'][d] * val(v['M'][d,t]) for d in DEPTS)
    return b


# Full 36-month solve
# --------------------------------------------------------------------------
def solve_milp(params, policy="Optimized", fixed_lines=None, time_limit=None):
    """Solve the 36-month model. Returns (result_dict, model).

    `fixed_lines` (optional): dict {month: number of active lines} that forces
    a given schedule, useful for the cost-versus-switch-month analysis.
    """
    cfg = _cfg(params)
    T = list(range(1, HORIZON + 1))
    model, v = build_model(cfg, T, cfg['demand'], policy)

    if fixed_lines is not None:
        for t, n in fixed_lines.items():
            model += v['N'][t] == n

    ok, code = solve_model(model, get_solver(time_limit))
    res = {'status': 'Optimal' if ok else f'Not Optimal (code {code})',
           'obj': None}
    if not ok:
        return res, model

    res['obj'] = pulp.value(model.objective)
    # Month-by-month cost ledger mirrors every term in the MILP objective.
    # This is used for like-for-like cash-flow payback against Always-Process.
    base_payroll = sum(cfg['base_workers'][d] * cfg['wage'][d] for d in DEPTS)
    monthly_costs = {}
    for t in T:
        row = {
            'shared_production': sum(cfg['cost_proc'][p] * val(v['Qp'][p,t]) for p in PRODUCTS),
            'dedicated_production': cfg['cost_ded_A'] * val(v['Qd']['A',t]),
            'installation': cfg['cost_install'] * val(v['Inst'][t]),
            'removal': cfg['cost_remove'] * val(v['Rem'][t]),
            'line_running': cfg['cost_run'] * val(v['N'][t]),
            'line_hired_payroll': cfg['wage']['Stitching'] * val(v['HiredLine'][t]),
            'shared_hiring_payroll': cfg['wage']['Stitching'] * val(v['HiredShared'][t]),
            'baseline_payroll': base_payroll,
            'inventory_holding': sum(cfg['cost_hold'][p] * val(v['Inv'][p,t]) for p in PRODUCTS),
            'shortage': sum(cfg['cost_short'][p] * val(v['Sh'][p,t]) for p in PRODUCTS),
            'expansion': sum(cfg['cost_exp'][d] * val(v['Exp'][d,t]) for d in DEPTS),
            'expansion_running': sum(cfg['cost_exp_run'][d] * val(v['M'][d,t]) for d in DEPTS),
            'layoff': cfg['cost_layoff'] * val(v['LayoffLine'][t]),
            'terminal': 0.0,
        }
        if t == HORIZON:
            row['terminal'] = cfg['cost_remove'] * val(v['N'][t]) + sum(
                cfg['cost_term'][d] * val(v['M'][d,t]) for d in DEPTS)
        row['total'] = sum(row.values())
        monthly_costs[t] = row

    res.update({
        'monthly_costs': monthly_costs,
        'N': {t: val(v['N'][t]) for t in T},
        'Install': {t: val(v['Inst'][t]) for t in T},
        'Remove': {t: val(v['Rem'][t]) for t in T},
        'M_stitch': {t: val(v['M']['Stitching', t]) for t in T},
        'M_cutting': {t: val(v['M']['Cutting', t]) for t in T},
        'M_finishing': {t: val(v['M']['Finishing', t]) for t in T},
        'Transferred_Stitching': {t: val(v['Transfer'][t]) for t in T},
        'Hired_Line_Operators': {t: val(v['HiredLine'][t]) for t in T},
        'Hired_Shared_Stitching': {t: val(v['HiredShared'][t]) for t in T},
        'Shared_Stitching_Operators': {
            t: cfg['ops_shared_stitch'] - val(v['Transfer'][t]) + val(v['HiredShared'][t])
            for t in T},
        'Line_Staffing_Check': {
            t: val(v['Transfer'][t]) + val(v['HiredLine'][t]) -
               cfg['L_line'] * val(v['N'][t]) for t in T},
        'M': {d: {t: val(v['M'][d, t]) for t in T} for d in DEPTS},
        'Q_ded_A': {t: val(v['Qd']['A', t]) for t in T},
        'Q_proc_A': {t: val(v['Qp']['A', t]) for t in T},
        'Shortage': {p: {t: val(v['Sh'][p, t]) for t in T} for p in PRODUCTS},
        'Inventory': {p: {t: val(v['Inv'][p, t]) for t in T} for p in PRODUCTS},
        'breakdown': _breakdown(cfg, v, T, True),
    })
    return res, model


def overhead_schedule(params, res):
    """Machine overhead per month from a solved plan (`res` from solve_milp).

    overhead = installed lines N[t] x monthly overhead of one line.  Overhead per
    unit divides by the dedicated units ACTUALLY produced that month (None when
    nothing was produced).  The split into electricity / maintenance /
    consumables is present only when a line design supplied machine counts.
    """
    cfg = _cfg(params)
    lo = cfg.get('line_overhead')
    rows = []
    for t in range(1, HORIZON + 1):
        n = res['N'][t]
        units = res['Q_ded_A'][t]
        total = cfg['cost_run'] * n
        rows.append({
            'Month': t, 'Lines': n,
            'Electricity': lo['electricity'] * n if lo else None,
            'Maintenance': lo['maintenance'] * n if lo else None,
            'Consumables': lo['consumables'] * n if lo else None,
            'Other line running': lo['other'] * n if lo else None,
            'Monthly overhead': total,
            'Dedicated units': units,
            'Overhead per unit (actual)': (total / units) if units > 1e-9 else None,
        })
    return rows


# --------------------------------------------------------------------------
# Break-even
# --------------------------------------------------------------------------
def calc_payback_from_results(baseline_result, transition_result):
    """Calculate investment payback from two solved 36-month plans.

    Compare the transition plan with the Always-Process baseline month by
    month. Operating savings include changes in production, baseline and
    incremental payroll, machine overhead, inventory, shortage penalties and
    expansion costs. Installation, removal and terminal decommissioning are
    treated as investment cash outflows. Transferred existing operators are
    not charged a second time: their wages remain in the common baseline
    payroll, while only incremental hires appear in the transition ledger.
    """
    for label, result in (("baseline", baseline_result),
                          ("transition", transition_result)):
        if result.get('status') != 'Optimal' or 'monthly_costs' not in result:
            raise ValueError(f"{label} result must be an Optimal solve with monthly_costs")

    rows = []
    cumulative_operating_savings = 0.0
    cumulative_investment = 0.0
    payback_month = None
    total_installed = 0.0
    for t in range(1, HORIZON + 1):
        b = baseline_result['monthly_costs'][t]
        tr = transition_result['monthly_costs'][t]
        baseline_operating = b['total'] - b['installation'] - b['removal'] - b['terminal']
        transition_operating = tr['total'] - tr['installation'] - tr['removal'] - tr['terminal']
        operating_saving = baseline_operating - transition_operating
        investment_outflow = tr['installation'] + tr['removal'] + tr['terminal']
        cumulative_operating_savings += operating_saving
        cumulative_investment += investment_outflow
        total_installed += tr['installation']
        net_cumulative = cumulative_operating_savings - cumulative_investment
        if payback_month is None and total_installed > 1e-9 and net_cumulative >= -1e-6:
            payback_month = t
        rows.append({
            'Month': t,
            'Baseline operating cost': baseline_operating,
            'Transition operating cost': transition_operating,
            'Net operating saving': operating_saving,
            'Investment outflow': investment_outflow,
            'Cumulative operating savings': cumulative_operating_savings,
            'Cumulative investment': cumulative_investment,
            'Cumulative net benefit': net_cumulative,
        })

    return {
        'payback_month': payback_month,
        'total_installation': total_installed,
        'total_investment_outflow': cumulative_investment,
        'cumulative_operating_savings': cumulative_operating_savings,
        'net_benefit_36m': cumulative_operating_savings - cumulative_investment,
        'monthly': rows,
        'has_installation': total_installed > 1e-9,
    }


def calc_break_even_detail(params):
    """Legacy static contribution-margin reference; not project payback.

    Kept for API compatibility. Use calc_payback_from_results() for the
    dashboard's investment payback, because that function compares complete
    solved cash flows and actual staffing decisions.
    """
    cfg = _cfg(params)
    saving = cfg['cost_proc']['A'] - cfg['cost_ded_A']
    out = {'saving_per_unit': saving}
    if saving <= 0:
        out.update(payback_units=float('inf'), monthly_volume=float('inf'),
                   dynamic_month=None)
        return out
    out['payback_units'] = cfg['cost_install'] / saving
    out['monthly_volume'] = (cfg['cost_run'] / saving) if saving > 0 else float('inf')
    out['dynamic_month'] = None
    out['note'] = ('Static reference only. Use calc_payback_from_results() '
                   'for 36-month payback based on solved baseline and transition cash flows.')
    return out


def calc_break_even(params):
    """Returns (static_break_even_units, dynamic_payback_month).

    static_break_even_units is the cumulative output needed to repay the
    installation cost (it is NOT a monthly volume).  See calc_break_even_detail
    for the monthly break-even volume as well.
    """
    d = calc_break_even_detail(params)
    return d['payback_units'], d['dynamic_month']


# --------------------------------------------------------------------------
# Rolling horizon
# --------------------------------------------------------------------------
def solve_rolling_horizon_detail(params, forecast_std=0.0, seed=0,
                                 time_limit=None):
    """Re-solve every month over the remaining horizon, commit only month t.

    Demand for month t is known; demand for later months is the true trend
    times (1 + N(0, forecast_std)).  With forecast_std = 0 the forecasts are
    perfect and the result should match the static plan.
    """
    cfg = _cfg(params)
    actual = cfg['demand']
    rng = np.random.default_rng(seed)
    T = list(range(1, HORIZON + 1))

    N_prev, M_prev = 0, {d: 0 for d in DEPTS}
    HLine_prev, HShared_prev = 0, 0
    Inv_prev = {p: 0.0 for p in PRODUCTS}
    spent = 0.0
    N_path, Inst_path, total = {}, {}, 0.0
    parts = dict.fromkeys(['shared_production', 'dedicated_production',
                           'installation', 'removal', 'line_running',
                           'line_hired_payroll', 'shared_hiring_payroll',
                           'baseline_payroll', 'inventory_holding', 'shortage',
                           'expansion', 'expansion_running', 'layoff', 'terminal'], 0.0)

    for t in T:
        window = list(range(t, HORIZON + 1))
        fc = {p: {tau: (actual[p][tau] if tau == t else
                        max(0.0, actual[p][tau] * (1 + rng.normal(0, forecast_std))))
                  for tau in window} for p in PRODUCTS}
        model, v = build_model(cfg, window, fc, "Optimized",
                               N0=N_prev, M0=M_prev, Inv0=Inv_prev,
                               spent0=spent, HLine0=HLine_prev, HShared0=HShared_prev)
        ok, code = solve_model(model, get_solver(time_limit))
        if not ok:
            raise RuntimeError(f"Rolling horizon failed in month {t} "
                               f"(status code {code})")

        # realised cost of month t's decisions
        b = _breakdown(cfg, v, [t], is_final=(t == HORIZON))
        for k, x in b.items():
            parts[k] += x
        # carry the state forward
        N_prev = int(round(val(v['N'][t])))
        HLine_prev = int(round(val(v['HiredLine'][t])))
        HShared_prev = int(round(val(v['HiredShared'][t])))
        M_prev = {d: int(round(val(v['M'][d, t]))) for d in DEPTS}
        Inv_prev = {p: val(v['Inv'][p, t]) for p in PRODUCTS}
        spent += (cfg['cost_install'] * val(v['Inst'][t])
                  + sum(cfg['cost_exp'][d] * val(v['Exp'][d, t]) for d in DEPTS))
        N_path[t] = N_prev
        Inst_path[t] = val(v['Inst'][t])

    return {'N': N_path, 'Install': Inst_path,
            'total_cost': sum(parts.values()), 'breakdown': parts}


def solve_rolling_horizon(params, forecast_std=0.0, seed=0):
    """Backward-compatible wrapper: returns (active_lines, installs)."""
    r = solve_rolling_horizon_detail(params, forecast_std, seed)
    return r['N'], r['Install']


# --------------------------------------------------------------------------
if __name__ == "__main__":
    params = {'demand': generate_demand()}
    try:                                   # use the budget-driven line design
        from line_balancing import design_line
        params = apply_line_design(params, design_line(700_000))
        print("Line design applied: cap_line", params['cap_line'],
              "L_line", params['L_line'], "install", params['cost_install'],
              "overhead/line/month", round(params['cost_run'], 2))
    except Exception as exc:               # falls back to the legacy defaults
        print("Line design not applied:", exc)
    out = {}
    for pol in POLICIES:
        res, _ = solve_milp(params, pol)
        out[pol] = res
        print(f"{pol:22s} {res['status']:8s} total cost Rs. {res['obj']:,.0f}")
    r = out["Optimized"]
    print("Active lines by month:", [int(r['N'][t]) for t in range(1, 37)])
    for k, x in r['breakdown'].items():
        print(f"   {k:22s} {x:>16,.0f}")
    print("Break-even:", calc_break_even_detail(params))
