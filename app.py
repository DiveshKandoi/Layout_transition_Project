import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from line_balancing import build_machines as lb_build_machines
from line_balancing import (MACHINES, OVERHEAD_ASSUMPTIONS, REFERENCE_MIX,
                            calculate_monthly_overhead,
                            calculate_overhead_per_unit, design_frontier,
                            design_line, exact_station_assignment,
                            minimum_budget, reference_investment, rpw_heuristic)
from optimization import (apply_line_design, calc_payback_from_results,
                          default_params, generate_demand,
                          line_design_warnings, overhead_schedule, solve_milp,
                          solve_rolling_horizon_detail, WAGES,
                          _cfg as fill_params)
from simulation import (check_line_consistency, line_capacity_check,
                        run_replications)

st.set_page_config(page_title="Layout Transition DSS", layout="wide")
st.title("Multi-Period Layout Transition Decision Support")
st.markdown("MILP optimisation, assembly-line balancing and discrete-event "
            "simulation for a process-to-product layout transition.")



def build_machines(machines_key, overhead_key):
    """Full machine records for line balancing (cost, purchase, setup, speed,
    power, consumables). Passing only cost and speed would silently give zero
    electricity and consumables, so the whole record is always built here."""
    _, _, _, kw, consumables = overhead_key
    return lb_build_machines(machines_key, kw, consumables)


@st.cache_data(show_spinner=False)
def design_cached(budget, max_stations, station_cost, line_cost, tiebreak,
                  machines_key, overhead_key):
    """Budget-constrained line balancing (exact, milliseconds)."""
    hours, tariff, rate, _, _ = overhead_key
    return design_line(budget, machines=build_machines(machines_key, overhead_key),
                       max_stations=int(max_stations), station_cost=station_cost,
                       line_cost=line_cost, tiebreak=tiebreak,
                       operating_hours=hours, electricity_tariff=tariff,
                       maintenance_rate=rate)


@st.cache_data(show_spinner=False)
def frontier_cached(budgets, max_stations, station_cost, line_cost, tiebreak,
                    machines_key, overhead_key):
    hours, tariff, rate, _, _ = overhead_key
    return design_frontier(budgets, machines=build_machines(machines_key, overhead_key),
                           max_stations=int(max_stations),
                           station_cost=station_cost, line_cost=line_cost,
                           tiebreak=tiebreak, operating_hours=hours,
                           electricity_tariff=tariff, maintenance_rate=rate)


# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Model Parameters")

sb.subheader("Demand and strategy")
base_A = sb.number_input("Starting Product A demand", value=2000, step=500)
max_A = sb.number_input("Ending Product A demand", value=20000, step=1000)
volatility = sb.slider("Demand volatility (noise)", 0.0, 0.2, 0.0)
seed = sb.number_input("Random seed (volatility)", value=1, step=1)
budget = sb.number_input("Cumulative capital budget (Rs)", value=10_000_000,
                         step=1_000_000)

sb.subheader("Dedicated line design")
derive_line = sb.checkbox(
    "Derive line from budget and machine costs", value=True,
    help="Line balancing picks stations and machines that minimise the cycle "
         "time within the budget. Its cycle time, stations and cost then replace "
         "line output, operators per line and install cost in the optimisation.")
line_budget = sb.number_input("Capital budget for ONE line (Rs)", value=700_000,
                              step=50_000, min_value=0)
max_stations = sb.number_input("Maximum stations (operators) per line", value=14,
                               min_value=2, max_value=14, step=1)
station_cost = sb.number_input("Extra cost per station (table, trolley) (Rs)",
                               value=0, step=5_000, min_value=0)
line_fixed_cost = sb.number_input("Fixed cost per line (power, layout) (Rs)",
                                  value=0, step=10_000, min_value=0)
tiebreak_label = sb.radio("Among equal cycle times prefer",
                          ["Fewest stations", "Lowest cost"], horizontal=True)
tiebreak = 'stations' if tiebreak_label == "Fewest stations" else 'cost'
with sb.expander("Machine costs and speeds"):
    st.caption("Source: machine_info.txt (indicative supplier prices, setup "
               "assumed Rs 3,000 per machine). Installed cost = purchase + setup. "
               "Speed 1.0 for every machine because no relative speeds were "
               "supplied. The manual station has no purchase cost and no power.")
    machine_df = st.data_editor(
        pd.DataFrame({'Machine': list(MACHINES),
                      'Purchase cost': [float(v['purchase_cost']) for v in MACHINES.values()],
                      'Setup cost': [float(v['setup_cost']) for v in MACHINES.values()],
                      'Speed': [float(v['speed']) for v in MACHINES.values()]}),
        hide_index=True, disabled=['Machine'], key="machine_table")
machines_key = tuple((r.Machine, float(r._2), float(r._3), float(r.Speed))
                     for r in machine_df.itertuples())
with sb.expander("Machine overhead assumptions"):
    st.caption("Monthly overhead per line = electricity + maintenance + "
               "consumables, recalculated from the machines the design selects.")
    oh_hours = st.number_input("Operating hours per month",
                               value=float(OVERHEAD_ASSUMPTIONS['operating_hours_month']),
                               min_value=0.0, step=4.0)
    oh_kw = st.number_input("Average power per powered machine (kW)", value=0.4,
                            min_value=0.0, step=0.05)
    oh_tariff = st.number_input("Electricity tariff (Rs/kWh)",
                                value=float(OVERHEAD_ASSUMPTIONS['electricity_tariff_per_kwh']),
                                min_value=0.0, step=0.5)
    oh_maint = st.number_input("Annual maintenance (% of purchase cost)",
                               value=100 * OVERHEAD_ASSUMPTIONS['maintenance_rate_annual'],
                               min_value=0.0, step=0.5) / 100
    oh_cons = st.number_input("Consumables per powered machine per month (Rs)",
                              value=250.0, min_value=0.0, step=25.0)
overhead_key = (float(oh_hours), float(oh_tariff), float(oh_maint), float(oh_kw),
                float(oh_cons))
if any(c < 0 or su < 0 or sp <= 0 for _, c, su, sp in machines_key):
    sb.error("Machine costs must be >= 0 and speeds > 0.")
    st.stop()
MACH = build_machines(machines_key, overhead_key)
LINE_KEY = (float(line_budget), int(max_stations), float(station_cost),
            float(line_fixed_cost), tiebreak, machines_key, overhead_key)

design_sb = None
if derive_line:
    design_sb = design_cached(*LINE_KEY)
    if design_sb['status'] != 'Optimal':
        need = design_sb.get('min_budget_needed')
        sb.error("The budget is too small for any complete line"
                 + (f" (minimum about Rs {need:,.0f})." if need else ".")
                 + " Raise the budget or switch the design off.")
        st.stop()
    sb.success(
        f"Line: {design_sb['cycle_time']:.3f} min cycle, "
        f"{design_sb['stations']} stations, Rs {design_sb['install_cost']:,.0f}, "
        f"{design_sb['cap_line']:,} units/month")

sb.subheader("Costs (Rs)")
if derive_line:
    cost_install = sb.number_input("Line install cost (from line design)",
                                   value=int(round(design_sb['install_cost'])),
                                   disabled=True)
else:
    cost_install = sb.number_input("Line install cost", value=1_200_000,
                                   step=100_000)
cost_remove = sb.number_input("Line removal cost", value=100_000, step=10_000)
if derive_line:
    _oh = design_sb['monthly_overhead']
    sb.number_input("Machine overhead per line per month (from line design)",
                    value=round(float(_oh['total_monthly_overhead']), 2),
                    disabled=True,
                    help="Electricity + maintenance + consumables of the machines "
                         "the design selected. Recalculated when the budget, "
                         "machine prices or overhead assumptions change.")
    cost_run_other = sb.number_input(
        "Other recurring cost per line per month (lighting, supervision)",
        value=0, step=1_000,
        help="Added to the machine overhead. 0 = none assumed.")
    cost_run = 0                      # replaced by the dynamic overhead in make_params
else:
    cost_run = sb.number_input(
        "Monthly line overhead (manual - no line design)", value=10_000, step=1_000,
        help="Used only because the line design is switched off, so there are no "
             "machine counts to calculate overhead from.")
    cost_run_other = 0

sb.subheader("Labour: monthly wage per worker (Rs)")
wage_cut = sb.number_input("Cutting - fabric cutter (skilled)",
                           value=WAGES['Cutting'], step=500)
wage_stitch = sb.number_input("Stitching - sewing operator (semi-skilled)",
                              value=WAGES['Stitching'], step=500)
wage_fin = sb.number_input("Finishing - trimming/packing helper (unskilled)",
                           value=WAGES['Finishing'], step=500)
process_eff = 0.50
sb.caption("Worker efficiency is fixed at 50% for the baseline capacity calculation.")

sb.subheader("Dedicated lines")
N_max = sb.number_input("Maximum number of dedicated lines", value=3,
                        min_value=1, max_value=10, step=1)
if "cap_line" not in st.session_state:
    st.session_state["cap_line"] = 20_800
if derive_line:                       # shown, not editable: it comes from the design
    st.session_state["cap_line"] = max(1_000, int(design_sb['cap_line']))
cap_line = sb.number_input("Output of one line (units/month)",
                           min_value=1_000, step=500, key="cap_line",
                           disabled=derive_line,
                           help="26 days x 8 h x 60 x 75 % uptime / cycle time. "
                                "With the line design on, the cycle time comes "
                                "from line balancing.")
l_line = sb.number_input("Operators per dedicated line", min_value=1, step=1,
                         value=int(design_sb['stations']) if derive_line else 11,
                         disabled=derive_line,
                         help="One operator per station.")

sb.subheader("Shared-capacity expansion")
allow_expansion = sb.checkbox("Allow conditional cutting/finishing expansion", value=False)
cost_exp_cut = sb.number_input("Cutting expansion capital per module (Rs)", value=50_000, step=10_000)
cost_exp_fin = sb.number_input("Finishing expansion capital per module (Rs)", value=40_000, step=10_000)
upkeep_cut = sb.number_input("Cutting module upkeep (Rs/month)", value=2_500, step=500)
upkeep_fin = sb.number_input("Finishing module upkeep (Rs/month)", value=1_500, step=500)
st.caption("Expansion modules add 15,000 standard minutes/month and require "
           "3 workers at 6,240 productive minutes per worker. Stitching expansion is disabled.")

sb.subheader("Labour and shared capacity")
labour_coupling = True
st.caption("The optimiser chooses transfers from the existing 24-person stitching crew "
           "and any additional dedicated/shared hires. Existing baseline staff remain paid.")
ded_uses_shared = sb.checkbox("Line output still needs shared cutting and finishing",
                              value=True)

sb.subheader("Inventory")
cost_hold = sb.number_input("Holding cost / unit", value=5)
cost_short = sb.number_input("Shortage penalty / unit", value=500)

SCALARS = dict(N_max=int(N_max), cap_line=cap_line, l_line=int(l_line),
               derive_line=derive_line, line_key=LINE_KEY, base_A=base_A, max_A=max_A, volatility=volatility, seed=int(seed),
               budget=budget, cost_install=cost_install, cost_remove=cost_remove,
               cost_run=cost_run, cost_run_other=cost_run_other,
               wage_cut=wage_cut, wage_stitch=wage_stitch, wage_fin=wage_fin,
               process_eff=process_eff, allow_expansion=allow_expansion,
               cost_exp_cut=cost_exp_cut, cost_exp_fin=cost_exp_fin,
               upkeep_cut=upkeep_cut, upkeep_fin=upkeep_fin,
               labour_coupling=labour_coupling, ded_uses_shared=ded_uses_shared,
               cost_hold=cost_hold, cost_short=cost_short)


def make_params(s):
    """Turn the sidebar values into model inputs (defaults fill the rest)."""
    p = default_params(generate_demand(s['base_A'], s['max_A'], s['volatility'],
                                       seed=s['seed']))
    p['B_t'] = s['budget']
    p['N_max'] = s['N_max']
    p['cap_line'] = s['cap_line']
    p['cost_install'] = s['cost_install']
    p['cost_remove'] = s['cost_remove']
    p['cost_run'] = s['cost_run']
    p['cost_run_other'] = s['cost_run_other']
    p['wage'] = {'Cutting': s['wage_cut'], 'Stitching': s['wage_stitch'],
                 'Finishing': s['wage_fin']}
    p['process_efficiency'] = s['process_eff']
    p['worker_efficiency'] = s['process_eff']
    p['cost_exp']['Cutting'] = s['cost_exp_cut']
    p['cost_exp']['Finishing'] = s['cost_exp_fin']
    p['cost_exp_upkeep']['Cutting'] = s['upkeep_cut']
    p['cost_exp_upkeep']['Finishing'] = s['upkeep_fin']
    p['cost_hold'] = {k: s['cost_hold'] for k in p['cost_hold']}
    p['cost_short'] = {k: s['cost_short'] for k in p['cost_short']}
    p['allow_expansion'] = s['allow_expansion']
    p['labour_coupling'] = s['labour_coupling']
    p['ded_uses_shared'] = s['ded_uses_shared']
    p['L_line'] = s['l_line']
    if s['derive_line']:
        # line balancing overrides cap_line, cost_install, L_line and cost_run
        # (cost_run = dynamic machine overhead of one line + cost_run_other)
        p = apply_line_design(p, design_cached(*s['line_key']))
    return p


# Cached solves: widget clicks no longer re-run the optimiser.
@st.cache_data(show_spinner=False)
def solve_cached(scalars, policy):
    res, _ = solve_milp(make_params(scalars), policy)
    return res


@st.cache_data(show_spinner=False)
def rolling_cached(scalars, forecast_std):
    return solve_rolling_horizon_detail(make_params(scalars), forecast_std)


params = make_params(SCALARS)
_cfg_now = fill_params(params)
sb.info(
    "Workforce baseline: 6 cutting + 24 stitching + 6 finishing workers. "
    "Each worker supplies 6,240 productive standard minutes/month "
    "(26 days x 8 hours x 60 x 50%). Dedicated-line staffing is a mix of "
    "transferred and hired workers, with incremental payroll included.")
money = st.column_config.NumberColumn(format="Rs. %.0f")
for note in line_design_warnings(params):
    st.warning(note)

tab1, tab2, tab3, tab4 = st.tabs(["Layout optimisation",
                                  "Policy comparison and break-even",
                                  "Line balancing", "Simulation"])

# ------------------------------------------------------------------- tab 1
with tab1:
    st.header("Optimal layout transition schedule")
    with st.spinner("Solving the MILP..."):
        res = solve_cached(SCALARS, "Optimized")

    if res['status'] != 'Optimal':
        st.error(f"No optimal solution found ({res['status']}). "
                 "Check the capital budget and capacities.")
    else:
        first = next((t for t, v in res['Install'].items() if v > 0), None)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total cost", f"Rs. {res['obj']:,.0f}")
        c2.metric("Most lines active", int(max(res['N'].values())))
        c3.metric("First install month", first if first else "None")

        df = pd.DataFrame({
            'Month': list(res['N']),
            'Active lines': list(res['N'].values()),
            'Cutting expansions': list(res['M_cutting'].values()),
            'Finishing expansions': list(res['M_finishing'].values()),
            'Dedicated volume': list(res['Q_ded_A'].values()),
            'Shared volume': list(res['Q_proc_A'].values()),
        })
        st.plotly_chart(px.bar(df, x='Month',
                               y=['Active lines', 'Cutting expansions', 'Finishing expansions'],
                               barmode='group',
                               title="Capacity decisions over time"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Month'], y=df['Dedicated volume'],
                                 name='Dedicated', stackgroup='one'))
        fig.add_trace(go.Scatter(x=df['Month'], y=df['Shared volume'],
                                 name='Shared', stackgroup='one'))
        fig.update_layout(title="Product A production allocation",
                          xaxis_title="Month", yaxis_title="Units")
        st.plotly_chart(fig)

        st.subheader("Cost breakdown")
        bd = pd.DataFrame({'Component': list(res['breakdown']),
                           'Rs': list(res['breakdown'].values())})
        st.dataframe(bd, column_config={'Rs': money}, hide_index=True)
        st.subheader("Monthly workforce and transfers")
        workforce = pd.DataFrame({
            'Month': list(res['N']),
            'Active lines': list(res['N'].values()),
            'Line operators transferred': list(res['Transferred_Stitching'].values()),
            'Line operators hired': list(res['Hired_Line_Operators'].values()),
            'Shared stitching hires': list(res['Hired_Shared_Stitching'].values()),
            'Shared stitching workforce': list(res['Shared_Stitching_Operators'].values()),
        })
        st.dataframe(workforce, hide_index=True)


        st.subheader("Machine overhead (recurring) and overhead per unit")
        st.caption("One-off machine investment is paid at installation and sits "
                   "in 'installation' above. This table is the recurring monthly "
                   "overhead: lines installed x overhead of one line. Overhead per "
                   "unit divides by the dedicated units actually produced.")
        oh_df = pd.DataFrame(overhead_schedule(params, res))
        total_oh = oh_df['Monthly overhead'].sum()
        total_units = oh_df['Dedicated units'].sum()
        o1, o2, o3 = st.columns(3)
        o1.metric("Overhead per line per month", f"Rs. {params['cost_run']:,.2f}")
        o2.metric("36-month machine overhead", f"Rs. {total_oh:,.0f}")
        o3.metric("Average overhead per dedicated unit",
                  f"Rs. {total_oh / total_units:,.3f}" if total_units > 0 else "n/a",
                  help="Total overhead over 36 months / total dedicated units. "
                       "n/a when no dedicated units are produced.")
        st.dataframe(oh_df, hide_index=True, column_config={
            'Lines': st.column_config.NumberColumn(format="%.0f"),
            'Dedicated units': st.column_config.NumberColumn(format="%.0f"),
            'Overhead per unit (actual)': st.column_config.NumberColumn(format="Rs. %.3f"),
            **{k: st.column_config.NumberColumn(format="Rs. %.2f")
               for k in ('Electricity', 'Maintenance', 'Consumables',
                         'Other line running', 'Monthly overhead')}})

        st.subheader("Rolling-horizon check")
        fstd = st.slider("Forecast error on future months (std. dev.)",
                         0.0, 0.3, 0.0, 0.05)
        if st.button("Run rolling horizon"):
            with st.spinner("Solving 36 sequential optimisations..."):
                rh = rolling_cached(SCALARS, fstd)
            dfr = pd.DataFrame({'Month': list(rh['N']),
                                'Static plan': list(res['N'].values()),
                                'Rolling horizon': list(rh['N'].values())})
            st.plotly_chart(px.line(dfr, x='Month',
                                    y=['Static plan', 'Rolling horizon'],
                                    title="Active lines: static vs rolling horizon"))
            c1, c2 = st.columns(2)
            c1.metric("Static plan cost", f"Rs. {res['obj']:,.0f}")
            c2.metric("Rolling-horizon cost", f"Rs. {rh['total_cost']:,.0f}",
                      delta=f"{rh['total_cost'] - res['obj']:,.0f}",
                      delta_color="inverse")

# ------------------------------------------------------------------- tab 2
with tab2:
    st.header("Baseline policy comparison")
    with st.spinner("Solving baselines..."):
        r_opt = solve_cached(SCALARS, "Optimized")
        r_proc = solve_cached(SCALARS, "Always-Process")
        r_ded = solve_cached(SCALARS, "Dedicated-From-Start")

    rows = [("Always-Process", r_proc), ("Dedicated-From-Start", r_ded),
            ("Optimized MILP", r_opt)]
    df_comp = pd.DataFrame({
        'Policy': [n for n, _ in rows],
        'Total cost (Rs)': [r['obj'] if r['status'] == 'Optimal' else None
                            for _, r in rows]})
    best_base = df_comp['Total cost (Rs)'][:2].min()
    df_comp['Saving vs best baseline (%)'] = [
        None, None,
        (100 * (best_base - r_opt['obj']) / best_base
         if r_opt['status'] == 'Optimal' else None)]
    st.dataframe(df_comp, hide_index=True, column_config={
        'Total cost (Rs)': money,
        'Saving vs best baseline (%)': st.column_config.NumberColumn(format="%.2f")})
    st.caption("Cutting/finishing expansion is permitted only when Product A has a shortage; total-cost minimisation decides whether it is worthwhile. B/C/D shortages alone cannot trigger expansion.")

    st.divider()
    st.subheader("Break-even analysis")
    payback = calc_payback_from_results(r_proc, r_opt)
    c1, c2, c3 = st.columns(3)
    c1.metric("36-month net operating savings",
              f"Rs. {payback['cumulative_operating_savings']:,.0f}")
    c2.metric("Investment outflow (incl. removal / terminal)",
              f"Rs. {payback['total_investment_outflow']:,.0f}")
    if not payback['has_installation']:
        payback_label = "No line installed"
    elif payback['payback_month'] is not None:
        payback_label = f"Month {payback['payback_month']}"
    else:
        payback_label = "Not within 36 months"
    c3.metric("Investment payback", payback_label,
              help="First month cumulative operating savings recover installation, removal and terminal outflows. Compares the Optimized MILP against the Always-Process baseline.")
    st.caption(f"36-month net benefit after investment: Rs. {payback['net_benefit_36m']:,.0f}. "
               "Transferred existing operators are not double-counted; only incremental line/shared hires are added to baseline payroll.")
    with st.expander("View month-by-month payback cash flows"):
        payback_df = pd.DataFrame(payback['monthly'])
        st.dataframe(payback_df, hide_index=True, column_config={
            col: money for col in payback_df.columns if col != 'Month'})

# ------------------------------------------------------------------- tab 3
with tab3:
    st.header("Budget-constrained line balancing")
    st.markdown("Given the budget for one line and the machine costs, the line "
                "is balanced and machines are chosen to **minimise the cycle "
                "time**. Its cycle time, stations and cost set the line output, "
                "operators and install cost used by the layout optimisation.")
    d = design_sb if design_sb is not None else design_cached(*LINE_KEY)
    if d['status'] != 'Optimal':
        need = d.get('min_budget_needed')
        st.error("The budget is too small for any complete line"
                 + (f" (minimum about Rs {need:,.0f})." if need else "."))
    else:
        if not derive_line:
            st.info("Preview only: the design switch in the sidebar is off, so "
                    "these values are NOT used by the optimisation.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cycle time", f"{d['cycle_time']:.3f} min")
        c2.metric("Stations = operators", d['stations'])
        c3.metric("Actual install cost", f"Rs. {d['install_cost']:,.0f}",
                  delta=f"{d['budget_used_pct'] - 100:.1f}% vs budget",
                  delta_color="off")
        c4.metric("Line output", f"{d['cap_line']:,} units/month")
        st.caption(
            f"Balance efficiency {d['balance_efficiency']:.1%}. "
            f"Solved exactly in {d['solve_seconds'] * 1000:.0f} ms ({d['method']}). "
            f"Fastest cycle time any budget could reach with these machines: "
            f"{d['cycle_time_floor']:.3f} min"
            + (" - already reached; more money will not speed the line up."
               if d['at_cycle_floor'] else "."))

        st.subheader("Passed to the layout optimisation")
        base = default_params()
        st.dataframe(pd.DataFrame({
            'Input': ["Line output (units/month)", "Install cost (Rs, one-off)",
                      "Operators per line",
                      "Monthly overhead per line (Rs, recurring)"],
            'Previous default': [base['cap_line'], base['cost_install'],
                                 base['L_line'], base['cost_run']],
            'From line design': [d['cap_line'], round(d['install_cost']),
                                 d['stations'],
                                 round(d['monthly_overhead']['total_monthly_overhead'], 2)],
            'Used now': [params['cap_line'], round(params['cost_install']),
                         params['L_line'], round(params['cost_run'], 2)]}),
            hide_index=True)

        st.subheader("Stations and machines")
        rows = []
        for s_no, tasks_here in d['assignment'].items():
            load = d['station_loads'][s_no - 1]
            rows.append({'Station': s_no,
                         'Tasks': ", ".join(str(t) for t in tasks_here),
                         'Machines': ", ".join(d['station_machines'][s_no]),
                         'Load (min)': round(load, 3),
                         'Utilisation': load / d['cycle_time']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            'Utilisation': st.column_config.ProgressColumn(
                format="%.0f%%", min_value=0, max_value=1)})
        for problem in d.get('machine_check', []):
            st.error(f"Line design inconsistency: {problem}")

        st.subheader("Machine investment and monthly overhead of one line")
        oh = d['monthly_overhead']
        rows = []
        for m, n in d['machine_counts'].items():
            pt = oh['per_type'][m]
            rows.append({'Machine': m, 'Count': n,
                         'Purchase (Rs)': n * MACH[m]['purchase_cost'],
                         'Setup (Rs)': n * MACH[m]['setup_cost'],
                         'Installed (Rs)': n * MACH[m]['cost'],
                         'Electricity /month': pt['electricity'],
                         'Maintenance /month': pt['maintenance'],
                         'Consumables /month': pt['consumables']})
        mc = pd.DataFrame(rows)
        tot = mc[mc.columns[1:]].sum()
        mc.loc[len(mc)] = ['Total', *tot.tolist()]
        st.dataframe(mc, hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in mc.columns[2:]})
        st.caption(f"{oh['powered_machine_count']} powered machines; the manual "
                   "station is not counted for electricity, maintenance or "
                   "consumables. One machine of a type serves every task of that "
                   "type placed at the same station.")

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Investment (installed)", f"Rs. {oh['installed_machine_value']:,.0f}",
                  help="Purchase + setup, one-off; also includes any extra station "
                       "or line cost set in the sidebar when added to the install "
                       "cost above.")
        i2.metric("Electricity /month", f"Rs. {oh['electricity_monthly']:,.2f}")
        i3.metric("Maintenance /month", f"Rs. {oh['maintenance_monthly']:,.2f}")
        i4.metric("Consumables /month", f"Rs. {oh['consumables_monthly']:,.2f}")
        j1, j2 = st.columns(2)
        j1.metric("Total monthly overhead", f"Rs. {oh['total_monthly_overhead']:,.2f}")
        out_units = j2.number_input(
            "Monthly output of this line for the per-unit figure (units)",
            min_value=0, value=int(d['cap_line']), step=500,
            help="Use the actual production. The default is the line's full "
                 "capacity, which gives only a full-capacity estimate.")
        per_unit = calculate_overhead_per_unit(oh['total_monthly_overhead'], out_units)
        label = ("Overhead per unit - FULL-CAPACITY estimate"
                 if out_units == d['cap_line'] else "Overhead per unit at this output")
        st.metric(label, f"Rs. {per_unit:,.3f}" if per_unit is not None else "n/a (no output)")
        st.caption("Actual overhead per unit for the optimised plan is in the "
                   "Layout optimisation tab (month by month).")

        with st.expander("Source discrepancy in machine_info.txt"):
            ref = reference_investment()
            ref_oh = calculate_monthly_overhead(REFERENCE_MIX)
            st.dataframe(pd.DataFrame({
                'Item': ["Purchase cost", "Installed cost (purchase + setup)"],
                'Itemised prices (used)': [ref['purchase_cost'], ref['installed_cost']],
                'Totals printed in source': [ref['source_purchase_cost'],
                                             ref['source_installed_cost']],
                'Difference': [ref['purchase_discrepancy'],
                               ref['installed_discrepancy']]}), hide_index=True)
            st.caption("The source totals are Rs 3,000 higher than the sum of its "
                       "own itemised prices. The itemised arithmetic is used and "
                       "the source file is not altered. Reference 11-machine "
                       f"overhead: Rs {ref_oh['total_monthly_overhead']:,.2f}/month "
                       "(source file shows Rs 6,716 with different electricity "
                       "and maintenance figures).")

    st.subheader("What does more capital buy?")
    if st.button("Show cycle time versus line budget"):
        low = minimum_budget(machines=MACH, station_cost=station_cost,
                             line_cost=line_fixed_cost)
        grid = tuple(round(low + k * (max(2 * line_budget, 2 * low) - low) / 14, -3)
                     for k in range(15))
        fr = pd.DataFrame(frontier_cached(grid, int(max_stations), *LINE_KEY[2:]))
        fr = fr[fr['status'] == 'Optimal']
        st.plotly_chart(px.line(fr, x='budget', y='cycle_time', markers=True,
                                title="Best cycle time for each line budget",
                                labels={'budget': 'Line budget (Rs)',
                                        'cycle_time': 'Cycle time (min)'}))
        st.dataframe(fr[['budget', 'install_cost', 'cycle_time', 'stations',
                         'cap_line', 'monthly_overhead',
                         'overhead_per_unit_at_capacity']], hide_index=True,
                     column_config={'budget': money, 'install_cost': money,
                                    'monthly_overhead': st.column_config.NumberColumn(
                                        format="Rs. %.2f"),
                                    'overhead_per_unit_at_capacity':
                                        st.column_config.NumberColumn(
                                            "overhead/unit (full capacity)",
                                            format="Rs. %.3f")})
        st.caption("A faster line is not automatically better: output beyond "
                   "demand is wasted, and every extra station takes an operator "
                   "from shared stitching. Compare budgets in the Layout "
                   "optimisation tab.")

    with st.expander("Benchmark: fixed cycle time, minimum stations "
                     "(no machine costs)"):
        ct = st.number_input("Cycle time (min)", value=0.45, min_value=0.45,
                             step=0.05, format="%.2f")
        stations, eff = rpw_heuristic(ct)
        b1, b2 = st.columns(2)
        b1.metric("RPW stations", len(stations))
        b2.metric("Balance efficiency", f"{eff:.2%}")
        st.json({f"Station {i + 1}": t for i, t in enumerate(stations)})
        if st.button("Run exact station assignment"):
            with st.spinner("Solving the exact model..."):
                n, eff_x, assignment = exact_station_assignment(ct)
            st.success(f"Exact solution: {n} stations at {eff_x:.2%} efficiency.")
            st.json({f"Station {s}": t for s, t in assignment.items()})

# ------------------------------------------------------------------- tab 4
with tab4:
    st.header("Discrete-event simulation (SimPy)")
    st.markdown("The simulation reads the same inputs as the optimiser (standard "
                "minutes, shared operators, process efficiency, operators per line).")

    for note in check_line_consistency(params):
        st.warning(note)
    st.caption("Cutting and finishing servers are simulation resources, not "
               "validated headcounts. Each station of the simulated line has its "
               "own machine of every type it needs, so machine counts do not "
               "limit the simulated line.")

    st.subheader("1. Does a dedicated line deliver the capacity the MILP assumes?")
    c1, c2, c3 = st.columns(3)
    cv = c1.slider("Processing-time variability (CV)", 0.05, 0.40, 0.10, 0.05)
    buf = c2.number_input("Buffer between stations (garments)", value=3,
                          min_value=1, max_value=10, step=1)
    reps_cap = c3.number_input("Replications ", value=6, min_value=1, step=1)
    if st.button("Run line-capacity check"):
        with st.spinner("Feeding a saturated line..."):
            st.session_state['cap_check'] = line_capacity_check(
                params, buffer=int(buf), cv=cv, reps=int(reps_cap))
    chk = st.session_state.get('cap_check')
    if chk:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Theoretical ({chk['bottleneck_min']:.2f} min bottleneck)",
                  f"{chk['theoretical']:,.0f}")
        m2.metric("Simulated", f"{chk['simulated']:,.0f}",
                  delta=f"{chk['ratio'] - 1:.1%}")
        m3.metric("MILP assumes", f"{chk['milp_cap_line']:,.0f}")
        m4.metric("Gap (simulated - MILP)", f"{chk['gap_vs_milp']:,.0f}")
        if params.get('line_overhead') and chk['simulated'] > 0:
            st.caption(f"Machine overhead per unit at the simulated output: Rs "
                       f"{params['line_overhead']['machine_total'] / chk['simulated']:,.3f} "
                       "(one line running at that output).")
        st.caption("The simulation uses 12,480 paid minutes/month. Dedicated-line output applies a separate 75% machine-uptime assumption; shared departments use 50% worker efficiency.")

        def apply_capacity(value):
            st.session_state['cap_line'] = int(round(value))

        st.button("Apply simulated capacity to the optimisation",
                  on_click=apply_capacity, args=(chk['simulated'],),
                  disabled=derive_line,
                  help="Switch off 'Derive line from budget' in the sidebar to "
                       "override the line output by hand." if derive_line else None)

    st.divider()
    st.subheader("2. Layout comparison for one month of the planned mix")
    d1, d2, d3 = st.columns(3)
    month = d1.slider("Month", 1, 36, 30)
    planned = 1
    if res.get('status') == 'Optimal':
        planned = max(1, int(round(res['N'][month])))
    n_lines = d2.number_input("Dedicated lines in hybrid layout", value=planned,
                              min_value=1, max_value=3, step=1)
    reps = d3.number_input("Replications", value=5, min_value=1, step=1)
    month_demand = {p: params['demand'][p][month] for p in 'ABCD'}
    st.caption("Demand this month: " + ", ".join(
        f"{p} {v:,.0f}" for p, v in month_demand.items()))

    if st.button("Run simulation comparison"):
        with st.spinner("Simulating garment flow..."):
            out = [run_replications("Process", n=int(reps), params=params,
                                    month_demand=month_demand, n_lines=0),
                   run_replications("Hybrid", n=int(reps), params=params,
                                    month_demand=month_demand,
                                    n_lines=int(n_lines),
                                    transfer_operators=(
                                        int(round(res['Transferred_Stitching'][month]))
                                        if res.get('status') == 'Optimal' and
                                        int(round(res['N'][month])) == int(n_lines)
                                        else min(params['ops_shared_stitch'],
                                                 params['L_line'] * int(n_lines))),
                                    shared_hires=(
                                        int(round(res['Hired_Shared_Stitching'][month]))
                                        if res.get('status') == 'Optimal' and
                                        int(round(res['N'][month])) == int(n_lines)
                                        else 0))]
        df_sim = pd.DataFrame(out)[['Layout', 'Lines', 'Shared_Ops', 'Offered',
                                    'Completed', 'Transferred_Operators', 'Hired_Line_Operators', 'Avg_Flow_Time', 'Flow_Time_SD',
                                    'WIP', 'Shared_Util']]
        st.dataframe(df_sim, hide_index=True, column_config={
            'Offered': st.column_config.NumberColumn(format="%.0f"),
            'Completed': st.column_config.NumberColumn(format="%.0f"),
            'Lines': st.column_config.NumberColumn(format="%.0f"),
            'Shared_Ops': st.column_config.NumberColumn(format="%.0f"),
            'Avg_Flow_Time': st.column_config.NumberColumn(format="%.2f min"),
            'Flow_Time_SD': st.column_config.NumberColumn(format="%.3f"),
            'WIP': st.column_config.NumberColumn(format="%.1f units"),
            'Shared_Util': st.column_config.NumberColumn(format="%.2f")})
        st.plotly_chart(px.bar(df_sim, x='Layout', y='Avg_Flow_Time',
                               color='Layout', error_y='Flow_Time_SD',
                               title="Average flow time"))
