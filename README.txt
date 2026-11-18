Corrected payback calculation for MILP_Layout_transition

Files:
  app.py
  optimization.py
  line_balancing.py
  simulation.py
  stage9_sensitivity.py

What changed:
- Payback is now calculated from the solved Always-Process baseline and the
  solved Optimized transition plan, month by month.
- The monthly cost ledger includes shared/dedicated production, baseline
  payroll, incremental line/shared hires, machine overhead, inventory,
  shortages, expansion costs, installation, removal and terminal costs.
- Existing transferred stitching operators are not charged again. Their wages
  remain in baseline payroll; only incremental hires are added separately.
- The dashboard shows 36-month operating savings, investment outflows, the
  first payback month (if reached), 36-month net benefit, and monthly cash flows.

Run:
  python -m pip install numpy pandas matplotlib plotly simpy pulp highspy
  streamlit run app.py

Stage 9 sensitivity analysis:
  python stage9_sensitivity.py

Note: Python syntax compilation was checked. A full MILP solve was not run in
this environment, so numerical payback results must be validated locally with
PuLP and a working HiGHS/CBC solver.
