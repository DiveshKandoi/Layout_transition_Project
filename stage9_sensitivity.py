#!/usr/bin/env python3
"""Stage 9: sensitivity analysis for the revised 36-month layout-transition MILP.

This script deliberately uses the project's current `optimization.py` and
`line_balancing.py`; it does not maintain a separate, simplified MILP.

The only sensitivity families retained are those in the final revised report:
  1. Budget available to design ONE dedicated line (machine/station budget).
  2. Penalty per unit of unmet demand (lost sales; no backorder carryover).
  3. Inventory holding cost per unit per month.

All other assumptions are fixed at the revised model defaults. The capital
budget B_t for the complete 36-month programme remains separate from the
single-line design budget and is not varied in this experiment.

Run from the directory containing this file, optimization.py and
line_balancing.py:
    python stage9_sensitivity.py

Dependencies: numpy, pandas, matplotlib, pulp, highspy.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optimization import default_params, solve_milp, apply_line_design
from line_balancing import design_line

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage9_outputs"
OUT.mkdir(parents=True, exist_ok=True)

# The final revised report retains only these three one-factor-at-a-time sweeps.
BASE_LINE_BUDGET = 700_000
LINE_BUDGETS = [200_000, 300_000, 400_000, 500_000,
                600_000, 700_000, 800_000, 900_000]
SHORTAGE_PENALTIES = [100, 250, 500, 750, 1_000, 1_500]
HOLDING_COSTS = [0, 1, 2.5, 5, 7.5, 10, 20]

SOLVER_TIME_LIMIT_SECONDS = 180


def _base_params():
    """Create the revised base case and apply the reference line design."""
    params = default_params()
    params["allow_expansion"] = False
    params["labour_coupling"] = True
    params["N_max"] = 3
    params["B_t"] = 10_000_000  # ₹1 crore cumulative programme capital budget

    design = design_line(BASE_LINE_BUDGET)
    if design.get("status") != "Optimal":
        raise RuntimeError(
            "The reference single-line design could not be generated at "
            f"₹{BASE_LINE_BUDGET:,.0f}: {design.get('status')}"
        )
    params = apply_line_design(params, design, days=26, hours=8, uptime=0.75)
    return params, design


def _solve(params):
    started = time.perf_counter()
    result, _model = solve_milp(
        copy.deepcopy(params), policy="Optimized",
        time_limit=SOLVER_TIME_LIMIT_SECONDS,
    )
    elapsed = time.perf_counter() - started
    if result.get("status") != "Optimal" or result.get("obj") is None:
        raise RuntimeError(
            f"MILP did not solve to optimality: {result.get('status')}"
        )
    return result, elapsed


def _summarise_result(name, family, tested_value, params, result, elapsed,
                      line_design=None):
    demand_a = sum(params["demand"]["A"].values())
    dedicated_a = sum(result["Q_ded_A"].values())
    total_shortage = sum(
        sum(result["Shortage"][p].values()) for p in ("A", "B", "C", "D")
    )
    total_inventory = sum(
        sum(result["Inventory"][p].values()) for p in ("A", "B", "C", "D")
    )
    install_schedule = [
        f"M{t}×{int(round(n))}"
        for t, n in result["Install"].items() if n > 1e-7
    ]
    active_schedule = [int(round(result["N"][t])) for t in range(1, 37)]
    row = {
        "scenario": name,
        "sensitivity_family": family,
        "tested_value": tested_value,
        "objective_cost_inr": float(result["obj"]),
        "total_lines_installed": int(round(sum(result["Install"].values()))),
        "installation_schedule": "; ".join(install_schedule) or "None",
        "terminal_active_lines": int(round(result["N"][36])),
        "dedicated_A_share_pct": (100 * dedicated_a / demand_a) if demand_a else 0.0,
        "total_shortage_units": float(total_shortage),
        "inventory_unit_months": float(total_inventory),
        "transferred_stitching_operator_months": float(
            sum(result["Transferred_Stitching"].values())
        ),
        "hired_line_operator_months": float(
            sum(result["Hired_Line_Operators"].values())
        ),
        "solver_status": result["status"],
        "solve_seconds": elapsed,
        "active_line_schedule": active_schedule,
    }
    if line_design is not None:
        row.update({
            "line_design_status": line_design.get("status"),
            "line_design_budget_inr": line_design.get("budget"),
            "line_install_cost_inr": line_design.get("install_cost"),
            "line_cycle_time_min": line_design.get("cycle_time"),
            "line_stations_operators": line_design.get("stations"),
            "line_capacity_units_month": line_design.get("cap_line"),
            "line_monthly_machine_overhead_inr": (
                line_design.get("monthly_overhead", {}).get("total_monthly_overhead")
            ),
        })
    return row, result


def _run_scenario(name, family, value, params, line_design=None):
    result, elapsed = _solve(params)
    return _summarise_result(name, family, value, params, result, elapsed,
                             line_design=line_design)


def _run_all():
    base_params, base_design = _base_params()

    # Reference solve for deltas and all chart baselines.
    base_row, base_result = _run_scenario(
        "Base case", "base_case", BASE_LINE_BUDGET,
        copy.deepcopy(base_params), line_design=base_design,
    )
    base_cost = base_row["objective_cost_inr"]

    rows = []
    details = {"Base case": base_result}

    # 1. Single-line design budget: re-design the line at each budget, then
    # re-optimise the same 36-month MILP with the resulting capacity, operators,
    # installation cost and machine-derived overhead.
    for budget in LINE_BUDGETS:
        design = design_line(budget)
        scenario_name = f"Single-line budget ₹{budget:,.0f}"
        if design.get("status") != "Optimal":
            rows.append({
                "scenario": scenario_name,
                "sensitivity_family": "single_line_budget",
                "tested_value": budget,
                "objective_cost_inr": np.nan,
                "solver_status": f"Line design {design.get('status')}",
                "line_design_status": design.get("status"),
                "line_design_budget_inr": budget,
                "line_minimum_budget_needed_inr": design.get("min_budget_needed"),
            })
            continue
        params = copy.deepcopy(base_params)
        params = apply_line_design(params, design, days=26, hours=8, uptime=0.75)
        row, result = _run_scenario(
            scenario_name, "single_line_budget", budget, params, design
        )
        rows.append(row)
        details[scenario_name] = result

    # 2. Shortage penalty: apply the same per-unit lost-sales penalty to A-D.
    for penalty in SHORTAGE_PENALTIES:
        params = copy.deepcopy(base_params)
        params["cost_short"] = {p: float(penalty) for p in ("A", "B", "C", "D")}
        scenario_name = f"Shortage penalty ₹{penalty:,.0f}/unit"
        row, result = _run_scenario(
            scenario_name, "shortage_penalty", penalty, params, base_design
        )
        rows.append(row)
        details[scenario_name] = result

    # 3. Inventory holding cost: same per-unit-per-month rate for A-D.
    for holding in HOLDING_COSTS:
        params = copy.deepcopy(base_params)
        params["cost_hold"] = {p: float(holding) for p in ("A", "B", "C", "D")}
        scenario_name = f"Inventory holding ₹{holding:g}/unit-month"
        row, result = _run_scenario(
            scenario_name, "inventory_holding_cost", holding, params, base_design
        )
        rows.append(row)
        details[scenario_name] = result

    results = pd.DataFrame(rows)
    results["cost_change_vs_base_inr"] = results["objective_cost_inr"] - base_cost
    results["cost_change_vs_base_pct"] = (
        100 * (results["objective_cost_inr"] / base_cost - 1)
    )
    results.to_csv(OUT / "sensitivity_results.csv", index=False)

    # Save a compact, machine-readable base reference and scenario results.
    base_reference = {
        "status": base_row["solver_status"],
        "objective_cost_inr": base_cost,
        "reference_single_line_budget_inr": BASE_LINE_BUDGET,
        "line_design": {
            "install_cost_inr": base_design.get("install_cost"),
            "cycle_time_min": base_design.get("cycle_time"),
            "stations_operators": base_design.get("stations"),
            "capacity_units_month": base_design.get("cap_line"),
            "monthly_machine_overhead_inr": base_design.get(
                "monthly_overhead", {}
            ).get("total_monthly_overhead"),
            "machine_counts": base_design.get("machine_counts"),
        },
        "active_line_schedule": base_row["active_line_schedule"],
        "installation_schedule": base_row["installation_schedule"],
        "dedicated_A_share_pct": base_row["dedicated_A_share_pct"],
        "total_shortage_units": base_row["total_shortage_units"],
        "inventory_unit_months": base_row["inventory_unit_months"],
        "cost_breakdown": base_result.get("breakdown", {}),
    }
    (OUT / "base_case_reference.json").write_text(
        json.dumps(base_reference, indent=2, default=_json_default), encoding="utf-8"
    )
    serial_details = {
        name: {
            "objective_cost_inr": result.get("obj"),
            "active_line_schedule": [
                int(round(result["N"][t])) for t in range(1, 37)
            ],
            "installation_schedule": {
                str(t): int(round(v)) for t, v in result["Install"].items()
                if v > 1e-7
            },
            "cost_breakdown": result.get("breakdown", {}),
        }
        for name, result in details.items()
    }
    (OUT / "scenario_details.json").write_text(
        json.dumps(serial_details, indent=2, default=_json_default), encoding="utf-8"
    )

    _make_charts(results, base_cost)
    _write_report(results, base_row, base_design, base_cost)
    return results, base_reference


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def _make_charts(results, base_cost):
    labels = {
        "single_line_budget": "Single-line design budget",
        "shortage_penalty": "Shortage penalty",
        "inventory_holding_cost": "Inventory holding cost",
    }
    for family, label in labels.items():
        sub = results[results["sensitivity_family"] == family].copy()
        sub = sub.dropna(subset=["objective_cost_inr"]).sort_values("tested_value")
        if sub.empty:
            continue
        plt.figure(figsize=(9, 5.2))
        plt.plot(sub["tested_value"], sub["objective_cost_inr"] / 1e7,
                 marker="o", linewidth=1.8)
        plt.axhline(base_cost / 1e7, linestyle="--", linewidth=1,
                    label="Base case (₹7 lakh line design)")
        plt.xlabel({
            "single_line_budget": "Capital budget for designing one line (₹)",
            "shortage_penalty": "Penalty per unit of unmet demand (₹)",
            "inventory_holding_cost": "Holding cost per unit per month (₹)",
        }[family])
        plt.ylabel("Optimal 36-month total cost (₹ crore)")
        plt.title(f"Stage 9 sensitivity: {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"sensitivity_{family}.png", dpi=180)
        plt.close()

    # Show how the optimal line-count path responds across all solved scenarios.
    solved = results.dropna(subset=["objective_cost_inr"])
    schedules = [x for x in solved.get("active_line_schedule", [])
                 if isinstance(x, list) and len(x) == 36]
    if schedules:
        mat = np.asarray(schedules, dtype=float)
        plt.figure(figsize=(13, max(5, 0.28 * len(solved))))
        plt.imshow(mat, aspect="auto", interpolation="nearest", vmin=0,
                   vmax=max(1, int(np.nanmax(mat))))
        plt.colorbar(label="Active dedicated lines")
        plt.yticks(range(len(solved)), solved["scenario"], fontsize=7)
        plt.xticks(range(0, 36, 3), [f"M{i + 1}" for i in range(0, 36, 3)])
        plt.xlabel("Planning month")
        plt.title("Optimal active-line schedule across retained sensitivity cases")
        plt.tight_layout()
        plt.savefig(OUT / "sensitivity_active_line_schedules.png", dpi=180)
        plt.close()


def _write_report(results, base_row, base_design, base_cost):
    families = [
        ("single_line_budget", "Single-line design budget", "₹ per line"),
        ("shortage_penalty", "Shortage penalty", "₹ per unmet unit"),
        ("inventory_holding_cost", "Inventory holding cost", "₹ per unit-month"),
    ]
    lines = [
        "# Stage 9 — Sensitivity analysis (revised)", "",
        "## Scope and method", "",
        "The sensitivity analysis retains only the three factors specified in the final revised report: the budget used to design one dedicated line, the penalty for unmet demand, and the inventory holding cost. All scenarios re-solve the same 36-month MILP imported from the project's revised `optimization.py`; the dedicated line is redesigned with `line_balancing.py` when the single-line design budget changes.", "",
        "The single-line design budget is distinct from the model's cumulative programme capital budget. The former changes the line's machine configuration, station count, cycle time, installation cost, capacity and machine-derived monthly overhead. The latter remains fixed at ₹1 crore in every scenario.", "",
        f"- Reference single-line design budget: ₹{BASE_LINE_BUDGET:,.0f}.",
        f"- Reference objective cost: ₹{base_cost:,.2f}.",
        f"- Reference line installation cost: ₹{base_design.get('install_cost', float('nan')):,.2f}.",
        f"- Reference cycle time: {base_design.get('cycle_time', float('nan')):.3f} minutes.",
        f"- Reference stations/operators per line: {base_design.get('stations', 'n/a')}.",
        f"- Reference dedicated-line capacity: {base_design.get('cap_line', 'n/a'):,} units/month.",
        f"- Reference active-line schedule: {base_row['active_line_schedule']}.",
        "",
        "## Factors examined", "",
    ]
    for family, label, unit in families:
        sub = results[results["sensitivity_family"] == family].dropna(
            subset=["objective_cost_inr"]
        ).sort_values("tested_value")
        lines.append(f"### {label}")
        lines.append("")
        if sub.empty:
            lines.append("No scenario solved successfully for this factor.")
        else:
            for _, row in sub.iterrows():
                lines.append(
                    f"- {row['tested_value']:g} {unit}: total cost "
                    f"₹{row['objective_cost_inr']:,.2f} "
                    f"({row['cost_change_vs_base_pct']:+.2f}% vs base); "
                    f"installation: {row.get('installation_schedule', 'n/a')}; "
                    f"dedicated A share: {row.get('dedicated_A_share_pct', float('nan')):.1f}%; "
                    f"shortage: {row.get('total_shortage_units', float('nan')):,.1f} units."
                )
        lines.append("")
    lines += [
        "## Model assumptions held constant", "",
        "- 36 monthly periods; Products A, B, C and D; A demand follows the revised deterministic 2,000-to-20,000 units/month trajectory.",
        "- Working calendar: 26 days × 8 hours; worker efficiency: 50%; productive standard time is derived from paid minutes and efficiency.",
        "- Existing workforce: 6 cutting, 24 stitching and 6 finishing workers. A dedicated line has one operator per station; its staffing can be supplied by transferred stitching workers and/or hired line operators under the revised staffing constraints.",
        "- Dedicated-line machine capacity uses 75% uptime. Cutting and finishing remain shared; stitching expansion is disabled and shared expansion is disabled in this sensitivity experiment.",
        "- Shortage is unmet demand (lost sales), not backorders. Inventory may carry between months and terminal inventory is constrained to zero.",
        "- The cumulative programme capital budget remains ₹1 crore. Other production costs, wages, demand, removal cost, line running-cost method and all remaining model inputs are unchanged.",
        "",
        "## Interpretation and limitations", "",
        "Each point is a separately re-optimised deterministic scenario, not a forecast. A changed objective value reflects both the changed cost assumption and any resulting change in the optimal production, inventory, staffing or line-installation decisions. The single-line budget sweep changes the physical line design as well as its economics; it is not a sweep of installation cost alone.",
        "",
        "Production, inventory and shortage quantities follow the variable domains in the revised MILP; integer line, staffing and expansion decisions are retained. Results are conditional on the illustrative demand and cost inputs and should not be interpreted as empirically validated factory savings.",
        "",
        "## Files generated", "",
        "- `sensitivity_results.csv` — scenario-level outcomes.",
        "- `base_case_reference.json` — reference design and baseline solution.",
        "- `scenario_details.json` — line schedules and objective breakdowns.",
        "- `sensitivity_single_line_budget.png`, `sensitivity_shortage_penalty.png`, `sensitivity_inventory_holding_cost.png` — factor charts.",
        "- `sensitivity_active_line_schedules.png` — active-line schedule comparison.",
    ]
    (OUT / "Stage9_Sensitivity_Report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    results, base = _run_all()
    print(json.dumps(base, indent=2, default=_json_default))
    print("\nSensitivity results:")
    cols = ["scenario", "objective_cost_inr", "cost_change_vs_base_pct",
            "installation_schedule", "dedicated_A_share_pct",
            "total_shortage_units", "solver_status"]
    print(results[[c for c in cols if c in results.columns]].to_string(index=False))
    print(f"\nOutputs saved to: {OUT}")


if __name__ == "__main__":
    main()
