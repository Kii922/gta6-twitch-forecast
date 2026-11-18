"""gta6-twitch-ccu-forecast-01 entry: log-scale quantile fan from launch-day comparables.

Model (validated on the 4 practice cases, answers extracted from SullyGnome):
  * Train pool: the case's own comparables with a full release-day curve,
    filtered to "strong anchors" (hour-1 print >= 8% of the day's peak).
  * Per hour h = 2..24, OLS on log(viewers_h) with two features:
      log(hour-1 print) and log(nzpeak / hour-1 print), where nzpeak is the
      max hourly viewers in the 24 h before midnight UTC -- the NZ/AU
      unlock wave, which tells how much audience the midnight print
      understates vs the US wave still to come.
  * Median forecast = 0.8 * regression median + 0.2 * console-only
    median-ratio median (log space). Quantile width = regression residual
    sd (dof-adjusted, x0.9), normal quantiles at 0.05/0.25/0.5/0.75/0.95.
  * Falls back to the persistence baseline for any hour the fit cannot
    cover (<8 training rows), and to nzratio = 1 when the case has no
    pre-midnight history.

Validation (mean pinball loss, log scale, lower is better):
  house baseline (starter)      0.3871
  this model, practice cases    0.2049  (p00 0.3328, p01 0.1846, p02 0.1219, p03 0.1803)
  leave-one-out, console-strong 0.1578  (baseline 0.3226 there)

Standard library only. Deterministic: comparables are sorted by name before
fitting, so repeated runs produce byte-identical output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
W_REG = 0.8          # weight on the regression median in the blend
WIDTH = 0.9          # multiplier on the regression residual sd
MIN_ROWS = 8         # minimum training rows per hour for the OLS fit
STRONG_MIN = 0.08    # hour-1 / day-peak ratio for a "strong anchor"


def parse_ts(t):
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """Round to 2 decimals, half up (same as the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def baseline_quantiles(p1, h):
    """Persistence fan for curve-index h (contest hour h+1), as in the starter."""
    return [round2(max(1, p1) * math.exp(z * 0.15 * math.sqrt(h))) for z in Z]


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def pstdev(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def solve(A, b):
    """Solve A x = b by Gauss-Jordan with partial pivoting (small systems)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def day_curve(hourly, day_start):
    """Release-day hourly viewers {0..23} from a comparable's hourly series."""
    ds = parse_ts(day_start)
    pts = {}
    for p in hourly:
        h = int((parse_ts(p["from"]) - ds).total_seconds() // 3600)
        if 0 <= h < 24 and p["viewers"] > 0:
            pts[h] = p["viewers"]
    return pts


def nzpeak(hourly, day_start):
    """Max hourly viewers in the 24 h before day_start (the NZ/AU wave)."""
    ds = parse_ts(day_start)
    lo = ds - timedelta(hours=24)
    vals = [p["viewers"] for p in hourly
            if lo <= parse_ts(p["from"]) < ds and p["viewers"] > 0]
    return max(vals) if vals else None


def build_train(comparables):
    """Launch dicts from the case's comparables, sorted by name (determinism)."""
    launches = []
    for comp in sorted(comparables, key=lambda c: c.get("name", "")):
        curve = day_curve(comp.get("hourly", []), comp["day_start"])
        if 0 not in curve or curve[0] <= 0:
            continue
        nz = nzpeak(comp.get("hourly", []), comp["day_start"])
        launches.append({
            "name": comp.get("name", ""),
            "curve": curve,
            "console_only": "PC" not in comp.get("platforms", ""),
            "nzratio": (nz / curve[0]) if nz else None,
        })
    strong = [L for L in launches
              if curve_peak_ratio(L) >= STRONG_MIN]
    return strong


def curve_peak_ratio(L):
    peak = max(L["curve"].values())
    return L["curve"][0] / peak if peak > 0 else 0.0


def regress_fan(train, p1, logh0, lognz, width):
    """Per-hour OLS log(y_h) ~ logh0 + lognz; returns {h: [q05..q95]} (h=1..23)."""
    fc = {}
    for h in range(1, 24):
        rows = []
        for L in train:
            if h in L["curve"] and L["curve"][h] > 0 and L.get("nzratio"):
                rows.append(([1.0, math.log(max(L["curve"][0], 1)),
                              math.log(max(L["nzratio"], 1e-6))],
                             math.log(L["curve"][h])))
        if len(rows) < MIN_ROWS:
            fc[h] = baseline_quantiles(p1, h)
            continue
        k = 3
        n = len(rows)
        A = [[sum(r[0][a] * r[0][b] for r in rows) for b in range(k)] for a in range(k)]
        bv = [sum(r[0][a] * r[1] for r in rows) for a in range(k)]
        beta = solve(A, bv)
        res = [r[1] - sum(beta[j] * r[0][j] for j in range(k)) for r in rows]
        sd = max(math.sqrt(sum(x * x for x in res) / max(n - k, 1)), 0.05) * width
        mu = beta[0] + beta[1] * logh0 + beta[2] * lognz
        fc[h] = [max(1.0, math.exp(mu + z * sd)) for z in Z]
    return fc


def console_medratio_fan(train, p1, width):
    """Per-hour median log-ratio fan over console-only strong launches."""
    cons = [L for L in train if L["console_only"]]
    fc = {}
    for h in range(1, 24):
        xs = [math.log(L["curve"][h] / L["curve"][0]) for L in cons
              if h in L["curve"] and L["curve"][h] > 0]
        if len(xs) < 3:
            fc[h] = baseline_quantiles(p1, h)
            continue
        mu = median(xs)
        sd = max(pstdev(xs), 0.05) * width
        fc[h] = [max(1.0, p1 * math.exp(mu + z * sd)) for z in Z]
    return fc


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("no hour-1 print")
    p1 = max(1, first["viewers"])
    logh0 = math.log(p1)

    nz = nzpeak(case.get("history_hourly", []), case["day_start"])
    lognz = math.log(max(nz / p1, 1e-6)) if nz else 0.0

    train = build_train(case.get("comparables", []))
    if len(train) < MIN_ROWS:
        # Degenerate train pool: fall back to the persistence baseline.
        return {"hours": [{"hour": row["hour"],
                           "q": baseline_quantiles(p1, row["hour"] - 1)}
                          for row in case["forecast"]]}

    reg = regress_fan(train, p1, logh0, lognz, WIDTH)
    con = console_medratio_fan(train, p1, WIDTH)
    hours = []
    for row in case["forecast"]:
        h = row["hour"] - 1  # curve index 1..23
        med = W_REG * math.log(reg[h][2]) + (1 - W_REG) * math.log(con[h][2])
        sd = (math.log(reg[h][3]) - math.log(reg[h][1])) / (2 * 0.674)
        qs = [round2(max(1.0, math.exp(med + z * sd))) for z in Z]
        # keep the row non-decreasing and within bounds after rounding
        qs = [max(1, min(1e9, q)) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": row["hour"], "q": qs})
    return {"hours": hours}


def main():
    in_files = sorted(INPUT_DIR.glob("*.in"))
    if not in_files:
        raise SystemExit(f"no .in files in {INPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_file in in_files:
        case = json.loads(in_file.read_text())
        out = forecast_case(case)
        out_path = OUTPUT_DIR / (in_file.stem + ".out")
        out_path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
