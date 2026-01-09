# stageq.py — overlap-safe comparisons, side-by-side monthly boxes, colored climatology
# Deps: requests, pandas, matplotlib, numpy
import os, json, argparse, datetime as dt, logging
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import xml.etree.ElementTree as ET

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(levelname)s - %(message)s")
LOG = logging.getLogger("stageq")

# --------------- Small utils -------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def day_str(x) -> str:
    return pd.Timestamp(x).date().isoformat()

def to_midnight_ts(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()

def yr_offset(ts: pd.Timestamp, years: int) -> pd.Timestamp:
    # accurate year offset (handles leap years)
    return (ts + pd.DateOffset(years=years))

# --------------- Config ------------------
def load_config(path: str) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        LOG.critical("Config not found: %s", path)
        return None
    with p.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            LOG.critical("Config JSON parse error: %s", e)
            return None

def normalize_sites(raw_sites: Any) -> List[Dict[str, Any]]:
    if raw_sites is None:
        return []
    out: List[Dict[str,Any]] = []
    if isinstance(raw_sites, list) and all(isinstance(x, str) for x in raw_sites):
        for s in raw_sites:
            sid = s.strip()
            if sid: out.append({"id": sid})
        return out
    if isinstance(raw_sites, list):
        for i, item in enumerate(raw_sites, 1):
            if isinstance(item, dict):
                sid = (item.get("id") or item.get("site") or item.get("site_id")
                       or item.get("usgs_id") or item.get("station"))
                if not sid:
                    LOG.warning("Site entry %d missing an id-like key: %r", i, item)
                    out.append(item.copy())
                else:
                    new = item.copy(); new["id"] = str(sid); out.append(new)
            else:
                LOG.warning("Unexpected site entry type at %d: %r", i, type(item))
        return out
    LOG.warning("Unsupported 'sites' structure in config: %r", type(raw_sites))
    return out

# --------------- USGS helpers -------------
BASE_DV = "https://waterservices.usgs.gov/nwis/dv"
BASE_IV = "https://nwis.waterservices.usgs.gov/nwis/iv/"

def _http_get(url: str, params: dict) -> str:
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code in (200, 301, 302): return r.text
            LOG.warning("GET %s -> %s", r.url, r.status_code)
        except requests.RequestException as e:
            LOG.warning("HTTP error (%d/3): %s", i+1, e)
    return ""

def _parse_waterml_values(xml_text: str) -> pd.DataFrame:
    if not xml_text:
        return pd.DataFrame(columns=["value"])
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame(columns=["value"])

    vals = []
    for v in root.findall(".//{*}value"):
        t = v.attrib.get("dateTime")
        s = (v.text or "").strip()
        if not t or s == "": continue
        try:
            vals.append((t, float(s)))
        except ValueError:
            continue
    if not vals:
        return pd.DataFrame(columns=["value"])

    times = pd.to_datetime([t for t,_ in vals], utc=True, errors="coerce")
    series = pd.Series([v for _,v in vals], index=times)
    series = series[series.index.notna()]
    if series.empty:
        return pd.DataFrame(columns=["value"])

    # tz-naive daily index at midnight
    series.index = series.index.tz_convert("UTC").tz_localize(None)
    series.index = pd.to_datetime(series.index.date)
    return series.to_frame("value")

def fetch_dv(site: str, param: str, start: str, end: str, stat: str) -> pd.DataFrame:
    params = {"format":"waterml,1.1","sites":site,"parameterCd":param,
              "startDT":start,"endDT":end,"statCd":stat}
    txt = _http_get(BASE_DV, params)
    df = _parse_waterml_values(txt)
    if df.empty:
        LOG.info("%s %s: DV statCd=%s returned no rows.", site, param, stat)
        return df
    LOG.info("%s %s: using DV statCd=%s with %d rows.", site, param, stat, len(df))
    return df

def _iv_cache_path(cache_dir: Path, site: str, param: str, s: dt.date, e: dt.date) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"iv_{site}_{param}_{s.isoformat()}_{e.isoformat()}.xml"

def fetch_iv_chunk(site: str, param: str, s: dt.date, e: dt.date, cache_dir: Optional[Path]) -> pd.DataFrame:
    params = {"format":"waterml,1.1","sites":site,"parameterCd":param,
              "startDT":s.isoformat(),"endDT":e.isoformat()}
    cache_hit, txt = False, ""
    if cache_dir:
        cp = _iv_cache_path(cache_dir, site, param, s, e)
        if cp.exists():
            txt = cp.read_text(encoding="utf-8", errors="ignore"); cache_hit = True
        else:
            txt = _http_get(BASE_IV, params)
            if txt: cp.write_text(txt, encoding="utf-8")
    if not txt:
        txt = _http_get(BASE_IV, params)
    if cache_hit:
        LOG.info("%s %s: IV chunk %s→%s (cache hit)", site, param, s, e)
    return _parse_waterml_values(txt)

def iv_to_daily_cached(site: str, param: str, start, end,
                       agg: str="mean", cache_dir: str="cache/usgs", chunk_days: int=31) -> pd.DataFrame:
    start_d = pd.Timestamp(start).date()
    end_d   = pd.Timestamp(end).date()
    cache_path = Path(cache_dir) if cache_dir else None

    parts = []
    cur = start_d
    chunks = []
    while cur <= end_d:
        nxt = min(cur + dt.timedelta(days=chunk_days), end_d)
        chunks.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)

    for i,(s,e) in enumerate(chunks,1):
        LOG.info("%s %s: IV chunk %d/%d %s→%s", site, param, i, len(chunks), s, e)
        df = fetch_iv_chunk(site, param, s, e, cache_path)
        if not df.empty: parts.append(df)

    if not parts: return pd.DataFrame(columns=["value"])
    iv = pd.concat(parts).sort_index()
    iv.index = pd.to_datetime(iv.index).tz_localize(None)
    rule = {"mean":"mean","min":"min","max":"max"}.get(agg.lower(),"mean")
    daily = iv.resample("D").agg({"value":rule})
    daily.index = pd.to_datetime(daily.index.date)
    LOG.info("%s %s: IV→daily produced %d rows.", site, param, len(daily))
    return daily

# ------------- Analysis helpers -------------
def fit_powerlaw_q_of_h(H: pd.Series, Q: pd.Series) -> Tuple[float, float, float, pd.Series]:
    s = pd.DataFrame({"H":H, "Q":Q}).replace([np.inf,-np.inf], np.nan).dropna()
    s = s[(s["H"]>0) & (s["Q"]>0)]
    if len(s) < 10:
        return (np.nan, np.nan, np.nan, pd.Series(index=H.index, dtype=float))
    X = np.log10(s["H"].values); Y = np.log10(s["Q"].values)
    B, logA = np.polyfit(X, Y, 1)
    A = 10**logA
    Qhat_fit = A * (s["H"]**B)
    ss_res = np.sum((s["Q"] - Qhat_fit)**2)
    ss_tot = np.sum((s["Q"] - s["Q"].mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot>0 else np.nan
    Qhat_full = pd.Series(index=H.index, dtype=float)
    Qhat_full.loc[s.index] = Qhat_fit.values
    return (A, B, R2, Qhat_full)

def flow_duration_curve(Q: pd.Series) -> pd.DataFrame:
    q = pd.Series(Q).dropna().sort_values(ascending=False).to_numpy()
    if q.size == 0:
        return pd.DataFrame(columns=["ExceedancePct","Q"])
    ranks = np.arange(1, q.size+1)
    exc = 100.0 * ranks / (q.size + 1)
    return pd.DataFrame({"ExceedancePct": exc, "Q": q})

# ------------- Plot helpers ----------------
def plot_hydrograph(site_label: str, df: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(12,6.8))
    ax.plot(df.index, df["Q_cfs"], lw=1.0, label="Q daily")
    ax.plot(df.index, df["Q_cfs"].rolling(7, min_periods=1).mean(), lw=2.0, label="Q 7-day mean")
    ax.set_yscale("log"); ax.set_ylabel("Discharge (cfs)"); ax.set_xlabel("Date")
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(df.index, df["H_ft"], lw=1.0, alpha=0.5, label="Stage (ft)")
    ax2.set_ylabel("Stage (ft)")
    ax.set_title(f"Hydrograph — {site_label}")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_rating_curve(site_label: str, df: pd.DataFrame, A: float, B: float, R2: float, out_png: Path):
    fig, ax = plt.subplots(figsize=(10.5,6.2))
    ax.scatter(df["H_ft"], df["Q_cfs"], s=15, alpha=0.55, label="Daily points")
    if np.isfinite(A) and np.isfinite(B):
        h = np.linspace(df["H_ft"].min(), df["H_ft"].max(), 300)
        ax.plot(h, A*(h**B), lw=2.2, label="Power-law fit")
    ax.set_xlabel("Stage / Gage Height (ft) [log]")
    ax.set_ylabel("Discharge (cfs) [log]")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_title(f"Rating Curve — {site_label}\nQ = {A:0.3g} · H^{B:0.3f}  (R² = {R2:0.3f})")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_residuals_time(df: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(14,4.2))
    ax.axhline(0, color="k", lw=1)
    ax.fill_between(df.index, -10, 10, color="0.9")
    ax.plot(df.index, df["resid_pct"], lw=1.0, label="Daily residual (%)")
    ax.plot(df.index, df["resid_pct"].rolling(30, min_periods=1).mean(),
            lw=2.5, label="30-day mean")
    ax.set_ylabel("Residuals (%)"); ax.set_xlabel("Date")
    ax.set_title("Rating-Curve Residuals (%): positive = measured Q > model")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_residuals_vs_stage(df: pd.DataFrame, out_png: Path):
    # binned median + IQR band (nice bias diagnostic)
    H = df["H_ft"]; R = df["resid_pct"]
    bins = np.linspace(H.min(), H.max(), 24)
    cats = pd.cut(H, bins, include_lowest=True)
    group = R.groupby(cats)
    mid = pd.Series([(b.left+b.right)/2 for b in group.groups.keys()], index=group.median().index)
    med = group.median().values
    q25 = group.quantile(0.25).values
    q75 = group.quantile(0.75).values

    fig, ax = plt.subplots(figsize=(13,6))
    ax.axhline(0, color="k", lw=1)
    ax.scatter(H, R, s=14, alpha=0.35, label="Daily")
    ax.plot(mid, med, lw=2.4, label="Binned median")
    ax.fill_between(mid, (q25+q75)/2 - 0.5*(q75-q25), (q25+q75)/2 + 0.5*(q75-q25),
                    alpha=0.25, label="±0.5 IQR")
    ax.set_xlabel("Stage / Gage Height (ft)"); ax.set_ylabel("Residuals (%)")
    ax.set_title("Residuals vs Stage — bias pattern by stage"); ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def _set_box_colors(ax, bp, face):
    for element in ["boxes","whiskers","caps","medians"]:
        for obj in bp[element]:
            obj.set_color("0.3")
            if element == "boxes":
                obj.set_facecolor(face)
                obj.set_alpha(0.6)

def plot_monthly_stage(df_full: pd.DataFrame, out_png: Path):
    d = df_full.copy(); d["Month"] = d.index.month
    vals_by_m = [d.loc[d["Month"]==m, "H_ft"].values for m in range(1,13)]
    counts = [len(v) for v in vals_by_m]
    labels = [f"{m:02d}\n(n={c})" for m,c in zip(range(1,13), counts)]
    fig, ax = plt.subplots(figsize=(12,6))
    bp = ax.boxplot(vals_by_m, tick_labels=labels, showfliers=False, patch_artist=True)
    _set_box_colors(ax, bp, "#bcd4f7")
    ax.set_title("Monthly Stage Distribution (full window)")
    ax.set_ylabel("Stage / Gage Height (ft)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_monthly_stage_compare(df_full: pd.DataFrame, df_recent: pd.DataFrame, recent_years: int, out_png: Path):
    # side-by-side boxes, both derived from the **same overlapped Q∩H window**
    def vals_by_month(df):
        d = df.copy(); d["Month"] = d.index.month
        return [d.loc[d["Month"]==m, "H_ft"].values for m in range(1,13)]

    v_full   = vals_by_month(df_full)
    v_recent = vals_by_month(df_recent)

    fig, ax = plt.subplots(figsize=(12.5,6.5))
    x = np.arange(1,13)
    pos_full   = x - 0.18
    pos_recent = x + 0.18

    bp1 = ax.boxplot(v_full, positions=pos_full, widths=0.30, showfliers=False, patch_artist=True)
    _set_box_colors(ax, bp1, "#bcd4f7")
    bp2 = ax.boxplot(v_recent, positions=pos_recent, widths=0.30, showfliers=False, patch_artist=True)
    _set_box_colors(ax, bp2, "#7aa6e8")

    ax.set_xticks(x); ax.set_xticklabels([f"{m:02d}" for m in x])
    ax.set_ylabel("Stage / Gage Height (ft)")
    ax.set_title(f"Monthly Stage — Full window vs last {recent_years} yrs (Q∩H overlap)")

    n_full   = sum(len(v) for v in v_full)
    n_recent = sum(len(v) for v in v_recent)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              [f"Full (n={n_full} days)", f"Last {recent_years} yrs (n={n_recent} days)"],
              loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_fdc(site_label: str, df: pd.DataFrame, out_png: Path, out_txt_dir: Path):
    f = flow_duration_curve(df["Q_cfs"])
    fig, ax = plt.subplots(figsize=(9.5,6))
    ax.plot(f["ExceedancePct"], f["Q"], lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Exceedance Probability (%)"); ax.set_ylabel("Discharge (cfs)")
    ax.set_title(f"Flow Duration Curve (daily Q) — {site_label}")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)
    for p in (10,50,90):
        qp = np.interp(p, f["ExceedancePct"], f["Q"])
        (out_txt_dir / f"fdc_Q{p}.txt").write_text(f"Q{p} = {qp:0.1f} cfs\n", encoding="utf-8")

def plot_climatology(site_label: str, df_full: pd.DataFrame, df_recent: pd.DataFrame, recent_years: int, out_png: Path):
    # median by DOY for Q and H — distinct colors for H
    def med_by_doy(df, col):
        d = df.copy()
        d["DOY"] = d.index.day_of_year
        # handle leap day by mapping 366->365
        d.loc[d["DOY"]==366, "DOY"] = 365
        return d.groupby("DOY")[col].median()

    q_full = med_by_doy(df_full, "Q_cfs")
    q_last = med_by_doy(df_recent, "Q_cfs")
    h_full = med_by_doy(df_full, "H_ft")
    h_last = med_by_doy(df_recent, "H_ft")

    fig, ax = plt.subplots(figsize=(13.5,6.8))
    ax.plot(q_full.index, q_full.values, color="C0", lw=1.8, label="Q median (full)")
    ax.plot(q_last.index, q_last.values, color="C0", lw=2.2, ls="--", label=f"Q median (last {recent_years})")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_xlabel("Day of Year")
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(h_full.index, h_full.values, color="C1", lw=1.8, label="H median (full)")
    ax2.plot(h_last.index, h_last.values, color="C1", lw=2.2, ls="--", label=f"H median (last {recent_years})")
    ax2.set_ylabel("Stage (ft)")
    ax.set_title(f"Seasonal Climatology — {site_label} (median by DOY)")
    # Build a combined legend
    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

# ------------- Core orchestration ----------------
def build_joined_daily(site_id: str, site_label: str, analysis: dict) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    q_param = analysis.get("q_param","00060"); q_stat = analysis.get("q_stat","00003")
    h_param = analysis.get("h_param","00065"); h_stat = analysis.get("h_stat","00003")
    start_ts = to_midnight_ts(analysis["_resolved_start"])
    end_ts   = to_midnight_ts(analysis["_resolved_end"])

    # Q DV
    df_q = fetch_dv(site_id, q_param, day_str(start_ts), day_str(end_ts), q_stat)
    if df_q.empty:
        raise RuntimeError(f"{site_id} {q_param}: no DV discharge values found.")
    df_q.index = pd.to_datetime(df_q.index).tz_localize(None)
    df_q.index = pd.to_datetime(df_q.index.date)
    df_q = df_q.rename(columns={"value":"Q_cfs"})

    # H DV else IV (limited to plot_years back from END)
    df_h = fetch_dv(site_id, h_param, day_str(start_ts), day_str(end_ts), h_stat)
    if df_h.empty:
        yrs = int(analysis.get("plot_years", 3))
        iv_start = max(start_ts, end_ts - pd.DateOffset(years=yrs) - pd.Timedelta(days=10))
        LOG.info("%s: Stage DV empty — trying IV fallback → daily for last %d yrs.", site_id, yrs)
        df_h = iv_to_daily_cached(site_id, h_param, iv_start, end_ts,
                                  agg=analysis.get("stage_daily_agg","mean"),
                                  cache_dir=analysis.get("cache_dir","cache/usgs"),
                                  chunk_days=int(analysis.get("iv_chunk_days",31)))
    df_h.index = pd.to_datetime(df_h.index).tz_localize(None)
    df_h.index = pd.to_datetime(df_h.index.date)
    df_h = df_h.rename(columns={"value":"H_ft"})

    # restrict to requested window
    df_q = df_q.loc[start_ts:end_ts]
    df_h = df_h.loc[start_ts:end_ts]

    # strict Q∩H overlap
    df = df_q.join(df_h, how="inner").dropna()
    if df.empty:
        raise RuntimeError(f"{site_id}: joined daily Q∩H is empty.")

    return df, df.index.min(), df.index.max()

def process_site(sc: Dict[str,Any], analysis: dict, plot_base: Path, export_csv: bool):
    site_id = (sc.get("id") or sc.get("site") or sc.get("site_id")
               or sc.get("usgs_id") or sc.get("station"))
    if not site_id:
        LOG.error("Site entry missing id-like key: %r", sc)
        raise KeyError("id")
    site_id   = str(site_id)
    site_label = sc.get("label", sc.get("name", site_id))
    LOG.info("--- Processing Site: %s ---", site_id)

    # propagate param/stat from site overrides if present
    for k in ("q_param","q_stat","h_param","h_stat"):
        if k in sc: analysis[k] = sc[k]

    out_dir = Path(plot_base) / site_id
    ensure_dir(out_dir)

    df, ov_start, ov_end = build_joined_daily(site_id, site_label, analysis)

    # power-law and residuals
    A, B, R2, _ = fit_powerlaw_q_of_h(df["H_ft"], df["Q_cfs"])
    if np.isfinite(A) and np.isfinite(B):
        df["Q_hat"] = A * (df["H_ft"] ** B)
        df["resid_pct"] = 100.0 * (df["Q_cfs"] - df["Q_hat"]) / df["Q_hat"]
    else:
        df["Q_hat"] = np.nan; df["resid_pct"] = np.nan

    # export joined series (optional)
    if export_csv:
        (out_dir / f"{site_id}_joined_daily.csv").write_text(df.to_csv(index_label="date"), encoding="utf-8")

    # recent subwindow for comparisons (strictly inside Q∩H)
    recent_years = int(analysis.get("compare_years", 3))
    recent_start = max(ov_start, yr_offset(ov_end, -recent_years))
    df_recent = df.loc[recent_start:ov_end]
    df_full   = df.copy()

    # Plots
    plot_hydrograph(site_label, df_full, out_dir / "hydrograph.png")
    plot_rating_curve(site_label, df_full, A, B, R2, out_dir / "rating_curve.png")
    plot_residuals_time(df_full, out_dir / "residuals_time.png")
    plot_residuals_vs_stage(df_full.dropna(subset=["resid_pct"]), out_dir / "residuals_vs_stage.png")
    plot_monthly_stage(df_full, out_dir / "monthly_stage_box_full.png")
    plot_monthly_stage_compare(df_full, df_recent, recent_years, out_dir / "monthly_stage_compare_full_vs_lastN.png")
    plot_fdc(site_label, df_full, out_dir / "flow_duration_curve.png", out_dir)
    plot_climatology(site_label, df_full, df_recent, recent_years, out_dir / "hydro_climatology_full_vs_lastN.png")

    LOG.info("Saved plots under: %s", out_dir)

    # Quick residuals explainer for you to reuse
    LOG.info(
        "Residuals explanation: residual(%%) = (Q_measured - Q_model) / Q_model * 100.\n"
        "• Positive residuals → the model (rating curve) UNDERESTIMATES discharge at those stages.\n"
        "• Negative residuals → the model OVERESTIMATES discharge.\n"
        "• A systematic slope in 'Residuals vs Stage' means stage-dependent bias (e.g., backwater, "
        "seasonal control changes, or survey datum shifts). The 30-day mean in the time series highlights "
        "seasonal trends, while the shaded ±10%% band shows a 'good fit' zone."
    )

# ------------- CLI / main -------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stage–Discharge analysis (separate PNGs).")
    p.add_argument("-c","--config", default="configs/config_stage_dischargeyakima.json", help="Path to config JSON.")
    p.add_argument("--plot-years", type=int, default=None, help="Years of history for stage IV fallback.")
    p.add_argument("--compare-years", type=int, default=None, help="Years for 'last N' comparisons.")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (overrides plot-years window).")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today).")
    p.add_argument("--plot-base", default="plots/q_stage_discharge", help="Base output dir for plots.")
    p.add_argument("--export-csv", action="store_true", help="Also export joined daily series to CSV.")
    return p.parse_args()

def main():
    LOG.info("--- Starting Stage–Discharge Script ---")
    args = parse_args()
    LOG.info("Base plot directory: %s", args.plot_base)

    cfg = load_config(args.config)
    if cfg is None:
        LOG.critical("No config — exiting.")
        return

    analysis = cfg.get("analysis", {}).copy()
    if args.plot_years is not None:   analysis["plot_years"]   = int(args.plot_years)
    if args.compare_years is not None: analysis["compare_years"] = int(args.compare_years)

    today  = to_midnight_ts(dt.date.today())
    end_ts = to_midnight_ts(args.end) if args.end else today
    if args.start: start_ts = to_midnight_ts(args.start)
    else:
        yrs = int(analysis.get("plot_years", 3))
        start_ts = end_ts - pd.DateOffset(years=yrs) - pd.Timedelta(days=10)

    analysis["_resolved_start"] = day_str(start_ts)
    analysis["_resolved_end"]   = day_str(end_ts)

    plot_base = Path(args.plot_base); ensure_dir(plot_base)

    sites = normalize_sites(cfg.get("sites"))
    if not sites:
        LOG.critical("No sites found in config — exiting.")
        return

    for sc in sites:
        try:
            process_site(sc, analysis, plot_base, export_csv=args.export_csv)
        except Exception as e:
            LOG.exception("Failed site %s: %s", sc.get("id","?"), e)

    LOG.info("--- Stage–Discharge Script Finished ---")

if __name__ == "__main__":
    main()