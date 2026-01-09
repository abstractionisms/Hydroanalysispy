# stageq2.py — robust chunked DV fetch + IV→daily stage + joint-window visuals
# Deps: requests, pandas, numpy, matplotlib
import argparse, datetime as dt, io, logging, math, os, random, time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import requests
import xml.etree.ElementTree as ET

LOG = logging.getLogger("stageq2")
logging.basicConfig(level=logging.INFO, format="%(asctime)s, %(levelname)s - %(message)s")

# ------------------------ Constants ------------------------
WS_DV = "https://waterservices.usgs.gov/nwis/dv/"
WS_IV = "https://nwis.waterservices.usgs.gov/nwis/iv/"
NWIS_MEAS = "https://waterdata.usgs.gov/nwis/measurements"   # RDB (optional)
OUT_BASE_DEFAULT = "plots/greene_stage_discharge"

# ------------------------ Small utils ----------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def today_midnight() -> pd.Timestamp:
    return pd.Timestamp(dt.date.today()).normalize()

def to_day(ts_like) -> pd.Timestamp:
    return pd.Timestamp(ts_like).normalize()

def daterange_years_back(end: pd.Timestamp, years: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    end = to_day(end); start = end - pd.Timedelta(days=int(years*365 + 10))
    return (start, end)

def http_get_text(url: str, params: Dict[str, Any], retries=6, base_sleep=0.8) -> str:
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.text
            LOG.warning("GET %s -> %s", r.url, r.status_code)
            last_err = requests.HTTPError(f"{r.status_code} for {r.url}")
        except requests.RequestException as e:
            last_err = e
            LOG.warning("HTTP error (%d/%d): %s", i+1, retries, e)
        time.sleep(base_sleep * (2**i) + random.random()*0.25)
    if last_err:
        raise last_err
    return ""

def try_parse_json(txt: str) -> Dict[str, Any]:
    if not txt: return {}
    try:
        return requests.models.complexjson.loads(txt)
    except Exception:
        return {}

# ------------------------ DV (Q) chunked -------------------
def _parse_dv_json_series(j: Dict[str, Any]) -> pd.DataFrame:
    try:
        series = j["value"]["timeSeries"][0]["values"][0]["value"]
    except Exception:
        return pd.DataFrame(columns=["value"])
    dates = pd.to_datetime([x.get("dateTime") for x in series], errors="coerce")
    vals  = pd.to_numeric([x.get("value") for x in series], errors="coerce")
    df = pd.DataFrame({"value": vals}, index=dates)
    df = df[~df.index.isna()]
    df.index = pd.to_datetime(df.index.date)
    return df.sort_index()

def _parse_dv_waterml(xml_text: str) -> pd.DataFrame:
    if not xml_text: return pd.DataFrame(columns=["value"])
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame(columns=["value"])
    ts, vs = [], []
    for v in root.findall(".//{*}value"):
        t = v.attrib.get("dateTime")
        s = (v.text or "").strip()
        if not t or s == "": continue
        ts.append(t); vs.append(s)
    if not ts: return pd.DataFrame(columns=["value"])
    idx = pd.to_datetime(ts, errors="coerce")
    vals = pd.to_numeric(vs, errors="coerce")
    s = pd.Series(vals, index=idx).dropna()
    if s.empty: return pd.DataFrame(columns=["value"])
    s.index = pd.to_datetime(s.index.date)
    return s.sort_index().to_frame("value")

def fetch_dv_daily_chunked(
    site: str,
    param: str="00060",
    stat: str="00003",
    start: Optional[pd.Timestamp]=None,
    end: Optional[pd.Timestamp]=None,
    chunk_years: int=5
) -> pd.DataFrame:
    """
    Robust DV fetch split into chunks to avoid 503 throttling.
    1) Try JSON per chunk (startDT/endDT).
    2) If a chunk fails, fall back to WaterML for that chunk.
    """
    if end is None: end = today_midnight()
    if start is None: start = end - pd.Timedelta(days=40*365 + 10)
    start = to_day(start); end = to_day(end)

    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur <= end:
        nxt = min(cur + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1), end)
        chunks.append((cur, nxt))
        cur = nxt + pd.Timedelta(days=1)

    parts: List[pd.DataFrame] = []
    for i, (s, e) in enumerate(chunks, 1):
        # 1) JSON attempt
        try:
            params_json = {
                "format": "json",
                "sites": site,
                "parameterCd": param,
                "statCd": stat,
                "startDT": s.date().isoformat(),
                "endDT": e.date().isoformat(),
            }
            txt = http_get_text(WS_DV, params_json, retries=5)
            j = try_parse_json(txt)
            dfj = _parse_dv_json_series(j)
            if not dfj.empty:
                LOG.info("DV JSON chunk %d/%d %s→%s rows=%d", i, len(chunks), s.date(), e.date(), len(dfj))
                parts.append(dfj); continue
        except Exception as ex_json:
            LOG.warning("DV JSON chunk %d failed (%s→%s): %s", i, s.date(), e.date(), ex_json)

        # 2) WaterML fallback
        try:
            params_xml = {
                "format": "waterml,1.1",
                "sites": site,
                "parameterCd": param,
                "statCd": stat,
                "startDT": s.date().isoformat(),
                "endDT": e.date().isoformat(),
            }
            xml = http_get_text(WS_DV, params_xml, retries=5)
            dfx = _parse_dv_waterml(xml)
            if not dfx.empty:
                LOG.info("DV XML chunk %d/%d %s→%s rows=%d", i, len(chunks), s.date(), e.date(), len(dfx))
                parts.append(dfx); continue
            else:
                LOG.warning("DV XML chunk %d empty (%s→%s).", i, s.date(), e.date())
        except Exception as ex_xml:
            LOG.warning("DV XML chunk %d failed (%s→%s): %s", i, s.date(), e.date(), ex_xml)

    if not parts:
        return pd.DataFrame(columns=["value"])
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]   # de-dup per day
    return out

# ------------------------ IV → Daily (Stage) ----------------
def parse_iv_xml_to_daily(xml_text: str, agg: str="mean") -> pd.DataFrame:
    if not xml_text: return pd.DataFrame(columns=["value"])
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return pd.DataFrame(columns=["value"])
    times, vals = [], []
    for v in root.findall(".//{*}value"):
        t = v.attrib.get("dateTime")
        s = (v.text or "").strip()
        if not t or s == "": continue
        times.append(t); vals.append(s)
    if not times: return pd.DataFrame(columns=["value"])
    t_index = pd.to_datetime(times, utc=True, errors="coerce")
    y = pd.to_numeric(vals, errors="coerce")
    s = pd.Series(y, index=t_index).dropna()
    if s.empty: return pd.DataFrame(columns=["value"])
    s.index = s.index.tz_convert("UTC").tz_localize(None)     # tz-naive
    rule = {"mean":"mean","min":"min","max":"max"}.get(agg.lower(),"mean")
    daily = s.resample("D").agg(rule)
    daily.index = pd.to_datetime(daily.index.date)
    return daily.to_frame("value")

def fetch_iv_daily(site: str, param: str="00065", start: pd.Timestamp=None, end: pd.Timestamp=None,
                   agg="mean", chunk_days=31, cache_dir: Optional[str]="cache/usgs") -> pd.DataFrame:
    assert start is not None and end is not None
    start_d = pd.Timestamp(start).date(); end_d = pd.Timestamp(end).date()
    parts = []; cur = start_d
    cache = Path(cache_dir) if cache_dir else None
    if cache: ensure_dir(cache)

    chunks: List[Tuple[dt.date, dt.date]] = []
    while cur <= end_d:
        nxt = min(cur + dt.timedelta(days=chunk_days), end_d)
        chunks.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)

    for i,(s,e) in enumerate(chunks, 1):
        params = {"format":"waterml,1.1","sites":site,"parameterCd":param,
                  "startDT":s.isoformat(),"endDT":e.isoformat()}
        txt = ""
        cache_path = cache / f"iv_{site}_{param}_{s}_{e}.xml" if cache else None
        if cache_path and cache_path.exists():
            txt = cache_path.read_text(encoding="utf-8", errors="ignore")
            LOG.info("%s %s: IV chunk %d/%d %s→%s (cache hit)", site, param, i, len(chunks), s, e)
        else:
            txt = http_get_text(WS_IV, params, retries=4)
            if cache_path and txt:
                cache_path.write_text(txt, encoding="utf-8")
        if txt:
            df = parse_iv_xml_to_daily(txt, agg=agg)
            if not df.empty: parts.append(df)

    if not parts: return pd.DataFrame(columns=["value"])
    out = pd.concat(parts).sort_index()
    out.index = pd.to_datetime(out.index.date)
    return out

# --------------------- Discrete Measurements (optional) ----
def fetch_discrete_measurements(site: str) -> pd.DataFrame:
    params = {"format":"rdb", "site_no":site, "agency_cd":"USGS"}
    try:
        txt = http_get_text(NWIS_MEAS, params, retries=4)
    except Exception:
        return pd.DataFrame(columns=["measurement_dt","date","Q_meas_cfs","H_meas_ft"])
    if not txt: return pd.DataFrame(columns=["measurement_dt","date","Q_meas_cfs","H_meas_ft"])
    df = pd.read_csv(io.StringIO(txt), sep="\t", comment="#", dtype=str)
    need = {"measurement_dt","discharge_va","gage_height_va"}
    if not need.issubset(set(df.columns)): return pd.DataFrame(columns=["measurement_dt","date","Q_meas_cfs","H_meas_ft"])
    df["measurement_dt"] = pd.to_datetime(df["measurement_dt"], format="ISO8601", errors="coerce")
    df["Q_meas_cfs"] = pd.to_numeric(df["discharge_va"], errors="coerce")
    df["H_meas_ft"] = pd.to_numeric(df["gage_height_va"], errors="coerce")
    df = df.dropna(subset=["measurement_dt","Q_meas_cfs","H_meas_ft"]).copy()
    df["date"] = pd.to_datetime(df["measurement_dt"].dt.date)
    return df[["measurement_dt","date","Q_meas_cfs","H_meas_ft"]].sort_values("measurement_dt")

# --------------------- Analysis helpers ---------------------
def fit_powerlaw_q_of_h(H: pd.Series, Q: pd.Series) -> Tuple[float, float, float, pd.Series]:
    """
    Fit Q = A * H^B on days where H>0 and Q>0.
    Returns A, B, R^2 (on the training subset), and a full-length Qhat Series.
    """
    df = pd.DataFrame({"H": H, "Q": Q}).replace([np.inf, -np.inf], np.nan).dropna()
    mask = (df["H"] > 0) & (df["Q"] > 0)
    train = df.loc[mask]
    if len(train) < 10:
        return (np.nan, np.nan, np.nan, pd.Series(np.nan, index=H.index, dtype=float))

    X = np.log10(train["H"].to_numpy())
    Y = np.log10(train["Q"].to_numpy())
    B, logA = np.polyfit(X, Y, 1)
    A = 10 ** logA

    Qhat_train = A * (train["H"].to_numpy() ** B)
    ss_res = np.sum((train["Q"].to_numpy() - Qhat_train) ** 2)
    ss_tot = np.sum((train["Q"].to_numpy() - train["Q"].mean()) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    Qhat_full = pd.Series(np.nan, index=H.index, dtype=float)
    pos = H > 0
    Qhat_full.loc[pos] = A * (H.loc[pos].to_numpy() ** B)
    return (A, B, R2, Qhat_full)

def fit_powerlaw_q_of_h_offset(H: pd.Series, Q: pd.Series, n_grid: int = 60
) -> tuple[float, float, float, float, pd.Series]:
    """
    Fit Q = A*(H-H0)^B by grid-searching H0 and doing log-log OLS for each H0.
    Returns A, B, H0, R^2 (train subset), and full-length Qhat.
    """
    df = pd.DataFrame({"H": H, "Q": Q}).replace([np.inf, -np.inf], np.nan).dropna()
    df = df[(df["H"] > 0) & (df["Q"] > 0)]
    if len(df) < 20:
        return (np.nan, np.nan, np.nan, np.nan, pd.Series(np.nan, index=H.index, dtype=float))

    hmin = df["H"].min()
    h10  = np.quantile(df["H"], 0.10)
    lo   = hmin - 0.5
    hi   = max(h10 - 0.02, hmin - 0.02)

    best = (np.inf, np.nan, np.nan, np.nan)  # sse, A, B, H0
    for H0 in np.linspace(lo, hi, n_grid):
        mask = df["H"] > H0 + 1e-6
        if mask.sum() < 10:
            continue
        X = np.log10((df.loc[mask, "H"] - H0).to_numpy())
        Y = np.log10(df.loc[mask, "Q"].to_numpy())
        B, logA = np.polyfit(X, Y, 1)
        pred = 10**(logA + B*X)
        sse = np.sum((10**Y - pred)**2)
        if sse < best[0]:
            best = (sse, 10**logA, B, H0)

    _, A, B, H0 = best
    if not np.isfinite(A):
        return (np.nan, np.nan, np.nan, np.nan, pd.Series(np.nan, index=H.index, dtype=float))

    m = df["H"] > H0 + 1e-6
    Qall = df.loc[m, "Q"].to_numpy()
    Qhat_train = A * ((df.loc[m, "H"].to_numpy() - H0) ** B)
    ss_res = np.sum((Qall - Qhat_train)**2)
    ss_tot = np.sum((Qall - Qall.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan

    Qhat_full = pd.Series(np.nan, index=H.index, dtype=float)
    pos = H > H0 + 1e-6
    Qhat_full.loc[pos] = A * ((H.loc[pos].to_numpy() - H0) ** B)
    return (A, B, H0, R2, Qhat_full)

def flow_duration_curve(Q: pd.Series) -> pd.DataFrame:
    q = pd.Series(Q).dropna().sort_values(ascending=False).to_numpy()
    if q.size == 0: return pd.DataFrame(columns=["ExceedancePct","Q"])
    ranks = np.arange(1, q.size+1)
    exc = 100.0 * ranks / (q.size + 1)
    return pd.DataFrame({"ExceedancePct": exc, "Q": q})

# ------------------------- Header helper --------------------
def build_header_text(df_joint: pd.DataFrame,
                      Q_dv: Optional[pd.DataFrame],
                      A: float, B: float, H0: float, R2: float) -> str:
    """Make a compact header string for tiny box."""
    if df_joint.empty:
        return ""
    w0 = df_joint.index.min().date()
    w1 = df_joint.index.max().date()
    n_days = len(df_joint)
    # Coverage relative to DV rows in the same window (if DV provided)
    if Q_dv is not None and not Q_dv.empty:
        dv_win = Q_dv.loc[df_joint.index.min():df_joint.index.max()]
        cov = 100.0 * n_days / max(1, len(dv_win))
    else:
        cov = 100.0
    win_txt = f"Data: {w0}–{w1}"
    n_txt = f"N days: {n_days:,}  |  Coverage: {cov:.1f}%"
    if np.isfinite(A) and np.isfinite(B):
        if np.isfinite(H0):
            fit_txt = f"Fit: Q = {A:.2f}·(H−H₀)^{B:.3f},  H₀={H0:.3f},  R²={R2:.3f}"
        else:
            fit_txt = f"Fit: Q = {A:.2f}·H^{B:.3f},  R²={R2:.3f}"
    else:
        fit_txt = "Fit: n/a"
    return f"{win_txt}  |  {n_txt}\n{fit_txt}"

# ------------------------- Plots ----------------------------
def plot_hydrograph(site_label: str, df: pd.DataFrame, meas: pd.DataFrame, out_png: Path):
    fig, ax = plt.subplots(figsize=(11,6))
    ax.plot(df.index, df["Q_cfs"], lw=1.4, label="Discharge (cfs)")
    ax.plot(df.index, df["Q_cfs"].rolling(7, min_periods=1).mean(), lw=1.8, label="Q 7-day mean")
    ax.set_yscale("log"); ax.set_ylabel("Discharge (cfs)"); ax.set_xlabel("Date")
    ax.grid(True, which="both", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(df.index, df["H_ft"], lw=1.0, alpha=0.6, label="Stage (ft)")
    ax2.set_ylabel("Stage (ft)")
    if not meas.empty:
        md = meas["date"].unique()
        y_on_days = df.loc[df.index.isin(md), "Q_cfs"]
        ax.scatter(y_on_days.index, y_on_days.values, s=36, marker="o", edgecolor="k", linewidths=0.6,
                   label="Discrete gaging days")
    ax.set_title(f"Hydrograph — {site_label}")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc="upper left", ncol=2, frameon=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_rating_with_measurements(site_label: str, df: pd.DataFrame,
                                  A: float, B: float, R2: float, H0: float,
                                  meas: pd.DataFrame, out_png: Path,
                                  header_text: str = ""):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(df["H_ft"], df["Q_cfs"], s=12, alpha=0.35, label="Daily pairs")
    if np.isfinite(A) and np.isfinite(B):
        h = np.linspace(max(1e-6, df["H_ft"].min()), df["H_ft"].max(), 400)
        if np.isfinite(H0):
            ax.plot(h, A*np.maximum(h-H0,1e-9)**B, lw=2.1,
                    label=f"Fit: Q={A:0.3g}·(H−{H0:0.3f})^{B:0.3f} (R²={R2:0.3f})")
        else:
            ax.plot(h, A*(h**B), lw=2.1,
                    label=f"Fit: Q={A:0.3g}·H^{B:0.3f} (R²={R2:0.3f})")
    if not meas.empty:
        ax.scatter(meas["H_meas_ft"], meas["Q_meas_cfs"], s=55, marker="*", edgecolor="k",
                   linewidths=0.6, label=f"Discrete gaugings (n={len(meas)})")
    ax.set_xlabel("Stage / Gage Height (ft)")
    ax.set_ylabel("Discharge (cfs)")
    ax.set_yscale("log"); ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(f"Rating Curve — {site_label}")

    # Tiny header box
    if header_text:
        ax.text(0.02, 0.98, header_text,
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_residuals_time(df: pd.DataFrame, out_png: Path):
    if "resid_pct" not in df.columns or df["resid_pct"].dropna().empty: return
    s = df["resid_pct"].copy()
    roll = s.rolling(30, min_periods=10, center=True).mean()
    fig, ax = plt.subplots(figsize=(11,4.3))
    ax.axhline(0, color="k", lw=1)
    ax.fill_between(s.index, -10, 10, color="grey", alpha=0.12, label="±10% band")
    ax.plot(s.index, s.values, lw=0.8, alpha=0.8, label="Residuals (daily)")
    ax.plot(roll.index, roll.values, lw=1.8, label="30-day mean")
    ax.set_ylabel("Residuals (%)"); ax.set_xlabel("Date")
    ax.set_title("Rating Residuals Through Time (published DV vs fitted)")
    ax.grid(True, which="both", alpha=0.25); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_residuals_vs_stage(df: pd.DataFrame, out_png: Path):
    if "resid_pct" not in df.columns or df["resid_pct"].dropna().empty: return
    H, R = df["H_ft"], df["resid_pct"]
    bins = np.linspace(H.min(), H.max(), 24)
    cats = pd.cut(H, bins, include_lowest=True)
    group = R.groupby(cats, observed=False)
    mid = pd.Series([(b.left+b.right)/2 for b in group.groups.keys()], index=group.median().index)
    med = group.median().values
    q25 = group.quantile(0.25).values
    q75 = group.quantile(0.75).values

    fig, ax = plt.subplots(figsize=(9.8,5.2))
    ax.axhline(0, color="k", lw=1)
    ax.scatter(H, R, s=12, alpha=0.35, label="Daily")
    ax.plot(mid, med, lw=2.0, label="Binned median")
    ax.fill_between(mid, q25, q75, alpha=0.18, label="IQR band")
    ax.set_xlabel("Stage / Gage Height (ft)"); ax.set_ylabel("Residuals (%)")
    ax.set_title("Residuals vs Stage — bias pattern by stage"); ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

# ---- Ratio-style diagnostics (with header box) ----
def plot_ratio_time(df: pd.DataFrame, out_png: Path, header_text: str = ""):
    if not {"Q_cfs","Q_hat"}.issubset(df.columns): return
    r = (df["Q_cfs"] / df["Q_hat"]).replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty: return
    roll = r.rolling(30, min_periods=10, center=True).median()
    fig, ax = plt.subplots(figsize=(11,4.3))
    ax.axhline(1.0, color="k", lw=1)
    ax.fill_between(r.index, 0.9, 1.1, alpha=0.12, label="±10% band")
    ax.plot(r.index, r.values, lw=0.8, alpha=0.85, label="Q_pub / Q_pred (daily)")
    ax.plot(roll.index, roll.values, lw=1.8, label="30-day median")
    ax.set_ylabel("Ratio (×)"); ax.set_xlabel("Date")
    ax.set_title("Performance Ratio Through Time")
    ax.grid(True, alpha=0.25); ax.legend(loc="upper left")

    # Tiny header box
    if header_text:
        ax.text(0.02, 0.98, header_text,
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_ratio_vs_stage(df: pd.DataFrame, out_png: Path, header_text: str = ""):
    if not {"Q_cfs","Q_hat","H_ft"}.issubset(df.columns): return
    ok = df[["H_ft","Q_cfs","Q_hat"]].replace([np.inf,-np.inf], np.nan).dropna()
    if ok.empty: return
    r, H = ok["Q_cfs"]/ok["Q_hat"], ok["H_ft"]
    bins = np.linspace(H.min(), H.max(), 24)
    cats = pd.cut(H, bins, include_lowest=True)
    med = r.groupby(cats, observed=False).median()
    q25 = r.groupby(cats, observed=False).quantile(0.25)
    q75 = r.groupby(cats, observed=False).quantile(0.75)
    centers = np.array([(b.left + b.right)/2 for b in med.index])

    fig, ax = plt.subplots(figsize=(9.8,5.2))
    ax.axhline(1.0, color="k", lw=1)
    ax.scatter(H, r, s=12, alpha=0.35, label="Daily")
    ax.plot(centers, med.values, lw=2.0, label="Binned median")
    ax.fill_between(centers, q25.values, q75.values, alpha=0.18, label="IQR band")
    ax.set_xlabel("Stage / Gage Height (ft)")
    ax.set_ylabel("Q_pub / Q_pred (×)")
    ax.set_title("Bias vs Stage (as Ratio)")
    ax.grid(True, alpha=0.25); ax.legend()

    # Tiny header box
    if header_text:
        ax.text(0.02, 0.98, header_text,
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_monthly_stage_compare(df_joint: pd.DataFrame, end: pd.Timestamp, lastN: int, out_png: Path):
    d = df_joint[["H_ft"]].dropna().copy()
    if d.empty: return
    d["Month"] = d.index.month
    full_med = d.groupby("Month")["H_ft"].median()
    startN = end - pd.Timedelta(days=int(lastN*365 + 10))
    dN = d.loc[startN:end]
    last_med = dN.groupby("Month")["H_ft"].median()
    months = np.arange(1,13)
    fig, ax = plt.subplots(figsize=(10.5,5.2))
    ax.plot(months, full_med.reindex(months), lw=2.0, label="Full joint median (H)")
    ax.plot(months, last_med.reindex(months), lw=2.0, ls="--", label=f"Last {lastN} yr median (H)")
    ax.set_xticks(months); ax.set_xlabel("Month"); ax.set_ylabel("Stage (ft)")
    ax.set_title("Monthly Stage Climatology — Full vs Recent (joint Q∩H)")
    ax.grid(True, alpha=0.25); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_monthly_stage_box(df_joint: pd.DataFrame, out_png: Path):
    d = df_joint[["H_ft"]].dropna().copy()
    if d.empty: return
    d["Month"] = d.index.month
    vals = [d.loc[d["Month"]==m, "H_ft"].values for m in range(1,13)]
    counts = [len(v) for v in vals]
    labels = [f"{m:02d}\n(n={c})" for m, c in zip(range(1,13), counts)]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot(vals, tick_labels=labels, showfliers=False)
    ax.set_title("Monthly Stage Distribution (Joint Q∩H)"); ax.set_ylabel("Stage (ft)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

# ---- New: monthly MEANS box (one point per month per year) ----
def plot_monthly_stage_box_means(daily: pd.DataFrame, out_png: Path):
    d = daily[["H_ft"]].dropna().copy()
    if d.empty: return
    m = d.resample("MS").mean()  # monthly mean stage
    m["Month"] = m.index.month
    vals = [m.loc[m["Month"]==k, "H_ft"].values for k in range(1,13)]
    labels = [f"{k:02d}\n(n={len(v)})" for k,v in zip(range(1,13), vals)]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.boxplot(vals, tick_labels=labels, showfliers=False)
    ax.set_title("Monthly Mean Stage Distribution (one point = one month)")
    ax.set_ylabel("Stage (ft)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

def plot_fdc(site_label: str, df_joint: pd.DataFrame, out_png: Path):
    if "Q_cfs" not in df_joint.columns: return
    f = flow_duration_curve(df_joint["Q_cfs"])
    if f.empty: return
    fig, ax = plt.subplots(figsize=(9.8,5.6))
    ax.plot(f["ExceedancePct"], f["Q"], lw=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Exceedance Probability (%)"); ax.set_ylabel("Discharge (cfs)")
    ax.set_title(f"Flow Duration Curve (daily) — {site_label}")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=220); plt.close(fig)

# -------------------------- Main ----------------------------
def main(site: str, lastN: int, iv_years: int, out_base: str, no_dv: bool):
    end = today_midnight()
    site_label = f"USGS {site}"

    # Stage via IV→daily across a long history
    start_iv, end_iv = daterange_years_back(end, iv_years)
    H = fetch_iv_daily(site, "00065", start=start_iv, end=end_iv, agg="mean", chunk_days=31, cache_dir="cache/usgs")
    if H.empty:
        raise RuntimeError("No IV→daily stage available.")
    H.columns = ["H_ft"]; H.index = pd.to_datetime(H.index).normalize()

    # Discharge DV daily (chunked) unless disabled
    if not no_dv:
        Q = fetch_dv_daily_chunked(site, "00060", "00003",
                                   start=end - pd.Timedelta(days=40*365 + 10),
                                   end=end, chunk_years=5)
        if Q.empty:
            LOG.warning("DV daily discharge still empty after chunking; proceeding with stage-only visuals.")
            no_dv = True
        else:
            Q.columns = ["Q_cfs"]; Q.index = pd.to_datetime(Q.index).normalize()
    else:
        Q = pd.DataFrame()

    # De-dup & sort right before the join
    if not no_dv and not Q.empty:
        Q = Q.sort_index().loc[~Q.index.duplicated(keep="last")]
    H = H.sort_index().loc[~H.index.duplicated(keep="last")]

    # Joint Q∩H for rating/residuals; else stage-only dataframe
    if not no_dv and not Q.empty:
        df = Q.join(H, how="inner").dropna()
        if df.empty:
            LOG.warning("Joint daily Q∩H is empty—falling back to stage-only visuals.")
            no_dv = True
    if no_dv:
        df = H.copy()

    out_dir = Path(out_base) / site
    ensure_dir(out_dir)

    # Fit and diagnostics
    if not no_dv:
        # Prefer offset fit; fall back to simple if needed
        try:
            A,B,H0,R2,Qhat = fit_powerlaw_q_of_h_offset(df["H_ft"], df["Q_cfs"])
        except Exception:
            A=B=H0=R2=np.nan; Qhat = pd.Series(np.nan, index=df.index, dtype=float)
        if not np.isfinite(A):
            A,B,R2,Qhat = fit_powerlaw_q_of_h(df["H_ft"], df["Q_cfs"])
            H0 = np.nan

        df["Q_hat"] = Qhat
        df["ratio"]  = df["Q_cfs"] / df["Q_hat"]   # 1.0 = perfect
        df["resid_pct"] = 100.0 * (df["Q_cfs"] - df["Q_hat"]) / df["Q_hat"]  # kept for CSV/back-compat

        LOG.info("Rating fit: A=%.6g, B=%.4f%s, R^2=%.3f, n=%d",
                 A, B, (f', H0={H0:.3f}' if np.isfinite(H0) else ""), R2,
                 df[["H_ft","Q_cfs"]].dropna().shape[0])

        # Build header text for tiny boxes
        header_text = build_header_text(df, Q, A, B, H0, R2)
    else:
        A=B=R2=H0=np.nan
        df["Q_hat"] = np.nan; df["ratio"] = np.nan; df["resid_pct"] = np.nan
        header_text = ""  # no DV/fit context

    # Optional: discrete measurements (non-blocking)
    try:
        meas = fetch_discrete_measurements(site)
        if not meas.empty and np.isfinite(A) and np.isfinite(B):
            meas["Q_fit_at_meas"] = (A * np.maximum(meas["H_meas_ft"] - (H0 if np.isfinite(H0) else 0.0), 1e-9) ** B
                                     if np.isfinite(H0) else A * (meas["H_meas_ft"] ** B))
            meas["resid_meas_vs_fit_pct"] = 100.0 * (meas["Q_meas_cfs"] - meas["Q_fit_at_meas"]) / meas["Q_fit_at_meas"]
            mjoin = meas.merge(
                df[["Q_cfs"]].reset_index().rename(columns={"index":"date"}),
                on="date", how="left"
            )
            meas["Q_dv_same_day"] = mjoin["Q_cfs"].values
            meas["resid_meas_vs_dv_pct"] = 100.0 * (meas["Q_meas_cfs"] - meas["Q_dv_same_day"]) / meas["Q_dv_same_day"]
            meas.to_csv(out_dir / "discrete_measurements_summary.csv", index=False)
    except Exception as e:
        LOG.warning("Measurements fetch failed (non-fatal): %s", e)
        meas = pd.DataFrame()

    # Plots
    plot_hydrograph(site_label,
                    df if "Q_cfs" in df else H.rename(columns={"H_ft":"Q_cfs"}),
                    meas,
                    out_dir / "01_hydrograph.png")
    if not no_dv:
        plot_rating_with_measurements(site_label, df, A, B, R2, H0, meas,
                                      out_dir / "02_rating_with_gaugings.png",
                                      header_text=header_text)
        # Ratio diagnostics (with tiny header box)
        plot_ratio_time(df, out_dir / "03_ratio_time.png", header_text=header_text)
        plot_ratio_vs_stage(df, out_dir / "04_ratio_vs_stage.png", header_text=header_text)
        # Optional legacy percent-residuals (commented out by default):
        # plot_residuals_time(df, out_dir / "03_residuals_time.png")
        # plot_residuals_vs_stage(df.dropna(subset=["resid_pct"]), out_dir / "04_residuals_vs_stage.png")
        plot_fdc(site_label, df, out_dir / "08_flow_duration_curve.png")

    # Monthly + climatology (stage-based)
    end_for_recent = df.index.max()
    plot_monthly_stage_compare(df if "H_ft" in df else H, end_for_recent, lastN, out_dir / "06_monthly_stage_climatology_full_vs_lastN.png")
    plot_monthly_stage_box(df if "H_ft" in df else H, out_dir / "07_monthly_stage_box.png")
    # New: monthly MEANS box (fixes Feb “short month” bias)
    plot_monthly_stage_box_means(df if "H_ft" in df else H, out_dir / "07b_monthly_stage_box_means.png")

    LOG.info("Saved plots under: %s", out_dir)
    if not no_dv:
        LOG.info(
            "Interpretation:\n"
            "  Rating uses Q = A*(H - H0)^B when possible (H0 allows non-zero effective stage).\n"
            "  Diagnostic ratio = Q_published / Q_pred. 1.0 = perfect; 1.2 = 20%% high; 0.8 = 20%% low."
        )
    else:
        LOG.info("Ran in stage-only mode (no DV). Use --no-dv to force this mode explicitly if desired.")

if __name__ == "__main__":
    import json

    ap = argparse.ArgumentParser(description="Stage–Discharge with robust DV chunking + offset rating + ratio diagnostics")
    ap.add_argument("--site", default="12422000", help="USGS site id")
    ap.add_argument("--lastN", type=int, default=5, help="Recent window for monthly compare (years)")
    ap.add_argument("--iv-years", type=int, default=25, help="How many years of IV→daily stage to pull")
    ap.add_argument("--out-base", default=OUT_BASE_DEFAULT, help="Base output folder")
    ap.add_argument("--no-dv", action="store_true", help="Skip DV fetch; produce stage-only visuals")
    ap.add_argument("--config", help="Path to JSON config with keys: site, lastN, iv_years, out_base, no_dv")

    args = ap.parse_args()

    # Defaults as declared above:
    defaults = {
        "site": "12422000",
        "lastN": 5,
        "iv_years": 25,
        "out_base": OUT_BASE_DEFAULT,
        "no_dv": False,
    }

    cfg = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            LOG.info("Loaded config from %s", args.config)
        except Exception as e:
            LOG.warning("Failed to load config %s (ignoring): %s", args.config, e)

    # Helper: if user didn’t override a CLI option (still at default),
    # take the value from config (if present); otherwise keep the CLI/default.
    def pick(key, cli_val):
        if key not in defaults:
            return cli_val
        used_default = (cli_val == defaults[key])
        return cfg.get(key, cli_val) if used_default else cli_val

    site = pick("site", args.site)
    lastN = int(pick("lastN", args.lastN))
    iv_years = int(pick("iv_years", args.iv_years))
    out_base = pick("out_base", args.out_base)

    # For booleans, if user passed --no-dv we keep True; otherwise use config value if provided.
    no_dv = args.no_dv if args.no_dv else bool(cfg.get("no_dv", defaults["no_dv"]))

    main(site, lastN, iv_years, out_base, no_dv)
