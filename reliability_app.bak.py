# reliability_app.py
# ----------------------------------------------------
# Site Reliability App (v1.8.2)
# Philosophy: ADD, never remove—this keeps your logic and upgrades safely.
#
# This build:
# - Fixes: NameError (tab1/df vars), pd.to_numeric TypeError (duplicate/missing cols)
# - Adds: robust numeric helpers, safer data-editor mapping, runtime <-> assets sync
# - UI: Available Life pie, status distribution pie, hydraulic system calculator + animation
# - Keeps: Your assets/runtime/history structures & workflow
# ----------------------------------------------------

import os, json, math, re, base64, logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from prophet import Prophet    # keep available for forecasts you plan to add
import plotly.graph_objects as go

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# ====== Page & CSS ======
st.set_page_config(page_title="Site Reliability App", layout="wide")

def inject_css():
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"]{
      background: radial-gradient(1200px 600px at 15% 10%, rgba(255,255,255,0.35), transparent 60%),
                  radial-gradient(900px 500px at 85% 20%, rgba(255,255,255,0.25), transparent 60%),
                  #87CEEB !important;
      color: #0A0F1A;
    }
    [data-testid="stSidebar"]{
      backdrop-filter: blur(8px);
      background: rgba(255,255,255,0.55) !important;
      border-right: 1px solid rgba(0,0,0,0.06);
    }
    html, body, [class*="css"] { color: #0A0F1A !important; }
    div[data-testid="stMetric"]{
      background: rgba(255,255,255,0.65);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 14px;
      padding: 14px 12px;
      box-shadow: 0 10px 24px rgba(0,0,0,0.15);
    }
    .stDataFrame{
      border-radius: 12px !important;
      overflow: hidden !important;
      box-shadow: 0 8px 20px rgba(0,0,0,0.15);
      background: rgba(255,255,255,0.85);
    }
    .stTabs [data-baseweb="tab-list"]{ gap: 6px; }
    .stTabs [data-baseweb="tab"]{
      background: rgba(255,255,255,0.7);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 10px;
      padding: 8px 12px;
    }
    .stDataFrame div[role="gridcell"] div, .stDataFrame td div {
      white-space: pre-wrap !important;
      line-height: 1.3;
    }
    </style>
    """, unsafe_allow_html=True)

def page_header():
    st.markdown("""
    <div style="
      padding: 18px 22px; border-radius: 16px;
      background: linear-gradient(135deg, rgba(10,111,255,0.20), rgba(10,111,255,0.06));
      border: 1px solid rgba(10,111,255,0.25);
      box-shadow: 0 12px 28px rgba(0,0,0,0.18);
      display:flex; align-items:center; gap:14px; margin-bottom:8px;">
      <div style="font-size:26px;">⚙️ <b>Site Reliability App</b></div>
      <div style="opacity:.8;">Pumps • Runtime • Maintenance • System Curves</div>
    </div>
    """, unsafe_allow_html=True)

def set_app_background(img_bytes: bytes, *, darken: float = 0.25, blur_px: int = 0, fixed: bool = True):
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    att = "fixed" if fixed else "scroll"
    overlay_rgba = f"rgba(0,0,0,{max(0.0, min(1.0, darken))})"
    st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background:
          linear-gradient({overlay_rgba}, {overlay_rgba}),
          url("data:image/jpeg;base64,{b64}") {att} center / cover no-repeat !important;
    }}
    [data-testid="stSidebar"], .block-container {{ backdrop-filter: blur({blur_px}px); }}
    </style>
    """, unsafe_allow_html=True)

inject_css()
page_header()

# ====== Files & Load/Save ======
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
ASSETS_JSON = os.path.join(DATA_DIR, "assets.json")
RUNTIME_CSV = os.path.join(DATA_DIR, "runtime.csv")
HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")
ATTACH_DIR = os.path.join(DATA_DIR, "attachments")
os.makedirs(ATTACH_DIR, exist_ok=True)

def load_assets():
    if os.path.exists(ASSETS_JSON):
        try:
            with open(ASSETS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
        except Exception:
            pass
    # Seed with sample
    return {
        "408-PP-001": {
            "Functional Location": "Primary Mill Discharge",
            "Model": "20/18 - AH-TU (WRT)",
            "Motor Power": "1450 kW",
            "Sealing": "L/F S-Box",
            "Drive Type": "GEARBOX COUPLED",
            "Motor Shaft Size (mm)": 140,
            "Pump Shaft Size (mm)": 150,
            "Max Speed (RPM)": 396.0,
            "Avg Speed (VSD RPM)": 324.72,
            "Max Capacity (m³/h)": 4200.0,
            "TDH (m)": 33.0,
            "BOM": ["T118-1-K31 • LANTERN RESTRICTOR • AM0525666",
                    "7189T1116-2018TUAH • PACKING SET OF 4 • AM1806819",
                    "H045M-E62 • GLAND BOLT • AM0533176",
                    "O-RING (69T164N439)",
                    "BEARING ASSEMBLY"],
            "CMMS Codes": ["AM0533180","AM0533179","AM0526665","AM0533181","AM0533170"],
        },
        "408-PP-008": {
            "Functional Location": "Secondary Mill Discharge",
            "Model": "550-MCR-TU",
            "Motor Power": "1450 kW",
            "Sealing": "L/F S-Box",
            "Drive Type": "COUPLED",
            "Motor Shaft Size (mm)": 140,
            "Pump Shaft Size (mm)": 150,
            "Max Speed (RPM)": 352.8,
            "Avg Speed (VSD RPM)": 289.3,
            "Max Capacity (m³/h)": 5000.0,
            "TDH (m)": 27.0,
            "BOM": ["UMC55044C23 WEIRMIN","UMC50872C3 WEIRMIN","M1800076C21 WEIRMIN"],
            "CMMS Codes": ["AM2454827","AM2454828","AM2454829"],
        },
    }

def save_assets(assets_dict: dict):
    with open(ASSETS_JSON, "w", encoding="utf-8") as f:
        json.dump(assets_dict, f, indent=2, ensure_ascii=False)

def load_runtime():
    if os.path.exists(RUNTIME_CSV):
        try:
            return pd.read_csv(RUNTIME_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "Asset Tag","Functional Location","Asset Model","MTBF (Hours)","Last Overhaul",
        "Running Hours Since Last Major Maintenance","Remaining Hours"
    ])

def save_runtime(df: pd.DataFrame):
    df.to_csv(RUNTIME_CSV, index=False)

def load_history():
    if os.path.exists(HISTORY_CSV):
        try:
            return pd.read_csv(HISTORY_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=[
        "Asset Tag","Functional Location","Asset Model","Date of Maintenance",
        "Reason","Spares Used","Hours Since Last","MTTR (hrs)"
    ])

def save_history(df: pd.DataFrame):
    df.to_csv(HISTORY_CSV, index=False)

# ====== Session & Utils ======
if "assets" not in st.session_state:
    st.session_state.assets = load_assets()
if "runtime_df" not in st.session_state:
    st.session_state.runtime_df = load_runtime()
if "history_df" not in st.session_state:
    st.session_state.history_df = load_history()

def to_num(x, default=0.0):
    """Coerce any cell-like value (scalars, strings, Series/arrays) to a clean float."""
    try:
        if isinstance(x, (np.ndarray, list, tuple)):
            if len(x) == 0:
                return default
            x = x[0]
        if isinstance(x, pd.Series):
            if x.empty:
                return default
            x = x.iloc[0]
        if x is None:
            return default
        if isinstance(x, (int, float, np.floating)):
            return float(x) if np.isfinite(x) else default
        s = str(x).strip().lower().replace(',', '')
        if s in {'', '-', '—', 'none', 'nan', 'n/a', 'na'}:
            return default
        val = float(s)
        return val if np.isfinite(val) else default
    except Exception:
        return default

def compute_remaining_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Remaining Hours = MTBF (Hours) - Running Hours Since Last Major Maintenance
    Robust to: non-DF, duplicate columns, missing columns.
    """
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame(columns=[
                "Asset Tag","Functional Location","Asset Model","MTBF (Hours)","Last Overhaul",
                "Running Hours Since Last Major Maintenance","Remaining Hours"
            ])
    df = df.copy()

    run_long = "Running Hours Since Last Major Maintenance"
    run_short = "Hrs Run Since"

    if run_short in df.columns and run_long not in df.columns:
        df = df.rename(columns={run_short: run_long})

    if (df.columns == run_long).sum() > 1:
        new_cols = []
        seen = 0
        for c in df.columns:
            if c == run_long:
                seen += 1
                new_cols.append(c if seen == 1 else f"{c} (dup{seen-1})")
            else:
                new_cols.append(c)
        df.columns = new_cols

    if "MTBF (Hours)" not in df.columns:
        df["MTBF (Hours)"] = 0.0
    if run_long not in df.columns:
        df[run_long] = 0.0

    def _series(frame, name, fallback=0.0):
        if name in frame.columns:
            obj = frame[name]
            if isinstance(obj, pd.DataFrame):
                s = obj.iloc[:, 0]
                s.name = name
                return s
            return obj
        return pd.Series([fallback] * len(frame), index=frame.index, name=name)

    mtbf_s = pd.to_numeric(_series(df, "MTBF (Hours)"), errors="coerce").fillna(0.0)
    run_s  = pd.to_numeric(_series(df, run_long), errors="coerce").fillna(0.0)

    df["Remaining Hours"] = mtbf_s.astype(float) - run_s.astype(float)
    return df

def get_numeric_series(df: pd.DataFrame, colname: str) -> pd.Series:
    """
    Always return a 1-D numeric Series for df[colname], even if the column is duplicated or missing.
    Fixes: pandas to_numeric TypeError (when value was a DataFrame or None).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    if colname not in df.columns:
        return pd.Series(dtype=float, index=df.index)
    s = df[colname]
    if isinstance(s, pd.DataFrame):
        if s.shape[1] == 0:
            return pd.Series(dtype=float, index=df.index)
        s = s.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")

BOM_COLS  = ["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE"]
HIST_COLS = [
    "Asset Tag","Functional Location","Asset Model","Date of Maintenance",
    "Reason","Spares Used","Hours Since Last","MTTR (hrs)",
    # New standardized fields:
    "Maintenance Code","Trigger","Running Hours Since Last Repair","Labour Hours",
    "Parts Used","Downtime Hours","Notes/Findings","Attachments"
]
TRIGGERS = {
    "PM01": [
        "Wear/age MTBF threshold reached",
        "Condition monitoring (vibration/thermography/ultrasound/oil)",
        "Performance drift (flow/head/power/efficiency)",
        "OEM recommended interval",
        "Time/usage based PM (calendar/hours)",
        "Opportunity-based (shutdown/turnaround)",
        "Operating context change (duty/start/stop/solids)",
        "Compliance/statutory requirement",
        "Project upgrade/commissioning follow-up",
        "LCC-Based",
        "Packings Replacement",
        "Other (specify)",
    ],
    "PM02": [
        "Functional failure occurred",
        "Potential failure (P–F interval) requires immediate action",
        "Operator report (noise/leak/heat/smell)",
        "Alarm/interlock/protection trip",
        "Seal leak/flush loss",
        "High vibration / imbalance / misalignment",
        "Over-temperature",
        "Electrical fault / motor protection trip",
        "Cavitation / suction starvation",
        "Blockage / foreign object",
        "Lubrication failure",
        "Instrument/PLC control fault",
        "Packings Replacement",
        "Other (specify)",
    ],
    "PM03": [
        "Visual walkdown / housekeeping",
        "Vibration route",
        "Thermography route",
        "Ultrasound route",
        "Lubrication check / top-up",
        "Alignment / fastener check",
        "Instrument reading verification",
        "Safety / guarding inspection",
        "Other (specify)",
    ],
    "PM00": [
        "Bearing seizure",
        "Mechanical seal catastrophic failure",
        "Shaft / coupling failure",
        "Impeller / volute damage",
        "Motor burn-out",
        "Cavitation damage leading to trip",
        "Suction blockage / no-flow",
        "Dry-run overheating",
        "Flooding / ingress damage",
        "Power quality / outage induced trip",
        "Gearbox failure",
        "Foundation/frame failure",
        "Worn Out Packings — Sliming Event",
        "Other (specify)",
    ],
}
def triggers_for(code: str):
    return TRIGGERS.get(str(code).strip(), [])

# Flat list for the grid dropdown (exclude "Other (specify)")
TRIG_PM01 = [t for t in TRIGGERS["PM01"] if t != "Other (specify)"]
TRIG_PM02 = [t for t in TRIGGERS["PM02"] if t != "Other (specify)"]
TRIG_PM03 = [t for t in TRIGGERS["PM03"] if t != "Other (specify)"]
TRIG_PM00 = [t for t in TRIGGERS["PM00"] if t != "Other (specify)"]

ALL_TRIGS = sorted(set(TRIG_PM01 + TRIG_PM02 + TRIG_PM03 + TRIG_PM00))

def triggers_for(code: str):
    return TRIGGERS.get(str(code).strip(), [])

# Flat list for the grid dropdown (exclude "Other" so we can append it once)
ALL_TRIGS = sorted({t for v in TRIGGERS.values() for t in v if t != "Other (specify)"})

def build_bom_table(asset: dict) -> pd.DataFrame:
    bt = asset.get("BOM Table")
    if isinstance(bt, list) and bt:
        rows=[]
        for r in bt:
            rows.append({
                "CMMS MATERIAL CODE": r.get("CMMS MATERIAL CODE","") or r.get("CMMS",""),
                "OEM PART NUMBER":    r.get("OEM PART NUMBER","") or r.get("OEM",""),
                "DESCRIPTION":        r.get("DESCRIPTION",""),
                "QUANTITY":           r.get("QUANTITY",1),
            })
        return pd.DataFrame(rows, columns=BOM_COLS)
    bom_list = asset.get("BOM", []) or []
    cmms_list = asset.get("CMMS Codes", []) or []
    n = max(len(bom_list), len(cmms_list))
    rows=[]
    for i in range(n):
        rows.append({
            "CMMS MATERIAL CODE": cmms_list[i] if i < len(cmms_list) else "",
            "OEM PART NUMBER":    bom_list[i]  if i < len(bom_list)  else "",
            "DESCRIPTION":        "",
            "QUANTITY":           1,
        })
    return pd.DataFrame(rows, columns=BOM_COLS)

def normalize_bom_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df, list):
        df = pd.DataFrame(df)
    elif isinstance(df, dict):
        if "data" in df and isinstance(df["data"], list):
            df = pd.DataFrame(df["data"])
        else:
            df = pd.DataFrame([df])

    df = df.copy() if df is not None else pd.DataFrame(columns=BOM_COLS)

    # Ensure required columns exist
    for c in BOM_COLS:
        if c not in df.columns:
            df[c] = "" if c != "QUANTITY" else 1

    # Types & cleaning
    df["QUANTITY"] = pd.to_numeric(df["QUANTITY"], errors="coerce").fillna(1).astype(int)
    if "PRICE" not in df.columns:
        df["PRICE"] = 0.0
    df["PRICE"] = pd.to_numeric(df["PRICE"], errors="coerce").fillna(0.0).astype(float)

    # Clean text fields (no ellipses)
   for c in ["CMMS MATERIAL CODE", "OEM PART NUMBER", "DESCRIPTION"]:
    if c in df.columns:
        df[c] = df[c].astype(str).fillna("")

    return df[BOM_COLS]

def bom_df_to_struct(df: pd.DataFrame) -> list:
    df = normalize_bom_df(df)
    return [{"CMMS MATERIAL CODE": r["CMMS MATERIAL CODE"], "OEM PART NUMBER": r["OEM PART NUMBER"],
             "DESCRIPTION": r["DESCRIPTION"], "QUANTITY": int(r["QUANTITY"])} for _, r in df.iterrows()]

def bom_df_to_legacy_lists(df: pd.DataFrame):
    df = normalize_bom_df(df)
    return df["OEM PART NUMBER"].astype(str).tolist(), df["CMMS MATERIAL CODE"].astype(str).tolist()

def normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy() if df is not None else pd.DataFrame(columns=HIST_COLS)
    for c in HIST_COLS:
        if c not in df.columns:
            df[c] = ""

    def _to_str_or_join(x):
        if isinstance(x, list): return "; ".join(map(str, x))
        return "" if pd.isna(x) else str(x)

    df["Spares Used"] = df["Spares Used"].apply(_to_str_or_join)
    df["Date of Maintenance"] = df["Date of Maintenance"].astype(str)
    df["Hours Since Last"] = pd.to_numeric(df["Hours Since Last"], errors="coerce").fillna(0.0)
    df["MTTR (hrs)"] = pd.to_numeric(df["MTTR (hrs)"], errors="coerce").fillna(0.0)

    # New numeric fields for Tab 3
    df["Running Hours Since Last Repair"] = pd.to_numeric(df["Running Hours Since Last Repair"], errors="coerce").fillna(0.0)
    df["Labour Hours"] = pd.to_numeric(df["Labour Hours"], errors="coerce").fillna(0.0)
    df["Downtime Hours"] = pd.to_numeric(df["Downtime Hours"], errors="coerce").fillna(0.0)

    return df[HIST_COLS]

def pretty_spares_multiline(s: str) -> str:
    parts = [p.strip() for p in str(s).split(";") if p and p.strip()]
    return "\n".join(parts)

def data_editor_compat(df, **kwargs):
    if hasattr(st, "data_editor"):
        return st.data_editor(df, **kwargs)
    return st.experimental_data_editor(df, **kwargs)

# ====== Sidebar ======
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("💾 Save All Now", key="save_all"):
        save_assets(st.session_state.assets)
        st.session_state.runtime_df = compute_remaining_hours(st.session_state.runtime_df)
        save_runtime(st.session_state.runtime_df)
        st.session_state.history_df = normalize_history_df(st.session_state.history_df)
        save_history(st.session_state.history_df)
        st.success("All data saved.")
    st.caption(f"Working dir: {os.getcwd()}")
    st.caption(f"Last opened: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    st.markdown("### ⬆️ Import")
    up_rt = st.file_uploader("Import Runtime CSV", type=["csv"], key="upload_runtime")
    if up_rt is not None:
        try:
            new_rt = pd.read_csv(up_rt)
            st.session_state.runtime_df = compute_remaining_hours(
                pd.concat([st.session_state.runtime_df,new_rt], ignore_index=True)
            )
            st.success("Runtime rows imported (appended).")
        except Exception as e:
            st.error(f"Could not import runtime: {e}")
    up_hist = st.file_uploader("Import History CSV", type=["csv"], key="upload_history")
    if up_hist is not None:
        try:
            new_h = pd.read_csv(up_hist)
            st.session_state.history_df = normalize_history_df(
                pd.concat([st.session_state.history_df,new_h], ignore_index=True)
            )
            st.success("History rows imported (appended).")
        except Exception as e:
            st.error(f"Could not import history: {e}")
    up_assets = st.file_uploader("Import Assets JSON", type=["json"], key="upload_assets")
    if up_assets is not None:
        try:
            incoming = json.loads(up_assets.read().decode("utf-8"))
            if isinstance(incoming, dict):
                st.session_state.assets.update(incoming)
                save_assets(st.session_state.assets)
                st.success("Assets merged.")
            else:
                st.error("JSON should be an object mapping tags to fields.")
        except Exception as e:
            st.error(f"Could not import assets: {e}")
    st.markdown("---")
    bg_file = st.file_uploader("Upload engineering photo (JPG/PNG)", type=["jpg","jpeg","png"], key="bg_image")
    if bg_file:
        set_app_background(bg_file.read(), darken=0.35, blur_px=4)

# ====== Tabs ======
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📌 Asset Explorer","🕒 Runtime Tracker","📝 Data Entry",
    "💧 Hydraulic System Calculator","💡 Financial Analysis","📉 Reliability Visualisation",
])

# ====== TAB 1: Asset Explorer ======
with tab1:
    import re as _re
    st.subheader("📌 Asset Explorer")

    left, right = st.columns([1, 2], gap="large")
    assets_dict = st.session_state.get("assets", {})
    all_tags = sorted([str(t) for t in assets_dict.keys()])

    with left:
        st.markdown("#### Select Asset")
        selected_tag = st.selectbox("Asset Tag", options=all_tags, index=(0 if all_tags else None), key="asset_select_dropdown")

        st.markdown("#### Assets by Section")
        if all_tags:
            section_counts = {}
            for t in all_tags:
                m = _re.match(r"^(\d{3})", str(t))
                sec = m.group(1) if m else "Other"
                section_counts[sec] = section_counts.get(sec, 0) + 1

            labels = list(section_counts.keys())
            values = [section_counts[k] for k in labels]
            total = max(sum(values), 1)

            fig_sec, ax_sec = plt.subplots(figsize=(4.8, 4.0))
            ax_sec.pie(
                values,
                labels=[f"{l} ({c}, {c/total*100:.0f}%)" for l, c in zip(labels, values)],
                startangle=90,
            )
            ax_sec.axis("equal")
            st.pyplot(fig_sec)
        else:
            st.info("No assets loaded yet. Add assets in **📝 Data Entry** or import Assets JSON.")

    with right:
        if selected_tag:
            a = assets_dict.get(selected_tag, {})
            st.markdown(f"### {selected_tag} — {a.get('Functional Location','')}")

            c1,c2,c3 = st.columns(3)
            c1.metric("Functional Location", a.get("Functional Location","—"))
            c2.metric("Model", a.get("Model","—"))
            c3.metric("Drive Type", a.get("Drive Type","—"))

            c4,c5,c6 = st.columns(3)
            c4.metric("Max Flow (m³/h)", f"{a.get('Max Capacity (m³/h)','—')}")
            c5.metric("TDH (m)", f"{a.get('TDH (m)','—')}")
            c6.metric("Avg Speed (VSD)", f"{a.get('Avg Speed (VSD RPM)','—')}")

            c7,c8,c9 = st.columns(3)
            c7.metric("Max Speed (RPM)", f"{a.get('Max Speed (RPM)','—')}")
            c8.metric("Motor Shaft (mm)", f"{a.get('Motor Shaft Size (mm)','—')}")
            c9.metric("Pump Shaft (mm)", f"{a.get('Pump Shaft Size (mm)','—')}")

            # Available Life pie
            st.markdown("### ⏳ Available Life")
            df_rt_tmp = compute_remaining_hours(st.session_state.runtime_df.copy())
            rt_row = df_rt_tmp[df_rt_tmp["Asset Tag"].astype(str) == str(selected_tag)]
            if not rt_row.empty:
                rem_raw  = rt_row.iloc[0].get("Remaining Hours", 0.0)
                mtbf_raw = rt_row.iloc[0].get("MTBF (Hours)", 0.0)
                rem  = float(np.nan_to_num(to_num(rem_raw),  nan=0.0, posinf=0.0, neginf=0.0))
                mtbf = float(np.nan_to_num(to_num(mtbf_raw), nan=0.0, posinf=0.0, neginf=0.0))

                m1, m2, m3 = st.columns([1,1,2])
                m1.metric("Remaining (hrs)", f"{rem:,.0f}")
                m2.metric("MTBF (hrs)", f"{mtbf:,.0f}")

                used = max(mtbf - rem, 0.0)
                rem  = max(rem, 0.0)
                total = used + rem
                if not np.isfinite(total) or total <= 0:
                    with m3:
                        st.info("No MTBF/runtime data yet to draw the pie.")
                else:
                    fig, ax = plt.subplots(figsize=(3.6, 3.6))
                    try:
                        ax.pie([used, rem],
                               labels=["Used", "Remaining"],
                               autopct=lambda p: f"{p:.0f}%",
                               startangle=90,
                               normalize=True)
                    except TypeError:
                        s = used + rem or 1.0
                        parts = [used/s, rem/s]
                        ax.pie(parts,
                               labels=["Used", "Remaining"],
                               autopct=lambda p: f"{p:.0f}%",
                               startangle=90)
                    ax.axis("equal")
                    with m3:
                        st.pyplot(fig)
            else:
                st.info("No runtime row found for this asset. Add one under **🕒 Runtime Tracker**.")

            # BOM
            st.markdown("### 🧰 BOM")
            df_bom = build_bom_table(a)
            if df_bom.empty:
                st.caption("No BOM recorded yet.")
            else:
                st.dataframe(df_bom, use_container_width=True)
                st.download_button("⬇️ Download BOM (CSV)", data=df_bom.to_csv(index=False),
                                   file_name=f"{selected_tag}_bom_table.csv", key="dl_bom_table_explorer_dropdown")

            # Maintenance History
            st.markdown("### 🛠️ Maintenance History")
            df_hist = normalize_history_df(st.session_state.history_df)
            df_hist_sel = df_hist[df_hist["Asset Tag"].astype(str) == str(selected_tag)].copy()
            if df_hist_sel.empty:
                st.caption("No maintenance history for this asset yet.")
            else:
                df_hist_sel["Spares Used (clean)"] = df_hist_sel["Spares Used"].apply(pretty_spares_multiline)
                st.dataframe(df_hist_sel, use_container_width=True)
        else:
            st.info("Select an asset to view details.")

# ====== TAB 2: Runtime Tracker ======
with tab2:
    st.subheader("🕒 Runtime Tracker")
    if st.session_state.runtime_df.empty:
        st.session_state.runtime_df = pd.DataFrame(columns=[
            "Asset Tag","Functional Location","Asset Model","MTBF (Hours)","Last Overhaul",
            "Running Hours Since Last Major Maintenance","Remaining Hours"
        ])

    run_long = "Running Hours Since Last Major Maintenance"
    run_short = "Hrs Run Since"
    df_rt = st.session_state.runtime_df.copy()

    # Auto-sync: ensure all assets appear at least once in runtime table
    assets = st.session_state.get("assets", {})
    asset_rows = [{
        "Asset Tag": str(tag),
        "Functional Location": a.get("Functional Location",""),
        "Asset Model": a.get("Model",""),
    } for tag, a in assets.items()] if isinstance(assets, dict) else []
    assets_df = pd.DataFrame(asset_rows, columns=["Asset Tag","Functional Location","Asset Model"]) if asset_rows else pd.DataFrame(columns=["Asset Tag","Functional Location","Asset Model"])
    if not assets_df.empty:
        merged = pd.merge(assets_df, df_rt, on="Asset Tag", how="left", suffixes=("_asset",""))
        merged["Functional Location"] = merged.get("Functional Location_asset", merged.get("Functional Location",""))
        merged["Asset Model"] = merged.get("Asset Model", merged.get("Asset Model_asset",""))
        df_rt = merged.drop(columns=[c for c in ["Functional Location_asset","Asset Model_asset"] if c in merged.columns])

    if run_short in df_rt.columns and run_long not in df_rt.columns:
        df_rt = df_rt.rename(columns={run_short: run_long})
    elif run_long not in df_rt.columns and len(df_rt.columns) >= 6:
        cols = df_rt.columns.tolist()
        cols[5] = run_long
        if run_long in df_rt.columns:
            for i, c in enumerate(cols):
                if c == run_long and i != 5:
                    cols[i] = f"{c} (dup{i})"
        df_rt.columns = cols

    df_rt = compute_remaining_hours(df_rt)

    # STATUS calc
    def _status_row(r):
        mtbf = float(np.nan_to_num(to_num(r.get("MTBF (Hours)", 0.0)), nan=0.0, posinf=0.0, neginf=0.0))
        run  = float(np.nan_to_num(to_num(r.get(run_long, 0.0)),  nan=0.0, posinf=0.0, neginf=0.0))
        if mtbf <= 0:
            return "🟢 Healthy"
        ratio = run / mtbf if mtbf > 0 else 0.0
        if ratio < 0.80:
            return "🟢 Healthy"
        elif ratio < 0.99:
            return "🟠 Plan for maintenance"
        else:
            return "🔴 Overdue for maintenance"
    df_rt["STATUS"] = df_rt.apply(_status_row, axis=1)

    # Display mapping
    rename_map = {
        "Asset Tag": "ASSET TAG",
        "Functional Location": "FUNCTIONAL LOCATION",
        "MTBF (Hours)": "MTBF (HOURS)",
        "Last Overhaul": "LAST MAJOR MAINTENNACE DATE",
        run_long: "RUNNING HOURS SINCE LAST MAJOR MAINTENANCE",
        "Remaining Hours": "REMAINING HOURS TO NEXT MAJOR MAINTENANCE",
    }
    df_rt_disp = df_rt.rename(columns=rename_map)
    desired_cols = [
        "ASSET TAG","FUNCTIONAL LOCATION","MTBF (HOURS)","LAST MAJOR MAINTENNACE DATE",
        "RUNNING HOURS SINCE LAST MAJOR MAINTENANCE","REMAINING HOURS TO NEXT MAJOR MAINTENANCE","STATUS"
    ]
    df_rt_disp = df_rt_disp[[c for c in desired_cols if c in df_rt_disp.columns]].copy()
    df_rt_disp = df_rt_disp.loc[:, ~pd.Index(df_rt_disp.columns).duplicated(keep='first')]

    all_cols = df_rt_disp.columns.tolist()
    editable_cols = [c for c in all_cols if c in ["FUNCTIONAL LOCATION","MTBF (HOURS)","LAST MAJOR MAINTENNACE DATE","RUNNING HOURS SINCE LAST MAJOR MAINTENANCE"]]
    disabled_cols = [c for c in all_cols if c not in editable_cols]

    edited_rt_disp = data_editor_compat(
        df_rt_disp,
        use_container_width=True,
        num_rows="dynamic",
        disabled=disabled_cols,
        key="runtime_editor"
    )

    # Map back to internal names for storage, recompute remaining
    back_rename = {
        "ASSET TAG": "Asset Tag",
        "FUNCTIONAL LOCATION": "Functional Location",
        "MTBF (HOURS)": "MTBF (Hours)",
        "LAST MAJOR MAINTENNACE DATE": "Last Overhaul",
        "RUNNING HOURS SINCE LAST MAJOR MAINTENANCE": run_long,
        "REMAINING HOURS TO NEXT MAJOR MAINTENANCE": "Remaining Hours",
    }
    edited_rt = edited_rt_disp.rename(columns={k: v for k, v in back_rename.items() if k in edited_rt_disp.columns})
    edited_rt = compute_remaining_hours(edited_rt)

    # Push Functional Location edits back into assets
    try:
        _assets = st.session_state.get('assets', {})
        for _, _r in edited_rt.iterrows():
            _tag = str(_r.get('Asset Tag','')).strip()
            _fl  = str(_r.get('Functional Location','')).strip()
            if _tag:
                if _tag not in _assets or not isinstance(_assets[_tag], dict):
                    _assets[_tag] = {}
                if _fl and _assets[_tag].get('Functional Location','') != _fl:
                    _assets[_tag]['Functional Location'] = _fl
        st.session_state.assets = _assets
    except Exception as _e:
        st.warning(f"Functional Location sync skipped: {_e}")

    st.session_state.runtime_df = edited_rt

    # Summary visuals
    df_view = st.session_state.runtime_df.copy()
    if "STATUS" not in df_view.columns and "Status" in df_view.columns:
        df_view["STATUS"] = df_view["Status"]

    status_counts = {
        "🟢 Healthy": int((df_view.get("STATUS","") == "🟢 Healthy").sum()) if "STATUS" in df_view.columns else 0,
        "🟠 Plan for maintenance": int((df_view.get("STATUS","") == "🟠 Plan for maintenance").sum()) if "STATUS" in df_view.columns else 0,
        "🔴 Overdue for maintenance": int((df_view.get("STATUS","") == "🔴 Overdue for maintenance").sum()) if "STATUS" in df_view.columns else 0,
    }
    labels = list(status_counts.keys())
    sizes = [status_counts[k] for k in labels]
    total = sum(sizes)

    if "STATUS" in df_view.columns:
        due_n = int(df_view["STATUS"].isin(["🟠 Plan for maintenance","🔴 Overdue for maintenance"]).sum())
    else:
        due_n = 0
    den = max(int(len(df_view)), 1)
    due_pct = float((due_n / den) * 100.0)

    col_pie, col_list = st.columns([2,1])
    with col_pie:
        if total <= 0:
            st.info("No data to summarize yet.")
        else:
            fig, ax = plt.subplots(figsize=(4.2, 4.2))
            try:
                ax.pie(sizes, labels=labels, autopct=lambda p: f"{p:.0f}%", startangle=90, normalize=True)
            except TypeError:
                s = sum(sizes) or 1
                ax.pie([x/s for x in sizes], labels=labels, autopct=lambda p: f"{p:.0f}%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

    with col_list:
        st.markdown("**Overdue assets**")
        overdue = []
        if "STATUS" in df_view.columns:
            overdue_df = df_view[df_view["STATUS"] == "🔴 Overdue for maintenance"]
            if "Asset Tag" in overdue_df.columns:
                overdue = overdue_df["Asset Tag"].dropna().astype(str).tolist()
            elif "ASSET TAG" in overdue_df.columns:
                overdue = overdue_df["ASSET TAG"].dropna().astype(str).tolist()
        if overdue:
            for t in overdue:
                st.write(f"• {t}")
        else:
            st.caption("None overdue right now.")

    c_rt1, c_rt2 = st.columns(2)
    if c_rt1.button("💾 Save Runtime Table", key="save_runtime"):
        save_runtime(st.session_state.runtime_df)
        save_assets(st.session_state.assets)
        st.success("Runtime table & assets saved.")
    c_rt2.download_button("⬇️ Download CSV",
        data=st.session_state.runtime_df.to_csv(index=False),
        file_name="runtime.csv",
        key="download_runtime"
    )

# ====== TAB 3: Data Entry ======
with tab3:
    st.subheader("📝 Data Entry")
    st.markdown("### Pump Asset — Add / Edit")
    mode = st.radio("Mode", ["Add new asset", "Edit existing"], horizontal=True, key="asset_mode_plus2")
    if mode == "Edit existing" and st.session_state.assets:
        tag_choice = st.selectbox("Select Asset Tag to edit", sorted(st.session_state.assets.keys()), key="asset_edit_select_plus2")
        data = st.session_state.assets.get(tag_choice, {})
    else:
        tag_choice, data = "", {}

    with st.form("asset_form_plus2", clear_on_submit=False):
        tag = st.text_input("Asset Tag", value=tag_choice, key="asset_tag_plus2")
        func_loc = st.text_input("Functional Location", value=data.get("Functional Location",""), key="asset_func_loc_plus2")
        model = st.text_input("Model", value=data.get("Model",""), key="asset_model_plus2")

        # Motor power
        std_motor_sizes = [
            0.37,0.55,0.75,1.1,1.5,2.2,3,4,5.5,7.5,9,11,15,18.5,22,30,37,45,55,75,90,110,132,160,200,220,250,280,315,355,400,450,500,560,660,710,800,900
        ]
        motor_opts = [f"{v} kW" for v in std_motor_sizes] + ["Other"]
        existing_mp = str(data.get("Motor Power","")).strip()
        def _match_default_mp():
            for idx, opt in enumerate(motor_opts[:-1]):
                if existing_mp.startswith(opt.split()[0]):
                    return idx
            return len(motor_opts)-1
        mp_sel = st.selectbox("Motor Power (standard 3-phase)", motor_opts, index=_match_default_mp(), key="motor_power_std_select")
        if mp_sel == "Other":
            m = re.search(r"([\d\.]+)", existing_mp)
            default_other = float(m.group(1)) if m else 0.0
            mp_other = st.number_input("Other motor power (kW)", min_value=0.0, value=default_other, step=0.1, key="motor_power_other")
            motor_power = f"{mp_other} kW" if mp_other > 0 else existing_mp
        else:
            motor_power = mp_sel

        # Sealing
        sealing_opts = ["Stuffing box seal","Expeller seal","Mechanical seal","Other"]
        def _seal_idx():
            cur = str(data.get("Sealing","")).strip().lower()
            for i, s in enumerate(sealing_opts[:-1]):
                if s.lower() in cur:
                    return i
            return len(sealing_opts)-1
        sealing_sel = st.selectbox("Sealing (select)", sealing_opts, index=_seal_idx(), key="sealing_select_plus2")
        sealing = st.text_input("Sealing (if 'Other')", value=data.get("Sealing","") if sealing_sel=="Other" else sealing_sel, key="sealing_other_box")

        # Drive type
        drive_opts = ["Pulley-belts driven","Directly coupled","Gearbox coupled","Other"]
        def _drive_idx():
            cur = str(data.get("Drive Type","")).strip().lower()
            for i, d in enumerate(drive_opts[:-1]):
                if d.lower() in cur:
                    return i
            return len(drive_opts)-1
        drive_sel = st.selectbox("Drive Type (select)", drive_opts, index=_drive_idx(), key="drive_select_plus2")
        drive = st.text_input("Drive Type (if 'Other')", value=data.get("Drive Type","") if drive_sel=="Other" else drive_sel, key="drive_other_box")

        motor_shaft = st.number_input("Motor Shaft Size (mm)", value=float(data.get("Motor Shaft Size (mm)",0)) if data.get("Motor Shaft Size (mm)") else 0.0, key="asset_motor_shaft_plus2")
        pump_shaft  = st.number_input("Pump Shaft Size (mm)",  value=float(data.get("Pump Shaft Size (mm)",0)) if data.get("Pump Shaft Size (mm)") else 0.0, key="asset_pump_shaft_plus2")
        max_speed   = st.number_input("Max Speed (RPM)", value=float(data.get("Max Speed (RPM)",0)) if data.get("Max Speed (RPM)") else 0.0, key="asset_max_speed_plus2")
        avg_speed   = st.number_input("Avg Speed (VSD RPM)", value=float(data.get("Avg Speed (VSD RPM)",0)) if data.get("Avg Speed (VSD RPM)") else 0.0, key="asset_avg_speed_plus2")
        max_flow    = st.number_input("Max Capacity (m³/h)", value=float(data.get("Max Capacity (m³/h)",0)) if data.get("Max Capacity (m³/h)") else 0.0, key="asset_max_flow_plus2")
        tdh         = st.number_input("TDH (m)", value=float(data.get("TDH (m)",0)) if data.get("TDH (m)") else 0.0, key="asset_tdh_plus2")

        # BOM Editor
        st.markdown("### 🧰 Bill of Materials (BOM)")
        df_bom_edit_default = build_bom_table(data)
        if df_bom_edit_default.empty:
            df_bom_edit_default = pd.DataFrame(
                [{"CMMS MATERIAL CODE": "", "OEM PART NUMBER": "", "DESCRIPTION": "", "QUANTITY": 1, "PRICE": 0.0} for _ in range(5)]
            )
        df_bom_edit_default = normalize_bom_df(df_bom_edit_default)

        df_bom_edit = data_editor_compat(
            df_bom_edit_default,
            key="bom_table_editor",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "CMMS MATERIAL CODE": st.column_config.TextColumn("CMMS MATERIAL CODE"),
                "OEM PART NUMBER":    st.column_config.TextColumn("OEM PART NUMBER"),
                "DESCRIPTION":        st.column_config.TextColumn("DESCRIPTION"),
                "QUANTITY":           st.column_config.NumberColumn("QUANTITY", min_value=1, step=1),
                "PRICE":            st.column_config.NumberColumn("PRICE", min_value=0.0, step=0.01, format="%.2f"),
            },
        )
        st.session_state["current_bom_df"] = normalize_bom_df(df_bom_edit)

        submitted = st.form_submit_button("Save Asset")
        if submitted:
            edited_df = df_bom_edit
            if not isinstance(edited_df, pd.DataFrame):
                ss_val = st.session_state.get("bom_table_editor", None)
                if isinstance(ss_val, pd.DataFrame):
                    edited_df = ss_val
                elif isinstance(ss_val, list):
                    edited_df = pd.DataFrame(ss_val)
                elif isinstance(ss_val, dict):
                    if "data" in ss_val and isinstance(ss_val["data"], list):
                        edited_df = pd.DataFrame(ss_val["data"])
                    else:
                        edited_df = pd.DataFrame([ss_val])
                else:
                    edited_df = build_bom_table(data)

            df_bom_clean = normalize_bom_df(edited_df)
            bom_table_struct = bom_df_to_struct(df_bom_clean)
            bom_list, cmms_list = bom_df_to_legacy_lists(df_bom_clean)

            if not tag.strip():
                st.error("Asset Tag is required.")
            else:
                st.session_state.assets[tag] = {
                    "Functional Location": func_loc, "Model": model, "Motor Power": motor_power,
                    "Sealing": sealing, "Drive Type": drive,
                    "Motor Shaft Size (mm)": motor_shaft, "Pump Shaft Size (mm)": pump_shaft,
                    "Max Speed (RPM)": max_speed, "Avg Speed (VSD RPM)": avg_speed,
                    "Max Capacity (m³/h)": max_flow, "TDH (m)": tdh,
                    "BOM Table": bom_table_struct,
                    "BOM": bom_list, "CMMS Codes": cmms_list,
                }
                save_assets(st.session_state.assets)
                st.success(f"Saved asset '{tag}' with BOM Table ({len(bom_table_struct)} rows).")

        # ===== Maintenance History — Standardized Table (single table) =====
    st.divider()
    st.markdown("### 🛠 Maintenance History — Standardized Table")

    # Current asset context
    current_tag  = st.session_state.get("asset_tag_plus2","").strip()
    current_model= st.session_state.get("asset_model_plus2","").strip()
    current_func = st.session_state.get("asset_func_loc_plus2","").strip()

    # Build BOM label options for parts picker
    def _bom_options_from_df(df: pd.DataFrame):
        df = normalize_bom_df(df)
        opts=[]
        for _, r in df.iterrows():
            oem  = r["OEM PART NUMBER"] or ""
            cmms = r["CMMS MATERIAL CODE"] or ""
            desc = r["DESCRIPTION"] or ""
            label = " • ".join([x for x in [desc, f"OEM: {oem}" if oem else "", f"CMMS: {cmms}" if cmms else ""] if x])
            opts.append(label if label else (oem or cmms))
        seen=set(); out=[]
        for x in opts:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    _bom_df_ctx = st.session_state.get("current_bom_df", pd.DataFrame(columns=BOM_COLS))
    if _bom_df_ctx.empty and isinstance(st.session_state.assets, dict) and current_tag in st.session_state.assets:
        _bom_df_ctx = normalize_bom_df(build_bom_table(st.session_state.assets.get(current_tag, {})))
    bom_opts = _bom_options_from_df(_bom_df_ctx) if not _bom_df_ctx.empty else []

    # Allowed triggers by code (CMRP depth, with "Other (specify)")

        # NOTE on the header naming you requested:
    st.caption("Trigger Field label varies by code: **PM01→RCM Trigger Type**, **PM02→Corrective Trigger Type**, **PM03→Inspection Type**, **PM00→Breakdown Event**.")

    # Load full history and pivot to the standardized columns for the current asset
    full_hist = normalize_history_df(st.session_state.history_df)

    keep_cols = [
        "Asset Tag","Maintenance Code","Trigger",
        "Running Hours Since Last Repair","Labour Hours","Parts Used","Downtime Hours",
        "Notes/Findings","Attachments"
    ]
    if current_tag:
        df_std = full_hist[full_hist["Asset Tag"].astype(str) == current_tag].copy()
    else:
        df_std = pd.DataFrame(columns=keep_cols)

    # Ensure required columns are present
    for col in keep_cols:
        if col not in df_std.columns:
            df_std[col] = ""

    df_std = df_std[keep_cols]

# Multiselect parts column if available
has_multiselect = hasattr(st.column_config, "MultiselectColumn")

def _parts_str_to_list(s: str):
    out = []
    for raw in str(s).split(";"):
        t = raw.strip()
        if not t:
            continue
        if " (x" in t and t.endswith(")"):
            t = t[:t.rfind(" (x")]
        out.append(t)
    return out

def _parts_list_to_str(v):
    if isinstance(v, list):
        return "; ".join(v)
    return "" if pd.isna(v) else str(v)

# Normalize Parts Used to match editor type
if has_multiselect:
    df_std["Parts Used"] = df_std["Parts Used"].apply(_parts_str_to_list)
else:
    df_std["Parts Used"] = df_std["Parts Used"].astype(str).fillna("")

# Force text columns to strings
for _c in ["Asset Tag", "Maintenance Code", "Trigger", "Notes/Findings", "Attachments"]:
    if _c in df_std.columns:
        df_std[_c] = df_std[_c].astype(str).fillna("")

# Force numeric columns to numbers
for _c in ["Running Hours Since Last Repair", "Labour Hours", "Downtime Hours"]:
    if _c in df_std.columns:
        df_std[_c] = pd.to_numeric(df_std[_c], errors="coerce").fillna(0.0)

# Build the column config for Parts Used
parts_col_cfg = (
    st.column_config.MultiselectColumn(
        "Parts Used",
        options=bom_opts,
        help="Choose parts from this asset’s BOM. Add qty like '… (x2)'.",
    )
    if has_multiselect else
    st.column_config.TextColumn(
        "Parts Used",
        help="Type parts; include qty like '… (x2)'.",
    )
)

col_cfg = {
    "Asset Tag": st.column_config.TextColumn("Asset Tag", disabled=True),
    "Maintenance Code": st.column_config.SelectboxColumn(
        "Maintenance Code", options=["","PM01","PM02","PM03","PM00"]
    ),
    "Trigger": st.column_config.SelectboxColumn(
        "Trigger Field", options=[""] + ALL_TRIGS + ["Other (specify)"]
    ),
    "Running Hours Since Last Repair": st.column_config.NumberColumn(
        "Running Hours Since Last Repair", min_value=0.0, step=1.0
    ),
    "Labour Hours": st.column_config.NumberColumn("Labour Hours", min_value=0.0, step=0.5),
    "Parts Used": parts_col_cfg,
    "Downtime Hours": st.column_config.NumberColumn("Downtime Hours", min_value=0.0, step=0.5),
    "Notes/Findings": st.column_config.TextColumn("Notes/Findings"),
    "Attachments": st.column_config.TextColumn(
        "Attachments", help="Enter file names or references (uploads not in table)."
    ),
}

    # Seed at least one editable row for this asset
    if df_std.empty:
        df_std = pd.DataFrame([{
            "Asset Tag": current_tag,
            "Maintenance Code": "",
            "Trigger": "",
            "Running Hours Since Last Repair": 0.0,
            "Labour Hours": 0.0,
            "Parts Used": ([] if has_multiselect else ""),
            "Downtime Hours": 0.0,
            "Notes/Findings": "",
            "Attachments": "",
        }])

    # Lock Asset Tag to current selection
    df_std["Asset Tag"] = current_tag

    edited_std = data_editor_compat(
        df_std,
        key="mh_std_table_editor",
        num_rows="dynamic",
        use_container_width=True,
        column_config=col_cfg,
        disabled=["Asset Tag"],
    )

    # Validation per row (trigger must match code’s set unless "Other")
    invalid_rows = []
    for i, r in edited_std.iterrows():
        code = str(r.get("Maintenance Code","")).strip()
        trig = str(r.get("Trigger","")).strip()
        if not code or not trig or trig == "Other (specify)":
            continue
        allowed = {
            "PM01": TRIG_PM01,
            "PM02": TRIG_PM02,
            "PM03": TRIG_PM03,
            "PM00": TRIG_PM00,
        }.get(code, [])

        if allowed and trig not in allowed:
            invalid_rows.append(i+1)  # 1-based for user

    if invalid_rows:
        st.warning(f"Trigger choice doesn’t match its Maintenance Code on row(s): {', '.join(map(str, invalid_rows))}. You can pick a code-appropriate option or use 'Other (specify)'.", icon="⚠️")

    # Save button (maps to legacy fields for compatibility and auto-stamps date)
    if st.button("💾 Save Standardized Maintenance Table", key="save_mh_std_table"):
        to_save = edited_std.copy()

        # Convert parts list back to string if multiselect used
        if has_multiselect and "Parts Used" in to_save.columns:
            to_save["Parts Used"] = to_save["Parts Used"].apply(_parts_list_to_str)

        # Inject context & legacy compatibility
        to_save["Asset Tag"] = current_tag
        to_save["Functional Location"] = current_func
        to_save["Asset Model"] = current_model
        to_save["Date of Maintenance"] = datetime.now().date().isoformat()

        to_save["Spares Used"] = to_save["Parts Used"]           # legacy mirror
        to_save["Reason"] = to_save["Notes/Findings"]             # legacy mirror
        to_save["Hours Since Last"] = to_save["Running Hours Since Last Repair"]  # legacy mirror
        to_save["MTTR (hrs)"] = 0.0

        # Drop truly empty rows
        def _nz(col): return pd.to_numeric(to_save[col], errors="coerce").fillna(0.0)
        keep_mask = (
            to_save["Maintenance Code"].astype(str).str.strip() != ""
        ) | (
            to_save["Parts Used"].astype(str).str.strip() != ""
        ) | (
            to_save["Notes/Findings"].astype(str).str.strip() != ""
        ) | (_nz("Running Hours Since Last Repair") > 0) | (_nz("Labour Hours") > 0) | (_nz("Downtime Hours") > 0)
        to_save = to_save[keep_mask].copy()

        # Merge back with other assets’ history
        remaining = full_hist[full_hist["Asset Tag"] != current_tag].copy()
        new_full = normalize_history_df(pd.concat([remaining, to_save], ignore_index=True))

        st.session_state.history_df = new_full
        save_history(new_full)
        st.success(f"Saved standardized maintenance table for {current_tag or 'current asset'}.")
        st.rerun()

# ====== TAB 4: Hydraulic System Calculator ======
def swamee_jain_f(Re, eps, D):
    if Re < 2000 and Re > 0: return 64.0 / Re
    if Re <= 0: return 0.02
    return 0.25 / (math.log10((eps/(3.7*D)) + (5.74/(Re**0.9)))**2)

def patm_kpa_from_elevation(elev_m): 
    return 101.325 * math.exp(-elev_m / 8434.0)

ROUGHNESS_MM = {
    "PVC / CPVC (smooth)": 0.0015, "HDPE (new)": 0.0015,
    "Stainless steel (new)": 0.015, "Copper (drawn)": 0.0015,
    "Carbon steel (new)": 0.045, "Commercial steel (used)": 0.09,
    "Galvanized iron": 0.15, "Ductile iron (cement-mortar lined)": 0.10,
    "Cast iron (new)": 0.26, "Cast iron (older)": 0.80,
    "FRP": 0.005, "PTFE-lined steel": 0.005, "Epoxy-lined steel": 0.01,
    "Rubber-lined steel (slurry)": 0.10, "Ceramic-lined (slurry severe)": 0.20,
    "Concrete (troweled)": 0.30, "Wood stave": 0.18,
    "Riveted steel": 1.00, "Corrugated metal": 3.00,
}
BASE_K = {
    "Entrance": 0.5, "Exit": 1.0, "Elbow 90°": 0.9, "Elbow 45°": 0.4,
    "Tee (run/through)": 0.6, "Tee (branch/into)": 1.8,
    "Gate valve (open)": 0.15, "Globe valve (open)": 10.0,
    "Ball valve (open)": 0.05, "Butterfly valve (open)": 0.9,
    "Check valve": 2.0, "Strainer": 2.0,
}
def k_expansion(beta):  # sudden expansion: Borda–Carnot
    beta = max(1e-6, min(1.0, float(beta)))
    return (1.0 - beta**2)**2
def k_contraction(beta):  # sudden contraction (quick estimate)
    beta = max(1e-6, min(1.0, float(beta)))
    return 0.5*((1.0/(beta**2)) - 1.0)
def fittings_panel(title, key_prefix):
    st.markdown(f"**{title} — Fittings (K method)**")
    cols = st.columns(4)
    counts = {}
    for i, (name, K) in enumerate(BASE_K.items()):
        col = cols[i % 4]
        counts[name] = col.number_input(f"{name} (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_base_{i}")
    st.caption("Reducers / Expanders (use d/D = downstream ID / upstream ID)")
    rc1, rc2, rc3, rc4 = st.columns(4)
    qty_contr = rc1.number_input("Sudden contraction (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_q_contr")
    beta_contr = rc2.number_input("Contraction d/D (β)", min_value=0.01, max_value=0.99, value=0.80, step=0.01, key=f"{key_prefix}_b_contr")
    qty_exp = rc3.number_input("Sudden expansion (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_q_exp")
    beta_exp = rc4.number_input("Expansion d/D (β)", min_value=0.01, max_value=0.99, value=0.80, step=0.01, key=f"{key_prefix}_b_exp")
    K_total = sum(counts[name] * BASE_K[name] for name in counts)
    K_total += qty_contr * k_contraction(beta_contr)
    K_total += qty_exp * k_expansion(beta_exp)
    st.caption(f"Sum K = {K_total:.3f}  •  (Contraction K≈{k_contraction(beta_contr):.3f}, Expansion K={k_expansion(beta_exp):.3f})")
    return K_total

with tab4:
    st.subheader("💧 Hydraulic System Calculator")

    all_tags = sorted(list(st.session_state.assets.keys()))
    col_pref1, col_pref2 = st.columns(2)
    prefill_tag = col_pref1.selectbox("Prefill from Asset (optional)", ["(none)"] + all_tags, key="prefill_asset_tag_v2")
    prefill_Q, prefill_TDH = 0.0, 0.0
    if prefill_tag != "(none)":
        a = st.session_state.assets.get(prefill_tag, {})
        try: prefill_Q = float(a.get("Max Capacity (m³/h)", 0)) or 0.0
        except: prefill_Q = 0.0
        try: prefill_TDH = float(a.get("TDH (m)", 0)) or 0.0
        except: prefill_TDH = 0.0
        col_pref2.info(f"Using {prefill_tag}: Design Q → {prefill_Q} m³/h, TDH → {prefill_TDH} m")

    st.markdown("### 📈 Process Details")
    p1, p2, p3, p4 = st.columns(4)
    site_elev_m = p1.number_input("Site Elevation (m)", min_value=0.0, value=0.0, step=10.0, key="hyd_site_elev")
    temp_C      = p2.number_input("Fluid Temperature (°C)", min_value=-10.0, value=20.0, step=1.0, key="hyd_tempC")
    design_Q    = p3.number_input("Design Flowrate (m³/h)", min_value=0.0, value=(prefill_Q or 1000.0), step=10.0, key="hyd_design_Q")
    density_kgL = p4.number_input("Fluid Density (kg/L)", min_value=0.2, value=1.00, step=0.01, key="hyd_density")

    p5, p6, p7, p8 = st.columns(4)
    use_mu      = p5.checkbox("Specify Dynamic Viscosity?", value=False, key="hyd_use_mu")
    mu_cP       = p5.number_input("Dynamic Viscosity (cP)", min_value=0.1, value=1.0, step=0.1, key="hyd_mu_cP", disabled=not use_mu)
    vapor_kpa   = p6.number_input("Vapor Pressure (kPa)", min_value=0.0, value=2.3, step=0.1, key="hyd_vapor_kpa")
    Qmin        = p7.number_input("Curve Minimum Flowrate (m³/h)", min_value=0.0, value=0.0, step=10.0, key="hyd_qmin")
    Qmax        = p8.number_input("Curve Maximum Flowrate (m³/h)", min_value=1.0, value=max(1.0, design_Q * 1.5), step=50.0, key="hyd_qmax")

    rho = density_kgL * 1000.0
    mu  = (mu_cP / 1000.0) if use_mu else 0.001  # Pa·s (default ~1 cP water)
    g   = 9.80665
    patm_kpa = patm_kpa_from_elevation(site_elev_m)

    st.markdown("### ⬅️ Suction Details")
    s1, s2, s3, s4 = st.columns(4)
    H_suction = s1.number_input("Suction static head (m) (+flooded, −lift)", value=0.0, key="hyd_H_suction")
    D_suc_mm  = s2.number_input("Suction Pipe ID (mm)", min_value=1.0, value=300.0, step=1.0, key="hyd_D_suc_mm")
    L_suc_m   = s3.number_input("Suction Pipe Length (m)", min_value=0.0, value=10.0, step=1.0, key="hyd_L_suc")
    P_suc_kpa = s4.number_input("Suction Residual Pressure (kPa)", min_value=0.0, value=0.0, step=1.0, key="hyd_P_suc")

    s5, s6, s7 = st.columns(3)
    lining_suc = s5.selectbox("Suction Pipe Lining Type", list(ROUGHNESS_MM.keys()), index=0, key="hyd_lining_suc")
    auto_suc   = s6.checkbox("Auto roughness from lining", value=True, key="hyd_auto_rough_suc")
    rough_suc_mm = s7.number_input("Override Roughness (mm)", value=ROUGHNESS_MM[lining_suc], step=0.001, key="hyd_rough_suc", disabled=auto_suc)

    K_suction = fittings_panel("Suction fittings", "hyd_sucfit")

    D_suc = D_suc_mm/1000.0
    A_suc = math.pi * (D_suc**2) / 4.0 if D_suc > 0 else 1e9
    V_suc = (design_Q/3600.0) / A_suc if A_suc>0 else 0.0
    st.caption(f"Suction velocity at design Q: {V_suc:.2f} m/s")

    st.markdown("### ➡️ Discharge Details")
    d1, d2, d3, d4 = st.columns(4)
    H_discharge = d1.number_input("Discharge static head (m) (+flooded, −lift)", value=10.0, key="hyd_H_discharge")
    D_dis_mm    = d2.number_input("Discharge Pipe ID (mm)", min_value=1.0, value=300.0, step=1.0, key="hyd_D_dis_mm")
    L_dis_m     = d3.number_input("Discharge Pipe Length (m)", min_value=0.0, value=200.0, step=1.0, key="hyd_L_dis")
    P_dis_kpa   = d4.number_input("Discharge Residual Pressure (kPa)", min_value=0.0, value=0.0, step=1.0, key="hyd_P_dis")

    d5, d6, d7 = st.columns(3)
    lining_dis = d5.selectbox("Discharge Pipe Lining Type", list(ROUGHNESS_MM.keys()), index=0, key="hyd_lining_dis")
    auto_dis   = d6.checkbox("Auto roughness from lining", value=True, key="hyd_auto_rough_dis")
    rough_dis_mm = d7.number_input("Override Roughness (mm)", value=ROUGHNESS_MM[lining_dis], step=0.001, key="hyd_rough_dis", disabled=auto_dis)

    K_discharge = fittings_panel("Discharge fittings", "hyd_disfit")

    D_dis = D_dis_mm/1000.0
    A_dis = math.pi * (D_dis**2) / 4.0 if D_dis > 0 else 1e9
    V_dis = (design_Q/3600.0) / A_dis if A_dis>0 else 0.0
    st.caption(f"Discharge velocity at design Q: {V_dis:.2f} m/s")

    eps_s = (rough_suc_mm if not auto_suc else ROUGHNESS_MM[lining_suc])/1000.0
    eps_d = (rough_dis_mm if not auto_dis else ROUGHNESS_MM[lining_dis])/1000.0
    H_static = H_discharge - H_suction
    H_press  = ((P_dis_kpa - P_suc_kpa) * 1000.0) / (rho * g)

    if Qmax <= Qmin:
        st.warning("Maximum Flow must be greater than Minimum Flow.")
    else:
        Qs_h = np.linspace(Qmin, Qmax, 140)           # m3/h
        Qs   = Qs_h / 3600.0                          # m3/s

        H_totals=[]; NPSHa_vals=[]; Hs_loss=[]; Hd_loss=[]
        for Q in Qs:
            V_s = Q / A_suc
            V_d = Q / A_dis
            Re_s = (rho*V_s*D_suc)/(mu if mu>0 else 1e-6) if D_suc>0 else 0.0
            Re_d = (rho*V_d*D_dis)/(mu if mu>0 else 1e-6) if D_dis>0 else 0.0
            f_s = swamee_jain_f(Re_s, eps_s, D_suc) if D_suc>0 else 0.0
            f_d = swamee_jain_f(Re_d, eps_d, D_dis) if D_dis>0 else 0.0
            h_s_pipe = ((f_s*(L_suc_m/max(D_suc,1e-6))) ) * (V_s**2)/(2*g)
            h_d_pipe = ((f_d*(L_dis_m/max(D_dis,1e-6))) ) * (V_d**2)/(2*g)
            h_s_fit  = K_suction  * (V_s**2)/(2*g)
            h_d_fit  = K_discharge * (V_d**2)/(2*g)
            Hs_loss.append(h_s_pipe + h_s_fit)
            Hd_loss.append(h_d_pipe + h_d_fit)

            H_totals.append(H_static + H_press + h_s_pipe + h_s_fit + h_d_pipe + h_d_fit)

            patm_m = (patm_kpa - vapor_kpa) * 1000.0 / (rho * g)
            h_s_total = h_s_pipe + h_s_fit
            NPSHa = patm_m + H_suction - h_s_total - (V_s**2)/(2*g)
            NPSHa_vals.append(NPSHa)

        idx_duty = int(np.argmin(np.abs(Qs_h - design_Q)))
        duty_H   = H_totals[idx_duty]
        duty_Hs  = Hs_loss[idx_duty]
        duty_Hd  = Hd_loss[idx_duty]

        csum1, csum2, csum3 = st.columns(3)
        csum1.metric("Suction Head Loss (m)", f"{duty_Hs:.2f}")
        csum2.metric("Discharge Head Loss (m)", f"{duty_Hd:.2f}")
        csum3.metric("Total Developed Head (m)", f"{duty_H:.2f}")

        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(Qs_h, H_totals, linewidth=3, label="System Curve (TDH)")
        ax.axhline(H_static, linestyle="--", linewidth=1.8, label=f"Static Head = {H_static:.1f} m")
        ax.scatter([design_Q],[duty_H], s=60, marker="o", label=f"Duty @ Q={design_Q:.0f} → H={duty_H:.1f} m")
        ax.set_xlabel("Flow (m³/h)"); ax.set_ylabel("Head (m)")
        ax.grid(True, alpha=0.3, linewidth=0.8)
        ax.legend()
        st.pyplot(fig)

        eff = st.slider("Pump efficiency (%) for power plot", min_value=20, max_value=90, value=75, step=1, key="hyd_eff")
        eta = max(0.01, eff/100.0)
        PkW = [ (rho*g*(Q)*(H))/ (eta*1000.0) for Q,H in zip(Qs, H_totals) ]
        duty_P = PkW[idx_duty]
        figP, axP = plt.subplots(figsize=(7,4))
        axP.plot(Qs_h, PkW, linewidth=2)
        axP.scatter([design_Q],[duty_P], s=50, marker="o", label=f"Duty Power ≈ {duty_P:.0f} kW")
        axP.set_title("Power Consumed (absorbed)")
        axP.set_xlabel("Flow (m³/h)"); axP.set_ylabel("kW")
        axP.grid(True, alpha=0.3, linewidth=0.8)
        axP.legend()
        st.pyplot(figP)

        st.markdown("### Cavitation Check: NPSH")
        NPSHr = st.number_input("Pump NPSHr at duty (m)", min_value=0.0, value=5.0, step=0.5, key="hyd_npshr")
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(Qs_h, NPSHa_vals, linewidth=2, label="NPSH Available")
        ax2.plot([Qs_h[0], Qs_h[-1]], [NPSHr, NPSHr], linestyle="--", linewidth=2, label="NPSH Required")
        ax2.scatter([design_Q],[NPSHa_vals[idx_duty]], s=40, marker="o")
        ax2.set_xlabel("Flow (m³/h)"); ax2.set_ylabel("NPSH (m)")
        ax2.grid(True, alpha=0.3, linewidth=0.8)
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("### 🌊 Plant Animation: Suction Tank → Pump → Delivery Tank")
        st.caption("Flow slider drives animation speed and tank levels.")
        _default_flow = float(design_Q)
        _qmin, _qmax = float(Qmin), float(Qmax)
        if _qmax <= _qmin: _qmax = _qmin + 1.0
        flow_m3h = st.slider("Plant Flow (m³/h)", min_value=float(_qmin), max_value=float(_qmax),
                             value=float(min(max(_default_flow, _qmin), _qmax)), step=10.0, key="plant_anim_flow")
        lvl = max(0.0, min(1.0, (flow_m3h - _qmin) / (_qmax - _qmin)))
        dur_pipe = max(0.6, 1.8 - 1.2*lvl)
        dur_imp  = max(0.4, 1.2 - 0.8*lvl)
        su_top   = 200 - int(70 * (0.15 + 0.65*(1.0 - lvl)))
        deliv_top= 200 - int(70 * (0.15 + 0.65*(lvl)))

        plant_svg = f"""
        <div style='display:flex;justify-content:center;'>
          <svg width="900" height="340" viewBox="0 0 900 340" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="gWater" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#69B9FF"/><stop offset="100%" stop-color="#2F86FF"/>
              </linearGradient>
              <clipPath id="clipSuction"><rect x="60" y="60" width="200" height="200" rx="10"/></clipPath>
              <clipPath id="clipDelivery"><rect x="640" y="60" width="200" height="200" rx="10"/></clipPath>
            </defs>
            <rect x="60" y="60" width="200" height="200" fill="none" stroke="#333" stroke-width="3" rx="10"/>
            <g clip-path="url(#clipSuction)">
              <rect x="60" y="{su_top}" width="200" height="{260 - su_top}" fill="url(#gWater)"/>
              <path d="M 60 {su_top-5} C 90 {su_top+5}, 120 {su_top-5}, 150 {su_top} S 210 {su_top+5}, 260 {su_top}"
                    stroke="#fff" stroke-width="2" fill="none" opacity="0.5">
                <animate attributeName="d"
                  dur="{dur_pipe}s" repeatCount="indefinite"
                  values="M 60 {su_top-5} C 90 {su_top+5}, 120 {su_top-5}, 150 {su_top} S 210 {su_top+5}, 260 {su_top};
                          M 60 {su_top} C 90 {su_top-5}, 120 {su_top+5}, 150 {su_top} S 210 {su_top-5}, 260 {su_top};
                          M 60 {su_top-5} C 90 {su_top+5}, 120 {su_top-5}, 150 {su_top} S 210 {su_top+5}, 260 {su_top}" />
              </path>
            </g>
            <text x="60" y="50" font-size="12" fill="#333">Suction Tank</text>
            <rect x="640" y="60" width="200" height="200" fill="none" stroke="#333" stroke-width="3" rx="10"/>
            <g clip-path="url(#clipDelivery)">
              <rect x="640" y="{deliv_top}" width="200" height="{260 - deliv_top}" fill="url(#gWater)"/>
              <path d="M 640 {deliv_top-5} C 670 {deliv_top+5}, 700 {deliv_top-5}, 730 {deliv_top} S 790 {deliv_top+5}, 840 {deliv_top}"
                    stroke="#fff" stroke-width="2" fill="none" opacity="0.5">
                <animate attributeName="d"
                  dur="{dur_pipe}s" repeatCount="indefinite"
                  values="M 640 {deliv_top-5} C 670 {deliv_top+5}, 700 {deliv_top-5}, 730 {deliv_top} S 790 {deliv_top+5}, 840 {deliv_top};
                          M 640 {deliv_top} C 670 {deliv_top-5}, 700 {deliv_top+5}, 730 {deliv_top} S 790 {deliv_top-5}, 840 {deliv_top};
                          M 640 {deliv_top-5} C 670 {deliv_top+5}, 700 {deliv_top-5}, 730 {deliv_top} S 790 {deliv_top+5}, 840 {deliv_top}" />
              </path>
            </g>
            <text x="640" y="50" font-size="12" fill="#333">Delivery Tank</text>
            <rect x="260" y="140" width="110" height="40" fill="#8b8f97" rx="8"/>
            <g opacity="0.85">
              <rect x="260" y="150" width="110" height="20" fill="#4DA3FF"/>
              <rect x="260" y="150" width="30" height="20" fill="#A8D4FF">
                <animate attributeName="x" from="260" to="370" dur="{dur_pipe}s" repeatCount="indefinite" />
              </rect>
              <rect x="245" y="150" width="30" height="20" fill="#A8D4FF">
                <animate attributeName="x" from="245" to="355" dur="{dur_pipe}s" repeatCount="indefinite" />
              </rect>
            </g>
            <circle cx="450" cy="160" r="46" fill="#e7e7e7" stroke="#333" stroke-width="3"/>
            <circle cx="450" cy="160" r="12" fill="#777"/>
            <g transform="translate(450,160)">
              <g>
                <path d="M0,-34 Q10,-6 0,-4 Q-10,-6 0,-34" fill="#4DA3FF"/>
                <path d="M0,34 Q-10,6 0,4 Q10,6 0,34" fill="#4DA3FF"/>
                <path d="M-34,0 Q-6,10 -4,0 Q-6,-10 -34,0" fill="#4DA3FF"/>
                <path d="M34,0 Q6,-10 4,0 Q6,10 34,0" fill="#4DA3FF"/>
                <animateTransform attributeName="transform" attributeType="XML" type="rotate"
                                  from="0 0 0" to="360 0 0" dur="{dur_imp}s" repeatCount="indefinite"/>
              </g>
            </g>
            <text x="430" y="120" font-size="12" fill="#333">Pump</text>
            <rect x="490" y="140" width="130" height="40" fill="#8b8f97" rx="8"/>
            <g opacity="0.85">
              <rect x="490" y="150" width="130" height="20" fill="#4DA3FF"/>
              <rect x="490" y="150" width="30" height="20" fill="#A8D4FF">
                <animate attributeName="x" from="490" to="620" dur="{max(0.4, dur_pipe*0.8)}s" repeatCount="indefinite" />
              </rect>
              <rect x="475" y="150" width="30" height="20" fill="#A8D4FF">
                <animate attributeName="x" from="475" to="605" dur="{max(0.4, dur_pipe*0.8)}s" repeatCount="indefinite" />
              </rect>
            </g>
            <path d="M 620 160 Q 630 150 640 160" fill="none" stroke="#4DA3FF" stroke-width="7" stroke-linecap="round">
              <animate attributeName="d"
                       values="M 620 160 Q 630 150 640 160;
                               M 620 160 Q 630 170 640 160;
                               M 620 160 Q 630 150 640 160"
                       dur="{max(0.4, dur_pipe*0.8)}s" repeatCount="indefinite"/>
            </path>
            <text x="60"  y="290" font-size="12" fill="#333">Flow = {flow_m3h:.0f} m³/h</text>
            <text x="360" y="290" font-size="12" fill="#333">Impeller ~ 1/{dur_imp:.2f}s per rev</text>
            <text x="640" y="290" font-size="12" fill="#333">Levels: Suction ↑/↓ · Delivery ↑</text>
          </svg>
        </div>
        """
        components.html(plant_svg, height=380)

        st.markdown("### Affinity Laws Calculator")
        a1,a2,a3 = st.columns(3)
        N1 = a1.number_input("Speed N1 (RPM)", min_value=1.0, value=1480.0, step=10.0, key="hyd_aff_N1")
        N2 = a2.number_input("Speed N2 (RPM)", min_value=1.0, value=1200.0, step=10.0, key="hyd_aff_N2")
        Q1 = a3.number_input("Known Flow at N1 (m³/h)", min_value=0.0, value=design_Q, step=10.0, key="hyd_aff_Q1")
        b1,b2,b3 = st.columns(3)
        H1 = b1.number_input("Known Head at N1 (m)", min_value=0.0, value=max(1.0, duty_H), step=1.0, key="hyd_aff_H1")
        P1 = b2.number_input("Known Power at N1 (kW) (optional)", min_value=0.0, value=max(1.0, duty_P), step=10.0, key="hyd_aff_P1")
        use_diam = b3.checkbox("Include Diameter ratio?", value=False, key="hyd_aff_use_d")
        if use_diam:
            c1,c2 = st.columns(2)
            D1 = c1.number_input("Impeller D1 (mm)", min_value=1.0, value=500.0, step=1.0, key="hyd_aff_D1")
            D2 = c2.number_input("Impeller D2 (mm)", min_value=1.0, value=450.0, step=1.0, key="hyd_aff_D2")
            ratio = (N2/N1) * (D2/D1)
        else:
            ratio = (N2/N1)
        Q2 = Q1 * ratio
        H2 = H1 * (ratio**2)
        P2 = (P1 * (ratio**3)) if P1 > 0 else (rho*g*(Q2/3600.0)*H2)/(max(0.01, st.session_state.get("hyd_eff", 75)/100.0)*1000.0)
        cQ,cH,cP = st.columns(3)
        cQ.metric("Q2 (m³/h)", f"{Q2:,.0f}")
        cH.metric("H2 (m)", f"{H2:,.1f}")
        cP.metric("P2 (kW)", f"{P2:,.0f}" if P2>0 else "—")

# ====== TAB 5: Financial Analysis ======
with tab5:
    st.subheader("💡 Financial Analysis")
    c0a,c0b,c0c,c0d = st.columns(4)
    tariff_r_per_kwh = c0a.number_input("Energy tariff (R/kWh)", min_value=0.0, value=2.20, step=0.01, key="fin_tariff")
    water_r_per_m3 = c0b.number_input("Water cost (R/m³) (optional)", min_value=0.0, value=12.00, step=0.50, key="fin_water")
    hours_per_day = c0c.number_input("Run hours / day", min_value=0.0, value=20.0, step=0.5, key="fin_hrs_day")
    days_per_month = c0d.number_input("Days / month", min_value=1, value=30, step=1, key="fin_days_m")

    b1,b2 = st.columns(2)
    with b1:
        st.markdown("**Baseline**")
        pwr_b = st.number_input("Motor input power (kW) – Baseline", min_value=0.0, value=900.0, step=10.0, key="fin_pwr_b")
        flow_b = st.number_input("Average flow (m³/h) – Baseline", min_value=0.0, value=3200.0, step=10.0, key="fin_flow_b")
        maint_b = st.number_input("Monthly spares & labour (R) – Baseline", min_value=0.0, value=180000.0, step=1000.0, key="fin_maint_b")
        downtime_b = st.number_input("Monthly downtime losses (R) – Baseline", min_value=0.0, value=250000.0, step=1000.0, key="fin_dt_b")
    with b2:
        st.markdown("**Proposed**")
        pwr_p = st.number_input("Motor input power (kW) – Proposed", min_value=0.0, value=760.0, step=10.0, key="fin_pwr_p")
        flow_p = st.number_input("Average flow (m³/h) – Proposed", min_value=0.0, value=3400.0, step=10.0, key="fin_flow_p")
        maint_p = st.number_input("Monthly spares & labour (R) – Proposed", min_value=0.0, value=120000.0, step=1000.0, key="fin_maint_p")
        downtime_p = st.number_input("Monthly downtime losses (R) – Proposed", min_value=0.0, value=120000.0, step=1000.0, key="fin_dt_p")

    kwh_month_b = pwr_b * hours_per_day * days_per_month
    kwh_month_p = pwr_p * hours_per_day * days_per_month
    cost_energy_b = kwh_month_b * tariff_r_per_kwh
    cost_energy_p = kwh_month_p * tariff_r_per_kwh

    m3_month_b = flow_b * hours_per_day * days_per_month
    m3_month_p = flow_p * hours_per_day * days_per_month

    include_water = st.checkbox("Include water cost in totals", value=False, key="fin_inc_water")
    cost_water_b = (m3_month_b * water_r_per_m3) if include_water else 0.0
    cost_water_p = (m3_month_p * water_r_per_m3) if include_water else 0.0

    mA,mB,mC = st.columns(3)
    mA.metric("Energy (kWh/month) – Baseline", f"{kwh_month_b:,.0f}")
    mB.metric("Energy (kWh/month) – Proposed", f"{kwh_month_p:,.0f}")
    mC.metric("Δ Energy (kWh/month)", f"{(kwh_month_b-kwh_month_p):,.0f}")

    nA,nB,nC = st.columns(3)
    nA.metric("Energy Cost (R/month) – Baseline", f"{cost_energy_b:,.0f}")
    nB.metric("Energy Cost (R/month) – Proposed", f"{cost_energy_p:,.0f}")
    nC.metric("Energy Cost Saving (R/month)", f"{(cost_energy_b-cost_energy_p):,.0f}")

    wA,wB,wC = st.columns(3)
    wA.metric("Water (m³/month) – Baseline", f"{m3_month_b:,.0f}")
    wB.metric("Water (m³/month) – Proposed", f"{m3_month_p:,.0f}")
    wC.metric("Δ Water (m³/month)", f"{(m3_month_p-m3_month_b):,.0f}")

    wwA,wwB,wwC = st.columns(3)
    wwA.metric("Water Cost (R/month) – Baseline", f"{cost_water_b:,.0f}")
    wwB.metric("Water Cost (R/month) – Proposed", f"{cost_water_p:,.0f}")
    wwC.metric("Δ Water Cost (R/month)", f"{(cost_water_p-cost_water_b):,.0f}")

    st.markdown("### TCO / NPV & Payback")
    cY,cZ,cW = st.columns(3)
    horizon_years = cY.number_input("Horizon (years)", min_value=1, value=5, step=1, key="fin_horizon")
    discount_rate = cZ.number_input("Discount rate (%)", min_value=0.0, value=10.0, step=0.5, key="fin_disc")
    capex_prop = cW.number_input("CAPEX for Proposed (R)", min_value=0.0, value=5_500_000.0, step=50_000.0, key="fin_capex")

    monthly_total_b = (cost_energy_b + (cost_water_b if include_water else 0) + maint_b + downtime_b)
    monthly_total_p = (cost_energy_p + (cost_water_p if include_water else 0) + maint_p + downtime_p)
    annual_b = monthly_total_b * 12
    annual_p = monthly_total_p * 12
    r = discount_rate/100.0

    def npv_stream(annual_cost, years, r_):
        return sum(annual_cost/((1+r_)**t) for t in range(1, years+1))

    npv_b = npv_stream(annual_b, horizon_years, r)
    npv_p = capex_prop + npv_stream(annual_p, horizon_years, r)
    saving_npv = npv_b - npv_p

    t1,t2,t3 = st.columns(3)
    t1.metric("Baseline NPV (R)", f"{npv_b:,.0f}")
    t2.metric("Proposed NPV incl. CAPEX (R)", f"{npv_p:,.0f}")
    t3.metric("NPV Benefit (R)", f"{saving_npv:,.0f}")

    delta_month = monthly_total_b - monthly_total_p
    payback_months = (capex_prop / delta_month) if delta_month > 0 else float("inf")
    pA,pB = st.columns(2)
    pA.metric("Monthly Saving (R)", f"{delta_month:,.0f}")
    pB.metric("Simple Payback (months)", "∞" if not np.isfinite(payback_months) else f"{payback_months:,.1f}")

# ====== TAB 6: Reliability Visualisation ======
with tab6:
    st.subheader("📉 Reliability Visualisation")
    st.caption("Trends from Runtime Tracker and Maintenance History.")
    df_rt_vis = st.session_state.runtime_df.copy()

    if not df_rt_vis.empty:
        # Clean numeric cols and recompute Remaining Hours
        for col in df_rt_vis.columns:
            if col in ["MTBF (Hours)","Running Hours Since Last Major Maintenance","Remaining Hours"]:
                obj = df_rt_vis[col]
                if isinstance(obj, pd.DataFrame):
                    obj = obj.iloc[:, 0]
                df_rt_vis[col] = pd.to_numeric(obj, errors="coerce")
        df_rt_vis = compute_remaining_hours(df_rt_vis)

        c1,c2,c3 = st.columns(3)
        c1.metric("Assets tracked", f"{len(df_rt_vis):,}")

        mtbf_vals = get_numeric_series(df_rt_vis, "MTBF (Hours)")
        mtbf_mean = float(mtbf_vals.replace([np.inf,-np.inf], np.nan).dropna().mean()) if len(mtbf_vals) else float("nan")
        c2.metric("Mean MTBF (hrs)", "-" if (pd.isna(mtbf_mean)) else f"{mtbf_mean:,.0f}")

        if "STATUS" in df_rt_vis.columns:
            _status_obj = df_rt_vis["STATUS"]
            if isinstance(_status_obj, pd.DataFrame):
                _status_obj = _status_obj.iloc[:,0]
            due_over = int(_status_obj.astype(str).isin(["🟠 Plan for maintenance","🔴 Overdue for maintenance"]).sum())
        else:
            due_over = 0
        den = max(int(len(df_rt_vis)), 1)
        c3.metric("Due/Overdue %", f"{(due_over/den*100):.1f}%")

        # Simple scatter: MTBF vs Remaining
        rem_vals = get_numeric_series(df_rt_vis, "Remaining Hours")
        fig_sc = go.Figure()
        try:
            tags = df_rt_vis["Asset Tag"].astype(str).tolist()
        except Exception:
            tags = [f"A{i}" for i in range(len(rem_vals))]
        fig_sc.add_trace(go.Scatter(
            x=mtbf_vals, y=rem_vals, mode="markers",
            text=tags, hovertemplate="Tag: %{text}<br>MTBF: %{x:.0f} h<br>Remaining: %{y:.0f} h<extra></extra>"
        ))
        fig_sc.update_layout(
            title="MTBF vs Remaining Hours",
            xaxis_title="MTBF (Hours)", yaxis_title="Remaining Hours",
            height=420, margin=dict(l=10,r=10,t=40,b=10)
        )
        st.plotly_chart(fig_sc, use_container_width=True)
    else:
        st.info("Add rows in **🕒 Runtime Tracker** to see visuals.")
