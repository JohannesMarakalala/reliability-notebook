"""
demo_app_hands_on.py — Hands-On Demo App (Sandboxed)
----------------------------------------------------
This module renders a complete demo experience parallel to the main app,
with realistic functionality and clearly enforced limitations.

Key constraints (all enforced here, never leaking into the main app):
- Max 8 assets total
- Max 10 BOM rows per asset (view & edit limited to first 10)
- Max 10 maintenance records per asset
- Planning Pack & Jobs Manager: LOCKED (upgrade wall)
- "Enable Asset Components Runtime Tracker": disabled + upgrade wall
- Project Manager tab: present but fully LOCKED (upgrade wall)
- Reliability Viz & Financial exports: disabled (upgrade wall)
- Saves go to ./data_demo/ sandbox only

State keys are all prefixed with 'demo_' to avoid collisions.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Optional, Iterable, Any, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from scipy.stats import linregress
import matplotlib.pyplot as plt
import math
import plotly.express as px

# ---- Compatibility shim for rerun (newer Streamlit uses st.rerun) ----
try:
    if hasattr(st, "rerun") and not hasattr(st, "experimental_rerun"):
        st.experimental_rerun = st.rerun  # type: ignore[attr-defined]
except Exception:
    pass

# =========================
# Demo Constants & Paths
# =========================
DATA_DIR = "data_demo"
ASSETS_JSON  = os.path.join(DATA_DIR, "assets.json")
RUNTIME_CSV  = os.path.join(DATA_DIR, "runtime.csv")
HISTORY_CSV  = os.path.join(DATA_DIR, "history.csv")
CONFIG_JSON  = os.path.join(DATA_DIR, "config.json")
ATTACH_DIR   = os.path.join(DATA_DIR, "attachments")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ATTACH_DIR, exist_ok=True)

DEMO_CAPS = {
    "max_assets": 8,
    "max_bom_items_per_asset": 10,
    "max_maintenance_records_per_asset": 10,
    "components_tracker_enabled": False,
    "planning_pack_enabled": False,
    "jobs_manager_enabled": False,
    "project_manager_enabled": False,  # whole tab locked
    "exports_enabled": False,          # viz & finance
    "allow_save": True                 # sandbox writes allowed
}

# =========================
# Aesthetics
# =========================
def _inject_css():
    st.markdown(
        """
        <style>
        .demo-ribbon {
          position: fixed; right: -50px; top: 14px; z-index: 9999;
          transform: rotate(45deg);
          background: #0ea5e9; color: #0b1220; font-weight: 700;
          padding: 6px 80px; box-shadow: 0 8px 18px rgba(0,0,0,.35);
        }
        .soft-card{
          padding:14px;border:1px solid rgba(0,0,0,0.08);border-radius:12px;
          background:rgba(255,255,255,0.85);box-shadow:0 8px 20px rgba(0,0,0,0.08);
        }
        .small-note { color:#334155; font-size:0.85rem; }
        .lock-card h4 { margin: 0 0 8px 0; }
        .upgrade-btn {
          background:#0ea5e9;color:#0b1220;border-radius:10px;
          padding:8px 12px;font-weight:700;border:1px solid rgba(14,165,233,0.5);
          display:inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def _header():
    st.markdown(
        """
        <div style="
          padding: 18px 22px; border-radius: 16px;
          background: linear-gradient(135deg, rgba(10,111,255,0.20), rgba(10,111,255,0.06));
          border: 1px solid rgba(10,111,255,0.25);
          box-shadow: 0 12px 28px rgba(0,0,0,0.18);
          display:flex; align-items:center; gap:14px; margin-bottom:8px;">
          <div style="font-size:26px;">⚙️ <b>VIGIL® — DEMO (Hands‑On)</b></div>
          <div style="opacity:.8;">Track • Plan • Analyse • Optimise — Sandbox</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="demo-ribbon">DEMO</div>', unsafe_allow_html=True)

# =========================
# Helpers & Compute
# =========================
def to_num(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x,str) and not x.strip()):
            return float(default)
        if isinstance(x,(int,float,np.floating)):
            return float(x)
        s=str(x).strip().lower().replace(",","")
        if s in {"","-","—","none","nan","n/a","na"}:
            return float(default)
        return float(s)
    except Exception:
        return float(default)

def compute_remaining_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = (df.copy() if isinstance(df,pd.DataFrame) else pd.DataFrame())
    if df.empty:
        return df
    run_col = "Running Hours Since Last Major Maintenance"
    if run_col not in df.columns: df[run_col] = 0.0
    if "MTBF (Hours)" not in df.columns: df["MTBF (Hours)"] = 0.0
    mtbf = pd.to_numeric(df["MTBF (Hours)"], errors="coerce").fillna(0.0)
    runh = pd.to_numeric(df[run_col], errors="coerce").fillna(0.0)
    df["Remaining Hours"] = (mtbf - runh).clip(lower=0.0)

    def _status(m,r):
        if m <= 0:
            return "🟢 Healthy"
        ratio = (r/m) if m>0 else 0.0
        return "🟢 Healthy" if ratio<0.80 else ("🟠 Plan for maintenance" if ratio<1.0 else "🔴 Overdue for maintenance")

    df["STATUS"] = [_status(m,r) for m,r in zip(mtbf,runh)]
    return df

BOM_COLS = ["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE","CRITICALITY","MTBF (H)"]

def build_bom_table(asset: dict) -> pd.DataFrame:
    df = pd.DataFrame(asset.get("BOM Table", []))
    if df.empty:
        legacy=[]
        bom=asset.get("BOM",[]) or []
        cmms=asset.get("CMMS Codes",[]) or []
        n=max(len(bom),len(cmms))
        for i in range(n):
            legacy.append({
                "CMMS MATERIAL CODE": cmms[i] if i<len(cmms) else "",
                "OEM PART NUMBER": bom[i] if i<len(bom) else "",
                "DESCRIPTION":"", "QUANTITY":1, "PRICE":0.0, "CRITICALITY":"Medium", "MTBF (H)":0.0
            })
        df=pd.DataFrame(legacy)
    for c in BOM_COLS:
        if c not in df.columns:
            df[c] = 0.0 if c in ("QUANTITY","PRICE","MTBF (H)") else ""
    df["QUANTITY"]=pd.to_numeric(df["QUANTITY"],errors="coerce").fillna(0).astype(int)
    df["PRICE"]=pd.to_numeric(df["PRICE"],errors="coerce").fillna(0.0)
    df["MTBF (H)"]=pd.to_numeric(df["MTBF (H)"],errors="coerce").fillna(0.0)
    df["CRITICALITY"]=df["CRITICALITY"].astype(str).replace({"":"Medium"})
    return df[BOM_COLS].copy()

def auto_table_height(n_rows: int, min_rows:int=3, max_rows:int=999,
                      row_px:int=38, header_px:int=42, padding_px:int=16) -> int:
    rows = min(max(n_rows, min_rows), max_rows)
    return int(header_px + padding_px + rows * row_px)

# =========================
# Demo Stores (sandbox)
# =========================
def _load_json(path:str, fallback: Any):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return fallback
    return fallback

def _save_json(path:str, obj: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save JSON: {e}")

def _load_csv(path:str, columns: List[str]) -> pd.DataFrame:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def _save_csv(path:str, df: pd.DataFrame):
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        st.error(f"Failed to save CSV: {e}")

def _demo_state_init():
    if "demo_caps" not in st.session_state:
        st.session_state.demo_caps = DEMO_CAPS.copy()
    if "demo_assets" not in st.session_state:
        st.session_state.demo_assets = _load_json(ASSETS_JSON, {})
    if "demo_runtime_df" not in st.session_state:
        st.session_state.demo_runtime_df = _load_csv(RUNTIME_CSV, [
            "Asset Tag","Functional Location","Asset Model","Criticality",
            "MTBF (Hours)","Last Overhaul","Running Hours Since Last Major Maintenance",
            "Remaining Hours","STATUS"
        ])
    if "demo_history_df" not in st.session_state:
        st.session_state.demo_history_df = _load_csv(HISTORY_CSV, [
            "Number","Asset Tag","Functional Location","WO Number","Date of Maintenance",
            "Maintenance Code","Spares Used","QTY","Hours Run Since Last Repair",
            "Labour Hours","Asset Downtime Hours","Notes and Findings","Attachments"
        ])
    if "demo_config" not in st.session_state:
        st.session_state.demo_config = _load_json(CONFIG_JSON, {
            "address_book":[{"name":"Planner","email":"planner@maraksreliability.com"}],
            "craft_presets":["Mechanical Technician","Electrician","Instrument Technician","Rigger","Boilermaker","Planner / Scheduler","Reliability Engineer","Other"],
            "dept_presets":["Mechanical","Electrical","Instrumentation","Rigging","Boiler / Fabrication","Process / Operations","Reliability","Planning / Scheduling","Engineering / Projects","Other"],
            "permit_presets":["Lifting Operation","Hot Work","Confined Space Entry","Electrical Isolation / LOTO","Working at Heights","Line Breaking","Pressure Testing"],
            "capacity_by_craft":{"Mechanical Technician":64,"Electrician":32,"Instrument Technician":24,"Rigger":24,"Boilermaker":24,"Planner / Scheduler":8,"Reliability Engineer":6,"Other":8},
            "sendgrid_from":"planner@maraksreliability.com",
            "default_asset_filter":[]
        })

def _seed_if_empty():
    """Seed small, believable demo dataset; called once on first run."""
    assets = st.session_state.demo_assets
    if assets:
        return
    demo_assets = {
        "408-PP-001":{
            "Functional Location":"Primary Mill Discharge",
            "Asset Type":"Pump","Model":"20/18 - AH-TU (WRT)","Criticality":"Medium","MTBF (Hours)":3600,
            "Technical Details":[{"Parameter":"Motor Power","Value":"1450","Unit":"kW","Notes":""}],
            "BOM Table":[
                {"CMMS MATERIAL CODE":"AM0525666","OEM PART NUMBER":"T118-1-K31","DESCRIPTION":"LANTERN RESTRICTOR","QUANTITY":1,"PRICE":150.0,"CRITICALITY":"Medium","MTBF (H)":1200},
                {"CMMS MATERIAL CODE":"AM1806819","OEM PART NUMBER":"7189T1116-2018TUAH","DESCRIPTION":"PACKING SET OF 4","QUANTITY":4,"PRICE":200.0,"CRITICALITY":"High","MTBF (H)":600},
                {"CMMS MATERIAL CODE":"AM2454801","OEM PART NUMBER":"WRT-IMPELLER","DESCRIPTION":"IMPELLER","QUANTITY":1,"PRICE":500.0,"CRITICALITY":"High","MTBF (H)":1000},
                {"CMMS MATERIAL CODE":"AM2454802","OEM PART NUMBER":"WRT-SHAFT","DESCRIPTION":"SHAFT","QUANTITY":1,"PRICE":800.0,"CRITICALITY":"High","MTBF (H)":3600},
                {"CMMS MATERIAL CODE":"AM2454803","OEM PART NUMBER":"WRT-GLAND","DESCRIPTION":"GLAND","QUANTITY":1,"PRICE":140.0,"CRITICALITY":"Medium","MTBF (H)":1800}
            ]
        },
        "408-PP-008":{
            "Functional Location":"Secondary Mill Discharge",
            "Asset Type":"Pump","Model":"550-MCR-TU","Criticality":"High","MTBF (Hours)":2800,
            "Technical Details":[{"Parameter":"Motor Power","Value":"1450","Unit":"kW","Notes":""}],
            "BOM Table":[
                {"CMMS MATERIAL CODE":"AM2454827","OEM PART NUMBER":"UMC55044C23 WEIRMIN","DESCRIPTION":"WEAR PLATE","QUANTITY":1,"PRICE":300.0,"CRITICALITY":"Medium","MTBF (H)":1500},
                {"CMMS MATERIAL CODE":"AM2454828","OEM PART NUMBER":"UMC50872C3 WEIRMIN","DESCRIPTION":"IMPELLER","QUANTITY":1,"PRICE":500.0,"CRITICALITY":"High","MTBF (H)":1000}
            ]
        },
        "CV-201":{
            "Functional Location":"Conveyor CV-201",
            "Asset Type":"Conveyor","Model":"CV-201","Criticality":"Medium","MTBF (Hours)":4200,
            "Technical Details":[{"Parameter":"Belt width","Value":"1200","Unit":"mm","Notes":""}],
            "BOM Table":[
                {"CMMS MATERIAL CODE":"CV201-01","OEM PART NUMBER":"IDL-STD","DESCRIPTION":"IDLER STANDARD","QUANTITY":6,"PRICE":80.0,"CRITICALITY":"Medium","MTBF (H)":3000},
                {"CMMS MATERIAL CODE":"CV201-02","OEM PART NUMBER":"MOTOR-75kW","DESCRIPTION":"DRIVE MOTOR","QUANTITY":1,"PRICE":3200.0,"CRITICALITY":"High","MTBF (H)":6000}
            ]
        }
    }
    st.session_state.demo_assets = demo_assets

    # seed runtime
    rt_rows = []
    for tag, meta in demo_assets.items():
        rt_rows.append({
            "Asset Tag": tag,
            "Functional Location": meta.get("Functional Location",""),
            "Asset Model": meta.get("Model",""),
            "Criticality": meta.get("Criticality","Medium"),
            "MTBF (Hours)": float(meta.get("MTBF (Hours)", 0.0)),
            "Last Overhaul":"",
            "Running Hours Since Last Major Maintenance": float(np.random.randint(100, 1200))
        })
    rt_df = compute_remaining_hours(pd.DataFrame(rt_rows))
    st.session_state.demo_runtime_df = rt_df

    # seed maintenance history
    hist_rows = []
    n = 1
    for tag in demo_assets.keys():
        for d, code in [("2025-01-05","Inspection"), ("2025-02-12","Lubrication"), ("2025-03-03","Observation")]:
            hist_rows.append({
                "Number": n, "Asset Tag": tag, "Functional Location":"",
                "WO Number": f"WO-{1000+n}", "Date of Maintenance": d,
                "Maintenance Code": code, "Spares Used":"", "QTY":0,
                "Hours Run Since Last Repair": np.random.randint(100, 600),
                "Labour Hours": np.random.choice([1,2,3]),
                "Asset Downtime Hours": np.random.choice([0.2,0.5,1.0]),
                "Notes and Findings": "Demo entry", "Attachments":""
            })
            n += 1
    st.session_state.demo_history_df = pd.DataFrame(hist_rows)

    # persist seed to sandbox
    _save_json(ASSETS_JSON, st.session_state.demo_assets)
    _save_csv(RUNTIME_CSV, st.session_state.demo_runtime_df)
    _save_json(CONFIG_JSON, st.session_state.demo_config)
    _save_csv(HISTORY_CSV, st.session_state.demo_history_df)

# =========================
# Upgrade wall (reusable)
# =========================
def upgrade_wall(title: str = "🔓 Upgrade to unlock",
                 bullets: Optional[List[str]] = None,
                 note: Optional[str] = None,
                 key_suffix: str = ""):
    bullets = bullets or [
        "Full Planning Pack & Jobs Manager",
        "Asset Components Runtime Tracker", 
        "Unlimited assets, BOM depth, and history"
    ]
    st.markdown('<div class="soft-card lock-card">', unsafe_allow_html=True)
    st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
    st.markdown("<ul>" + "".join([f"<li>{b}</li>" for b in bullets]) + "</ul>", unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="small-note">{note}</div>', unsafe_allow_html=True)
    cols = st.columns([1,3])
    with cols[0]:
        # Use a unique key with the provided suffix
        key = f"upg_{title}_{key_suffix}" if key_suffix else f"upg_{title}"
        if st.button("🔑 Sign in to Full", key=key):
            # Exit demo and go to landing
            for k in list(st.session_state.keys()):
                if k.startswith("demo_") or k in ("is_demo","demo_flavor","demo_caps"):
                    st.session_state.pop(k, None)
            st.session_state.pop("is_demo", None)
            st.toast("Exited Demo. Back to landing.", icon="👋")
            try:
                st.experimental_rerun()
            except Exception:
                if hasattr(st, "rerun"): st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Demo Sync & Caps
# =========================
def _sync_runtime_from_demo_assets():
    assets = st.session_state.get("demo_assets", {})
    rt = st.session_state.get("demo_runtime_df", pd.DataFrame()).copy()
    need=["Asset Tag","Functional Location","Asset Model","Criticality","MTBF (Hours)","Last Overhaul","Running Hours Since Last Major Maintenance","Remaining Hours","STATUS"]
    for c in need:
        if c not in rt.columns:
            rt[c] = np.nan if c in ("MTBF (Hours)","Running Hours Since Last Major Maintenance","Remaining Hours") else ""
    for t, meta in assets.items():
        mask = (rt["Asset Tag"].astype(str)==str(t))
        if not mask.any():
            rt = pd.concat([rt, pd.DataFrame([{
                "Asset Tag": t,
                "Functional Location": meta.get("Functional Location",""),
                "Asset Model": meta.get("Model",""),
                "Criticality": meta.get("Criticality","Medium"),
                "MTBF (Hours)": float(to_num(meta.get("MTBF (Hours)",0.0),0.0)),
                "Last Overhaul":"",
                "Running Hours Since Last Major Maintenance":0.0
            }])], ignore_index=True)
        else:
            if meta.get("Functional Location"): rt.loc[mask,"Functional Location"]=meta["Functional Location"]
            if meta.get("Model"): rt.loc[mask,"Asset Model"]=meta["Model"]
            if meta.get("Criticality"): rt.loc[mask,"Criticality"]=meta["Criticality"]
            # MTBF controlled from asset meta here
            rt.loc[mask,"MTBF (Hours)"]=float(to_num(meta.get("MTBF (Hours)",0.0),0.0))

    rt = compute_remaining_hours(rt)
    st.session_state.demo_runtime_df = rt
    _save_csv(RUNTIME_CSV, rt)

def _enforce_caps():
    caps = st.session_state.get("demo_caps", DEMO_CAPS)
    assets = st.session_state.get("demo_assets", {})
    # assets cap
    if len(assets) > caps["max_assets"]:
        keep = dict(list(assets.items())[:caps["max_assets"]])
        st.session_state.demo_assets = keep
        _save_json(ASSETS_JSON, keep)

    # BOM cap (truncate view only; we store full but show 10 max)
    # We'll truncate when displaying

    # Maintenance cap (store only first N entries per asset)
    h = st.session_state.get("demo_history_df", pd.DataFrame()).copy()
    if not h.empty and "Asset Tag" in h.columns:
        h["__ord"] = range(1, len(h)+1)
        h = h.sort_values(["Asset Tag","__ord"]).groupby("Asset Tag").head(caps["max_maintenance_records_per_asset"]).drop(columns="__ord")
        st.session_state.demo_history_df = h.reset_index(drop=True)
        _save_csv(HISTORY_CSV, st.session_state.demo_history_df)

# =========================
# TAB 1: Asset Hub (IN)
# =========================
def _tab_asset_hub():
    st.subheader("📌 Asset Hub — Demo Sandbox")
    caps = st.session_state.demo_caps
    assets: Dict[str, dict] = st.session_state.get("demo_assets", {})

    a_tab1, a_tab2 = st.tabs(["Browse & Metrics", "Explore Asset"])

    with a_tab1:
        c1, c2 = st.columns([1,2])
        with c1:
            q = st.text_input("Search by Asset Tag", key="demo_search_assets")
            tags = sorted([t for t in assets.keys() if not q or q.lower() in t.lower()])
            tag = st.selectbox("Select Asset", options=([""]+tags) if tags else [""], index=0, key="demo_asset_sel")
            st.caption(f"{len(tags)} result(s)")
        with c2:
            st.markdown("**Assets Overview**")
            overview = []
            for t in tags:
                meta = assets.get(t,{})
                overview.append({
                    "Asset Tag": t,
                    "Location": meta.get("Functional Location",""),
                    "Model": meta.get("Model",""),
                    "Crit": meta.get("Criticality","Medium"),
                    "MTBF (Hours)": meta.get("MTBF (Hours)", 0)
                })
            st.dataframe(pd.DataFrame(overview), use_container_width=True, height=auto_table_height(len(overview)))

        st.markdown("---")
        st.markdown("### Add Asset (Sandbox)")
        if len(assets) >= caps["max_assets"]:
            upgrade_wall("🔒 Asset limit reached in Demo",
                         ["Add unlimited assets in the full version",
                          "Deeper BOMs and components tracking",
                          "End-to-end Planning Pack & Jobs Manager"],
                         note=f"Max {caps['max_assets']} assets in demo.")
        else:
            with st.form("demo_add_asset_form", clear_on_submit=True):
                tcol1, tcol2, tcol3 = st.columns(3)
                with tcol1:
                    tag_new = st.text_input("Asset Tag *")
                    fl = st.text_input("Functional Location")
                    a_type = st.selectbox("Asset Type", ["Pump","Conveyor","Crusher","Screen","Other"], index=0)
                with tcol2:
                    model = st.text_input("Model")
                    crit = st.selectbox("Criticality", ["Low","Medium","High"], index=1)
                with tcol3:
                    mtbf = st.number_input("MTBF (Hours)", min_value=0.0, step=100.0, value=2000.0)
                    add_bom_rows = st.number_input("Initial BOM rows (0–10)", min_value=0, max_value=10, step=1, value=0)

                submitted = st.form_submit_button("➕ Add Asset")
                if submitted:
                    tag_new = (tag_new or "").strip()
                    if not tag_new:
                        st.error("Asset Tag is required.")
                    elif tag_new in assets:
                        st.error("Asset tag already exists.")
                    else:
                        assets[tag_new] = {
                            "Functional Location": fl,
                            "Asset Type": a_type,
                            "Model": model,
                            "Criticality": crit,
                            "MTBF (Hours)": float(mtbf),
                            "Technical Details":[{"Parameter":"", "Value":"", "Unit":"", "Notes":""}],
                            "BOM Table":[
                                {"CMMS MATERIAL CODE":"", "OEM PART NUMBER":"", "DESCRIPTION":"", "QUANTITY":1, "PRICE":0.0, "CRITICALITY":"Medium", "MTBF (H)":0.0}
                                for _ in range(int(add_bom_rows))
                            ]
                        }
                        st.session_state.demo_assets = assets
                        _save_json(ASSETS_JSON, assets)
                        _sync_runtime_from_demo_assets()
                        st.success(f"Asset '{tag_new}' added to demo.")
                        st.experimental_rerun()

        if tag:
            st.markdown("---")
            meta = assets.get(tag, {})
            st.markdown(f"### {tag} — Details")
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("Criticality", meta.get("Criticality","Medium"))
            with m2: st.metric("MTBF (Hours)", meta.get("MTBF (Hours)",0))
            with m3: st.metric("Model", meta.get("Model",""))
            with m4: st.metric("Location", meta.get("Functional Location",""))

            st.markdown("#### BOM (showing up to 10 rows in demo)")
            bom_df = build_bom_table(meta)
            visible = bom_df.head(st.session_state.demo_caps["max_bom_items_per_asset"]).copy()
            if len(bom_df) > st.session_state.demo_caps["max_bom_items_per_asset"]:
                st.caption(f"Showing first {st.session_state.demo_caps['max_bom_items_per_asset']} of {len(bom_df)} components — demo limit.")
                upgrade_wall("🔒 BOM depth limited in Demo",
                             ["Track all components in full version",
                              "Drive component-level runtime tracking",
                              "Smart spares optimization & alerts"])

            st.dataframe(visible, use_container_width=True, height=auto_table_height(len(visible)))

    with a_tab2:
        st.markdown("### Explore Asset — What‑if")
        if not tag:
            st.info("Select an asset in 'Browse & Metrics' to explore.")
        else:
            meta = assets.get(tag, {})
            rt = st.session_state.get("demo_runtime_df", pd.DataFrame())
            row = rt[rt["Asset Tag"]==tag]
            base_mtbf = float(meta.get("MTBF (Hours)", 0.0))
            base_run = float(row["Running Hours Since Last Major Maintenance"].iloc[0]) if not row.empty else 0.0

            col1, col2 = st.columns([1,2])
            with col1:
                mtbf_adj = st.slider("What‑if MTBF (hours)", min_value=0, max_value=int(max(1000, base_mtbf*2+1)), value=int(base_mtbf), step=50)
                run_adj  = st.slider("What‑if Running Hours", min_value=0, max_value=int(max(100, base_mtbf*1.5+1)), value=int(base_run), step=10)
                rem = max(mtbf_adj - run_adj, 0)
                st.metric("Remaining Hours (what‑if)", rem)
                st.caption("This does not change saved data — explore impact before committing.")

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=rem,
                    title={"text": "Remaining Hours (What‑if)"},
                    gauge={"axis":{"range":[0, max(1, mtbf_adj)]}}
                ))
                fig.update_layout(height=260, margin=dict(t=20,b=10,l=10,r=10))
                st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2: Planning Hub
# =========================
def _tab_planning_hub():
    st.subheader("🕒 Planning Hub — Demo Sandbox")

    p1, p2, p3 = st.tabs(["Universal Runtime Tracker", "Planning Pack", "Jobs Manager"])

    with p1:
        st.markdown("### Universal Runtime Tracker")
        # ... existing code ...
        
        st.markdown("---")
        st.markdown("#### Components Runtime Tracker (disabled in demo)")
        st.checkbox("Enable Asset Components Runtime Tracker", value=False, disabled=True, key="demo_comp_rt_chk")
        upgrade_wall("🔒 Components Runtime Tracker is locked in Demo",
                     ["Enable per-component tracking",
                      "Automated alerts when components approach MTBF",
                      "Deeper analytics & spares planning"],
                     note="Unlock in the full version.",
                     key_suffix="components_tracker")

    with p2:
        st.markdown("### Planning Pack")
        upgrade_wall("🔒 Planning Pack is locked in Demo",
                     ["Build weekly/shift plans with capacity checks",
                      "Kitting readiness & permit workflows", 
                      "Execution tracking and schedule adherence KPIs"],
                     note="Sign in to access the full Planning Pack.",
                     key_suffix="planning_pack")

    with p3:
        st.markdown("### Jobs Manager")
        upgrade_wall("🔒 Jobs Manager is locked in Demo",
                     ["Create work orders and track statuses",
                      "Auto-link history & spares usage",
                      "Bulk import/export & CMMS sync"],
                     note="Sign in to access Jobs Manager.",
                     key_suffix="jobs_manager")

# =========================
# TAB 3: Maintenance Records (IN)
# =========================
def _tab_maintenance_records():
    st.subheader("🧾 Maintenance Records — Demo Sandbox")

    h = st.session_state.get("demo_history_df", pd.DataFrame()).copy()
    assets = st.session_state.get("demo_assets", {})
    caps = st.session_state.get("demo_caps", DEMO_CAPS)

    # Logger
    st.markdown("### Maintenance Logger")
    with st.form("demo_mr_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            tag = st.selectbox("Asset Tag", options=[""]+sorted(assets.keys()), index=0, key="demo_mr_tag")
        with c2:
            wo = st.text_input("WO Number")
        with c3:
            when = st.text_input("Date of Maintenance (YYYY-MM-DD)", value=datetime.now().strftime("%Y-%m-%d"))
        with c4:
            code = st.selectbox("Maintenance Code", ["Inspection","Lubrication","Repair","Overhaul","Observation"], index=0)

        c5, c6, c7 = st.columns(3)
        with c5:
            hrs_since = st.number_input("Hours Since Last Repair", min_value=0.0, step=10.0)
        with c6:
            labour = st.number_input("Labour Hours", min_value=0.0, step=0.5)
        with c7:
            dt = st.number_input("Asset Downtime Hours", min_value=0.0, step=0.1)

        notes = st.text_area("Notes and Findings", height=80)

        submit = st.form_submit_button("➕ Add Record")
        if submit:
            if not tag:
                st.error("Select an Asset Tag.")
            else:
                # cap per asset
                cnt = len(h[h["Asset Tag"]==tag]) if not h.empty else 0
                if cnt >= caps["max_maintenance_records_per_asset"]:
                    st.warning(f"Demo limit reached: {caps['max_maintenance_records_per_asset']} records for {tag}.")
                    upgrade_wall("🔒 Maintenance records limit in Demo",
                                 ["Unlimited history logging",
                                  "Link to Jobs Manager & CMMS close-out",
                                  "Export full history for audits"],
                                 note="Sign in to keep full maintenance history.")
                else:
                    new_row = {
                        "Number": (int(h["Number"].max())+1) if not h.empty else 1,
                        "Asset Tag": tag,
                        "Functional Location":"",
                        "WO Number": wo,
                        "Date of Maintenance": when,
                        "Maintenance Code": code,
                        "Spares Used":"", "QTY":0,
                        "Hours Run Since Last Repair": float(hrs_since),
                        "Labour Hours": float(labour),
                        "Asset Downtime Hours": float(dt),
                        "Notes and Findings": notes,
                        "Attachments": ""
                    }
                    h = pd.concat([h, pd.DataFrame([new_row])], ignore_index=True)
                    st.session_state.demo_history_df = h
                    _enforce_caps()
                    _save_csv(HISTORY_CSV, st.session_state.demo_history_df)
                    st.success("Record added (demo sandbox).")
                    st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Records (showing up to 10 per asset in demo)")
    h_view = h.copy()
    if not h_view.empty and "Asset Tag" in h_view.columns:
        h_view["__ord"] = range(1, len(h_view)+1)
        h_view = h_view.sort_values(["Asset Tag","__ord"]).groupby("Asset Tag").head(DEMO_CAPS["max_maintenance_records_per_asset"]).drop(columns="__ord")
    st.dataframe(h_view, use_container_width=True, height=auto_table_height(min(len(h_view), 20)))

# =========================
# TAB 4: Reliability Visualisation (IN)
# =========================
def _tab_reliability_viz():
    st.subheader("📉 Reliability Visualisation — Demo Sandbox")

    h = st.session_state.get("demo_history_df", pd.DataFrame()).copy()
    rt = st.session_state.get("demo_runtime_df", pd.DataFrame()).copy()

    # Filters
    left, right = st.columns([2,1])
    with left:
        tag_filter = st.multiselect("Filter by Asset", options=sorted(rt["Asset Tag"].astype(str).unique().tolist()) if not rt.empty else [])
    with right:
        st.write(" ")
        st.write(" ")
        if st.button("⬇️ Export Charts (disabled)"):
            st.info("Export is disabled in demo.")
            upgrade_wall("🔒 Export locked in Demo",
                         ["Download raw datasets and charts",
                          "Automate weekly KPI packs",
                          "Share interactive dashboards with stakeholders"])

    if tag_filter:
        rt = rt[rt["Asset Tag"].isin(tag_filter)]
        if not h.empty:
            h = h[h["Asset Tag"].isin(tag_filter)]

    # Pareto of downtime hours by asset
    st.markdown("#### Pareto: Total Downtime Hours by Asset")
    if not h.empty:
        pareto = h.groupby("Asset Tag")["Asset Downtime Hours"].sum().sort_values(ascending=False)
        fig = go.Figure(go.Bar(x=pareto.index.tolist(), y=pareto.values.tolist()))
        fig.update_layout(height=320, margin=dict(t=30,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No maintenance data yet. Add records in the Maintenance tab.")

    # Trend: maintenance events per month
    st.markdown("#### Trend: Maintenance Events per Month")
    if not h.empty and "Date of Maintenance" in h.columns:
        df = h.copy()
        try:
            df["Date"] = pd.to_datetime(df["Date of Maintenance"], errors="coerce")
            ts = df.dropna(subset=["Date"]).groupby(pd.Grouper(key="Date", freq="M"))["Number"].count()
            fig2 = go.Figure(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers"))
            fig2.update_layout(height=320, margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute trend: {e}")
    else:
        st.info("No dated maintenance events yet.")

    st.markdown("---")
    st.markdown("#### Connect CMMS (disabled)")
    upgrade_wall("🔒 Connect CMMS locked in Demo",
                 ["Live sync with SAP/Pragma/EAM systems",
                  "Auto-close jobs into maintenance history",
                  "End-to-end reliability reporting"])

# =========================
# TAB 5: Financial Analysis (IN, redacted)
# =========================
def _tab_financials():
    st.subheader("💡 Financial Analysis — Demo Sandbox")

    # Redacted baselines
    baseline_downtime_cost_per_hour = 12000  # obfuscated number
    baseline_events_per_month = 8            # obfuscated
    baseline_repair_cost_avg = 45000         # obfuscated

    colA, colB, colC = st.columns(3)
    with colA:
        avoid_hours = st.slider("Avoided downtime (hours/month)", 0, 120, 16, step=2)
    with colB:
        improve_pct = st.slider("Availability improvement (%)", 0, 50, 8, step=1)
    with colC:
        program_cost = st.slider("Program cost (per month, ZAR)", 0, 250000, 60000, step=5000)

    monthly_savings = int((avoid_hours * baseline_downtime_cost_per_hour) * (1 + improve_pct/100))
    annual_savings = monthly_savings * 12
    payback_months = (program_cost / monthly_savings) if monthly_savings > 0 else float("inf")

    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Monthly Savings (redacted)", f"~R{round(monthly_savings, -2):,}")
    with m2: st.metric("Annual Savings (redacted)", f"~R{round(annual_savings, -3):,}")
    with m3: st.metric("Payback (months)", f"{payback_months:.1f}" if payback_months != float('inf') else "N/A")

    st.markdown("---")
    cols = st.columns([1,1])
    with cols[0]:
        if st.button("⬇️ Export Financials (disabled)"):
            st.info("Export is disabled in demo.")
            upgrade_wall("🔒 Financial exports locked in Demo",
                         ["Download full financial models",
                          "Scenario packs & stakeholder-ready PDFs",
                          "Deep cost drill-down with BOM-linked parts"])

# =========================
# TAB 6: Engineering Tool (IN, fully usable)
# =========================

def _tab_engineering_tools():
    st.subheader("🧰 Engineering Tool")
    st.caption("Quick calculations, verifications, and simple designs for site engineers.")

    # ---- Shared Constants & Helpers ----
    ROUGHNESS_MM = {
        "PVC / CPVC (smooth)": 0.0015, "HDPE (new)": 0.0015, "Stainless steel (new)": 0.015,
        "Copper (drawn)": 0.0015, "Carbon steel (new)": 0.045, "Commercial steel (used)": 0.09,
        "Galvanized iron": 0.15, "Ductile iron (cement-mortar lined)": 0.10, "Cast iron (new)": 0.26,
        "Cast iron (older)": 0.80, "FRP": 0.005, "PTFE-lined steel": 0.005, "Epoxy-lined steel": 0.01,
        "Rubber-lined steel (slurry)": 0.10, "Ceramic-lined (slurry severe)": 0.20, "Concrete (troweled)": 0.30,
        "Wood stave": 0.18, "Riveted steel": 1.00, "Corrugated metal": 3.00, "Aluminum (smooth)": 0.001,
        "PEX tubing": 0.007, "Brass (new)": 0.0015, "Titanium (smooth)": 0.005, "Glass-lined": 0.003,
    }
    BASE_K = {
        "Entrance (sharp)": 0.5, "Entrance (rounded)": 0.05, "Exit": 1.0,
        "Elbow 90° (standard)": 0.9, "Elbow 90° (long radius)": 0.45, "Elbow 45°": 0.4,
        "Tee (run/through)": 0.6, "Tee (branch/into)": 1.8, "Gate valve (open)": 0.15,
        "Globe valve (open)": 10.0, "Ball valve (open)": 0.05, "Butterfly valve (open)": 0.9,
        "Check valve (swing)": 2.0, "Check valve (ball)": 4.5, "Strainer (clean)": 2.0,
        "Orifice plate (sharp)": 0.5, "Bend (miter 90°)": 1.1, "Bend (smooth 90°)": 0.3,
        "Venturi (long)": 0.2, "Diaphragm valve": 2.0, "Plug valve": 0.1,
    }
    FLUID_PROPS = {
        "Water (20°C)": {"density_kgL": 0.998, "viscosity_cP": 1.002, "specific_heat_kJkgK": 4.184},
        "Oil (light, 40°C)": {"density_kgL": 0.85, "viscosity_cP": 20.0, "specific_heat_kJkgK": 2.0},
        "Slurry (fine, 20°C)": {"density_kgL": 1.2, "viscosity_cP": 5.0, "specific_heat_kJkgK": 3.5},
        "Slurry (coarse, 20°C)": {"density_kgL": 1.5, "viscosity_cP": 50.0, "specific_heat_kJkgK": 3.0},
        "Air (20°C)": {"density_kgL": 0.0012, "viscosity_cP": 0.018, "specific_heat_kJkgK": 1.005},
        "Ethanol (25°C)": {"density_kgL": 0.789, "viscosity_cP": 1.074, "specific_heat_kJkgK": 2.44},
        "Glycerin (20°C)": {"density_kgL": 1.26, "viscosity_cP": 1500.0, "specific_heat_kJkgK": 2.43},
        "Custom": {"density_kgL": 1.0, "viscosity_cP": 1.0, "specific_heat_kJkgK": 4.184},
    }
    MATERIAL_PROPS = {
        "Steel (mild)": {"E_GPa": 200, "yield_MPa": 250, "density_kgm3": 7850, "thermal_cond_WmK": 50},
        "Aluminum": {"E_GPa": 70, "yield_MPa": 95, "density_kgm3": 2700, "thermal_cond_WmK": 205},
        "Concrete": {"E_GPa": 30, "yield_MPa": 20, "density_kgm3": 2400, "thermal_cond_WmK": 1.4},
        "Wood (pine)": {"E_GPa": 10, "yield_MPa": 40, "density_kgm3": 500, "thermal_cond_WmK": 0.15},
        "Copper": {"E_GPa": 110, "yield_MPa": 70, "density_kgm3": 8960, "thermal_cond_WmK": 400},
        "Custom": {"E_GPa": 200, "yield_MPa": 250, "density_kgm3": 7850, "thermal_cond_WmK": 50},
    }
    g = 9.80665

    def patm_kpa_from_elevation(elev_m: float) -> float:
        return 101.325 * math.exp(-elev_m / 8434.0)

    def swamee_jain_f(Re: float, eps: float, D: float) -> float:
        if Re <= 0 or D <= 0:
            return 0.02
        if Re < 2000:
            return 64.0 / Re
        return 0.25 / (math.log10((eps/(3.7*D)) + (5.74/(Re**0.9)))**2)

    def k_expansion(beta: float) -> float:
        b = max(1e-6, min(1.0, float(beta)))
        return (1.0 - b**2) ** 2

    def k_contraction(beta: float) -> float:
        b = max(1e-6, min(1.0, float(beta)))
        return 0.5 * ((1.0 / (b**2)) - 1.0)

    def fittings_panel(title: str, key_prefix: str) -> float:
        st.markdown(f"**{title}**")
        cols = st.columns(4)
        counts = {}
        for i, (name, K) in enumerate(BASE_K.items()):
            col = cols[i % 4]
            counts[name] = col.number_input(f"{name} (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_base_{i}")
        st.caption("Reducers / Expanders (β = smaller ID / larger ID)")
        rc1, rc2, rc3, rc4 = st.columns(4)
        qty_contr = rc1.number_input("Contraction (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_q_contr")
        beta_contr = rc2.number_input("Contraction β", min_value=0.01, max_value=0.99, value=0.80, step=0.01, key=f"{key_prefix}_b_contr")
        qty_exp = rc3.number_input("Expansion (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_q_exp")
        beta_exp = rc4.number_input("Expansion β", min_value=0.01, max_value=0.99, value=0.80, step=0.01, key=f"{key_prefix}_b_exp")
        counts = {k: int(v) for k, v in counts.items()}
        K_total = sum(counts.get(name, 0) * BASE_K[name] for name in BASE_K)
        K_total += qty_contr * k_contraction(beta_contr)
        K_total += qty_exp * k_expansion(beta_exp)
        st.caption(f"Sum K = {K_total:.3f}")
        return float(K_total)

    # ===== Units toggle (top-most)
    is_us = st.checkbox("Use US units (gpm, ft, psi, in)", value=False, key="eng_units_us")
    # 🔧 make 'is_us' also available to nested blocks that read from session_state
    st.session_state["is_us"] = is_us

    # --- legacy compatibility: we removed search boxes; keep downstream filters happy ---
    tool_search = ""

    def u_len_label(): return "ft" if is_us else "m"
    def u_head_label(): return "ft" if is_us else "m"
    def u_press_label(): return "psi" if is_us else "kPa"
    def u_flow_label(): return "gpm" if is_us else "m³/h"
    def u_diam_label(): return "in" if is_us else "mm"
    def u_rough_label(): return "in" if is_us else "mm"
    def u_vel_label(): return "ft/s" if is_us else "m/s"
    def u_power_label(): return "hp" if is_us else "kW"
    def u_force_label(): return "lb" if is_us else "N"
    def u_stress_label(): return "psi" if is_us else "MPa"
    def u_torque_label(): return "lb-ft" if is_us else "N-m"
    def to_si_flow(q): return (q * 0.00378541 / 60.0) if is_us else (q / 3600.0)
    def from_si_flow_m3h(q_m3s): return (q_m3s * 60.0 / 0.00378541) if is_us else (q_m3s * 3600.0)
    def to_si_len(x): return (x * 0.3048) if is_us else x
    def from_si_len_m(x_m): return (x_m / 0.3048) if is_us else x_m
    def to_si_head(h): return (h * 0.3048) if is_us else h
    def from_si_head_m(h_m): return (h_m / 0.3048) if is_us else h_m
    def to_si_press(p): return (p * 6.89476) if is_us else p
    def from_si_press_kpa(p_kpa): return (p_kpa / 6.89476) if is_us else p_kpa
    def to_si_diam(d): return (d * 0.0254) if is_us else (d / 1000.0)
    def to_si_rough(r): return (r * 0.0254) if is_us else (r / 1000.0)
    def from_si_vel(v): return (v / 0.3048) if is_us else v
    def from_si_power_kw(p_kw): return (p_kw * 1.3410229) if is_us else p_kw
    def to_si_power_kw(p): return (p / 1.3410229) if is_us else p
    def to_si_force(f): return (f * 4.44822) if is_us else f
    def to_si_stress(s): return (s * 6.89476 / 1000.0) if is_us else s
    def to_si_torque(t): return (t * 1.35582) if is_us else t
    def to_si_temp(t, from_unit='C'):
        if from_unit == 'F': return (t - 32) * 5 / 9
        if from_unit == 'K': return t - 273.15
        return t
    def from_si_temp(t_c, to_unit='C'):
        if to_unit == 'F': return t_c * 9 / 5 + 32
        if to_unit == 'K': return t_c + 273.15
        return t_c

    # ===== Sub-tabs (immediately under units)
    sub_hyd, sub_mech, sub_elec, sub_unit, sub_sci = st.tabs([
        "💧  Hydraulics Toolbox", "🔧 Mechanical", "⚡ Electrical/Thermal", "🔄 Unit Converter", "🧮 Scientific Calculator"
    ])

    # ============ HYDRAULICS ============
    with sub_hyd:
        st.markdown("### 💧 Hydraulics Toolbox")
        st.caption("Industrial-grade tools for fluid systems.")

        # Add missing unit conversion functions for hydraulics section
        def from_si_len(x_m):
            return (x_m / 0.3048) if is_us else x_m

        def from_si_vel(v_ms):
            return (v_ms / 0.3048) if is_us else v_ms

        def from_si_power_kw(p_kw):
            return (p_kw * 1.3410229) if is_us else p_kw

        def from_si_torque(t_nm):
            return (t_nm / 1.35582) if is_us else t_nm

        def from_si_force(f_n):
            return (f_n / 4.44822) if is_us else f_n

        # Tool selector (single dropdown for a clean subtab)
        tool_option = st.selectbox(
            "Select Tool",
            [
                "Hydraulic System Calculator",
                "Pipe Sizing Assistant", 
                "Valve Sizing",
                "Orifice Sizing",
            ],
            index=0,
            key="hyd_system_tool",
        )

        # ---------- Shared UI helpers used by hydraulics tools ----------
        def pill(label: str, value_text: str, tooltip: str):
            st.markdown(
                f"""
                <span title="{tooltip}"
                      style="display:inline-block;padding:4px 10px;border-radius:9999px;background:#eef3ff;
                             color:#0b2950;font-size:0.92rem;border:1px solid #d9e3ff;margin:2px 6px 6px 0;">
                    <strong style="margin-right:6px;">{label}:</strong>{value_text}
                </span>
                """,
                unsafe_allow_html=True,
            )

        # Dense fittings panel (8 inputs per row)
        def fittings_panel_dense(title: str, key_prefix: str, cols_per_row: int = 8) -> float:
            st.markdown(f"**{title}**")
            cols = st.columns(cols_per_row)
            counts = {}
            for i, (name, K) in enumerate(BASE_K.items()):
                col = cols[i % cols_per_row]
                counts[name] = col.number_input(
                    f"{name} (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_dense_{i}"
                )
            st.caption("Reducers / Expanders (β = smaller ID / larger ID)")
            rc1, rc2, rc3, rc4 = st.columns(4)
            qty_contr = rc1.number_input("Contraction (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_q_contr8")
            beta_contr = rc2.number_input("Contraction β", min_value=0.01, max_value=0.99, value=0.80, step=0.01, key=f"{key_prefix}_b_contr8")
            qty_exp = rc3.number_input("Expansion (qty)", min_value=0, value=0, step=1, key=f"{key_prefix}_q_exp8")
            beta_exp = rc4.number_input("Expansion β", min_value=0.01, max_value=0.99, value=0.80, step=0.01, key=f"{key_prefix}_b_exp8")
            counts = {k: int(v) for k, v in counts.items()}
            K_total = sum(counts.get(name, 0) * BASE_K[name] for name in BASE_K)
            K_total += qty_contr * k_contraction(beta_contr)
            K_total += qty_exp * k_expansion(beta_exp)
            st.caption(f"Sum K = {K_total:.3f}")
            return float(K_total)

        def suction_block_local(prefix="hc_suc8", design_flow=None):
            s1, s2, s3, s4 = st.columns(4)
            Hs = s1.number_input(f"Suction static head (+flooded, −lift) ({u_head_label()})", value=0.0, key=f"{prefix}_Hs")
            Ds = s2.number_input(f"Suction Pipe ID ({u_diam_label()})", min_value=0.1, value=300.0, step=1.0, key=f"{prefix}_Ds")
            Ls = s3.number_input(f"Suction Pipe Length ({u_len_label()})", min_value=0.0, value=10.0, step=1.0, key=f"{prefix}_Ls")
            Ps = s4.number_input(f"Suction Residual Pressure ({u_press_label()})", min_value=0.0, value=0.0, step=0.01, key=f"{prefix}_Ps")
            t1, t2, t3 = st.columns(3)
            lining = t1.selectbox("Suction lining / material", list(ROUGHNESS_MM.keys()), index=11, key=f"{prefix}_lining")
            auto_r = t2.checkbox("Use preset roughness", value=True, key=f"{prefix}_auto_r")
            rough = t3.number_input(f"Override roughness ({u_rough_label()})", value=ROUGHNESS_MM[lining], step=0.001, key=f"{prefix}_rough", disabled=auto_r)
            K_s = fittings_panel_dense("Suction Fittings", prefix, cols_per_row=8)
            Dm = to_si_diam(Ds)
            Am = math.pi * (Dm**2) / 4.0 if Dm > 0 else 1e9
            V_suc = from_si_vel((to_si_flow(design_flow))/Am) if (design_flow and Am>0) else 0.0
            st.caption(f"Suction velocity @ design flow: {V_suc:.2f} {u_vel_label()}")
            eps_m = to_si_rough(rough) if not auto_r else ROUGHNESS_MM[lining]/1000.0
            return {"Hs_m": to_si_head(Hs), "Ds_m": Dm, "Ls_m": to_si_len(Ls), "Ps_kpa": to_si_press(Ps), "eps_m": eps_m, "Ksum": float(K_s)}

        def discharge_block_local(prefix="hc_dis8", design_flow=None):
            d1, d2, d3, d4 = st.columns(4)
            Hd = d1.number_input(f"Discharge static head ({u_head_label()})", value=10.0, key=f"{prefix}_Hd")
            Dd = d2.number_input(f"Discharge Pipe ID ({u_diam_label()})", min_value=0.1, value=300.0, step=1.0, key=f"{prefix}_Dd")
            Ld = d3.number_input(f"Discharge Pipe Length ({u_len_label()})", min_value=0.0, value=200.0, step=1.0, key=f"{prefix}_Ld")
            Pd = d4.number_input(f"Discharge Residual Pressure ({u_press_label()})", min_value=0.0, value=0.0, step=0.01, key=f"{prefix}_Pd")
            t1, t2, t3 = st.columns(3)
            lining = t1.selectbox("Discharge lining / material", list(ROUGHNESS_MM.keys()), index=4, key=f"{prefix}_lining")
            auto_r = t2.checkbox("Use preset roughness", value=True, key=f"{prefix}_auto_r")
            rough = t3.number_input(f"Override roughness ({u_rough_label()})", value=ROUGHNESS_MM[lining], step=0.001, key=f"{prefix}_rough", disabled=auto_r)
            K_d = fittings_panel_dense("Discharge Fittings", prefix, cols_per_row=8)
            Dm = to_si_diam(Dd)
            Am = math.pi * (Dm**2) / 4.0 if Dm > 0 else 1e9
            V_dis = from_si_vel((to_si_flow(design_flow))/Am) if (design_flow and Am>0) else 0.0
            st.caption(f"Discharge velocity @ design flow: {V_dis:.2f} {u_vel_label()}")
            eps_m = to_si_rough(rough) if not auto_r else ROUGHNESS_MM[lining]/1000.0
            return {"Hd_m": to_si_head(Hd), "Dd_m": Dm, "Ld_m": to_si_len(Ld), "Pd_kpa": to_si_press(Pd), "eps_m": eps_m, "Ksum": float(K_d)}

        # ---------- A) HYDRAULIC SYSTEM CALCULATOR ----------
        if tool_option == "Hydraulic System Calculator":
            st.markdown("### Hydraulic System Calculator")

            # Top row: Design/Min/Max
            c_top1, c_top2, c_top3 = st.columns(3)
            Q_design = c_top1.number_input(f"Design Flow ({u_flow_label()})", min_value=0.0, value=1000.0, step=10.0, key="hc_Q")
            Qmin = c_top2.number_input(f"Curve Minimum Flow ({u_flow_label()})", min_value=0.0, value=0.0, step=10.0, key="hc_qmin_inline")
            Qmax = c_top3.number_input(f"Curve Maximum Flow ({u_flow_label()})", min_value=1.0, value=1500.0, step=50.0, key="hc_qmax_inline")

            # Fluid/site
            c1, c2, c3, c4 = st.columns(4)
            elev = c1.number_input(f"Site Elevation ({u_len_label()})", min_value=0.0, value=0.0, step=10.0, key="hc_elev_inline")
            tempC = c2.number_input("Fluid Temperature (°C)", min_value=-10.0, value=20.0, step=1.0, key="hc_tempC_inline")
            fluid = c3.selectbox("Fluid", list(FLUID_PROPS.keys()), key="hc_fluid_inline")
            props = FLUID_PROPS[fluid]
            dens_kgL = c3.number_input("Density (kg/L)", min_value=0.2, value=props["density_kgL"], step=0.01, key="hc_rho_inline", disabled=(fluid!="Custom"))
            mu_cP = c4.number_input("Viscosity (cP)", min_value=0.05, value=props["viscosity_cP"], step=0.05, key="hc_mu_cP_inline", disabled=(fluid!="Custom"))
            vap_disp = c4.number_input(f"Vapor Pressure ({u_press_label()})", min_value=0.0, value=2.3, step=0.01, key="hc_vpk_inline")

            # SI props
            rho = dens_kgL * 1000.0
            mu = mu_cP / 1000.0
            patm_kpa = patm_kpa_from_elevation(to_si_len(elev))
            vap_kpa = to_si_press(vap_disp)

            # Mode/arrangement row
            r_mode1, r_mode2, r_mode3 = st.columns(3)
            mode = r_mode1.selectbox("Mode", ["Single pump", "Multiple pumps"], index=0, key="hc_mode")
            arrang = r_mode2.selectbox("Arrangement", ["Series", "Parallel"], index=0, key="hc_arr", disabled=(mode == "Single pump"))
            n_pumps = r_mode3.slider("Number of pumps", min_value=2, max_value=8, value=2, step=1, key="hc_n", disabled=(mode == "Single pump"))

            # Suction / Discharge blocks
            st.markdown("### ⬅️ Suction Details")
            suc = suction_block_local(prefix="hc_suc8", design_flow=Q_design)
            st.markdown("### ➡️ Discharge Details")
            dis = discharge_block_local(prefix="hc_dis8", design_flow=Q_design)

            # Pump curve
            st.markdown("### Pump curve")
            p1a, p1b, p1c = st.columns(3)
            H0 = p1a.number_input(f"Shutoff head ({u_head_label()}) at Q=0", min_value=0.0, value=60.0, step=1.0, key="hc_H0")
            Qb = p1b.number_input(f"BEP Flow ({u_flow_label()})", min_value=1.0, value=1800.0, step=10.0, key="hc_Qb")
            Hb = p1c.number_input(f"Head at BEP ({u_head_label()})", min_value=0.0, value=45.0, step=1.0, key="hc_Hb")

            cb1, cb2 = st.columns(2)
            lock_bep_to_design = cb1.checkbox("Lock BEP to pass through Design Point", value=False, key="hc_lock_bep")
            use_three_point_fit = cb2.checkbox("3-point pump fit (add runout)", value=True, key="hc_3pt")

            c3a, c3b, c3c, c3d = st.columns(4)
            runout_mult = c3a.slider("Runout flow multiplier (×BEP)", 1.1, 2.0, 1.3, 0.05, key="hc_run_mult")
            runout_head_pct = c3b.slider("Runout head (% of shutoff)", 0.0, 50.0, 10.0, 1.0, key="hc_run_pct")
            eta_pct = c3c.slider("Pump efficiency (%)", 30, 95, 75, 1, key="hc_eta")
            speed_pct = c3d.slider("Pump Speed (%)", 40, 120, 100, 1, key="hc_speed_pct")
            N_ref_rpm = c3d.number_input("Ref Speed (RPM)", min_value=100, max_value=20000, value=2950, step=10, key="hc_ref_rpm")

            if Qmax <= Qmin:
                st.warning("Maximum Flow must be greater than Minimum Flow.")
            else:
                # Friction helper
                def friction_losses_segment(Q_si, D_m, L_m, eps_m, Ksum):
                    if D_m <= 0:
                        return 0.0, 0.0, 0.0
                    A = math.pi * (D_m**2) / 4.0
                    V = Q_si / A if A > 0 else 0.0
                    Re = (rho * V * D_m) / max(mu, 1e-9)
                    f = swamee_jain_f(Re, eps_m, D_m)
                    h_pipe = f * (L_m / max(D_m, 1e-9)) * (V**2) / (2 * g)
                    h_minor = Ksum * (V**2) / (2 * g)
                    return h_pipe, h_minor, V

                # Static+pressure head (allow negative)
                H_static_levels_m = dis["Hd_m"] - suc["Hs_m"]
                H_press_m = ((dis["Pd_kpa"] - suc["Ps_kpa"]) * 1000.0) / (rho * g)
                H_static_total_m = H_static_levels_m + H_press_m

                # Flow axis
                Q_axis_disp = np.linspace(Qmin, Qmax, 500)
                Q_axis_si = np.array([to_si_flow(q) for q in Q_axis_disp])

                # System head & NPSHa
                H_sys_m = np.zeros_like(Q_axis_si, dtype=float)
                NPSHa_m = np.zeros_like(Q_axis_si, dtype=float)

                for i, Q_si in enumerate(Q_axis_si):
                    hs_pipe, hs_minor, V_s = friction_losses_segment(Q_si, suc["Ds_m"], suc["Ls_m"], suc["eps_m"], suc["Ksum"])
                    hd_pipe, hd_minor, _   = friction_losses_segment(Q_si, dis["Dd_m"], dis["Ld_m"], dis["eps_m"], dis["Ksum"])
                    H_sys_m[i] = H_static_total_m + hs_pipe + hs_minor + hd_pipe + hd_minor

                    patm_Pa = patm_kpa * 1000.0
                    P_s_Pa = suc["Ps_kpa"] * 1000.0
                    P_vap_Pa = vap_kpa * 1000.0
                    NPSHa_m[i] = (patm_Pa + P_s_Pa - P_vap_Pa) / (rho * g) + suc["Hs_m"] - (hs_pipe + hs_minor) - (V_s**2) / (2 * g)

                # Exact system head at design
                Qd_si = to_si_flow(Q_design)
                hs_pipe_d, hs_minor_d, V_s_d = friction_losses_segment(Qd_si, suc["Ds_m"], suc["Ls_m"], suc["eps_m"], suc["Ksum"])
                hd_pipe_d, hd_minor_d, _     = friction_losses_segment(Qd_si, dis["Dd_m"], dis["Ld_m"], dis["eps_m"], dis["Ksum"])
                H_sys_at_design_m = H_static_total_m + hs_pipe_d + hs_minor_d + hd_pipe_d + hd_minor_d

                # Pump reference curve (quadratic or 3-pt)
                Qb_disp, Hb_disp = float(Qb), float(Hb)
                if lock_bep_to_design:
                    Qb_disp = float(Q_design)
                    Hb_disp = float(from_si_head_m(H_sys_at_design_m))
                H0_m = to_si_head(H0)
                Qb_si = to_si_flow(Qb_disp)
                Hb_m = to_si_head(Hb_disp)

                if use_three_point_fit:
                    Qr_si = max(Qb_si * float(runout_mult), Qb_si + 1e-9)
                    Hr_m = H0_m * (float(runout_head_pct) / 100.0)
                    A = np.array([[0.0**2, 0.0, 1.0],[Qb_si**2, Qb_si, 1.0],[Qr_si**2, Qr_si, 1.0]])
                    y = np.array([H0_m, Hb_m, Hr_m])
                    a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
                else:
                    k = (H0_m - Hb_m) / max(Qb_si**2, 1e-12)
                    a, b, c = (-k, 0.0, H0_m)

                def H_ref_m(Q_si): return a*(Q_si**2) + b*Q_si + c

                # Required speed to hit design
                def f_speed(s):
                    s = max(s, 1e-6)
                    return (s**2)*H_ref_m(Qd_si/s) - H_sys_at_design_m

                s_scan = np.linspace(0.5, 1.5, 401)
                f_vals = np.array([f_speed(s) for s in s_scan])
                s_required = float(s_scan[np.argmin(np.abs(f_vals))])
                N_req_rpm = s_required * float(N_ref_rpm)
                eta = max(eta_pct/100.0, 1e-6)

                # Apply current speed
                s = speed_pct/100.0
                H_pump_scaled_m = np.array([(s**2)*H_ref_m(Q/max(s,1e-9)) for Q in Q_axis_si])
                H_pump_scaled_m = np.maximum(H_pump_scaled_m, 0.0)

                # Duty intersection (approx)
                idx_int = int(np.argmin(np.abs(H_pump_scaled_m - H_sys_m)))
                Q_int_disp = float(Q_axis_disp[idx_int])
                H_int_disp = float(from_si_head_m(H_sys_m[idx_int]))

                # Power
                P_h_kw = (rho * g * Q_axis_si * H_pump_scaled_m)/1000.0
                P_sh_kw = P_h_kw/eta

                # Pills
                p_row1a, p_row1b = st.columns(2)
                with p_row1a:
                    pill(
                        "System Head at Design",
                        f"{from_si_head_m(H_sys_at_design_m):.2f} {u_head_label()}",
                        "Total head the pump must overcome at the design flow (static ± friction + pressure delta)."
                    )
                with p_row1b:
                    pill(
                        "Required Pump Speed to Hit Design",
                        f"{N_req_rpm:.0f} RPM",
                        f"Speed via affinity laws to intersect the system at design flow. Uses reference {float(N_ref_rpm):.0f} RPM."
                    )

                # Three plots side-by-side
                p1, p2, p3 = st.columns(3)
                flow_lab = f"Flow ({u_flow_label()})"
                head_lab = f"Head ({u_head_label()})"
                power_lab = f"Power ({u_power_label()})"

                with p1:
                    st.subheader("System Curve")
                    df_sys = pd.DataFrame({
                        "Flow": Q_axis_disp,
                        f"System Head ({u_head_label()})": from_si_head_m(H_sys_m),
                        f"Pump Head @ {speed_pct}% ({u_head_label()})": from_si_head_m(H_pump_scaled_m)
                    })
                    fig_sys = px.line(
                        df_sys, x="Flow",
                        y=[f"System Head ({u_head_label()})", f"Pump Head @ {speed_pct}% ({u_head_label()})"],
                        labels={"Flow": flow_lab, "variable": "Curve", "value": head_lab}
                    )
                    fig_sys.add_scatter(x=[Q_design], y=[from_si_head_m(H_sys_at_design_m)], mode="markers", name="Design @ System")
                    fig_sys.add_scatter(x=[Q_int_disp], y=[H_int_disp], mode="markers", name="Duty (Intersection)")
                    fig_sys.add_hline(y=from_si_head_m(H_static_total_m), line_dash="dot", annotation_text="Static head", annotation_position="top left")
                    yvals = np.concatenate([
                        from_si_head_m(H_sys_m),
                        from_si_head_m(H_pump_scaled_m),
                        [from_si_head_m(H_static_total_m)]
                    ])
                    ymin, ymax = np.min(yvals), np.max(yvals)
                    fig_sys.update_yaxes(range=[0, ymax*1.1])
                    st.plotly_chart(fig_sys, use_container_width=True)

                with p2:
                    st.subheader("NPSH")
                    df_npsh = pd.DataFrame({
                        "Flow": Q_axis_disp,
                        f"NPSHa ({u_head_label()})": from_si_head_m(NPSHa_m)
                    })
                    fig_npsh = px.line(df_npsh, x="Flow", y=f"NPSHa ({u_head_label()})", labels={"Flow": flow_lab, "value": f"NPSHa ({u_head_label()})"})
                    fig_npsh.add_hrect(y0=0, y1=from_si_head_m(3.0), fillcolor="red", opacity=0.2, annotation_text="Low NPSH risk", annotation_position="top left")
                    fig_npsh.add_hrect(y0=from_si_head_m(3.0), y1=from_si_head_m(6.0), fillcolor="yellow", opacity=0.2, annotation_text="Medium risk")
                    fig_npsh.add_hrect(y0=from_si_head_m(6.0), y1=from_si_head_m(1000), fillcolor="green", opacity=0.2, annotation_text="Safe")
                    fig_npsh.update_yaxes(range=[0, from_si_head_m(20.0)])
                    st.plotly_chart(fig_npsh, use_container_width=True)

                with p3:
                    st.subheader("Power")
                    df_power = pd.DataFrame({
                        "Flow": Q_axis_disp,
                        f"Power ({u_power_label()})": from_si_power_kw(P_sh_kw)
                    })
                    fig_power = px.line(df_power, x="Flow", y=f"Power ({u_power_label()})", labels={"Flow": flow_lab, "value": power_lab})
                    fig_power.update_yaxes(range=[0, from_si_power_kw(np.max(P_sh_kw))*1.1])
                    st.plotly_chart(fig_power, use_container_width=True)

                # Summary table
                st.markdown("### 📊 Summary")
                sum_df = pd.DataFrame({
                    "Parameter": [
                        "Duty Flow", "Duty Head", "Static Head", "Friction Loss @ Design", "NPSHa @ Design",
                        "Pump Speed", "Hydraulic Power", "Shaft Power", "Motor Power (est.)"
                    ],
                    "Value": [
                        f"{Q_int_disp:.1f} {u_flow_label()}",
                        f"{H_int_disp:.1f} {u_head_label()}",
                        f"{from_si_head_m(H_static_total_m):.1f} {u_head_label()}",
                        f"{from_si_head_m(hs_pipe_d + hs_minor_d + hd_pipe_d + hd_minor_d):.1f} {u_head_label()}",
                        f"{from_si_head_m(NPSHa_m[idx_int]):.1f} {u_head_label()}",
                        f"{speed_pct}% ({N_req_rpm:.0f} RPM req.)",
                        f"{from_si_power_kw(P_h_kw[idx_int]):.2f} {u_power_label()}",
                        f"{from_si_power_kw(P_sh_kw[idx_int]):.2f} {u_power_label()}",
                        f"{from_si_power_kw(P_sh_kw[idx_int]*1.1):.2f} {u_power_label()}"
                    ]
                })
                st.table(sum_df)

        # ---------- B) PIPE SIZING ASSISTANT ----------
        elif tool_option == "Pipe Sizing Assistant":
            st.markdown("### Pipe Sizing Assistant")

            # Inputs
            c1, c2, c3, c4 = st.columns(4)
            Q_pipe = c1.number_input(f"Flow Rate ({u_flow_label()})", min_value=0.1, value=100.0, step=10.0, key="pipe_Q")
            L_pipe = c2.number_input(f"Pipe Length ({u_len_label()})", min_value=0.0, value=100.0, step=10.0, key="pipe_L")
            dH_allow = c3.number_input(f"Allowable Head Loss ({u_head_label()})", min_value=0.1, value=5.0, step=0.5, key="pipe_dH")
            lining = c4.selectbox("Pipe Material", list(ROUGHNESS_MM.keys()), index=4, key="pipe_mat")

            # Fluid
            c5, c6, c7 = st.columns(3)
            fluid_pipe = c5.selectbox("Fluid", list(FLUID_PROPS.keys()), key="pipe_fluid")
            props_pipe = FLUID_PROPS[fluid_pipe]
            dens_pipe = c6.number_input("Density (kg/L)", min_value=0.2, value=props_pipe["density_kgL"], step=0.01, key="pipe_rho", disabled=(fluid_pipe!="Custom"))
            mu_pipe = c7.number_input("Viscosity (cP)", min_value=0.05, value=props_pipe["viscosity_cP"], step=0.05, key="pipe_mu", disabled=(fluid_pipe!="Custom"))

            # SI conversions
            rho_pipe = dens_pipe * 1000.0
            mu_pipe_si = mu_pipe / 1000.0
            eps_m = ROUGHNESS_MM[lining] / 1000.0
            Q_si = to_si_flow(Q_pipe)
            L_si = to_si_len(L_pipe)
            dH_allow_m = to_si_head(dH_allow)

            # Iterative sizing
            D_min = 0.001
            D_max = 2.0
            D_guess = (D_min + D_max) / 2.0
            for _ in range(50):
                A = math.pi * (D_guess**2) / 4.0
                V = Q_si / A
                Re = (rho_pipe * V * D_guess) / max(mu_pipe_si, 1e-9)
                f = swamee_jain_f(Re, eps_m, D_guess)
                hf = f * (L_si / D_guess) * (V**2) / (2 * g)
                if hf < dH_allow_m:
                    D_max = D_guess
                else:
                    D_min = D_guess
                D_guess = (D_min + D_max) / 2.0

            D_req_m = D_guess
            V_final = Q_si / (math.pi * (D_req_m**2) / 4.0)

            # Display results
            st.markdown("### 📏 Recommended Pipe Size")
            col1, col2, col3 = st.columns(3)
            col1.metric(f"Required Diameter", f"{from_si_len(D_req_m*1000):.1f} {u_diam_label()}")
            col2.metric(f"Velocity", f"{from_si_vel(V_final):.2f} {u_vel_label()}")
            col3.metric(f"Actual Head Loss", f"{from_si_head_m(hf):.2f} {u_head_label()}")

            # Velocity guidelines
            st.markdown("### 📊 Velocity Guidelines")
            guide_col1, guide_col2 = st.columns(2)
            with guide_col1:
                st.markdown("**Water & Low-Viscosity Fluids:**")
                st.markdown("- Suction: 0.6–1.5 m/s (2–5 ft/s)")
                st.markdown("- Discharge: 1.5–3.0 m/s (5–10 ft/s)")
            with guide_col2:
                st.markdown("**Slurries & Viscous Fluids:**")
                st.markdown("- Suction: 0.3–1.0 m/s (1–3 ft/s)")
                st.markdown("- Discharge: 1.0–2.0 m/s (3–6 ft/s)")

        # ---------- C) VALVE SIZING ----------
        elif tool_option == "Valve Sizing":
            st.markdown("### Valve Sizing")

            # Inputs
            c1, c2, c3, c4 = st.columns(4)
            Q_valve = c1.number_input(f"Flow Rate ({u_flow_label()})", min_value=0.1, value=100.0, step=10.0, key="valve_Q")
            dP_valve = c2.number_input(f"Pressure Drop ({u_press_label()})", min_value=0.1, value=10.0, step=1.0, key="valve_dP")
            valve_type = c3.selectbox("Valve Type", ["Globe", "Ball", "Butterfly", "Gate", "Check"], index=0, key="valve_type")
            fluid_valve = c4.selectbox("Fluid", list(FLUID_PROPS.keys()), key="valve_fluid")

            # Fluid properties
            props_valve = FLUID_PROPS[fluid_valve]
            dens_valve = props_valve["density_kgL"]
            rho_valve = dens_valve * 1000.0

            # SI conversions
            Q_si_valve = to_si_flow(Q_valve)
            dP_si_valve = to_si_press(dP_valve) * 1000.0  # Convert to Pa

            # Valve coefficient calculation
            Cv = Q_si_valve * math.sqrt(rho_valve / 1000.0) / math.sqrt(dP_si_valve / 100000.0)

            # Display results
            st.markdown("### 📊 Valve Sizing Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Required Cv", f"{Cv:.2f}")
            col2.metric("Flow Coefficient (Kv)", f"{Cv * 0.865:.2f}")
            col3.metric("Pressure Drop", f"{dP_valve:.1f} {u_press_label()}")

            # Valve selection guidelines
            st.markdown("### 📋 Valve Selection Guidelines")
            guide_col1, guide_col2 = st.columns(2)
            with guide_col1:
                st.markdown("**Valve Type Applications:**")
                st.markdown("- **Globe:** Throttling, precise control")
                st.markdown("- **Ball:** On/off, low pressure drop")
                st.markdown("- **Butterfly:** Large diameters, moderate control")
                st.markdown("- **Gate:** Isolation, minimal pressure drop")
                st.markdown("- **Check:** Prevent backflow")
            with guide_col2:
                st.markdown("**Cv Sizing Tips:**")
                st.markdown("- Select valve with Cv 20-80% of max")
                st.markdown("- Avoid operating near fully open")
                st.markdown("- Consider cavitation risk at high ΔP")

        # ---------- D) ORIFICE SIZING ----------
        elif tool_option == "Orifice Sizing":
            st.markdown("### Orifice Sizing")

            # Inputs
            c1, c2, c3, c4 = st.columns(4)
            Q_orifice = c1.number_input(f"Flow Rate ({u_flow_label()})", min_value=0.1, value=100.0, step=10.0, key="orifice_Q")
            dP_orifice = c2.number_input(f"Pressure Drop ({u_press_label()})", min_value=0.1, value=10.0, step=1.0, key="orifice_dP")
            pipe_diam = c3.number_input(f"Pipe Diameter ({u_diam_label()})", min_value=0.1, value=6.0, step=0.1, key="orifice_pipe_diam")
            fluid_orifice = c4.selectbox("Fluid", list(FLUID_PROPS.keys()), key="orifice_fluid")

            # Fluid properties
            props_orifice = FLUID_PROPS[fluid_orifice]
            dens_orifice = props_orifice["density_kgL"]
            rho_orifice = dens_orifice * 1000.0

            # SI conversions
            Q_si_orifice = to_si_flow(Q_orifice)
            dP_si_orifice = to_si_press(dP_orifice) * 1000.0  # Convert to Pa
            D_pipe_si = to_si_diam(pipe_diam)

            # Orifice calculation (simplified)
            A_pipe = math.pi * (D_pipe_si**2) / 4.0
            beta_guess = 0.5
            C_d = 0.61  # Discharge coefficient for sharp-edged orifice

            # Iterative solution for beta
            for _ in range(50):
                A_orifice = A_pipe * beta_guess**2
                Q_calc = C_d * A_orifice * math.sqrt(2 * dP_si_orifice / rho_orifice)
                error = Q_calc - Q_si_orifice
                if abs(error) < 1e-9:
                    break
                if error > 0:
                    beta_guess *= 0.99
                else:
                    beta_guess *= 1.01

            orifice_diam = D_pipe_si * beta_guess

            # Display results
            st.markdown("### 📊 Orifice Sizing Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Orifice Diameter", f"{from_si_len(orifice_diam*1000):.2f} {u_diam_label()}")
            col2.metric("Beta Ratio (d/D)", f"{beta_guess:.3f}")
            col3.metric("Pressure Drop", f"{dP_orifice:.1f} {u_press_label()}")

            # Guidelines
            st.markdown("### 📋 Orifice Plate Guidelines")
            guide_col1, guide_col2 = st.columns(2)
            with guide_col1:
                st.markdown("**Beta Ratio Ranges:**")
                st.markdown("- Recommended: 0.2 - 0.7")
                st.markdown("- Avoid: <0.2 or >0.75")
                st.markdown("- Ideal: 0.4 - 0.6")
            with guide_col2:
                st.markdown("**Installation Tips:**")
                st.markdown("- Minimum 10D upstream, 5D downstream")
                st.markdown("- Use flange taps for pressure measurement")
                st.markdown("- Consider vena contracta effects")
    # ============ MECHANICAL ============
    with sub_mech:
        st.markdown("### 🔧 Mechanical Tools")
        st.caption("Beam analysis, shaft sizing, bolt calculations, etc.")

        # Add missing unit conversion functions for mechanical section
        def from_si_len_m(x_m):
            return (x_m / 0.3048) if is_us else x_m

        def from_si_force(f_n):
            return (f_n / 4.44822) if is_us else f_n

        def from_si_torque(t_nm):
            return (t_nm / 1.35582) if is_us else t_nm

        mech_tool = st.selectbox(
            "Select Mechanical Tool",
            ["Beam Deflection Calculator", "Shaft Sizing", "Bolt Torque Calculator", "Spring Design"],
            key="mech_tool_select"
        )

        if mech_tool == "Beam Deflection Calculator":
            st.markdown("### Beam Deflection Calculator")

            c1, c2, c3 = st.columns(3)
            beam_type = c1.selectbox("Beam Type", ["Simply Supported", "Cantilever"], key="beam_type")
            load_type = c2.selectbox("Load Type", ["Point Load", "Distributed Load"], key="load_type")
            material = c3.selectbox("Material", list(MATERIAL_PROPS.keys()), key="beam_material")

            props = MATERIAL_PROPS[material]
            E = props["E_GPa"] * 1e9  # Convert to Pa

            c4, c5, c6 = st.columns(3)
            L_beam = c4.number_input(f"Length ({u_len_label()})", min_value=0.1, value=5.0, step=0.5, key="beam_L")
            if load_type == "Point Load":
                P = c5.number_input(f"Load ({u_force_label()})", min_value=0.1, value=1000.0, step=100.0, key="beam_P")
                a = c6.number_input(f"Load Position from left ({u_len_label()})", min_value=0.0, max_value=float(L_beam), value=float(L_beam)/2, step=0.5, key="beam_a")
            else:
                w = c5.number_input(f"Distributed Load ({u_force_label()}/{u_len_label()})", min_value=0.1, value=200.0, step=10.0, key="beam_w")

            # Cross-section
            c7, c8 = st.columns(2)
            section_type = c7.selectbox("Cross-section", ["Rectangular", "Circular", "I-beam"], key="beam_section")
            if section_type == "Rectangular":
                b = c8.number_input(f"Width ({u_diam_label()})", min_value=0.1, value=4.0, step=0.1, key="beam_b")
                h = c8.number_input(f"Height ({u_diam_label()})", min_value=0.1, value=6.0, step=0.1, key="beam_h")
                I = (b * h**3) / 12.0
            elif section_type == "Circular":
                d = c8.number_input(f"Diameter ({u_diam_label()})", min_value=0.1, value=4.0, step=0.1, key="beam_d")
                I = math.pi * d**4 / 64.0
            else:  # I-beam
                # Simplified I-beam moment of inertia
                h = c8.number_input(f"Height ({u_diam_label()})", min_value=0.1, value=6.0, step=0.1, key="beam_I_h")
                bf = c8.number_input(f"Flange Width ({u_diam_label()})", min_value=0.1, value=4.0, step=0.1, key="beam_bf")
                tf = c8.number_input(f"Flange Thickness ({u_diam_label()})", min_value=0.1, value=0.5, step=0.1, key="beam_tf")
                tw = c8.number_input(f"Web Thickness ({u_diam_label()})", min_value=0.1, value=0.3, step=0.1, key="beam_tw")
                I = (bf * h**3 - (bf - tw) * (h - 2*tf)**3) / 12.0

            # Convert to SI
            L_si = to_si_len(L_beam)
            if section_type != "I-beam":  # I-beam inputs are already in consistent units
                I_si = I * (to_si_diam(1))**4  # Convert moment of inertia
            else:
                I_si = I * (to_si_len(1))**4  # Convert for I-beam

            # Calculate deflection
            if beam_type == "Simply Supported":
                if load_type == "Point Load":
                    # Convert load position
                    a_si = to_si_len(a)
                    if a_si <= L_si/2:
                        x_si = a_si
                    else:
                        x_si = L_si - a_si
                    P_si = to_si_force(P)
                    deflection = (P_si * x_si * (L_si**2 - x_si**2)**1.5) / (9 * math.sqrt(3) * E * I_si * L_si)
                else:
                    w_si = to_si_force(w) / to_si_len(1)  # Convert distributed load
                    deflection = (5 * w_si * L_si**4) / (384 * E * I_si)
            else:  # Cantilever
                if load_type == "Point Load":
                    P_si = to_si_force(P)
                    deflection = (P_si * L_si**3) / (3 * E * I_si)
                else:
                    w_si = to_si_force(w) / to_si_len(1)
                    deflection = (w_si * L_si**4) / (8 * E * I_si)

            # Display results
            st.markdown("### 📊 Beam Analysis Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Maximum Deflection", f"{from_si_len_m(deflection*1000):.3f} {u_diam_label()}")
            col2.metric("Deflection/Length Ratio", f"{(deflection/L_si)*100:.3f}%")
            
            # Stress calculation
            if beam_type == "Simply Supported" and load_type == "Point Load":
                M_max = (P_si * a_si * (L_si - a_si)) / L_si
            elif beam_type == "Simply Supported" and load_type == "Distributed Load":
                M_max = (w_si * L_si**2) / 8
            elif beam_type == "Cantilever" and load_type == "Point Load":
                M_max = P_si * L_si
            else:  # Cantilever distributed load
                M_max = (w_si * L_si**2) / 2

            if section_type == "Rectangular":
                c = h/2
                stress = (M_max * to_si_diam(c)) / I_si
            elif section_type == "Circular":
                c = d/2
                stress = (M_max * to_si_diam(c)) / I_si
            else:  # I-beam
                c = h/2
                stress = (M_max * to_si_len(c)) / I_si

            col3.metric("Maximum Bending Stress", f"{from_si_press_kpa(stress/1000):.1f} {u_stress_label()}")

        elif mech_tool == "Shaft Sizing":
            st.markdown("### Shaft Sizing Calculator")

            c1, c2, c3 = st.columns(3)
            power = c1.number_input(f"Power ({u_power_label()})", min_value=0.1, value=10.0, step=1.0, key="shaft_power")
            rpm = c2.number_input("RPM", min_value=10, value=1800, step=100, key="shaft_rpm")
            material_shaft = c3.selectbox("Shaft Material", list(MATERIAL_PROPS.keys()), index=0, key="shaft_material")

            props_shaft = MATERIAL_PROPS[material_shaft]
            tau_allow = props_shaft["yield_MPa"] * 1e6 * 0.3  # 30% of yield for shear

            # Convert power to torque
            power_si = to_si_power_kw(power) * 1000  # Convert to Watts
            torque_si = power_si / (2 * math.pi * rpm / 60)

            # Shaft diameter from torsion
            d_shaft = ((16 * torque_si) / (math.pi * tau_allow)) ** (1/3)

            st.markdown("### 📊 Shaft Sizing Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Required Diameter", f"{from_si_len_m(d_shaft*1000):.1f} {u_diam_label()}")
            col2.metric("Torque", f"{from_si_torque(torque_si):.0f} {u_torque_label()}")
            col3.metric("Shear Stress", f"{from_si_press_kpa(tau_allow/1000):.0f} {u_stress_label()}")

        elif mech_tool == "Bolt Torque Calculator":
            st.markdown("### Bolt Torque Calculator")

            c1, c2, c3 = st.columns(3)
            bolt_size = c1.selectbox("Bolt Size", ["M6", "M8", "M10", "M12", "M16", "M20", "1/4\"", "3/8\"", "1/2\"", "3/4\""], key="bolt_size")
            bolt_class = c2.selectbox("Bolt Class", ["4.6", "8.8", "10.9", "12.9"], key="bolt_class")
            friction = c3.number_input("Friction Coefficient", min_value=0.05, value=0.15, step=0.01, key="bolt_friction")

            # Bolt properties (simplified)
            bolt_props = {
                "M6": {"area_mm2": 20.1, "proof_load_kN": 6.72},
                "M8": {"area_mm2": 36.6, "proof_load_kN": 12.2},
                "M10": {"area_mm2": 58.0, "proof_load_kN": 19.3},
                "M12": {"area_mm2": 84.3, "proof_load_kN": 28.1},
                "M16": {"area_mm2": 157, "proof_load_kN": 52.4},
                "M20": {"area_mm2": 245, "proof_load_kN": 81.7},
            }

            if bolt_size in bolt_props:
                area = bolt_props[bolt_size]["area_mm2"] * 1e-6  # Convert to m²
                proof_load = bolt_props[bolt_size]["proof_load_kN"] * 1000  # Convert to N
                
                # Torque calculation
                preload = proof_load * 0.75  # 75% of proof load
                torque = preload * friction * (to_si_diam(1) if is_us else 0.001)  # Simplified

                st.markdown("### 📊 Bolt Torque Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Recommended Torque", f"{from_si_torque(torque):.0f} {u_torque_label()}")
                col2.metric("Preload Force", f"{from_si_force(preload):.0f} {u_force_label()}")
                col3.metric("Friction Coefficient", f"{friction:.2f}")

    # ============ ELECTRICAL/THERMAL ============
    with sub_elec:
        st.markdown("### ⚡ Electrical & Thermal Tools")
        st.caption("Motor sizing, cable sizing, heat transfer calculations.")

        elec_tool = st.selectbox(
            "Select Electrical/Thermal Tool",
            ["Motor Sizing", "Cable Sizing", "Heat Exchanger Basic", "Pipe Insulation"],
            key="elec_tool_select"
        )

        if elec_tool == "Motor Sizing":
            st.markdown("### Motor Sizing Calculator")

            c1, c2, c3 = st.columns(3)
            power_req = c1.number_input(f"Required Power ({u_power_label()})", min_value=0.1, value=10.0, step=1.0, key="motor_power")
            efficiency = c2.number_input("Efficiency (%)", min_value=10, max_value=99, value=85, step=1, key="motor_eff")
            service_factor = c3.number_input("Service Factor", min_value=1.0, max_value=2.0, value=1.15, step=0.05, key="motor_sf")

            # Motor size calculation
            motor_power = (power_req * 100 / efficiency) * service_factor

            st.markdown("### 📊 Motor Sizing Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Required Motor Power", f"{motor_power:.1f} {u_power_label()}")
            col2.metric("Efficiency", f"{efficiency}%")
            col3.metric("Service Factor", f"{service_factor}")

            # Standard motor sizes
            st.markdown("### 📋 Standard Motor Sizes")
            if is_us:
                std_sizes = ["1 hp", "2 hp", "3 hp", "5 hp", "7.5 hp", "10 hp", "15 hp", "20 hp", "25 hp", "30 hp", "40 hp", "50 hp"]
            else:
                std_sizes = ["0.75 kW", "1.1 kW", "1.5 kW", "2.2 kW", "3 kW", "4 kW", "5.5 kW", "7.5 kW", "11 kW", "15 kW", "18.5 kW", "22 kW"]
            
            recommended = next((size for size in std_sizes if float(size.split()[0]) >= motor_power), std_sizes[-1])
            st.info(f"**Recommended motor size: {recommended}**")

        elif elec_tool == "Cable Sizing":
            st.markdown("### Cable Sizing Calculator")

            c1, c2, c3 = st.columns(3)
            current = c1.number_input("Current (A)", min_value=1.0, value=50.0, step=5.0, key="cable_current")
            voltage = c2.number_input("Voltage (V)", min_value=24.0, value=480.0, step=24.0, key="cable_voltage")
            length = c3.number_input(f"Length ({u_len_label()})", min_value=1.0, value=100.0, step=10.0, key="cable_length")

            # Simplified cable sizing
            # Assuming copper cable, 75°C, 3% voltage drop
            area_mm2 = (current * to_si_len(length) * 0.0172 * 2) / (voltage * 0.03)

            st.markdown("### 📊 Cable Sizing Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Required Cable Size", f"{area_mm2:.1f} mm²")
            col2.metric("Current Rating", f"{current:.0f} A")
            col3.metric("Voltage Drop", "3%")

            # Standard cable sizes
            st.markdown("### 📋 Standard Cable Sizes")
            std_cables = ["1.5 mm²", "2.5 mm²", "4 mm²", "6 mm²", "10 mm²", "16 mm²", "25 mm²", "35 mm²", "50 mm²", "70 mm²", "95 mm²"]
            recommended = next((cable for cable in std_cables if float(cable.split()[0]) >= area_mm2), std_cables[-1])
            st.info(f"**Recommended cable: {recommended}**")

    # ============ UNIT CONVERTER ============
    with sub_unit:
        st.markdown("### 🔄 Unit Converter")
        st.caption("Convert between common engineering units.")

        conv_col1, conv_col2 = st.columns(2)

        with conv_col1:
            conversion_type = st.selectbox(
                "Conversion Type",
                ["Length", "Pressure", "Flow", "Temperature", "Force", "Power"],
                key="conv_type"
            )

        with conv_col2:
            if conversion_type == "Length":
                value = st.number_input("Value", value=1.0, key="conv_val_length")
                from_unit = st.selectbox("From", ["m", "mm", "in", "ft"], key="conv_from_length")
                to_unit = st.selectbox("To", ["m", "mm", "in", "ft"], key="conv_to_length")
                
                # Conversion factors to meters
                to_meter = {"m": 1.0, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
                result = value * to_meter[from_unit] / to_meter[to_unit]

            elif conversion_type == "Pressure":
                value = st.number_input("Value", value=1.0, key="conv_val_pressure")
                from_unit = st.selectbox("From", ["Pa", "kPa", "bar", "psi", "inH2O", "ftH2O"], key="conv_from_pressure")
                to_unit = st.selectbox("To", ["Pa", "kPa", "bar", "psi", "inH2O", "ftH2O"], key="conv_to_pressure")
                
                # Conversion factors to Pa
                to_pascal = {"Pa": 1.0, "kPa": 1000.0, "bar": 100000.0, "psi": 6894.76, "inH2O": 248.84, "ftH2O": 2989.07}
                result = value * to_pascal[from_unit] / to_pascal[to_unit]

            elif conversion_type == "Flow":
                value = st.number_input("Value", value=1.0, key="conv_val_flow")
                from_unit = st.selectbox("From", ["m³/s", "m³/h", "L/s", "gpm", "cfm"], key="conv_from_flow")
                to_unit = st.selectbox("To", ["m³/s", "m³/h", "L/s", "gpm", "cfm"], key="conv_to_flow")
                
                # Conversion factors to m³/s
                to_m3s = {"m³/s": 1.0, "m³/h": 1/3600.0, "L/s": 0.001, "gpm": 0.00006309, "cfm": 0.000471947}
                result = value * to_m3s[from_unit] / to_m3s[to_unit]

            elif conversion_type == "Temperature":
                value = st.number_input("Value", value=0.0, key="conv_val_temp")
                from_unit = st.selectbox("From", ["°C", "°F", "K"], key="conv_from_temp")
                to_unit = st.selectbox("To", ["°C", "°F", "K"], key="conv_to_temp")
                
                # Temperature conversions
                if from_unit == "°C":
                    if to_unit == "°F": result = value * 9/5 + 32
                    elif to_unit == "K": result = value + 273.15
                    else: result = value
                elif from_unit == "°F":
                    if to_unit == "°C": result = (value - 32) * 5/9
                    elif to_unit == "K": result = (value - 32) * 5/9 + 273.15
                    else: result = value
                else:  # K
                    if to_unit == "°C": result = value - 273.15
                    elif to_unit == "°F": result = (value - 273.15) * 9/5 + 32
                    else: result = value

            elif conversion_type == "Force":
                value = st.number_input("Value", value=1.0, key="conv_val_force")
                from_unit = st.selectbox("From", ["N", "kN", "lbf"], key="conv_from_force")
                to_unit = st.selectbox("To", ["N", "kN", "lbf"], key="conv_to_force")
                
                # Conversion factors to N
                to_newton = {"N": 1.0, "kN": 1000.0, "lbf": 4.44822}
                result = value * to_newton[from_unit] / to_newton[to_unit]

            elif conversion_type == "Power":
                value = st.number_input("Value", value=1.0, key="conv_val_power")
                from_unit = st.selectbox("From", ["W", "kW", "hp"], key="conv_from_power")
                to_unit = st.selectbox("To", ["W", "kW", "hp"], key="conv_to_power")
                
                # Conversion factors to W
                to_watt = {"W": 1.0, "kW": 1000.0, "hp": 745.7}
                result = value * to_watt[from_unit] / to_watt[to_unit]

        st.metric("Converted Value", f"{result:.6g} {to_unit}")

    # ============ SCIENTIFIC CALCULATOR ============
    with sub_sci:
        st.markdown("### 🧮 Scientific Calculator")
        st.caption("Advanced calculations for engineering analysis.")

        calc_col1, calc_col2 = st.columns(2)

        with calc_col1:
            calc_input = st.text_input("Enter calculation", "sin(pi/2) + log10(100)", key="calc_input")
            
            # Common constants
            st.markdown("**Common Constants:**")
            const_col1, const_col2 = st.columns(2)
            with const_col1:
                st.write("π = 3.141593")
                st.write("e = 2.718282")
                st.write("g = 9.80665 m/s²")
            with const_col2:
                st.write("R = 8.314 J/mol·K")
                st.write("c = 299792458 m/s")
                st.write("h = 6.626e-34 J·s")

        with calc_col2:
            try:
                # Safe evaluation with common math functions
                allowed_names = {
                    **{name: getattr(math, name) for name in dir(math) if not name.startswith('_')},
                    'abs': abs, 'max': max, 'min': min, 'sum': sum, 'round': round
                }
                
                # Evaluate the expression
                result = eval(calc_input, {"__builtins__": {}}, allowed_names)
                st.metric("Result", f"{result:.6g}")
                
                # Show additional representations
                if isinstance(result, (int, float)):
                    st.write(f"Exponential: {result:.4e}")
                    if result > 0:
                        st.write(f"Logarithm: log₁₀ = {math.log10(result):.4f}, ln = {math.log(result):.4f}")
                    
            except Exception as e:
                st.error(f"Calculation error: {str(e)}")

        # Quick calculations section
        st.markdown("### 🔢 Quick Calculations")
        quick_col1, quick_col2, quick_col3 = st.columns(3)
        
        with quick_col1:
            if st.button("Area of Circle"):
                radius = st.number_input("Radius", value=1.0, key="circle_radius")
                area = math.pi * radius ** 2
                st.write(f"Area: {area:.4f}")

        with quick_col2:
            if st.button("Volume of Sphere"):
                radius = st.number_input("Radius", value=1.0, key="sphere_radius")
                volume = (4/3) * math.pi * radius ** 3
                st.write(f"Volume: {volume:.4f}")

        with quick_col3:
            if st.button("Quadratic Roots"):
                st.write("ax² + bx + c = 0")
                a = st.number_input("a", value=1.0, key="quad_a")
                b = st.number_input("b", value=0.0, key="quad_b")
                c = st.number_input("c", value=-1.0, key="quad_c")
                
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    root1 = (-b + math.sqrt(discriminant)) / (2*a)
                    root2 = (-b - math.sqrt(discriminant)) / (2*a)
                    st.write(f"Roots: {root1:.4f}, {root2:.4f}")
                else:
                    real = -b/(2*a)
                    imag = math.sqrt(-discriminant)/(2*a)
                    st.write(f"Roots: {real:.4f} ± {imag:.4f}i")

# =========================
# TAB 7: Project Manager (LOCKED)
# =========================
def _tab_project_manager():
    st.subheader("📂 My Project Manager — Demo")
    upgrade_wall("🔒 Project Manager is locked in Demo",
                 ["Unified Gantt (Tasks & Milestones)",
                  "Linked assets, analytics, & FMECA hub",
                  "Exports and stakeholder share links"],
                 note="Sign in to access the full Project Manager.",
                 key_suffix="project_manager_main")

    # Render subtabs placeholders with walls for clarity
    t1, t2, t3 = st.tabs(["Tasks & Milestones", "Analytics", "Linked Items"])
    
    with t1:
        upgrade_wall("🔒 Subtab locked in Demo",
                     ["Edit and view all project structures",
                      "Interactive Gantt and dependencies", 
                      "Export and share project packages"],
                     key_suffix="tasks_milestones")
    
    with t2:
        upgrade_wall("🔒 Subtab locked in Demo", 
                     ["Edit and view all project structures",
                      "Interactive Gantt and dependencies",
                      "Export and share project packages"],
                     key_suffix="analytics")
    
    with t3:
        upgrade_wall("🔒 Subtab locked in Demo",
                     ["Edit and view all project structures",
                      "Interactive Gantt and dependencies",
                      "Export and share project packages"], 
                     key_suffix="linked_items")

# =========================
# Sidebar (Exit Demo + Sandbox Save)
# =========================
def _sidebar_demo():
    with st.sidebar:
        st.header("⚙️ Demo Controls")
        st.caption("Sandbox writes to ./data_demo; your main data is untouched.")
        if st.button("Exit Demo → Sign In"):
            # purge demo state and go back
            for k in list(st.session_state.keys()):
                if k.startswith("demo_") or k in ("is_demo","demo_flavor","demo_caps"):
                    st.session_state.pop(k, None)
            st.toast("Exited Demo. Back to landing.", icon="👋")
            try:
                st.experimental_rerun()
            except Exception:
                if hasattr(st, "rerun"): st.rerun()
        st.markdown("---")
        if st.session_state.get("demo_caps", DEMO_CAPS).get("allow_save", True):
            if st.button("💾 Save All (Sandbox)"):
                _save_json(ASSETS_JSON, st.session_state.demo_assets)
                _save_csv(RUNTIME_CSV, st.session_state.demo_runtime_df)
                _save_csv(HISTORY_CSV, st.session_state.demo_history_df)
                _save_json(CONFIG_JSON, st.session_state.demo_config)
                st.success("Saved to demo sandbox.")

        st.caption(f"Demo opened: {datetime.now().strftime('%d %b %Y, %H:%M')}")
        st.caption(f"Sandbox path: {os.path.abspath(DATA_DIR)}")

# =========================
# Public entrypoint
# =========================
def render():
    _inject_css()
    _header()
    _demo_state_init()
    _seed_if_empty()
    _enforce_caps()
    _sidebar_demo()

    # Create DEMO tabs (same 7 tabs as main, but with enforced limits)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📌 Asset Hub", "🕒 Planning Hub", "🧾 Maintenance Records",
        "📉 Reliability Visualisation", "💡 Financial Analysis", "🧰 Engineering Tool", "📂 My Project Manager"
    ])

    with tab1:
        _tab_asset_hub()
    with tab2:
        _tab_planning_hub()
    with tab3:
        _tab_maintenance_records()
    with tab4:
        _tab_reliability_viz()
    with tab5:
        _tab_financials()
    with tab6:
        _tab_engineering_tools()
    with tab7:
        _tab_project_manager()

# If someone runs the module directly (for testing)
if __name__ == "__main__":
    render()
