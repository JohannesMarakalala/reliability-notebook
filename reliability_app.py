# reliability_app.py — Landing + Auth + Demo gating (Top → Tabs creation)
# ----------------------------------------------------------------------------

import os, json, sqlite3, re, hashlib, secrets
from datetime import datetime
from typing import Optional, Any, Iterable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit as _st_internal  # keep an alias to the original module

# ---- Simple beta gate ----
if not st.session_state.get("beta_ok"):
    code = st.text_input("Enter beta access code", type="password")
    if st.button("Enter"):
        if code.strip() == st.secrets.get("BETA_ACCESS_CODE", ""):
            st.session_state.beta_ok = True
            st.rerun()
        else:
            st.error("Invalid access code")
    st.stop()
# --------------------------

try:
    # If we're on a new version (st.rerun exists) but experimental_rerun doesn't,
    # alias it so existing code works without edits.
    if hasattr(st, "rerun") and not hasattr(st, "experimental_rerun"):
        st.experimental_rerun = st.rerun
except Exception:
    pass
# ---------------------------------------------------------------------------

# ======== GLOBAL WIDGET HARDENING (drop-in replacement) ========
# Put this right after: import streamlit as st

# Keep originals
__orig_multiselect = _st_internal.multiselect
__orig_selectbox   = _st_internal.selectbox

def __to_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def __filter_defaults(default_vals: Any, options: Iterable) -> list:
    """
    Returns only those defaults that actually exist in options (by equality).
    Works for strings and other comparable objects.
    """
    opts_list = list(options or [])
    safe = []
    for dv in __to_list(default_vals):
        if any(ov == dv for ov in opts_list):
            safe.append(dv)
    return safe

def __safe_multiselect(label: str, options: Iterable, default: Any = None, **kwargs):
    """
    Drop-in replacement for st.multiselect:
    - Removes any default items that aren't in options so Streamlit won't crash.
    - If ALL defaults are invalid, it shows with an empty default.
    - Never throws because of mismatched defaults.
    """
    try:
        clean_default = __filter_defaults(default, options)
        return __orig_multiselect(label, options=options, default=clean_default, **kwargs)
    except Exception:
        return __orig_multiselect(label, options=list(options or []), default=[], **kwargs)

def __safe_selectbox(label: str, options: Iterable, index: int = 0, **kwargs):
    """
    Robust selectbox:
    - If index is invalid (or options empty), clamps to a safe value.
    """
    try:
        opts = list(options or [])
        if 'index' in kwargs:
            index = kwargs.pop('index')
        if not opts:
            opts = [""]
            index = 0
        if not isinstance(index, int) or index < 0 or index >= len(opts):
            index = 0
        return __orig_selectbox(label, options=opts, index=index, **kwargs)
    except Exception:
        return __orig_selectbox(label, options=[""], index=0, **kwargs)

# Monkey-patch globally
st.multiselect = __safe_multiselect
st.selectbox   = __safe_selectbox
# ======== END GLOBAL WIDGET HARDENING ========

# =========================
# Feature gates (kept for clarity)
# =========================
FEATURE_RUNTIME_TRACKER_READY = True
FEATURE_MAINTENANCE_RECORDS_READY = False

# =========================
# Page config & aesthetics
# =========================
st.set_page_config(page_title="VIGIL", layout="wide")

# 2) Always-visible beta banner (HTML, not affected by st themes)
st.markdown("""
<div style="
  margin: 10px 0 16px 0;
  padding: 10px 14px;
  border-radius: 10px;
  background: rgba(14,165,233,0.12);
  border: 1px solid rgba(14,165,233,0.35);
  color: #0b1220;
  font-weight: 600;
">
  VIGIL • Public Beta v0.1.0 — If anything looks off, please let us know.
</div>
""", unsafe_allow_html=True)

# (Optional) Keep a Streamlit-native notice too (safe to remove if you prefer)
# st.info("VIGIL • Public Beta v0.1.0 — If anything looks off, please let us know.")

# 3) Component styling (tabs)
st.markdown("""
<style>
div[data-testid="stTabs"] div[data-testid="stTabs"] button[role="tab"] {
  background: #1e293b !important;
  color: #e2e8f0 !important;
  border-radius: 999px !important;
  margin-right: 6px !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
}
div[data-testid="stTabs"] div[data-testid="stTabs"] button[aria-selected="true"] {
  background: #0ea5e9 !important;
  color: #0b1220 !important;
  border-color: rgba(14,165,233,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# 4) Global CSS injector
def _inject_css():
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
      border-radius: 12px !important; overflow: hidden !important;
      box-shadow: 0 8px 20px rgba(0,0,0,0.15); background: rgba(255,255,255,0.85);
    }
    .stTabs [data-baseweb="tab-list"]{ gap: 6px; }
    .stTabs [data-baseweb="tab"]{
      background: rgba(255,255,255,0.7); border: 1px solid rgba(0,0,0,0.08);
      border-radius: 10px; padding: 8px 12px;
    }
    .soft-card{
      padding:14px;border:1px solid rgba(0,0,0,0.08);border-radius:12px;
      background:rgba(255,255,255,0.8);box-shadow:0 8px 20px rgba(0,0,0,0.08);
    }
    .small-note { color:#334155; font-size:0.85rem; }
    .demo-ribbon {
      position: fixed; right: -50px; top: 14px; z-index: 9999;
      transform: rotate(45deg);
      background: #0ea5e9; color: #0b1220; font-weight: 700;
      padding: 6px 80px; box-shadow: 0 8px 18px rgba(0,0,0,.35);
    }
    </style>
    """, unsafe_allow_html=True)

# 5) App header (only show INSIDE the app, not on landing)
def _header():
    st.markdown("""
    <div style="
      padding: 18px 22px; border-radius: 16px;
      background: linear-gradient(135deg, rgba(10,111,255,0.20), rgba(10,111,255,0.06));
      border: 1px solid rgba(10,111,255,0.25);
      box-shadow: 0 12px 28px rgba(0,0,0,0.18);
      display:flex; align-items:center; gap:14px; margin-bottom:8px;">
      <div style="font-size:26px;">⚙️ <b>VIGIL®</b></div>
      <div style="opacity:.8;">Track • Plan • Analyse • Optimise</div>
    </div>
    """, unsafe_allow_html=True)

# 6) Apply CSS now (defer header)
_inject_css()
# Call _header() later when user is in the app, e.g.:
# if st.session_state.get("logged_in") or st.session_state.get("mode") in ("Demo","App"):
#     _header()
# ... then build your main tabs/layout

# =========================
# Paths & config
# =========================
DATA_DIR = "data"; os.makedirs(DATA_DIR, exist_ok=True)
ASSETS_JSON  = os.path.join(DATA_DIR, "assets.json")
RUNTIME_CSV  = os.path.join(DATA_DIR, "runtime.csv")
HISTORY_CSV  = os.path.join(DATA_DIR, "history.csv")
ATTACH_DIR   = os.path.join(DATA_DIR, "attachments"); os.makedirs(ATTACH_DIR, exist_ok=True)
DB_PATH      = os.path.join(DATA_DIR, "reliability.db")
USERS_JSON   = os.path.join(DATA_DIR, "users.json")   # NEW

DEFAULT_CONFIG = {
    "address_book":[{"name":"Planner","email":"planner@maraksreliability.com"}],
    "craft_presets":["Mechanical Technician","Electrician","Instrument Technician","Rigger","Boilermaker","Planner / Scheduler","Reliability Engineer","Other"],
    "dept_presets":["Mechanical","Electrical","Instrumentation","Rigging","Boiler / Fabrication","Process / Operations","Reliability","Planning / Scheduling","Engineering / Projects","Other"],
    "permit_presets":["Lifting Operation","Hot Work","Confined Space Entry","Electrical Isolation / LOTO","Working at Heights","Line Breaking","Pressure Testing"],
    "capacity_by_craft":{"Mechanical Technician":64,"Electrician":32,"Instrument Technician":24,"Rigger":24,"Boilermaker":24,"Planner / Scheduler":8,"Reliability Engineer":6,"Other":8},
    "sendgrid_from":"planner@maraksreliability.com",
    "default_asset_filter":[],
    "next_wo_seq":1
}
CONFIG_JSON = os.path.join(DATA_DIR, "config.json")

# =========================
# Auth store (users.json)
# =========================
def _load_users()->list:
    try:
        if os.path.exists(USERS_JSON):
            with open(USERS_JSON,"r",encoding="utf-8") as f: d=json.load(f)
            return d if isinstance(d, list) else []
        return []
    except Exception:
        return []

def _save_users(users:list):
    try:
        os.makedirs(os.path.dirname(USERS_JSON), exist_ok=True)
        with open(USERS_JSON,"w",encoding="utf-8") as f: json.dump(users,f,indent=2,ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save users: {e}")

def _hash_pbkdf2_sha256(pw:str, salt_bytes:bytes)->str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt_bytes, 120_000).hex()

def _pwd_hash(pw:str, salt:Optional[str]=None)->tuple[str,str]:
    salt = salt or secrets.token_hex(16)  # hex string
    try:
        salt_bytes = bytes.fromhex(salt)
    except ValueError:
        # tolerate non-hex salt strings gracefully
        salt_bytes = str(salt).encode("utf-8")
    return salt, _hash_pbkdf2_sha256(pw, salt_bytes)

def _find_user(email:str)->Optional[dict]:
    email = (email or "").strip().lower()
    for u in _load_users():
        if (u.get("email","").strip().lower()==email):
            return u
    return None

def _create_user(name:str, email:str, pw:str)->tuple[bool,str]:
    # Trim before validation to avoid false "empty" due to spaces
    name=name.strip(); email=email.strip().lower(); pw = pw or ""
    if not name or not email or not pw:
        return False, "Please fill all fields."
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email format."
    users=_load_users()
    if any((u.get("email","").lower()==email) for u in users):
        return False,"Account already exists."
    salt, h = _pwd_hash(pw)
    users.append({
        "name":name,"email":email,"salt":salt,"hash":h,
        "algo":"pbkdf2_sha256_v1",
        "created":datetime.now().isoformat(timespec="seconds")
    })
    _save_users(users)
    return True,"Account created."

def _verify_login(email:str, pw:str)->tuple[bool,Optional[dict],str]:
    u=_find_user(email)
    if not u:
        return False, None, "Account not found."
    try:
        salt=u.get("salt",""); hh=u.get("hash",""); algo=u.get("algo","pbkdf2_sha256_v1")
        # Compute candidate hash (robust to salt format)
        try:
            salt_bytes = bytes.fromhex(salt)
        except ValueError:
            salt_bytes = str(salt).encode("utf-8")
        if algo == "pbkdf2_sha256_v1" or not algo:
            h = _hash_pbkdf2_sha256(pw, salt_bytes)
        elif algo == "sha256_legacy":
            h = hashlib.sha256((str(salt) + pw).encode("utf-8")).hexdigest()
        else:
            h = _hash_pbkdf2_sha256(pw, salt_bytes)
        if h==hh:
            return True, {"name":u.get("name",""),"email":u.get("email","")}, "Welcome."
        return False, None, "Invalid credentials."
    except Exception:
        return False, None, "Login failed."

# =========================
# DB (SQLite)
# =========================
class _DB:
    def __init__(self, path:str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._init_schema()
    def _init_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS assets(
            tag TEXT PRIMARY KEY,
            details_json TEXT
        );
        CREATE TABLE IF NOT EXISTS runtime_assets (
            asset_tag TEXT PRIMARY KEY,
            functional_location TEXT,
            asset_model TEXT,
            criticality TEXT,
            mtbf_h REAL,
            last_overhaul TEXT,
            run_hours REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS runtime_components (
            asset_tag TEXT,
            component_key TEXT,
            cmms TEXT,
            oem_part TEXT,
            description TEXT,
            qty INTEGER,
            criticality TEXT,
            mtbf_h REAL,
            last_major_date TEXT,
            run_hours_since REAL,
            updated_at TEXT,
            PRIMARY KEY (asset_tag, component_key)
        );
        """); self.conn.commit()
    def assets_list(self, search: Optional[str]) -> pd.DataFrame:
        q="SELECT tag, details_json FROM assets WHERE 1=1"; p=[]
        if search:
            q+=" AND tag LIKE ?"; p.append(f"%{search}%")
        try: return pd.read_sql_query(q, self.conn, params=p)
        except Exception: return pd.DataFrame(columns=["tag","details_json"])

db = _DB(DB_PATH)

@st.cache_data(ttl=15, show_spinner=False)
def fetch_assets_from_db(search: Optional[str], rev:int) -> pd.DataFrame:
    return db.assets_list(search)

# =========================
# Lightweight stores
# =========================
def _load_assets()->dict:
    if os.path.exists(ASSETS_JSON):
        try:
            with open(ASSETS_JSON,"r",encoding="utf-8") as f: d=json.load(f)
            return d if isinstance(d,dict) else {}
        except Exception as e:
            st.error(f"Failed to load assets JSON: {e}")
    return {}

def _save_assets(d:dict):
    if st.session_state.get("is_demo"):
        st.info("💡 Demo mode: saving to disk is disabled. Create an account to keep your data.")
        return
    try:
        with open(ASSETS_JSON,"w",encoding="utf-8") as f: json.dump(d,f,indent=2,ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save assets JSON: {e}")

def _load_runtime()->pd.DataFrame:
    if os.path.exists(RUNTIME_CSV):
        try: return pd.read_csv(RUNTIME_CSV)
        except Exception as e: st.error(f"Failed to load runtime CSV: {e}")
    return pd.DataFrame(columns=[
        "Asset Tag","Functional Location","Asset Model","Criticality",
        "MTBF (Hours)","Last Overhaul","Running Hours Since Last Major Maintenance",
        "Remaining Hours","STATUS"
    ])

def _save_runtime(df:pd.DataFrame):
    if st.session_state.get("is_demo"):
        st.info("💡 Demo mode: saving to disk is disabled. Create an account to keep your data.")
        return
    try: df.to_csv(RUNTIME_CSV, index=False)
    except Exception as e: st.error(f"Failed to save runtime CSV: {e}")

def _load_history()->pd.DataFrame:
    if os.path.exists(HISTORY_CSV):
        try: return pd.read_csv(HISTORY_CSV)
        except Exception as e: st.error(f"Failed to load history CSV: {e}")
    return pd.DataFrame(columns=[
        "Number","Asset Tag","Functional Location","WO Number","Date of Maintenance",
        "Maintenance Code","Spares Used","QTY","Hours Run Since Last Repair",
        "Labour Hours","Asset Downtime Hours","Notes and Findings","Attachments"
    ])

def _load_config()->dict:
    try:
        if os.path.exists(CONFIG_JSON):
            with open(CONFIG_JSON,"r",encoding="utf-8") as f: d=json.load(f)
        else: d={}
        cfg=DEFAULT_CONFIG.copy(); cfg.update(d if isinstance(d,dict) else {})
        return cfg
    except Exception as e:
        st.warning(f"Config load issue: {e}"); return DEFAULT_CONFIG.copy()

def _save_config(cfg:dict):
    if st.session_state.get("is_demo"):
        st.info("💡 Demo mode: saving to disk is disabled. Create an account to keep your settings.")
        return
    try:
        with open(CONFIG_JSON,"w",encoding="utf-8") as f: json.dump(cfg,f,indent=2,ensure_ascii=False)
    except Exception as e:
        st.error(f"Failed to save config: {e}")

# =========================
# Session
# =========================
if "data_rev"   not in st.session_state: st.session_state.data_rev = 0
if "assets"     not in st.session_state: st.session_state.assets = _load_assets()
if "runtime_df" not in st.session_state: st.session_state.runtime_df = _load_runtime()
if "history_df" not in st.session_state: st.session_state.history_df = _load_history()
if "config"     not in st.session_state: st.session_state.config = _load_config()
if "search_query" not in st.session_state: st.session_state.search_query = {}
if "component_life" not in st.session_state: st.session_state.component_life = {}

def _bump_rev(): st.session_state.data_rev += 1

# =========================
# Helpers
# =========================
def to_num(x, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x,str) and not x.strip()): return float(default)
        if isinstance(x,(int,float,np.floating)): return float(x)
        s=str(x).strip().lower().replace(",","")
        if s in {"","-","—","none","nan","n/a","na"}: return float(default)
        return float(s)
    except Exception: return float(default)

def compute_remaining_hours(df: pd.DataFrame) -> pd.DataFrame:
    df = (df.copy() if isinstance(df,pd.DataFrame) else pd.DataFrame())
    if df.empty: return df
    run_col = "Running Hours Since Last Major Maintenance"
    if run_col not in df.columns: df[run_col] = 0.0
    if "MTBF (Hours)" not in df.columns: df["MTBF (Hours)"] = 0.0
    mtbf = pd.to_numeric(df["MTBF (Hours)"], errors="coerce").fillna(0.0)
    runh = pd.to_numeric(df[run_col], errors="coerce").fillna(0.0)
    df["Remaining Hours"] = (mtbf - runh).clip(lower=0.0)
    def _status(m,r):
        if m<=0: return "🟢 Healthy"
        ratio = (r/m) if m>0 else 0.0
        return "🟢 Healthy" if ratio<0.80 else ("🟠 Plan for maintenance" if ratio<1.0 else "🔴 Overdue for maintenance")
    df["STATUS"] = [_status(m,r) for m,r in zip(mtbf,runh)]
    return df

def auto_table_height(
    n_rows: int,
    min_rows: int = 3,
    max_rows: int = 999,
    row_px: int = 38,
    header_px: int = 42,
    padding_px: int = 16
) -> int:
    rows = min(max(n_rows, min_rows), max_rows)
    return int(header_px + padding_px + rows * row_px)

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
        if c not in df.columns: df[c] = 0.0 if c in ("QUANTITY","PRICE","MTBF (H)") else ""
    df["QUANTITY"]=pd.to_numeric(df["QUANTITY"],errors="coerce").fillna(0).astype(int)
    df["PRICE"]=pd.to_numeric(df["PRICE"],errors="coerce").fillna(0.0)
    df["MTBF (H)"]=pd.to_numeric(df["MTBF (H)"],errors="coerce").fillna(0.0)
    df["CRITICALITY"]=df["CRITICALITY"].astype(str).replace({"":"Medium"})
    return df[BOM_COLS].copy()

def sync_components_runtime_from_bom(tag: str, bom_df: pd.DataFrame, strict: bool = False) -> None:
    bom_df = bom_df.copy() if isinstance(bom_df,pd.DataFrame) else pd.DataFrame()
    for c in ["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","CRITICALITY","MTBF (H)"]:
        if c not in bom_df.columns: bom_df[c] = "" if c in ("CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","CRITICALITY") else 0.0
    bom_df["QUANTITY"]=pd.to_numeric(bom_df["QUANTITY"],errors="coerce").fillna(0).astype(int)
    bom_df["MTBF (H)"]=pd.to_numeric(bom_df["MTBF (H)"],errors="coerce").fillna(0.0)
    bom_df["CRITICALITY"]=bom_df["CRITICALITY"].astype(str).replace({"":"Medium"})

    cl = st.session_state.get("component_life") or {}
    a = cl.get(tag) or {"components": {}}
    comps = a.get("components") or {}
    seen=set()
    for _,r in bom_df.iterrows():
        cmms=str(r["CMMS MATERIAL CODE"]).strip()
        oem=str(r["OEM PART NUMBER"]).strip()
        desc=str(r["DESCRIPTION"]).strip()
        key=cmms if cmms else f"{oem}__{desc}".strip("_")
        if not key: continue
        comps[key]={"name":desc or cmms or oem,"cmms":cmms,"criticality":str(r["CRITICALITY"] or "Medium"),"mtbf_h":float(r["MTBF (H)"]) }
        seen.add(key)
    if strict:
        for k in list(comps.keys()):
            if k not in seen: comps.pop(k,None)
    a["components"]=comps; cl[tag]=a
    st.session_state.component_life = cl

def sync_runtime_from_assets_non_destructive(tags=None, overwrite_mtbf=True):
    assets=st.session_state.get("assets",{})
    rt=st.session_state.get("runtime_df",pd.DataFrame()).copy()
    need=["Asset Tag","Functional Location","Asset Model","Criticality","MTBF (Hours)","Last Overhaul","Running Hours Since Last Major Maintenance","Remaining Hours","STATUS"]
    for c in need:
        if c not in rt.columns: rt[c] = np.nan if c in ("MTBF (Hours)","Running Hours Since Last Major Maintenance","Remaining Hours") else ""
    tags = tags or list(assets.keys())
    for t in tags:
        meta=assets.get(t,{})
        mask = (rt["Asset Tag"].astype(str)==str(t))
        if not mask.any():
            rt = pd.concat([rt, pd.DataFrame([{
                "Asset Tag": t,
                "Functional Location": meta.get("Functional Location",""),
                "Asset Model": meta.get("Model",""),
                "Criticality": meta.get("Criticality","Medium"),
                "MTBF (Hours)": float(to_num(meta.get("MTBF (Hours)",0.0),0.0)),
                "Last Overhaul":"", "Running Hours Since Last Major Maintenance":0.0
            }])], ignore_index=True)
        else:
            if meta.get("Functional Location"): rt.loc[mask,"Functional Location"]=meta["Functional Location"]
            if meta.get("Model"):               rt.loc[mask,"Asset Model"]=meta["Model"]
            if meta.get("Criticality"):         rt.loc[mask,"Criticality"]=meta["Criticality"]
            if overwrite_mtbf:
                rt.loc[mask,"MTBF (Hours)"]=float(to_num(meta.get("MTBF (Hours)",0.0),0.0))
    rt = compute_remaining_hours(rt); st.session_state.runtime_df = rt
    try: _save_runtime(rt)
    except Exception: pass

# =========================
# DEMO capabilities (legacy)
# =========================
DEMO_CAPABILITIES = {
    "can_write": False,
    "allow_import": False,
    "asset_max_count": 8,
    "bom_items_max": 10,
    "mr_max_per_asset": 10,
    "tasks_add_limit": 2,
    "planning_pack_enabled": False,
    "jobs_manager_enabled": False,
    "components_tracker_enabled": False,
    "export_enabled": False,
}

def _exit_demo():
    for k in ["is_demo","capabilities"]:
        st.session_state.pop(k, None)
    st.toast("Exited Demo. Back to landing.", icon="👋")
    st.experimental_rerun()

# =========================
# Sidebar
# =========================
def _sidebar():
    with st.sidebar:
        st.header("⚙️ Controls")
        if st.session_state.get("is_demo"):
            st.markdown('<div class="demo-ribbon">DEMO</div>', unsafe_allow_html=True)

        if st.session_state.get("is_demo"):
            st.button("💾 Save All Now", disabled=True, help="Saving is disabled in demo. Create an account to keep your data.")
        else:
            if st.button("💾 Save All Now", key="save_all"):
                _save_assets(st.session_state.assets)
                st.session_state.runtime_df = compute_remaining_hours(st.session_state.runtime_df)
                _save_runtime(st.session_state.runtime_df)
                _save_config(st.session_state.config)
                st.success("All data saved.")

        st.caption(f"Working dir: {os.getcwd()}")
        st.caption(f"Last opened: {datetime.now().strftime('%d %b %Y, %H:%M')}")

        st.markdown("### ⬆️ Import")
        if st.session_state.get("is_demo"):
            st.file_uploader("Import Runtime CSV", type=["csv"], disabled=True, help="Disabled in demo")
            st.file_uploader("Import Assets JSON", type=["json"], disabled=True, help="Disabled in demo")
        else:
            up_rt = st.file_uploader("Import Runtime CSV", type=["csv"], key="upload_runtime")
            if up_rt is not None:
                try:
                    new_rt = pd.read_csv(up_rt)
                    st.session_state.runtime_df = compute_remaining_hours(
                        pd.concat([st.session_state.runtime_df, new_rt], ignore_index=True)
                    )
                    st.success("Runtime rows imported (appended).")
                except Exception as e:
                    st.error(f"Could not import runtime: {e}")

            up_assets = st.file_uploader("Import Assets JSON", type=["json"], key="upload_assets")
            if up_assets is not None:
                try:
                    incoming = json.loads(up_assets.read().decode("utf-8"))
                    if isinstance(incoming, dict):
                        st.session_state.assets.update(incoming)
                        _save_assets(st.session_state.assets)
                        st.success("Assets merged.")
                    else:
                        st.error("JSON should be an object mapping tags to fields.")
                except Exception as e:
                    st.error(f"Could not import assets: {e}")

        st.markdown("---")
        user = st.session_state.get("auth_user")
        if user:
            st.markdown(f"**Signed in:** {user.get('name','')}  \n<small>{user.get('email','')}</small>", unsafe_allow_html=True)
            if st.button("Sign out"):
                st.session_state.pop("auth_user", None)
                st.experimental_rerun()
        if st.session_state.get("is_demo"):
            if st.button("Exit Demo"):
                _exit_demo()

# =========================
# Landing + Auth views - ENHANCED
# =========================
def _landing_view():
    # Enhanced CSS for landing page
    st.markdown("""
    <style>
    .landing-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .hero-section {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border-radius: 20px;
        padding: 40px 30px;
        margin-bottom: 30px;
        color: white;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #0ea5e9, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-bottom: 25px;
    }
    .metrics-container {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin: 30px 0;
    }
    .metric-box {
        background: rgba(255,255,255,0.1);
        padding: 15px 25px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0ea5e9;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        margin: 40px 0;
    }
    .feature-card {
        background: rgba(255,255,255,0.95);
        padding: 25px 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
    }
    .feature-title {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 10px;
    }
    .feature-desc {
        color: #475569;
        font-size: 0.9rem;
    }
    .auth-section {
        background: rgba(255,255,255,0.9);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
    }
    .demo-section {
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        color: white;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .form-container {
        background: white;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin: 15px 0;
    }
    .section-title {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 20px;
        font-size: 1.4rem;
    }
    .trust-badge {
        text-align: center;
        margin: 20px 0;
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section - SIMPLIFIED to avoid HTML conflicts
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">VIGIL®</div>
        <div class="hero-subtitle">Reliability Command Center — Transform Maintenance from Reactive to Predictive</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics using Streamlit columns instead of HTML
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">47%</div>
            <div class="metric-label">Fewer Downtimes</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">28%</div>
            <div class="metric-label">Cost Reduction</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-value">89%</div>
            <div class="metric-label">Asset Reliability</div>
        </div>
        """, unsafe_allow_html=True)

    # Features Grid
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Asset Intelligence</div>
            <div class="feature-desc">Real-time MTBF tracking, predictive analytics, and AI-powered failure forecasting</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🛠️</div>
            <div class="feature-title">Smart Planning</div>
            <div class="feature-desc">Optimized maintenance scheduling, resource allocation, and capacity planning</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Cost Optimization</div>
            <div class="feature-desc">ROI tracking, budget forecasting, and spend analysis with actionable insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Content Columns
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        # Authentication Section - Using Streamlit containers instead of raw HTML
        with st.container():
            st.markdown("""
            <div class="auth-section">
                <div style="display: flex; gap: 20px;">
            """, unsafe_allow_html=True)
            
            # Two sub-columns for login and signup
            auth_col1, auth_col2 = st.columns(2, gap="medium")
            
            with auth_col1:
                st.markdown('<div class="section-title">🔐 Sign In</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="form-container">', unsafe_allow_html=True)
                    email = st.text_input("Email", key="login_email", placeholder="your.email@company.com")
                    pw = st.text_input("Password", type="password", key="login_pw", placeholder="Enter your password")
                    if st.button("Sign In →", key="btn_login", use_container_width=True):
                        ok, user, msg = _verify_login((email or "").strip(), pw or "")
                        if ok:
                            st.session_state.auth_user = user
                            st.session_state.pop("is_demo", None)
                            st.success("Welcome back! Redirecting...")
                            st.experimental_rerun()
                        else:
                            st.error(msg)
                    st.markdown('</div>', unsafe_allow_html=True)

            with auth_col2:
                st.markdown('<div class="section-title">🚀 Create Account</div>', unsafe_allow_html=True)
                with st.container():
                    st.markdown('<div class="form-container">', unsafe_allow_html=True)
                    name2 = st.text_input("Full name", key="reg_name", placeholder="John Smith")
                    email2 = st.text_input("Email address", key="reg_email", placeholder="john.smith@company.com")
                    pw2 = st.text_input("Password", type="password", key="reg_pw", placeholder="Create a password")
                    pw3 = st.text_input("Confirm password", type="password", key="reg_cpw", placeholder="Confirm your password")
                    if st.button("Create Account →", key="btn_create", use_container_width=True):
                        n = (name2 or "").strip()
                        e = (email2 or "").strip()
                        p2 = pw2 or ""
                        p3 = pw3 or ""
                        if not n or not e or not p2 or not p3:
                            st.error("Please fill all fields and confirm password.")
                        elif p2 != p3:
                            st.error("Passwords do not match.")
                        else:
                            ok, msg = _create_user(n, e, p2)
                            if ok:
                                st.success("Account created! You can sign in now.")
                            else:
                                st.error(msg)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Demo Section
        st.markdown("""
        <div class="demo-section">
            <div style="font-size: 2.5rem; margin-bottom: 15px;">🚀</div>
            <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 10px;">Try Interactive Demo</div>
            <div style="opacity: 0.9; margin-bottom: 20px; font-size: 0.95rem;">
                Experience full platform capabilities with sample data
            </div>
            <ul style="text-align: left; opacity: 0.9; font-size: 0.9rem; margin: 20px 0;">
                <li>Sample assets & maintenance data</li>
                <li>Full functionality access</li>
                <li>No commitment required</li>
                <li>Instant setup</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Launch Demo Experience", key="btn_demo", use_container_width=True, type="primary"):
            st.session_state.is_demo = True
            st.session_state.capabilities = DEMO_CAPABILITIES.copy()
            st.session_state.pop("auth_user", None)
            st.success("Launching demo experience...")
            st.experimental_rerun()

    # Trust Badge
    st.markdown("""
    <div class="trust-badge">
        <hr style="margin: 30px 0; opacity: 0.3;">
        Trusted by maintenance teams worldwide • Enterprise-grade security • 99.9% Uptime
    </div>
    """, unsafe_allow_html=True)

    # First-run Admin Creation (only if no users exist)
    if not _load_users():
        st.markdown("---")
        st.markdown("### 👑 First Run — Create Admin Account")
        st.info("Welcome to VIGIL! Since this is your first time, please create an administrator account.")
        
        admin_col1, admin_col2, admin_col3 = st.columns(3)
        with admin_col1:
            name = st.text_input("Admin name", key="adm_name", placeholder="Admin User")
        with admin_col2:
            email = st.text_input("Admin email", key="adm_email", placeholder="admin@company.com")
        with admin_col3:
            pw = st.text_input("Admin password", type="password", key="adm_pw", placeholder="Strong password")
            pwc = st.text_input("Confirm password", type="password", key="adm_pwc", placeholder="Confirm password")
        
        if st.button("Create Admin Account", key="btn_admin", use_container_width=True):
            n = (name or "").strip()
            e = (email or "").strip()
            p = pw or ""
            pc = pwc or ""
            if not n or not e or not p or not pc:
                st.error("Please fill all fields and confirm password.")
            elif p != pc:
                st.error("Passwords do not match.")
            else:
                ok, msg = _create_user(n, e, p)
                if ok:
                    st.success("Admin account created! Please sign in above.")
                else:
                    st.error(msg)

def _gate_to_app():
    """True if user is authenticated or demo."""
    return bool(st.session_state.get("auth_user") or st.session_state.get("is_demo"))

# =========================
# ROUTER — choose Demo vs Main
# =========================

# If not authenticated and not demo → show landing/login and stop
if not _gate_to_app():
    _landing_view()
    st.stop()

# If demo → call the separate hands-on demo module and stop (prevents main tabs rendering)
demo_app = None
try:
    import demo_app_hands_on as demo_app
except Exception:
    demo_app = None

if st.session_state.get("is_demo"):
    if demo_app is None:
        st.error("Demo module not found. Place 'demo_app_hands_on.py' next to this file.")
        st.stop()
    demo_app.render()
    st.stop()

# =========================
# Sidebar (main app only from here)
# =========================
_sidebar()

# =========================
# Tabs (7) — your app continues below
# =========================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📌 Asset Hub", "🕒 Planning Hub", "🧾 Maintenance Records",
    "📉 Reliability Visualisation", "💡 Financial Analysis", "🧰 Engineering Tool", "📂 My Project Manager"
])

# ============================================================
# TAB 1 — Explore (first) and Asset Master (second)
# ============================================================
with tab1:
    sub_explore, sub_master = st.tabs(["🔎 Explore", "🛠 Asset Master"])

    # -------------------------
    # Explore
    # -------------------------
    with sub_explore:
        st.subheader("📌 Asset Explorer")

        _tab1_search_in = st.session_state.get("search_query", {}).get("tab1", "")
        with st.sidebar:
            st.session_state.search_query["tab1"] = st.text_input("Search Assets", value=_tab1_search_in, key="t1_ex_search")
        _tab1_search = st.session_state.get("search_query", {}).get("tab1", "")

        try:
            assets_df_db = fetch_assets_from_db(_tab1_search, st.session_state.data_rev)
        except Exception:
            assets_df_db = pd.DataFrame()

        if isinstance(assets_df_db, pd.DataFrame) and not assets_df_db.empty and "details_json" in assets_df_db.columns:
            db_map = {}
            for _, r in assets_df_db.iterrows():
                tag = str(r.get("tag","")).strip()
                if not tag: continue
                try: details = json.loads(r["details_json"]) if r["details_json"] else {}
                except Exception: details = {}
                db_map[tag] = details
            if db_map:
                st.session_state.assets = db_map

        assets_dict = st.session_state.get("assets", {}) or {}
        all_tags = sorted(map(str, assets_dict.keys()))
        if _tab1_search:
            try:
                rx = re.compile(re.escape(_tab1_search), re.IGNORECASE)
                all_tags = [t for t in all_tags if rx.search(t)]
            except Exception:
                pass

        st.caption(f"Assets loaded: {len(all_tags)}")

        if not all_tags:
            st.info("No assets found. Use Asset Master to add or Import Assets JSON.")
        else:
            left, right = st.columns([1, 2], gap="large")

            with left:
                st.markdown("#### Select Asset")
                prev = st.session_state.get("t1_ex_selected_tag")
                idx = all_tags.index(prev) if prev in all_tags else 0
                selected_tag = st.selectbox("Asset Tag", options=all_tags, index=idx, key="t1_ex_asset_select")
                st.session_state["t1_ex_selected_tag"] = selected_tag

                st.markdown("#### Assets by Section")
                section_counts = {}
                for t in all_tags:
                    m = re.match(r"^(\d{3})", str(t))
                    sec = m.group(1) if m else "Other"
                    section_counts[sec] = section_counts.get(sec, 0) + 1
                labels = list(section_counts.keys())
                values = [section_counts[k] for k in labels]
                total = max(sum(values), 1)

                fig = go.Figure(data=[go.Pie(
                    labels=[f"{l} ({c}, {c/total*100:.0f}%)" for l, c in zip(labels, values)],
                    values=values, hole=0.3, textinfo="label+percent",
                    hovertemplate="%{label}<br>Count: %{value}<br>%{percent}<extra></extra>"
                )])
                fig.update_layout(showlegend=True, height=300, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True, key="t1_ex_assets_by_section")

            with right:
                a = assets_dict.get(selected_tag, {})
                st.markdown(f"### {selected_tag} — {a.get('Functional Location','')}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Functional Location", a.get("Functional Location", "—"))
                c2.metric("Asset Type", a.get("Asset Type", "—"))
                c3.metric("Model", a.get("Model", "—"))

                st.markdown("### 🔧 Technical Details")
                df_tech = pd.DataFrame(a.get("Technical Details", []))
                if df_tech.empty:
                    st.caption("No technical details recorded yet.")
                else:
                    st.dataframe(df_tech, use_container_width=True)

                st.markdown("### ⏳ Available Life")
                if not FEATURE_RUNTIME_TRACKER_READY:
                    st.info("No MTBF/runtime data yet to draw the chart.")
                else:
                    df_rt_tmp = compute_remaining_hours(st.session_state.runtime_df.copy())
                    rt_row = df_rt_tmp[df_rt_tmp["Asset Tag"].astype(str) == str(selected_tag)]
                    if not rt_row.empty:
                        rem = float(np.nan_to_num(to_num(rt_row.iloc[0].get("Remaining Hours", 0.0)), nan=0.0))
                        mtb = float(np.nan_to_num(to_num(rt_row.iloc[0].get("MTBF (Hours)", 0.0)),  nan=0.0))
                        used = max(mtb - rem, 0.0); rem = max(rem, 0.0); total = used + rem
                        if total > 0:
                            fig2 = go.Figure(data=[go.Pie(labels=["Used","Remaining"], values=[used, rem], hole=0.5,
                                            textinfo="label+percent",
                                            hovertemplate="%{label}: %{value:.0f}h (%{percent})<extra></extra>")])
                            fig2.update_layout(showlegend=True, height=300, margin=dict(t=10, b=10, l=10, r=10))
                            st.plotly_chart(fig2, use_container_width=True, key=f"t1_ex_avlife_{selected_tag}")
                        else:
                            st.info("No MTBF/runtime data yet to draw the chart.")
                    else:
                        st.info("No MTBF/runtime data yet to draw the chart.")

                st.markdown("### 🧰 BOM (View Only)")
                df_bom = build_bom_table(a)
                if not df_bom.empty:
                    df_bom["Total Price"] = df_bom["QUANTITY"] * df_bom["PRICE"]
                    st.dataframe(df_bom, use_container_width=True)
                    st.download_button("⬇️ Download BOM (CSV)",
                                       data=df_bom.to_csv(index=False),
                                       file_name=f"{selected_tag}_bom_table.csv",
                                       key=f"t1_ex_dl_bom_{selected_tag}")
                else:
                    st.caption("No BOM recorded yet.")

                st.markdown("### 🛠️ Maintenance History (View Only)")
                if not FEATURE_MAINTENANCE_RECORDS_READY:
                    st.caption("No maintenance history for this asset yet.")
                else:
                    df_hist = st.session_state.get("history_df", pd.DataFrame()).copy()
                    df_hist_sel = df_hist[df_hist["Asset Tag"].astype(str) == str(selected_tag)].copy()
                    if not df_hist_sel.empty:
                        df_hist_sel["Number"] = df_hist_sel.index + 1
                        st.dataframe(df_hist_sel, use_container_width=True)
                    else:
                        st.caption("No maintenance history for this asset yet.")

    # -------------------------
    # Asset Master (Add/Edit)
    # -------------------------
    with sub_master:
        st.subheader("🛠 Asset Master — Central Data Hub")
        st.caption("Add or edit assets and their BOM. Explore remains read-only.")

        mode = st.selectbox("Action", ["Add Asset", "Edit Asset"], index=0, key="t1_am_mode")

        # ADD
        if mode == "Add Asset":
            st.markdown("### 📋 Asset Details (Add)")
            c1,c2,c3 = st.columns([1,1,1])
            with c1: new_tag = st.text_input("Asset Tag (e.g., 408-PP-001)", key="t1_am_add_tag")
            with c2: new_location = st.text_input("Functional Location", key="t1_am_add_loc")
            with c3: new_type = st.selectbox("Asset Type", ["Pump","Motor","Valve","Other"], key="t1_am_add_type")

            c4,c5,c6 = st.columns([1,1,1])
            with c4: new_model = st.text_input("Model", key="t1_am_add_model")
            with c5: new_crit = st.selectbox("Criticality", ["Low","Medium","High"], index=1, key="t1_am_add_crit")
            with c6: new_mtbf = st.number_input("MTBF (Hours)", min_value=0.0, value=0.0, step=1.0, key="t1_am_add_mtbf")

            st.markdown("### 🔧 Technical Details")
            tech_cols=["Parameter","Value","Unit","Notes"]
            tech_cfg={"Parameter":st.column_config.TextColumn(),
                      "Value":st.column_config.NumberColumn(min_value=0.0,step=0.1),
                      "Unit":st.column_config.TextColumn(),
                      "Notes":st.column_config.TextColumn()}
            edited_tech = st.data_editor(pd.DataFrame(columns=tech_cols), column_config=tech_cfg,
                                         num_rows="dynamic", use_container_width=True, key="t1_am_add_tech", height=150)

            st.markdown("### 🧰 BOM Editor")
            bom_cols=["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE","CRITICALITY","MTBF (H)"]
            bom_cfg={
                "CMMS MATERIAL CODE": st.column_config.TextColumn(help="Feeds Components: CMMS"),
                "OEM PART NUMBER":    st.column_config.TextColumn(help="Feeds Components: OEM"),
                "DESCRIPTION":        st.column_config.TextColumn(help="Feeds Components: Name/Desc"),
                "QUANTITY":           st.column_config.NumberColumn(min_value=1, step=1),
                "PRICE":              st.column_config.NumberColumn(min_value=0.0, step=1.0),
                "CRITICALITY":        st.column_config.SelectboxColumn(options=["Low","Medium","High"], default="Medium"),
                "MTBF (H)":           st.column_config.NumberColumn(min_value=0.0, step=1.0),
            }
            edited_bom = st.data_editor(pd.DataFrame(columns=bom_cols), column_config=bom_cfg,
                                        num_rows="dynamic", use_container_width=True, key="t1_am_add_bom", height=220)

            if not edited_bom.empty:
                tmp = edited_bom.copy()
                tmp["Total Price"] = (pd.to_numeric(tmp["QUANTITY"], errors="coerce").fillna(0).astype(int) *
                                      pd.to_numeric(tmp["PRICE"], errors="coerce").fillna(0.0))
                st.metric("Est. BOM Total Cost (View-Only)", f"${tmp['Total Price'].sum():,.2f}")
            else:
                st.caption("Add BOM rows to see total estimated cost.")

            if st.button("💾 Save Asset", use_container_width=True, key="t1_am_add_save"):
                if not str(new_tag or "").strip():
                    st.error("Asset Tag is required.")
                else:
                    tag = str(new_tag).strip().upper()
                    if tag in st.session_state.assets:
                        st.warning(f"Tag '{tag}' exists—appending '-NEW'."); tag=f"{tag}-NEW"
                    st.session_state.assets[tag] = {
                        "Functional Location": new_location.strip(),
                        "Asset Type": new_type,
                        "Model": new_model.strip(),
                        "Criticality": new_crit,
                        "MTBF (Hours)": float(to_num(new_mtbf,0.0)),
                        "Technical Details": edited_tech.to_dict("records") if not edited_tech.empty else [],
                        "BOM Table": [
                            {
                                "CMMS MATERIAL CODE": str(r.get("CMMS MATERIAL CODE","")),
                                "OEM PART NUMBER":    str(r.get("OEM PART NUMBER","")),
                                "DESCRIPTION":        str(r.get("DESCRIPTION","")),
                                "QUANTITY":           int(to_num(r.get("QUANTITY",1),1)),
                                "PRICE":              float(to_num(r.get("PRICE",0.0),0.0)),
                                "CRITICALITY":        str(r.get("CRITICALITY","Medium")),
                                "MTBF (H)":           float(to_num(r.get("MTBF (H)",0.0),0.0)),
                            } for _, r in edited_bom.iterrows()
                        ] if not edited_bom.empty else [],
                        "BOM": [], "CMMS Codes": []
                    }
                    _save_assets(st.session_state.assets)
                    sync_runtime_from_assets_non_destructive([tag], overwrite_mtbf=True)
                    sync_components_runtime_from_bom(tag, pd.DataFrame(st.session_state.assets[tag]["BOM Table"]), strict=False)
                    _bump_rev()
                    st.success(f"Added {tag}."); st.rerun()

        # EDIT
        else:
            all_tags = sorted(st.session_state.assets.keys())
            if not all_tags:
                st.warning("No assets to edit. Add one first.")
            else:
                selected_tag = st.selectbox("Select Asset", all_tags, key="t1_am_edit_select")
                asset = json.loads(json.dumps(st.session_state.assets[selected_tag]))

                st.markdown("### 📋 Asset Details (Edit)")
                c1,c2,c3 = st.columns([1,1,1])
                with c1:
                    asset["Functional Location"] = st.text_input("Functional Location", value=asset.get("Functional Location",""), key=f"t1_am_edit_loc_{selected_tag}")
                with c2:
                    types=["Pump","Motor","Valve","Other"]
                    idx=types.index(asset.get("Asset Type","Pump")) if asset.get("Asset Type") in types else 0
                    asset["Asset Type"] = st.selectbox("Asset Type", types, index=idx, key=f"t1_am_edit_type_{selected_tag}")
                with c3:
                    asset["Model"] = st.text_input("Model", value=asset.get("Model",""), key=f"t1_am_edit_model_{selected_tag}")

                c4,c5,_ = st.columns([1,1,1])
                with c4:
                    crits=["Low","Medium","High"]
                    cidx=crits.index(asset.get("Criticality","Medium")) if asset.get("Criticality") in crits else 1
                    asset["Criticality"] = st.selectbox("Criticality", crits, index=cidx, key=f"t1_am_edit_crit_{selected_tag}")
                with c5:
                    asset["MTBF (Hours)"] = st.number_input("MTBF (Hours)", min_value=0.0,
                                                            value=float(to_num(asset.get("MTBF (Hours)",0.0),0.0)),
                                                            step=1.0, key=f"t1_am_edit_mtbf_{selected_tag}")

                st.markdown("### 🔧 Technical Details")
                tech_cols=["Parameter","Value","Unit","Notes"]
                tech_cfg={"Parameter":st.column_config.TextColumn(),
                          "Value":st.column_config.NumberColumn(min_value=0.0,step=0.1),
                          "Unit":st.column_config.TextColumn(),
                          "Notes":st.column_config.TextColumn()}
                tech_df = pd.DataFrame(asset.get("Technical Details", []))
                if tech_df.empty: tech_df = pd.DataFrame(columns=tech_cols)
                edited_tech = st.data_editor(tech_df, column_config=tech_cfg, num_rows="dynamic",
                                             use_container_width=True, key=f"t1_am_edit_tech_{selected_tag}", height=150)

                st.markdown("### 🧰 BOM Editor")
                bom_cols=["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE","CRITICALITY","MTBF (H)"]
                bom_cfg={
                    "CMMS MATERIAL CODE": st.column_config.TextColumn(help="Feeds Components: CMMS"),
                    "OEM PART NUMBER":    st.column_config.TextColumn(help="Feeds Components: OEM"),
                    "DESCRIPTION":        st.column_config.TextColumn(help="Feeds Components: Name/Desc"),
                    "QUANTITY":           st.column_config.NumberColumn(min_value=1, step=1),
                    "PRICE":              st.column_config.NumberColumn(min_value=0.0, step=1.0),
                    "CRITICALITY":        st.column_config.SelectboxColumn(options=["Low","Medium","High"], default="Medium"),
                    "MTBF (H)":           st.column_config.NumberColumn(min_value=0.0, step=1.0),
                }
                bom_df_prefill = pd.DataFrame(asset.get("BOM Table", []))
                if bom_df_prefill.empty: bom_df_prefill = pd.DataFrame(columns=bol_cols)  # noqa: F821 (fixed next line)
                # fix typo:
                if 'bol_cols' in locals(): del bol_cols
                if bom_df_prefill.empty: bom_df_prefill = pd.DataFrame(columns=bom_cols)

                edited_bom = st.data_editor(bom_df_prefill, column_config=bom_cfg, num_rows="dynamic",
                                            use_container_width=True, key=f"t1_am_edit_bom_{selected_tag}", height=220)

                if not edited_bom.empty:
                    tmp = edited_bom.copy()
                    tmp["Total Price"] = (pd.to_numeric(tmp["QUANTITY"], errors="coerce").fillna(0).astype(int) *
                                          pd.to_numeric(tmp["PRICE"], errors="coerce").fillna(0.0))
                    st.metric("Est. BOM Total Cost (View-Only)", f"${tmp['Total Price'].sum():,.2f}")
                else:
                    st.caption("Edit BOM rows to see total cost.")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("💾 Update Asset", use_container_width=True, key=f"t1_am_edit_update_{selected_tag}"):
                        asset["Technical Details"] = edited_tech.to_dict("records") if not edited_tech.empty else []
                        asset["BOM Table"] = [
                            {
                                "CMMS MATERIAL CODE": str(r.get("CMMS MATERIAL CODE","")),
                                "OEM PART NUMBER":    str(r.get("OEM PART NUMBER","")),
                                "DESCRIPTION":        str(r.get("DESCRIPTION","")),
                                "QUANTITY":           int(to_num(r.get("QUANTITY",1),1)),
                                "PRICE":              float(to_num(r.get("PRICE",0.0),0.0)),
                                "CRITICALITY":        str(r.get("CRITICALITY","Medium")),
                                "MTBF (H)":           float(to_num(r.get("MTBF (H)",0.0),0.0)),
                            } for _, r in edited_bom.iterrows()
                        ] if not edited_bom.empty else []
                        asset["CMMS Codes"] = [row["CMMS MATERIAL CODE"] for row in asset["BOM Table"] if row.get("CMMS MATERIAL CODE")]
                        asset["BOM"] = [f"{row['OEM PART NUMBER']} • {row['DESCRIPTION']} • {row['CMMS MATERIAL CODE']}".strip(" •") for row in asset["BOM Table"]]

                        st.session_state.assets[selected_tag] = asset
                        _save_assets(st.session_state.assets)
                        sync_runtime_from_assets_non_destructive([selected_tag], overwrite_mtbf=True)
                        sync_components_runtime_from_bom(selected_tag, pd.DataFrame(asset["BOM Table"]), strict=False)
                        _bump_rev()
                        st.success(f"Updated {selected_tag}."); st.rerun()
                with col_btn2:
                    if st.button("🗑️ DELETE ASSET", use_container_width=True, key=f"t1_am_edit_del_{selected_tag}"):
                        st.session_state[f"t1_am_del_confirm_{selected_tag}"] = True
                    if st.session_state.get(f"t1_am_del_confirm_{selected_tag}", False):
                        st.error("This will permanently delete the asset and associated runtime/components.")
                        cA, cB = st.columns([1,1])
                        with cA:
                            if st.button("✔️ CONFIRM DELETE", key=f"t1_am_del_yes_{selected_tag}"):
                                st.session_state.assets.pop(selected_tag, None); _save_assets(st.session_state.assets)
                                if isinstance(st.session_state.runtime_df, pd.DataFrame) and not st.session_state.runtime_df.empty:
                                    st.session_state.runtime_df = st.session_state.runtime_df[st.session_state.runtime_df["Asset Tag"].astype(str)!=selected_tag].reset_index(drop=True)
                                    _save_runtime(st.session_state.runtime_df)
                                if isinstance(st.session_state.component_life, dict):
                                    st.session_state.component_life.pop(selected_tag, None)
                                st.session_state[f"t1_am_del_confirm_{selected_tag}"] = False
                                _bump_rev(); st.success(f"Deleted {selected_tag}."); st.rerun()
                        with cB:
                            if st.button("Cancel", key=f"t1_am_del_no_{selected_tag}"):
                                st.session_state[f"t1_am_del_confirm_{selected_tag}"] = False
                                st.info("Deletion cancelled.")

        st.markdown("<p class='small-note'>Explore is intentionally limited until Planning Hub & Maintenance Records are built.</p>", unsafe_allow_html=True)

# =========================
# TAB 2 — Planning Hub
# =========================
with tab2:
    sub_rt, sub_pack, sub_jobs = st.tabs(["⏱ Runtime Tracker", "📦 Planning Pack", "🧰 Jobs Manager"])

    # ---------- Runtime Tracker (FULL) ----------
    with sub_rt:
        # --- Helpers ---
        def _status_from_ratio(r):
            if r < 0.80: return "🟢 Healthy"
            if r < 1.00: return "🟠 Plan for maintenance"
            return "🔴 Overdue for maintenance"

        def _compute_remaining_and_status(df, mtbf_col, run_col, rem_col, status_col):
            if df.empty:
                df[rem_col] = []
                df[status_col] = []
                return df
            m = pd.to_numeric(df[mtbf_col], errors="coerce").fillna(0.0)
            r = pd.to_numeric(df[run_col],  errors="coerce").fillna(0.0)
            rem = (m - r).clip(lower=0.0)
            df[rem_col] = rem
            used_ratio = np.divide(r, m, out=np.zeros_like(r, dtype=float), where=m>0)
            df[status_col] = [_status_from_ratio(x) for x in used_ratio]
            return df

        def _avg_daily_hours_for(tag: str, default_hours: float = 16.0) -> float:
            try:
                a = st.session_state.assets.get(tag, {})
                for row in a.get("Technical Details", []) or []:
                    if str(row.get("Parameter","")).strip().lower() == "avg daily (h)":
                        val = to_num(row.get("Value", default_hours), default_hours)
                        return float(val) if val > 0 else default_hours
            except Exception:
                pass
            return float(default_hours)

        def _seed_runtime_assets_from_assets_dict(overwrite_mtbf: bool = True):
            try:
                assets = st.session_state.get("assets", {}) or {}
                now = datetime.now().isoformat()
                cur = db.conn.cursor()
                for tag, meta in assets.items():
                    fl   = str(meta.get("Functional Location",""))
                    mdl  = str(meta.get("Model",""))
                    crit = str(meta.get("Criticality","Medium"))
                    mtbf = float(to_num(meta.get("MTBF (Hours)", 0.0), 0.0)) if overwrite_mtbf else None
                    row = cur.execute("SELECT asset_tag FROM runtime_assets WHERE asset_tag=?", (tag,)).fetchone()
                    if row is None:
                        cur.execute("""INSERT INTO runtime_assets(asset_tag,functional_location,asset_model,criticality,mtbf_h,
                                                                  last_overhaul,run_hours,updated_at)
                                       VALUES(?,?,?,?,?,?,?,?)""",
                                    (tag, fl, mdl, crit, (mtbf if mtbf is not None else 0.0), None, 0.0, now))
                    else:
                        if mtbf is not None:
                            cur.execute("""UPDATE runtime_assets
                                              SET functional_location=?, asset_model=?, criticality=?, mtbf_h=?, updated_at=?
                                            WHERE asset_tag=?""",
                                        (fl, mdl, crit, mtbf, now, tag))
                        else:
                            cur.execute("""UPDATE runtime_assets
                                              SET functional_location=?, asset_model=?, criticality=?, updated_at=?
                                            WHERE asset_tag=?""",
                                        (fl, mdl, crit, now, tag))
                db.conn.commit()
            except Exception as e:
                st.warning(f"Seeding runtime_assets failed: {e}")

        def _load_runtime_assets_df() -> pd.DataFrame:
            try:
                df = pd.read_sql_query("SELECT * FROM runtime_assets", db.conn)
            except Exception:
                df = pd.DataFrame(columns=["asset_tag","functional_location","asset_model","criticality",
                                           "mtbf_h","last_overhaul","run_hours","updated_at"])
            if "last_overhaul" in df.columns:
                df["last_overhaul"] = pd.to_datetime(df["last_overhaul"], errors="coerce")
            for c in ["mtbf_h","run_hours"]:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            df = _compute_remaining_and_status(df, "mtbf_h", "run_hours", "remaining_hours", "status")
            return df

        def _save_runtime_assets_changes(edited_df: pd.DataFrame, original_df: pd.DataFrame):
            try:
                orig = original_df.set_index("asset_tag")[["last_overhaul","run_hours"]]
                edit = edited_df.set_index("asset_tag")[["last_overhaul","run_hours"]]
                def _norm_date(s):
                    if pd.isna(s): return None
                    if isinstance(s, (pd.Timestamp, datetime)): return s.strftime("%Y-%m-%d")
                    try: return pd.to_datetime(s).strftime("%Y-%m-%d")
                    except Exception: return None
                changed = []
                for tag in edit.index:
                    eo = _norm_date(edit.loc[tag,"last_overhaul"])
                    oo = _norm_date(orig.loc[tag,"last_overhaul"]) if tag in orig.index else None
                    er = float(to_num(edit.loc[tag,"run_hours"], 0.0))
                    orr = float(to_num(orig.loc[tag,"run_hours"], 0.0)) if tag in orig.index else None
                    if eo != oo or abs(er - (orr if orr is not None else -9999.0)) > 1e-9:
                        changed.append((tag, eo, er))
                if not changed: return 0
                now = datetime.now().isoformat()
                cur = db.conn.cursor()
                for tag, d, h in changed:
                    cur.execute("""UPDATE runtime_assets
                                      SET last_overhaul=?, run_hours=?, updated_at=?
                                    WHERE asset_tag=?""", (d, h, now, tag))
                db.conn.commit()
                return len(changed)
            except Exception as e:
                st.error(f"Failed to save runtime assets: {e}")
                return 0

        def _refresh_runtime_df_session():
            df = _load_runtime_assets_df()
            out = pd.DataFrame({
                "Asset Tag": df.get("asset_tag", pd.Series(dtype=str)),
                "Functional Location": df.get("functional_location", pd.Series(dtype=str)),
                "Asset Model": df.get("asset_model", pd.Series(dtype=str)),
                "MTBF (Hours)": df.get("mtbf_h", pd.Series(dtype=float)),
                "Last Overhaul": df.get("last_overhaul", pd.Series(dtype="datetime64[ns]")),
                "Running Hours Since Last Major Maintenance": df.get("run_hours", pd.Series(dtype=float)),
            })
            out = compute_remaining_hours(out)
            st.session_state.runtime_df = out

        def _component_key(cmms: str, oem: str, desc: str) -> str:
            cmms = (cmms or "").strip()
            if cmms: return cmms
            return f"{(oem or '').strip()}__{(desc or '').strip()}".strip("_")

        def _seed_components_from_bom_non_destructive(asset_tag: str):
            a = st.session_state.get("assets", {}).get(asset_tag, {})
            bom = build_bom_table(a)
            if bom is None or bom.empty: return 0
            now = datetime.now().isoformat()
            cur = db.conn.cursor()
            n = 0
            for _, r in bom.iterrows():
                cmms = str(r.get("CMMS MATERIAL CODE","") or "")
                oem  = str(r.get("OEM PART NUMBER","") or "")
                desc = str(r.get("DESCRIPTION","") or "")
                qty  = int(to_num(r.get("QUANTITY", r.get("QTY", 0)), 0))
                crit = str(r.get("CRITICALITY","Medium") or "Medium")
                mtbf = float(to_num(r.get("MTBF (H)", 0.0), 0.0))
                key  = _component_key(cmms, oem, desc)
                if not key: continue
                row = cur.execute("""SELECT 1 FROM runtime_components WHERE asset_tag=? AND component_key=?""",
                                  (asset_tag, key)).fetchone()
                if row is None:
                    cur.execute("""INSERT INTO runtime_components(asset_tag,component_key,cmms,oem_part,description,qty,
                                                                  criticality,mtbf_h,last_major_date,run_hours_since,updated_at)
                                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                                (asset_tag, key, cmms, oem, desc, qty, crit, mtbf, None, 0.0, now))
                    n += 1
                else:
                    cur.execute("""UPDATE runtime_components
                                      SET cmms=?, oem_part=?, description=?, qty=?, criticality=?, mtbf_h=?, updated_at=?
                                    WHERE asset_tag=? AND component_key=?""",
                                (cmms, oem, desc, qty, crit, mtbf, now, asset_tag, key))
                    n += 1
            db.conn.commit()
            return n

        def _load_components_df(asset_tag: str) -> pd.DataFrame:
            try:
                df = pd.read_sql_query("""
                    SELECT asset_tag, component_key, cmms, oem_part, description, qty, criticality, mtbf_h,
                           last_major_date, run_hours_since
                      FROM runtime_components
                     WHERE asset_tag=?
                     ORDER BY description, cmms
                """, db.conn, params=(asset_tag,))
            except Exception:
                df = pd.DataFrame(columns=["asset_tag","component_key","cmms","oem_part","description","qty",
                                           "criticality","mtbf_h","last_major_date","run_hours_since"])
            df["last_major_date"] = pd.to_datetime(df["last_major_date"], errors="coerce")
            df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
            df["mtbf_h"] = pd.to_numeric(df["mtbf_h"], errors="coerce").fillna(0.0)
            df["run_hours_since"] = pd.to_numeric(df["run_hours_since"], errors="coerce").fillna(0.0)
            df = _compute_remaining_and_status(df, "mtbf_h", "run_hours_since",
                                               "remaining_hours_to_next", "status")
            return pd.DataFrame({
                "CMMS MATERIAL CODE": df["cmms"],
                "OEM PART NUMBER": df["oem_part"],
                "DESCRIPTION": df["description"],
                "QTY": df["qty"],
                "CRITICALITY": df["criticality"],
                "MTBF (H)": df["mtbf_h"],
                "LAST REPLACEMENT DATE": df["last_major_date"],
                "HOURS SINCE LAST REPLACEMENT": df["run_hours_since"],
                "REMAINING HOURS TO NEXT REPLACEMENT": df["remaining_hours_to_next"],
                "STATUS": df["status"],
                "__asset_tag": df["asset_tag"],
                "__component_key": df["component_key"],
            })

        def _save_components_changes(edited_df: pd.DataFrame) -> int:
            try:
                if edited_df.empty: return 0
                now = datetime.now().isoformat()
                cur = db.conn.cursor()
                n = 0
                for _, r in edited_df.iterrows():
                    tag = str(r.get("__asset_tag",""))
                    key = str(r.get("__component_key",""))
                    if not tag or not key: continue
                    d = r.get("LAST REPLACEMENT DATE", None)
                    if pd.isna(d): d_str = None
                    elif isinstance(d, (pd.Timestamp, datetime)): d_str = d.strftime("%Y-%m-%d")
                    else:
                        try: d_str = pd.to_datetime(d).strftime("%Y-%m-%d")
                        except Exception: d_str = None
                    hrs = float(to_num(r.get("HOURS SINCE LAST REPLACEMENT", 0.0), 0.0))
                    if hrs < 0: hrs = 0.0
                    cur.execute("""UPDATE runtime_components
                                      SET last_major_date=?, run_hours_since=?, updated_at=?
                                    WHERE asset_tag=? AND component_key=?""",
                                (d_str, hrs, now, tag, key))
                    n += 1
                db.conn.commit()
                return n
            except Exception as e:
                st.error(f"Failed to save components: {e}")
                return 0

        # --- Seed base runtime from assets ---
        _seed_runtime_assets_from_assets_dict(overwrite_mtbf=True)

        st.subheader("⏱ Runtime Tracker")
        st.markdown("### Universal Asset Runtime Tracker")

        # Universal table
        df_rt_orig = _load_runtime_assets_df()
        view = pd.DataFrame({
            "Asset Tag": df_rt_orig["asset_tag"],
            "Functional Location": df_rt_orig["functional_location"],
            "Asset Model": df_rt_orig["asset_model"],
            "Criticality": df_rt_orig["criticality"],
            "MTBF (Hours)": df_rt_orig["mtbf_h"],
            "Last Overhaul": df_rt_orig["last_overhaul"],
            "Running Hours Since Last Major Maintenance": df_rt_orig["run_hours"],
            "Remaining Hours": df_rt_orig["remaining_hours"],
            "STATUS": df_rt_orig["status"],
        })
        if "Last Overhaul" in view.columns:
            view["Last Overhaul"] = pd.to_datetime(view["Last Overhaul"], errors="coerce")
        for c in ["MTBF (Hours)","Running Hours Since Last Major Maintenance","Remaining Hours"]:
            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0.0)

        colcfg = {
            "Asset Tag": st.column_config.TextColumn(disabled=True),
            "Functional Location": st.column_config.TextColumn(disabled=True),
            "Asset Model": st.column_config.TextColumn(disabled=True),
            "Criticality": st.column_config.TextColumn(disabled=True),
            "MTBF (Hours)": st.column_config.NumberColumn(disabled=True),
            "Last Overhaul": st.column_config.DateColumn(help="Editable"),
            "Running Hours Since Last Major Maintenance": st.column_config.NumberColumn(min_value=0.0, step=1.0, help="Editable"),
            "Remaining Hours": st.column_config.NumberColumn(disabled=True),
            "STATUS": st.column_config.TextColumn(disabled=True),
        }
        h_rt = auto_table_height(view.shape[0], min_rows=3, max_rows=999)
        edited_view = st.data_editor(
            view,
            column_config=colcfg,
            num_rows="fixed",
            use_container_width=True,
            height=h_rt,
            key="t2_rt_universal_editor",
        )

        # Save button → refresh Tab 1 immediately (no extra buttons)
        if st.button("💾 Save Runtime Updates", use_container_width=True, key="t2_rt_universal_save"):
            back = pd.DataFrame({
                "asset_tag": edited_view["Asset Tag"],
                "functional_location": edited_view["Functional Location"],
                "asset_model": edited_view["Asset Model"],
                "criticality": edited_view["Criticality"],
                "mtbf_h": edited_view["MTBF (Hours)"],
                "last_overhaul": edited_view["Last Overhaul"],
                "run_hours": edited_view["Running Hours Since Last Major Maintenance"],
            })
            changed = _save_runtime_assets_changes(back, df_rt_orig)
            if changed >= 0:
                _refresh_runtime_df_session()
                st.session_state["data_rev"] = st.session_state.get("data_rev", 0) + 1
                st.session_state["__explorer_refresh_token"] = datetime.now().isoformat()
            st.success(f"Saved {changed} row(s).")
            st.rerun()

        # ===== BELOW SAVE: 3 visuals side-by-side =====
        kpi_df = _load_runtime_assets_df()
        total_assets = int(kpi_df.shape[0])
        green = int((kpi_df["status"] == "🟢 Healthy").sum())
        amber = int((kpi_df["status"] == "🟠 Plan for maintenance").sum())
        red   = int((kpi_df["status"] == "🔴 Overdue for maintenance").sum())

        # Due ≤ 30 days
        due = 0
        for _, r in kpi_df.iterrows():
            mtbf = float(to_num(r.get("mtbf_h", 0.0), 0.0))
            runh = float(to_num(r.get("run_hours", 0.0), 0.0))
            rem  = max(mtbf - runh, 0.0)
            avgd = _avg_daily_hours_for(str(r.get("asset_tag","")).strip(), 16.0)
            days = (rem / avgd) if avgd > 0 else 9999
            if days <= 30.0: due += 1
        not_due = max(total_assets - due, 0)

        st.markdown("### Fleet Status & Criticality — Universal View")
        v1, v2, v3 = st.columns(3)

        # (1) Fleet Status donut
        with v1:
            try:
                fig_status = go.Figure(data=[go.Pie(
                    labels=["🟢 Healthy","🟠 Plan","🔴 Overdue"],
                    values=[green, amber, red],
                    hole=0.5,
                    textinfo="label+percent",
                    hovertemplate="%{label}: %{value}<extra></extra>"
                )])
                fig_status.update_layout(
                    title="Fleet Status",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                    height=260, margin=dict(t=30,b=50,l=10,r=10)
                )
                st.plotly_chart(fig_status, use_container_width=True, key="t2_rt_fleet_status")
            except Exception:
                st.caption("Status chart unavailable.")

        # (2) Assets by Criticality bar
        with v2:
            try:
                crit_counts = kpi_df["criticality"].fillna("Medium").astype(str).value_counts().sort_index()
                fig_crit = go.Figure(data=[go.Bar(name="Assets", x=crit_counts.index.tolist(), y=crit_counts.values.tolist())])
                fig_crit.update_layout(
                    title="Assets by Criticality",
                    height=260, margin=dict(t=30,b=50,l=10,r=10),
                    xaxis_title="Criticality", yaxis_title="Count",
                    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_crit, use_container_width=True, key="t2_rt_assets_by_crit")
            except Exception:
                st.caption("Criticality chart unavailable.")

        # (3) Due ≤ 30 days donut (visual KPI)
        with v3:
            try:
                fig_due = go.Figure(data=[go.Pie(
                    labels=["Due ≤30d", "Not Due"],
                    values=[due, not_due],
                    hole=0.6,
                    textinfo="label+value",
                    hovertemplate="%{label}: %{value}<extra></extra>"
                )])
                fig_due.update_layout(
                    title=f"Due ≤30 days (Total: {total_assets})",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
                    height=260, margin=dict(t=30,b=50,l=10,r=10)
                )
                st.plotly_chart(fig_due, use_container_width=True, key="t2_rt_due_kpi")
            except Exception:
                st.caption("Due KPI chart unavailable.")

        st.divider()

        # ===== Asset Components Runtime Tracker =====
        enable_comp = st.checkbox("Enable Asset Components Runtime Tracker", key="t2_rt_comp_enable", value=False)
        if enable_comp:
            tags = sorted(st.session_state.get("assets", {}).keys())
            if not tags:
                st.info("No assets available. Add assets in Asset Hub first.")
            else:
                sel_tag = st.selectbox("Select Asset", tags, key="t2_rt_comp_asset")

                col_seed, _ = st.columns([1,1])
                with col_seed:
                    if st.button("🔄 Re-seed from BOM (non-destructive)", key="t2_rt_comp_seed"):
                        n = _seed_components_from_bom_non_destructive(sel_tag)
                        st.success(f"Seeded/updated {n} component row(s) from BOM.")
                _seed_components_from_bom_non_destructive(sel_tag)

                comp_df_ui = _load_components_df(sel_tag)
                cfg_comp = {
                    "CMMS MATERIAL CODE": st.column_config.TextColumn(disabled=True),
                    "OEM PART NUMBER":    st.column_config.TextColumn(disabled=True),
                    "DESCRIPTION":        st.column_config.TextColumn(disabled=True),
                    "QTY":                st.column_config.NumberColumn(disabled=True),
                    "CRITICALITY":        st.column_config.TextColumn(disabled=True),
                    "MTBF (H)":           st.column_config.NumberColumn(disabled=True),
                    "LAST REPLACEMENT DATE": st.column_config.DateColumn(help="Editable"),
                    "HOURS SINCE LAST REPLACEMENT": st.column_config.NumberColumn(min_value=0.0, step=1.0, help="Editable"),
                    "REMAINING HOURS TO NEXT REPLACEMENT": st.column_config.NumberColumn(disabled=True),
                    "STATUS":             st.column_config.TextColumn(disabled=True),
                }
                h_comp = auto_table_height(comp_df_ui.shape[0], min_rows=3, max_rows=999)
                edited_comp = st.data_editor(
                    comp_df_ui,
                    column_config=cfg_comp,
                    num_rows="fixed",
                    use_container_width=True,
                    height=h_comp,
                    key=f"t2_rt_comp_editor_{sel_tag}",
                )

                if st.button("💾 Update Component Runtime", use_container_width=True, key=f"t2_rt_comp_update_{sel_tag}"):
                    edited_comp["__asset_tag"] = sel_tag
                    if "__component_key" not in edited_comp.columns and "__component_key" in comp_df_ui.columns:
                        edited_comp["__component_key"] = comp_df_ui["__component_key"]
                    changed = _save_components_changes(edited_comp)
                    st.success(f"Updated {changed} component row(s).")

                # Component visual: MTBF vs Remaining (legend at bottom)
                if not edited_comp.empty:
                    try:
                        x_labels = edited_comp["DESCRIPTION"].fillna("").astype(str)
                        y_mtbf   = pd.to_numeric(edited_comp["MTBF (H)"], errors="coerce").fillna(0.0)
                        y_rem    = pd.to_numeric(edited_comp["REMAINING HOURS TO NEXT REPLACEMENT"], errors="coerce").fillna(0.0)
                        fig_cmp = go.Figure()
                        fig_cmp.add_bar(name="MTBF (H)", x=x_labels, y=y_mtbf)
                        fig_cmp.add_bar(name="Remaining (H)", x=x_labels, y=y_rem)
                        fig_cmp.update_layout(
                            title=f"MTBF vs Remaining — {sel_tag}",
                            barmode="group",
                            xaxis_title="Component",
                            yaxis_title="Hours",
                            height=320,
                            margin=dict(t=30, b=80, l=10, r=10),
                            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
                        )
                        st.plotly_chart(fig_cmp, use_container_width=True, key=f"t2_rt_comp_chart_{sel_tag}")
                    except Exception:
                        st.caption("Component chart unavailable.")

# === END OF RUNTIME TRACKER TAB ===

# ======================= Planning Pack (sub_pack) — FULL BLOCK =======================
# reliability_app.py — Tab 2 / Subtab: 📦 Planning Pack (sub_pack) — FULL DROP-IN (scoped to sub_pack)

# ---------- Minimal guard imports ----------
import os, json, uuid, shutil, subprocess, sqlite3, base64, re
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------- Paths (idempotent) ----------
if 'DATA_DIR' not in globals():
    DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

if 'JOBS_DIR' not in globals():
    JOBS_DIR = os.path.join(DATA_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

if 'ATTACH_DIR' not in globals():
    ATTACH_DIR = os.path.join(DATA_DIR, "attachments")
os.makedirs(ATTACH_DIR, exist_ok=True)

_PP_INDEX_PATH = os.path.join(JOBS_DIR, "jobs_index.json")
RELIAB_DB      = os.path.join(DATA_DIR, "reliability.db")
HIST_CSV       = os.path.join(DATA_DIR, "history.csv")
_WIP_DIR       = os.path.join(JOBS_DIR, "_wip")
os.makedirs(_WIP_DIR, exist_ok=True)

# ---------- Config defaults (non-destructive) ----------
cfg = st.session_state.get("config", {}) or {}
cfg.setdefault("address_book", [])
cfg.setdefault("dept_presets", [
    "Engineering / Projects","Electrical","Instrumentation","Mechanical",
    "Boiler / Fabrication","Rigging","Civil / Structural","Reliability",
    "Supply Chain / Stores","Process / Operations","Planning / Scheduling","Safety (HSE)"])
cfg.setdefault("craft_presets", [
    "Mechanical Technician","Electrical Technician","Rigger","Boilermaker","Instrument Tech"])
cfg.setdefault("permit_presets", ["Hot Work","Confined Space","Working at Heights","Lockout/Tagout"])
cfg.setdefault("capacity_by_craft", {"Mechanical Technician": 16, "Electrical Technician": 12})
cfg.setdefault("verified_domains", ["em7701.maraksreliability.com"])   # SendGrid verified domains
cfg.setdefault("sendgrid_from", "planner@em7701.maraksreliability.com")
cfg.setdefault("sendgrid_reply_to", "")
cfg.setdefault("require_verified_from", True)
st.session_state.config = cfg

# ---------- Small helpers ----------
def _b64(x: bytes) -> str:
    return base64.b64encode(x).decode("ascii")

def _pdate(s):
    try: return datetime.strptime(str(s), "%Y-%m-%d").date()
    except Exception: return None

def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")

def _auto_table_height(df: pd.DataFrame, min_h: int = 120, row_h: int = 32, max_h: int = 480) -> int:
    rows = int(df.shape[0] if isinstance(df, pd.DataFrame) else 0)
    h = min_h + rows * row_h
    return int(max(min(h, max_h), min_h))

# Guarded numeric
if 'to_num' not in globals():
    def to_num(x, default=0):
        try:
            if x is None: return default
            if isinstance(x, (int, float, np.floating)): return float(x)
            s = str(x).strip().replace(",", "")
            if s.lower() in {"", "none", "nan", "n/a", "-"}: return default
            return float(s) if any(ch in s for ch in ".eE") else float(int(s))
        except Exception:
            return default

# ---- BOM guards (use your real ones if present) ----
if 'BOM_COLS' not in globals():
    BOM_COLS = ["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE"]
if 'build_bom_table' not in globals():
    def build_bom_table(asset: dict) -> pd.DataFrame:
        return pd.DataFrame(columns=BOM_COLS)

# ---- Save config guard ----
if 'save_config' not in globals():
    def save_config(cfg_in: dict):
        if '_save_cfg' in globals():
            try: _save_cfg(cfg_in); return
            except Exception: pass
        st.session_state.config = cfg_in

# ---------- Index / jobs I/O ----------
def _normalize_index(raw):
    norm = {"jobs": {}, "next_job_seq": 1, "next_wo_int": 1}
    if not isinstance(raw, dict): return norm
    for k in ("next_job_seq","next_wo_int"):
        try: norm[k] = int(raw.get(k, 1) or 1)
        except Exception: pass
    jobs = raw.get("jobs", {})
    if isinstance(jobs, dict):
        for jid, meta in jobs.items():
            if not isinstance(meta, dict): continue
            m = dict(meta)
            m["id"] = m.get("id") or str(jid)
            m.setdefault("status", "Draft")
            m.setdefault("archived", False)
            m.setdefault("trashed", False)
            m.setdefault("activated", bool(meta.get("activated", False)))
            norm["jobs"][m["id"]] = m
    elif isinstance(jobs, list):
        for meta in jobs:
            if not isinstance(meta, dict): continue
            jid = meta.get("id") or meta.get("persist_id")
            if not jid: continue
            m = dict(meta); m["id"] = str(jid)
            m.setdefault("status", "Draft")
            m.setdefault("archived", False)
            m.setdefault("trashed", False)
            m.setdefault("activated", bool(meta.get("activated", False)))
            norm["jobs"][m["id"]] = m
    return norm

def _read_index():
    try:
        if os.path.exists(_PP_INDEX_PATH):
            with open(_PP_INDEX_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = {"jobs": {}, "next_job_seq": 1, "next_wo_int": 1}
    except Exception:
        raw = {"jobs": {}, "next_job_seq": 1, "next_wo_int": 1}
    return _normalize_index(raw)

def _write_index(idx):
    try:
        norm = _normalize_index(idx if isinstance(idx, dict) else {})
        with open(_PP_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(norm, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def _job_path(job_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-","_") else "_" for ch in str(job_id))
    return os.path.join(JOBS_DIR, f"{safe}.json")

def load_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    return None

def _index_upsert(job: dict):
    idx = _read_index()
    jid = job.get("persist_id") or job.get("id")
    if not jid: return
    meta = {
        "id": jid,
        "status": job.get("status","Draft"),
        "archived": bool(job.get("archived", False)),
        "trashed": bool(job.get("trashed", False)),
        "activated": bool(job.get("activated", False)),
        "asset_tag": job.get("asset_tag",""),
        "pm_code": job.get("pm_code",""),
        "priority": job.get("priority",""),
        "lead_role": job.get("lead_role",""),
        "required_end_date": job.get("required_end_date",""),
        "wo_number": job.get("wo_number",""),
        "last_updated": job.get("last_updated", datetime.now().isoformat())
    }
    idx["jobs"][jid] = meta
    _write_index(idx)

def new_job_id() -> str:
    idx = _read_index()
    seq = int(idx.get("next_job_seq", 1) or 1)
    today = datetime.now().strftime("%Y%m%d")
    jid = f"JOB-{today}-{seq:03d}"
    idx["next_job_seq"] = seq + 1
    _write_index(idx)
    return jid

def save_job(job: dict, persist_id: str | None = None):
    if not isinstance(job, dict): return False, None
    job = json.loads(json.dumps(job))  # deep copy
    if persist_id: job["id"] = persist_id
    jid = job.get("id") or job.get("persist_id")
    if not jid: return False, None
    job["last_updated"] = datetime.now().isoformat()
    p = _job_path(jid)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(job, f, indent=2, ensure_ascii=False)
        _index_upsert(job | {"persist_id": jid})
        return True, jid
    except Exception as e:
        st.error(f"Failed to save job: {e}")
        return False, None

def set_status(job_id: str, status: str):
    job = load_job(job_id)
    if not job: return False
    if str(job.get("status")) == "Completed": return False
    job["status"] = status
    if status == "Submitted" and not str(job.get("wo_number","")).strip():
        idx = _read_index()
        n = int(idx.get("next_wo_int", 1) or 1)
        job["wo_number"] = f"WO-{n:06d}"
        idx["next_wo_int"] = n + 1
        _write_index(idx)
    ok, _ = save_job(job, persist_id=job_id)
    return ok

def archive_job(job_id: str, archived: bool = True):
    job = load_job(job_id)
    if not job: return False
    job["archived"] = bool(archived)
    ok, _ = save_job(job, persist_id=job_id)
    return ok

def mark_completed(job_id: str, completed_by: str = ""):
    job = load_job(job_id)
    if not job: return False, "Job not found."
    job["status"] = "Completed"
    job["completed_at"] = datetime.now().isoformat()
    job["completed_by"] = completed_by
    ok, _ = save_job(job, persist_id=job_id)
    return (ok, "OK" if ok else "Failed to mark completed")

# ---------- WIP bootstrappers ----------
def _new_wip_from_scratch(prefill_asset: str | None = None):
    wid = f"WIP-{uuid.uuid4().hex[:8]}"
    wip = {
        "id": wid, "persist_id": "", "created_at": datetime.now().isoformat(),
        "status": "Draft", "archived": False, "trashed": False,
        "asset_tag": prefill_asset or "", "pm_code": "", "task_description": "",
        "creator": "", "required_end_date": date.today().strftime("%Y-%m-%d"),
        "priority": "", "requested_by": "", "wo_number": "",
        "lead_role": "", "co_leads": [], "lead_department": "",
        "spares": [], "delivery_note": "",
        "labour": [],
        "lifting": {
            "cranage_required": False, "crane_type": "", "mobile_capacity": "", "overhead_spec": "",
            "forklift_required": False, "forklift_capacity": "", "scaffolding_required": False
        },
        "logistics": {
            "pickup_point": "", "delivery_location": "",
            "transport_needed": False, "transport_type": "",
            "permits": [], "responsible_person": ""
        },
        "planned_duration_hours": 0.0,
        "planned_start": date.today().strftime("%Y-%m-%d"),
        "planned_end": (date.today()+timedelta(days=6)).strftime("%Y-%m-%d"),
        "capacity_fit": [],
        "kitting_readiness": {"pct_ok": 0.0, "warnings": []},
        "attachments": [],
        "currency_symbol": "R",
        "export": {"csv_path": "", "ics_path": "", "pdf_path": "", "html_path": ""}
    }
    try:
        with open(os.path.join(_WIP_DIR, f"{wid}.json"), "w", encoding="utf-8") as f:
            json.dump(wip, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return wip

def _wip_from_persisted(job_id: str):
    job = load_job(job_id) or {}
    wid = f"WIP-{uuid.uuid4().hex[:8]}"
    wip = json.loads(json.dumps(job))
    wip["id"] = wid
    wip["persist_id"] = job_id
    try:
        with open(os.path.join(_WIP_DIR, f"{wid}.json"), "w", encoding="utf-8") as f:
            json.dump(wip, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return wip

def _ensure_wip():
    if "pp_wip" not in st.session_state or not st.session_state.pp_wip:
        st.session_state.pp_wip = _new_wip_from_scratch()

def _persist_wip_to_job():
    wip = st.session_state.pp_wip

    # Merge live BOM picks if present
    bom_snap = st.session_state.get("pp_bom_snapshot")
    if isinstance(bom_snap, pd.DataFrame):
        need = {"Include","Pick QTY","Price"}
        if need.issubset(bom_snap.columns):
            inc = bom_snap[(bom_snap["Include"] == True) &
                           (pd.to_numeric(bom_snap["Pick QTY"], errors="coerce").fillna(0) > 0)].copy()
            lines = []
            for _, r in inc.iterrows():
                qtyi = int(to_num(r.get("Pick QTY", 0)))
                pricef = float(to_num(r.get("Price", 0.0)))
                lines.append({
                    "cmms": str(r.get("CMMS","")), "oem": str(r.get("OEM/Part","")), "desc": str(r.get("Description","")),
                    "qty": qtyi, "price": pricef, "line_total": float(qtyi * pricef)
                })
            wip["spares"] = lines

    # Assign persisted id if new
    if not wip.get("persist_id"):
        jid = new_job_id()
        wip["persist_id"] = jid
    else:
        jid = wip["persist_id"]

    # Move attachments folder WIP → job
    wip_attach_dir = os.path.join(ATTACH_DIR, "plans", wip["id"])
    job_attach_dir = os.path.join(ATTACH_DIR, "plans", jid)
    if os.path.isdir(wip_attach_dir):
        os.makedirs(os.path.dirname(job_attach_dir), exist_ok=True)
        if os.path.isdir(job_attach_dir):
            shutil.rmtree(job_attach_dir, ignore_errors=True)
        os.rename(wip_attach_dir, job_attach_dir)
        wip["attachments"] = [os.path.join(job_attach_dir, os.path.basename(p)) for p in wip.get("attachments", [])]

    # Save persisted copy under job id
    to_save = json.loads(json.dumps(wip)); to_save["id"] = jid
    ok, _ = save_job(to_save, persist_id=jid)

    # Keep WIP file too
    try:
        with open(os.path.join(_WIP_DIR, f"{wip['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(wip, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return ok, jid

# ---------- Date validation ----------
def _validate_dates(planned_start: date | None, planned_end: date | None, required_end: date | None):
    errs = []
    if planned_start and planned_end and planned_end < planned_start:
        errs.append("Planned End cannot be **before** Planned Start.")
    if required_end and planned_start and planned_start > required_end:
        errs.append("Planned Start cannot be **after** Required End Date.")
    if required_end and planned_end and planned_end > required_end:
        errs.append("Planned End cannot be **after** Required End Date.")
    return errs

def _validate_schedule_now(job_like: dict):
    ps = _pdate(job_like.get("planned_start",""))
    pe = _pdate(job_like.get("planned_end",""))
    re = _pdate(job_like.get("required_end_date",""))
    return _validate_dates(ps, pe, re)

# ---------- History (CSV + SQLite mirror) ----------
def _ensure_hist_csv_header():
    if not os.path.exists(HIST_CSV):
        cols = ["Number","Asset Tag","Functional Location","WO Number","Date of Maintenance",
                "Maintenance Code","Spares Used","QTY","Hours Run Since Last Repair",
                "Labour Hours","Asset Downtime Hours","Notes and Findings","Attachments"]
        pd.DataFrame(columns=cols).to_csv(HIST_CSV, index=False)

def _history_sqlite_init():
    with sqlite3.connect(RELIAB_DB) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_tag TEXT,
            functional_location TEXT,
            wo_number TEXT,
            date_of_maintenance TEXT,
            maintenance_code TEXT,
            spare_cmms TEXT,
            spare_desc TEXT,
            qty INTEGER,
            hours_run_since REAL,
            labour_hours REAL,
            downtime_hours REAL,
            notes TEXT,
            attachments_json TEXT,
            created_at TEXT
        )
        """)
        conn.commit()

def _history_sqlite_insert(rows: list[dict]):
    if not rows: return
    _history_sqlite_init()
    with sqlite3.connect(RELIAB_DB) as conn:
        cur = conn.cursor()
        for r in rows:
            cur.execute("""
            INSERT INTO maintenance_history(
                asset_tag,functional_location,wo_number,date_of_maintenance,maintenance_code,
                spare_cmms,spare_desc,qty,hours_run_since,labour_hours,downtime_hours,
                notes,attachments_json,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.get("Asset Tag",""), r.get("Functional Location",""), r.get("WO Number",""),
                r.get("Date of Maintenance",""), r.get("Maintenance Code",""),
                r.get("Spares Used",""), r.get("Spares Desc",""),
                int(r.get("QTY",0) or 0),
                float(r.get("Hours Run Since Last Repair",0) or 0.0),
                float(r.get("Labour Hours",0) or 0.0),
                float(r.get("Asset Downtime Hours",0) or 0.0),
                r.get("Notes and Findings",""),
                r.get("Attachments",""),
                datetime.now().isoformat()
            ))
        conn.commit()

def _append_history_rows_csv_and_sqlite(job: dict, closeout: dict, used_lines: list):
    _ensure_hist_csv_header()
    asset_tag = job.get("asset_tag","")
    asset = (st.session_state.get("assets") or {}).get(asset_tag, {})
    floc = asset.get("Functional Location","")
    wo = job.get("wo_number","")
    date_maint = closeout.get("finish_date","")
    mcode = closeout.get("maintenance_code","")
    hrs_since = float(to_num(closeout.get("hours_run_since", 0)))
    lab_hrs = float(to_num(closeout.get("labour_hours", 0)))
    dt_hrs = float(to_num(closeout.get("downtime_hours", 0)))
    notes = closeout.get("notes","")
    attach_list = closeout.get("attachments", [])
    attach_txt = ";".join([os.path.basename(x) for x in attach_list]) if attach_list else ""

    csv_rows = []
    if used_lines:
        for ul in used_lines:
            qty = int(to_num(ul.get("used_qty",0)))
            if qty <= 0: continue
            cmms = ul.get("cmms","")
            desc = ul.get("desc","")
            csv_rows.append({
                "Number": job.get("persist_id") or job.get("id",""),
                "Asset Tag": asset_tag,
                "Functional Location": floc,
                "WO Number": wo,
                "Date of Maintenance": date_maint,
                "Maintenance Code": mcode,
                "Spares Used": cmms or desc,
                "QTY": qty,
                "Hours Run Since Last Repair": hrs_since,
                "Labour Hours": lab_hrs,
                "Asset Downtime Hours": dt_hrs,
                "Notes and Findings": notes,
                "Attachments": attach_txt
            })
    if not csv_rows:
        csv_rows.append({
            "Number": job.get("persist_id") or job.get("id",""),
            "Asset Tag": asset_tag,
            "Functional Location": floc,
            "WO Number": wo,
            "Date of Maintenance": date_maint,
            "Maintenance Code": mcode,
            "Spares Used": "",
            "QTY": 0,
            "Hours Run Since Last Repair": hrs_since,
            "Labour Hours": lab_hrs,
            "Asset Downtime Hours": dt_hrs,
            "Notes and Findings": notes,
            "Attachments": attach_txt
        })
    # CSV append
    pd.DataFrame(csv_rows).to_csv(HIST_CSV, mode="a", header=False, index=False)
    try:
        st.session_state.history_df = pd.read_csv(HIST_CSV)
    except Exception:
        pass
    # SQLite mirror
    sql_rows = []
    for r in csv_rows:
        sql_rows.append({
            "Asset Tag": r["Asset Tag"],
            "Functional Location": r["Functional Location"],
            "WO Number": r["WO Number"],
            "Date of Maintenance": r["Date of Maintenance"],
            "Maintenance Code": r["Maintenance Code"],
            "Spares Used": r["Spares Used"],
            "Spares Desc": r["Spares Used"],
            "QTY": r["QTY"],
            "Hours Run Since Last Repair": r["Hours Run Since Last Repair"],
            "Labour Hours": r["Labour Hours"],
            "Asset Downtime Hours": r["Asset Downtime Hours"],
            "Notes and Findings": r["Notes and Findings"],
            "Attachments": r["Attachments"]
        })
    _history_sqlite_insert(sql_rows)

# ---------- Runtime reset helpers ----------
def _reset_runtime_asset(asset_tag: str, finish_date: str):
    df = st.session_state.get("runtime_df", pd.DataFrame()).copy()
    if df.empty or "Asset Tag" not in df.columns: return
    mask = (df["Asset Tag"].astype(str) == str(asset_tag))
    if "Last Overhaul" in df.columns:
        df.loc[mask, "Last Overhaul"] = finish_date
    run_col = "Running Hours Since Last Major Maintenance"
    if run_col in df.columns:
        df.loc[mask, run_col] = 0.0
    if "MTBF (Hours)" in df.columns:
        mtbf = pd.to_numeric(df["MTBF (Hours)"], errors="coerce").fillna(0.0)
        runh = pd.to_numeric(df.get(run_col, 0.0), errors="coerce").fillna(0.0)
        df["Remaining Hours"] = (mtbf - runh).clip(lower=0.0)
    st.session_state.runtime_df = df
    try:
        if "save_runtime" in globals(): save_runtime(df)  # why: persist if your app supports it
    except Exception:
        pass

def _reset_runtime_components(asset_tag: str, used_lines: list, finish_date: str):
    df = st.session_state.get("components_runtime_df", None)
    replaced_keys = set()
    for ul in (used_lines or []):
        if not ul.get("replaced", False): continue
        cm = str(ul.get("cmms","")).strip()
        ds = str(ul.get("desc","")).strip()
        if cm: replaced_keys.add(("cmms", cm))
        elif ds: replaced_keys.add(("desc", ds))
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.copy()
        if "Asset Tag" not in df.columns: return
        mask_asset = (df["Asset Tag"].astype(str) == str(asset_tag))
        for kind, key in replaced_keys:
            if kind == "cmms":
                m = mask_asset & (df.get("CMMS MATERIAL CODE","").astype(str).str.strip() == key)
            else:
                m = mask_asset & (df.get("DESCRIPTION","").astype(str).str.strip() == key)
            if not m.any(): continue
            if "LAST REPLACEMENT DATE" in df.columns:
                df.loc[m, "LAST REPLACEMENT DATE"] = finish_date
            if "HOURS SINCE LAST REPLACEMENT" in df.columns:
                df.loc[m, "HOURS SINCE LAST REPLACEMENT"] = 0.0
            if "MTBF (H)" in df.columns and "REMAINING HOURS TO NEXT MAINTENANCE" in df.columns:
                mtbf = pd.to_numeric(df.loc[m, "MTBF (H)"], errors="coerce").fillna(0.0)
                df.loc[m, "REMAINING HOURS TO NEXT MAINTENANCE"] = mtbf
            if {"STATUS","MTBF (H)","HOURS SINCE LAST REPLACEMENT"}.issubset(df.columns):
                def _stat_row(r):
                    mtbf = float(pd.to_numeric([r["MTBF (H)"]], errors="coerce").fillna(0.0)[0])
                    ran  = float(pd.to_numeric([r["HOURS SINCE LAST REPLACEMENT"]], errors="coerce").fillna(0.0)[0])
                    if mtbf <= 0: return "🟢 Healthy"
                    ratio = ran/mtbf if mtbf>0 else 0
                    return "🟢 Healthy" if ratio<0.80 else ("🟠 Plan for maintenance" if ratio<1.0 else "🔴 Overdue for maintenance")
                df.loc[m, "STATUS"] = df.loc[m, ["MTBF (H)","HOURS SINCE LAST REPLACEMENT"]].apply(_stat_row, axis=1)
        st.session_state.components_runtime_df = df
        return
    # fallback dict
    cl = st.session_state.get("component_life", {})
    asset = cl.get(asset_tag, {})
    comp_map = asset.get("components", {})
    if not comp_map: return
    for kind, key in replaced_keys:
        for k, c in comp_map.items():
            ok = (kind=="cmms" and str(c.get("cmms","")).strip()==key) or (kind=="desc" and (str(c.get("name","")).strip()==key or str(k).strip()==key))
            if not ok: continue
            c["last_replacement_date"] = finish_date
            c["hours_since"] = 0.0
            comp_map[k] = c
    asset["components"] = comp_map
    cl[asset_tag] = asset
    st.session_state.component_life = cl

# ---------- Job card export helpers ----------
try:
    from jobcard_template import render_jobcard_html as _tpl_render_jobcard_html
except Exception:
    _tpl_render_jobcard_html = None

def render_jobcard_html(job: dict, assets: dict, currency: str, fx_rate: float) -> str:
    if _tpl_render_jobcard_html:
        return _tpl_render_jobcard_html(job, assets, currency, fx_rate)
    # minimal fallback
    return f"<html><body><h2>WO {job.get('wo_number','')}</h2><p>Asset: {job.get('asset_tag','')}</p><p>Due: {job.get('required_end_date','')}</p></body></html>"

def _wkhtmltopdf_path() -> str:
    cfgx = st.session_state.get("config", {}) or {}
    p = (cfgx.get("wkhtmltopdf_path") or "wkhtmltopdf")
    if os.path.isabs(p) and os.path.isfile(p): return p
    try:
        import shutil as _sh
        found = _sh.which(p)
    except Exception:
        found = None
    return found or ""

def generate_job_card_pdf(job: dict, assets: dict, currency: str, fx_rate: float):
    """
    Returns (ok, file_path, msg).
    Always writes HTML under ATTACH_DIR/plans/<job_id>/<wo>.html
    Tries wkhtmltopdf for PDF; falls back to HTML if missing.
    """
    try:
        html = render_jobcard_html(job, assets, currency, fx_rate)
    except Exception as e:
        return False, "", f"Failed to render job card HTML: {e}"

    job_id = job.get("persist_id") or job.get("id") or f"WIP-{datetime.now().strftime('%H%M%S')}"
    wip_folder = os.path.join(ATTACH_DIR, "plans", str(job_id))
    os.makedirs(wip_folder, exist_ok=True)
    wo = (job.get("wo_number") or job_id)
    html_path = os.path.join(wip_folder, f"{wo}.html")
    pdf_path = os.path.join(wip_folder, f"{wo}.pdf")

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as e:
        return False, "", f"Failed to save HTML: {e}"

    bin_path = _wkhtmltopdf_path()
    if not bin_path:
        return False, html_path, "wkhtmltopdf not found; saved HTML only."
    try:
        subprocess.run([bin_path, "--quiet", html_path, pdf_path], check=True)
        if os.path.isfile(pdf_path) and os.path.getsize(pdf_path) > 0:
            return True, pdf_path, "PDF generated."
        return False, html_path, "wkhtmltopdf ran but PDF missing; using HTML."
    except Exception as e:
        return False, html_path, f"wkhtmltopdf failed: {e}"

# --- CSV/ICS export fallbacks (non-recursive) ---
try:
    from jobcard_template import build_job_csv as _tpl_build_job_csv
except Exception:
    _tpl_build_job_csv = None
try:
    from jobcard_template import build_ics as _tpl_build_ics
except Exception:
    _tpl_build_ics = None

if 'build_job_csv' not in globals():
    def build_job_csv(job: dict, assets: dict) -> str:
        if _tpl_build_job_csv:
            return _tpl_build_job_csv(job, assets)
        import io, csv
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Field","Value"])
        w.writerow(["WO Number", job.get("wo_number","")])
        w.writerow(["Job ID", job.get("persist_id") or job.get("id","")])
        w.writerow(["Asset Tag", job.get("asset_tag","")])
        w.writerow(["Required End Date", job.get("required_end_date","")])
        w.writerow(["Priority", job.get("priority","")])
        w.writerow(["Creator", job.get("creator","")])
        w.writerow(["Requested By", job.get("requested_by","")])
        w.writerow(["PM Code", job.get("pm_code","")])
        w.writerow([])
        w.writerow(["Task Description"])
        w.writerow([job.get("task_description","")])
        w.writerow([])
        w.writerow(["Labour (craft, qty, hours, dept, names, notes)"])
        for r in job.get("labour", []):
            w.writerow([
                str(r.get("craft","")), int(to_num(r.get("qty",0))),
                float(to_num(r.get("hours",0.0))), str(r.get("dept","")),
                str(r.get("names","")), str(r.get("notes",""))
            ])
        w.writerow([])
        w.writerow(["Spares (cmms, oem, desc, qty, price, line_total)"])
        for s in job.get("spares", []):
            w.writerow([
                s.get("cmms",""), s.get("oem",""), s.get("desc",""),
                int(to_num(s.get("qty",0))),
                float(to_num(s.get("price",0.0))),
                float(to_num(s.get("line_total", 0.0))),
            ])
        return buf.getvalue()

if 'build_ics' not in globals():
    def build_ics(job: dict) -> str:
        if _tpl_build_ics:
            return _tpl_build_ics(job)
        uid = (job.get("persist_id") or job.get("id") or f"WIP-{uuid.uuid4().hex[:8]}")
        dt = job.get("required_end_date") or date.today().strftime("%Y-%m-%d")
        dtics = dt.replace("-", "")
        summary = f"WO {job.get('wo_number','')} • {job.get('asset_tag','')}"
        desc = (job.get("task_description","") or "").replace("\n", "\\n")
        return (
            "BEGIN:VCALENDAR\r\n"
            "VERSION:2.0\r\n"
            "PRODID:-//ReliabilityNotebook//PlannerPack//EN\r\n"
            "CALSCALE:GREGORIAN\r\n"
            "METHOD:PUBLISH\r\n"
            "BEGIN:VEVENT\r\n"
            f"UID:{uid}@reliability\r\n"
            f"DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\r\n"
            f"DTSTART;VALUE=DATE:{dtics}\r\n"
            f"SUMMARY:{summary}\r\n"
            f"DESCRIPTION:{desc}\r\n"
            "END:VEVENT\r\n"
            "END:VCALENDAR\r\n"
        )

# --- SENDGRID: key getter + API shim ---
def _get_sendgrid_key():
    cfg2 = st.session_state.get("config", {}) or {}
    return (
        cfg2.get("sendgrid_api_key") or
        os.environ.get("SENDGRID_API_KEY") or
        os.environ.get("SG_API_KEY") or
        ""
    )

if 'send_email_sendgrid' not in globals():
    try:
        import requests
        def send_email_sendgrid(from_email, to_list, subject, html_body, attachments=None, reply_to=None):
            api_key = _get_sendgrid_key()
            if not api_key:
                return False, "Missing SENDGRID_API_KEY (or config['sendgrid_api_key']).", {}
            # Deduplicate recipients (case-insensitive)
            seen=set(); tos=[]
            for x in (to_list or []):
                em=(x.get("email","") or "").strip()
                if not em: continue
                k=em.lower()
                if k in seen: continue
                seen.add(k)
                tos.append({"email": em, "name": (x.get("name","") or "").strip()})
            if not tos:
                return False, "No recipient emails provided.", {}
            payload = {
                "from": {"email": from_email},
                "personalizations": [{"to": tos}],
                "subject": subject or "",
                "content": [{"type": "text/html", "value": html_body or "<p>(no body)</p>"}],
            }
            if reply_to:
                payload["reply_to"] = {"email": reply_to}
            if attachments:
                payload["attachments"] = [
                    {"filename": a.get("filename","attachment.bin"),
                     "type": a.get("type","application/octet-stream"),
                     "content": a.get("content","")} for a in attachments if a
                ]
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            try:
                r = requests.post("https://api.sendgrid.com/v3/mail/send", headers=headers, json=payload, timeout=20)
            except Exception as e:
                return False, f"HTTP error posting to SendGrid: {e}", {}
            if r.status_code in (200, 202):
                return True, "Queued with SendGrid.", dict(r.headers)
            return False, f"{r.status_code}: {r.text}", dict(r.headers)
    except Exception:
        import urllib.request, urllib.error
        def send_email_sendgrid(from_email, to_list, subject, html_body, attachments=None, reply_to=None):
            api_key = _get_sendgrid_key()
            if not api_key:
                return False, "Missing SENDGRID_API_KEY (or config['sendgrid_api_key']).", {}
            seen=set(); tos=[]
            for x in (to_list or []):
                em=(x.get("email","") or "").strip()
                if not em: continue
                k=em.lower()
                if k in seen: continue
                seen.add(k)
                tos.append({"email": em, "name": (x.get("name","") or "").strip()})
            if not tos:
                return False, "No recipient emails provided.", {}
            payload = {
                "from": {"email": from_email},
                "personalizations": [{"to": tos}],
                "subject": subject or "",
                "content": [{"type": "text/html", "value": html_body or "<p>(no body)</p>"}],
            }
            if reply_to:
                payload["reply_to"] = {"email": reply_to}
            if attachments:
                payload["attachments"] = [
                    {"filename": a.get("filename","attachment.bin"),
                     "type": a.get("type","application/octet-stream"),
                     "content": a.get("content","")} for a in attachments if a
                ]
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                "https://api.sendgrid.com/v3/mail/send",
                data=data,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    headers = dict(resp.headers)
                    return True, "Queued with SendGrid.", headers
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                return False, f"{e.code}: {body}", dict(getattr(e, "headers", {}))
            except Exception as e:
                return False, f"HTTP error posting to SendGrid: {e}", {}

# ---------- UI: Planning Pack (Subtab 2) ----------
PP_VER = "v7"   # widget key suffix to avoid collisions
if "assets" not in st.session_state:
    st.session_state.assets = {}

# NOTE: sub_pack must already exist from your Tab 2 declaration.
with sub_pack:
    # ---------- header: dropdown with gating ----------
    def _include_in_dropdown(j):
        status = str(j.get("status","Draft"))
        if status == "Draft": return True
        if status == "Submitted": return True
        return False

    idx = _read_index()
    jobs_meta = list(idx.get("jobs", {}).values()) if isinstance(idx.get("jobs", {}), dict) else []
    jobs_ids = sorted([j["id"] for j in jobs_meta
                       if isinstance(j, dict) and (not j.get("archived", False)) and
                       (not j.get("trashed", False)) and _include_in_dropdown(j)],
                      key=lambda x: x, reverse=True)

    if "pp_selected_job_id" not in st.session_state:
        st.session_state.pp_selected_job_id = "— WIP (unsaved) —"
    if "pp_after_save_select_id" not in st.session_state:
        st.session_state.pp_after_save_select_id = ""

    staged_sel = st.session_state.pp_after_save_select_id
    if staged_sel:
        st.session_state.pp_selected_job_id = staged_sel if staged_sel in jobs_ids else "— WIP (unsaved) —"
        st.session_state.pp_after_save_select_id = ""

    sel_id = st.selectbox(
        "Job (Draft + Submitted)",
        options=["— WIP (unsaved) —"] + jobs_ids,
        index=(["— WIP (unsaved) —"] + jobs_ids).index(st.session_state.pp_selected_job_id)
        if st.session_state.pp_selected_job_id in (["— WIP (unsaved) —"] + jobs_ids) else 0,
        key="pp_job_select_"+PP_VER
    )

    top_l, top_r = st.columns([3,2])
    with top_r:
        col_a, col_b, col_c = st.columns(3)
        new_clicked = col_a.button("➕ New WIP", use_container_width=True, key="pp_new_wip_"+PP_VER)
        dup_clicked = col_b.button("🧬 Duplicate to WIP", use_container_width=True,
                                   disabled=(sel_id in [None,"— WIP (unsaved) —"]), key="pp_dup_to_wip_"+PP_VER)
        arc_clicked = col_c.button("🗃️ Archive", use_container_width=True,
                                   disabled=(sel_id in [None,"— WIP (unsaved) —"]), key="pp_archive_btn_"+PP_VER)

    if new_clicked:
        prefill = st.session_state.get("asset_filter",[None])
        prefill = prefill[0] if (isinstance(prefill, list) and len(prefill)==1) else ""
        st.session_state.pp_wip = _new_wip_from_scratch(prefill_asset=prefill)
        st.session_state.pp_selected_job_id = "— WIP (unsaved) —"
        st.success("New WIP ready.")
    elif dup_clicked and sel_id not in [None,"— WIP (unsaved) —"]:
        base = load_job(sel_id)
        st.session_state.pp_wip = _new_wip_from_scratch()
        if base:
            for k,v in base.items():
                if k == "id": continue
                st.session_state.pp_wip[k] = v
        st.session_state.pp_wip["persist_id"] = ""  # new copy
        st.session_state.pp_selected_job_id = "— WIP (unsaved) —"
        st.success("Duplicated into WIP (not saved).")
    elif arc_clicked and sel_id not in [None,"— WIP (unsaved) —"]:
        if archive_job(sel_id, archived=True):
            st.success("Archived.")
            st.session_state.pp_selected_job_id = "— WIP (unsaved) —"
            st.rerun()
        else:
            st.error("Archive failed.")
    else:
        if sel_id != st.session_state.pp_selected_job_id:
            st.session_state.pp_selected_job_id = sel_id
            if sel_id not in [None,"— WIP (unsaved) —"]:
                st.session_state.pp_wip = _wip_from_persisted(sel_id)
            else:
                _ensure_wip()
        else:
            _ensure_wip()

    job = st.session_state.pp_wip
    if not job:
        st.info("Create or select something to begin.")
        st.stop()
    locked = (job.get("status") == "Completed")
    lock_note = " (read-only — Completed)" if locked else ""

    # ---------- A) Job Setup ----------
    st.markdown(f"### A) Job Setup{lock_note}")
    all_tags = sorted(list(st.session_state.assets.keys()))
    col1, col2 = st.columns(2)
    with col1:
        at_opts = [""] + all_tags
        st.session_state.pp_wip["asset_tag"] = st.selectbox(
            "Asset Tag", options=at_opts,
            index=(at_opts.index(job.get("asset_tag","")) if job.get("asset_tag","") in at_opts else 0),
            disabled=locked, key="pp_asset_tag_"+PP_VER
        )
    with col2:
        pm_opts = ["","PM00","PM01","PM02","PM03","PM04"]
        st.session_state.pp_wip["pm_code"] = st.selectbox(
            "PM Code", options=pm_opts,
            index=(pm_opts.index(job.get("pm_code","")) if job.get("pm_code","") in pm_opts else 0),
            disabled=locked, key="pp_pm_code_"+PP_VER
        )

    st.session_state.pp_wip["task_description"] = st.text_area(
        "Task Description (what & how)", value=job.get("task_description",""), height=80,
        disabled=locked, key="pp_task_desc_"+PP_VER
    )

    names = [c["name"] for c in st.session_state.config.get("address_book", []) if "name" in c and c["name"]]
    c3, c4, c5 = st.columns(3)
    with c3:
        idx_name = (names.index(job.get("creator",""))+1 if job.get("creator","") in names else 0)
        st.session_state.pp_wip["creator"] = st.selectbox("Creator", options=[""]+names, index=idx_name,
                                                          disabled=locked, key="pp_creator_"+PP_VER)
        if not locked:
            new_creator = st.text_input("Add new creator (name)", key="pp_add_creator_"+PP_VER)
            if new_creator:
                if not any(c.get("name")==new_creator for c in st.session_state.config["address_book"]):
                    st.session_state.config["address_book"].append({"name": new_creator, "email": ""})
                    save_config(st.session_state.config)
                    st.success("Creator added. Re-open dropdown to pick.")
    with c4:
        cur_date = _pdate(job.get("required_end_date","")) or date.today()
        sel_date = st.date_input("Required End Date", value=cur_date, disabled=locked, key="pp_req_end_"+PP_VER)
        st.session_state.pp_wip["required_end_date"] = sel_date.strftime("%Y-%m-%d")
    with c5:
        pr_opts = ["","Low","Normal","High","Critical"]
        st.session_state.pp_wip["priority"] = st.selectbox(
            "Priority", options=pr_opts,
            index=(pr_opts.index(job.get("priority","")) if job.get("priority","") in pr_opts else 0),
            disabled=locked, key="pp_priority_"+PP_VER
        )
        st.session_state.pp_wip["requested_by"] = st.text_input(
            "Requested By", value=job.get("requested_by",""), disabled=locked, key="pp_requested_by_"+PP_VER
        )
        st.text_input("Works Order Number", value=job.get("wo_number",""), disabled=True, key="pp_wo_display_"+PP_VER)

    st.markdown("#### Lead Role & Co-leads")
    lr1, lr2 = st.columns([2,1])
    role_options = [
        "Department Foreman — Mechanical","Department Foreman — Electrical","Department Foreman — Boiler/Fabrication",
        "Department Foreman — Rigging","Department Foreman — Instrumentation",
        "Section Engineer — Mechanical","Section Engineer — Electrical","Section Engineer — Instrumentation","Section Engineer — Boiler/Fabrication","Section Engineer — Civil/Structural",
        "Project Engineer","Reliability Engineer","Electrical Engineer (Design/Systems)","Instrumentation Engineer","Mechanical Engineer (Design)","Civil/Structural Engineer",
        "Engineering Specialist — Mechanical","Engineering Specialist — Electrical","Engineering Specialist — Instrumentation","Engineering Specialist — Boiler/Fabrication",
        "Section Engineering Manager","Plant/Operations Manager","Supply Chain Supervisor","Resource Coordinator","Planner / Scheduler","Safety Officer (HSE)",
        "Process/Operations Supervisor","Metallurgy / Process Engineer","Process Operator / Processor","Other"
    ]
    st.session_state.pp_wip["lead_role"] = lr1.selectbox(
        "Lead Role", options=[""]+role_options,
        index=([""]+role_options).index(job.get("lead_role","")) if job.get("lead_role","") in ([""]+role_options) else 0,
        disabled=locked, key="pp_lead_role_"+PP_VER
    )
    dept_opts = st.session_state.config.get("dept_presets", [])
    _role_lower = str(job.get("lead_role","")).lower()
    default_dept = "Engineering / Projects"
    if "elect" in _role_lower: default_dept = "Electrical"
    elif "instr" in _role_lower: default_dept = "Instrumentation"
    elif "rigging" in _role_lower or "crane" in _role_lower: default_dept = "Rigging"
    elif "boiler" in _role_lower or "weld" in _role_lower: default_dept = "Boiler / Fabrication"
    elif "civil" in _role_lower or "struct" in _role_lower: default_dept = "Civil / Structural"
    elif "reliab" in _role_lower: default_dept = "Reliability"
    elif "supply chain" in _role_lower: default_dept = "Supply Chain / Stores"
    elif "plant" in _role_lower or "process" in _role_lower: default_dept = "Process / Operations"
    elif "safety" in _role_lower or "hse" in _role_lower: default_dept = "Safety (HSE)"
    elif "planner" in _role_lower or "schedule" in _role_lower: default_dept = "Planning / Scheduling"
    st.session_state.pp_wip["lead_department"] = lr2.selectbox(
        "Lead Department", options=dept_opts,
        index=(dept_opts.index(job.get("lead_department", default_dept)) if job.get("lead_department", default_dept) in dept_opts else 0),
        disabled=locked, key="pp_lead_dept_"+PP_VER
    )
    st.session_state.pp_wip["co_leads"] = st.multiselect(
        "Co-leads", options=role_options, default=job.get("co_leads", []),
        disabled=locked, key="pp_co_leads_"+PP_VER
    )

    st.markdown("---")

    # ---------- B) Spares & Kitting ----------
    def _money_fmt_sym(x: float, sym: str) -> str:
        return f"{sym} {x:,.2f}"

    def _compute_kitting_metrics(disp_df: pd.DataFrame):
        inc = disp_df[disp_df["Include"] == True] if "Include" in disp_df.columns else pd.DataFrame()
        warnings = []
        if not inc.empty and any(pd.to_numeric(inc["Price"], errors="coerce").fillna(0.0) <= 0):
            warnings.append(f"{int((pd.to_numeric(inc['Price'], errors='coerce').fillna(0.0) <= 0).sum())} line(s) with no price.")
        if not inc.empty and any(pd.to_numeric(inc["Pick QTY"], errors="coerce").fillna(0) <= 0):
            warnings.append(f"{int((pd.to_numeric(inc['Pick QTY'], errors='coerce').fillna(0) <= 0).sum())} line(s) with Pick QTY = 0.")
        valid = inc[(pd.to_numeric(inc["Pick QTY"], errors="coerce").fillna(0) > 0) &
                    (pd.to_numeric(inc["Price"], errors="coerce").fillna(0.0) > 0.0)]
        pct_ok = (100.0 * len(valid) / max(1, len(inc))) if len(inc) else 0.0
        total = float(pd.to_numeric(inc.get("Line Total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not inc.empty else 0.0
        return pct_ok, warnings, total

    st.markdown(f"### B) Spares & Kitting{lock_note}")
    st.session_state.pp_wip["delivery_note"] = st.text_input(
        "Delivery note (e.g., Deliver to Bay X)", value=job.get("delivery_note",""),
        disabled=locked, key="pp_delivery_note_"+PP_VER
    )
    col_sym, _ = st.columns([1,3])
    with col_sym:
        sym = st.radio("Currency symbol", ["R", "$"], index=(0 if job.get("currency_symbol","R")=="R" else 1),
                       horizontal=True, key="pp_curr_sym_"+PP_VER)
    st.session_state.pp_wip["currency_symbol"] = sym

    bom_df = pd.DataFrame(columns=BOM_COLS)
    if job.get("asset_tag"):
        asset = st.session_state.assets.get(job["asset_tag"], {})
        bom_df = build_bom_table(asset)

    if bom_df.empty:
        st.info("No BOM found for this asset.")
        st.session_state.pp_bom_snapshot = None
    else:
        selected = {(s.get("cmms",""), s.get("oem",""), s.get("desc","")): int(to_num(s.get("qty",0))) for s in job.get("spares", [])}
        disp = pd.DataFrame({
            "Include": False,
            "CMMS": bom_df["CMMS MATERIAL CODE"].astype(str),
            "OEM/Part": bom_df["OEM PART NUMBER"].astype(str),
            "Description": bom_df["DESCRIPTION"].astype(str),
            "Pick QTY": pd.to_numeric(bom_df["QUANTITY"], errors="coerce").fillna(0).astype(int),
            "Price": pd.to_numeric(bom_df["PRICE"], errors="coerce").fillna(0.0).astype(float),
        })
        for i in disp.index:
            keyt = (disp.at[i,"CMMS"], disp.at[i,"OEM/Part"], disp.at[i,"Description"])
            if keyt in selected and selected[keyt] > 0:
                disp.at[i, "Include"] = True
                disp.at[i, "Pick QTY"] = selected[keyt]

        if locked:
            tmp = disp.copy()
            tmp["Line Total"] = pd.to_numeric(tmp["Pick QTY"], errors="coerce").fillna(0) * pd.to_numeric(tmp["Price"], errors="coerce").fillna(0.0)
            pct_ok, warnings, total_spares_cost = _compute_kitting_metrics(tmp)
            gcol, tcol = st.columns([1,1])
            with gcol:
                col_fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=pct_ok, number={'suffix': "%"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#2ca02c" if pct_ok>=85 else "#ff7f0e" if pct_ok>=60 else "#d62728"},
                           'steps': [{'range': [0, 60], 'color': "#ffe6e6"},
                                     {'range': [60, 85], 'color': "#fff2e0"},
                                     {'range': [85, 100], 'color': "#e8f6e8"}] }
                ))
                col_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), title="Kitting Readiness")
                st.plotly_chart(col_fig, use_container_width=True)
            with tcol:
                st.metric("Total Spares Cost", _money_fmt_sym(total_spares_cost, sym))
            st.dataframe(tmp, use_container_width=True, height=_auto_table_height(tmp))
            st.session_state.pp_bom_snapshot = tmp.copy()
        else:
            colA, colB = st.columns(2)
            if colA.button("Select all", key="pp_bom_sel_all_"+PP_VER):
                disp["Include"] = True
            if colB.button("Select none", key="pp_bom_sel_none_"+PP_VER):
                disp["Include"] = False

            edited_bom = st.data_editor(
                disp,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Include": st.column_config.CheckboxColumn(required=False),
                    "Pick QTY": st.column_config.NumberColumn(min_value=0, step=1),
                    "Price": st.column_config.NumberColumn(min_value=0.0, step=1.0, help="Base price (no FX)"),
                },
                key="pp_bom_editor_"+PP_VER,
                height=_auto_table_height(disp)
            )
            edited_bom = edited_bom.copy()
            edited_bom["Line Total"] = pd.to_numeric(edited_bom["Pick QTY"], errors="coerce").fillna(0) * pd.to_numeric(edited_bom["Price"], errors="coerce").fillna(0.0)
            st.session_state.pp_bom_snapshot = edited_bom.copy()

            pct_ok, warnings, total_spares_cost = _compute_kitting_metrics(edited_bom)
            gcol, mcol = st.columns([1,1])
            with gcol:
                col_fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=pct_ok, number={'suffix': "%"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#2ca02c" if pct_ok>=85 else "#ff7f0e" if pct_ok>=60 else "#d62728"},
                           'steps': [{'range': [0, 60], 'color': "#ffe6e6"},
                                     {'range': [60, 85], 'color': "#fff2e0"},
                                     {'range': [85, 100], 'color': "#e8f6e8"}] }
                ))
                col_fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), title="Kitting Readiness")
                st.plotly_chart(col_fig, use_container_width=True)
            with mcol:
                if warnings:
                    st.warning(" • ".join(warnings))
                st.metric("Total Spares Cost", _money_fmt_sym(total_spares_cost, sym))
            st.dataframe(edited_bom, use_container_width=True, height=_auto_table_height(edited_bom))

            if st.button("💾 Save picked spares to WIP", key="pp_save_spares_wip_"+PP_VER):
                lines = []
                for _, r in edited_bom.iterrows():
                    if bool(r["Include"]) and int(to_num(r["Pick QTY"])) > 0:
                        qtyi = int(to_num(r["Pick QTY"]))
                        pricef = float(to_num(r["Price"]))
                        lines.append({
                            "cmms": str(r["CMMS"]),
                            "oem": str(r["OEM/Part"]),
                            "desc": str(r["Description"]),
                            "qty": qtyi,
                            "price": pricef,
                            "line_total": float(qtyi * pricef)
                        })
                st.session_state.pp_wip["spares"] = lines
                st.session_state.pp_wip["kitting_readiness"] = {"pct_ok": pct_ok, "warnings": warnings}
                st.session_state.pp_wip["currency_symbol"] = sym
                st.success("Picked spares saved to WIP (not persisted).")

    st.markdown("---")

    # ---------- C) Resources & Logistics ----------
    st.markdown(f"### C) Resources & Logistics{lock_note}")
    st.markdown("**C1. Craft labour**")
    craft_opts = st.session_state.config.get("craft_presets", [])
    dept_opts  = st.session_state.config.get("dept_presets", [])

    if not job.get("labour"):
        st.session_state.pp_wip["labour"] = [{"craft":"Mechanical Technician", "qty":1, "hours":4.0, "dept":"Mechanical", "notes":"", "names":""}]

    labour_df = pd.DataFrame(st.session_state.pp_wip["labour"])
    for c in ["craft","qty","hours","dept","notes","names"]:
        if c not in labour_df.columns:
            labour_df[c] = "" if c in ["craft","dept","notes","names"] else 0
    labour_df["qty"]   = pd.to_numeric(labour_df["qty"], errors="coerce").fillna(0).astype(int)
    labour_df["hours"] = pd.to_numeric(labour_df["hours"], errors="coerce").fillna(0.0).astype(float)

    edited_labour = st.data_editor(
        labour_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "craft": st.column_config.SelectboxColumn("Craft", options=craft_opts, required=False, disabled=locked),
            "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1, disabled=locked),
            "hours": st.column_config.NumberColumn("Hours", min_value=0.0, step=0.5, disabled=locked),
            "dept": st.column_config.SelectboxColumn("Dept", options=dept_opts, required=False, disabled=locked),
            "names": st.column_config.TextColumn("Names (comma-separated)", disabled=locked),
            "notes": st.column_config.TextColumn("Notes", disabled=locked)
        },
        key="pp_labour_editor_"+PP_VER,
        height=_auto_table_height(labour_df)
    )

    st.markdown("**C2. Lifting & access**")
    cranage_required = st.checkbox("Cranage required", value=bool(job.get("lifting",{}).get("cranage_required", False)),
                                   disabled=locked, key=f"pp_cranage_req_{job['id']}_{PP_VER}")
    crane_type      = job.get("lifting",{}).get("crane_type","")
    mobile_capacity = job.get("lifting",{}).get("mobile_capacity","50t")
    overhead_spec   = job.get("lifting",{}).get("overhead_spec","")
    if cranage_required:
        crane_type = st.selectbox("Crane type", ["Overhead crane","Mobile crane"],
                                  index=(0 if crane_type=="Overhead crane" else 1 if crane_type=="Mobile crane" else 0),
                                  disabled=locked, key=f"pp_crane_type_{job['id']}_{PP_VER}")
        if crane_type == "Mobile crane":
            mopts = ["20t","25t","35t","50t","70t","80t","90t","100t","120t","160t","200t","225t","250t","300t","400t"]
            mobile_capacity = st.selectbox("Mobile crane capacity", mopts,
                                           index=(mopts.index(mobile_capacity) if mobile_capacity in mopts else 3),
                                           disabled=locked, key=f"pp_mobile_cap_{job['id']}_{PP_VER}")
        else:
            overhead_spec = st.text_input("Overhead crane (bay/ton)", value=overhead_spec, disabled=locked,
                                          key=f"pp_oh_spec2_{job['id']}_{PP_VER}")
    else:
        crane_type, mobile_capacity, overhead_spec = "", "", ""

    forklift_required = st.checkbox("Forklift required", value=bool(job.get("lifting",{}).get("forklift_required", False)),
                                    disabled=locked, key=f"pp_fl_req_{job['id']}_{PP_VER}")
    forklift_capacity = job.get("lifting",{}).get("forklift_capacity","3t")
    if forklift_required:
        fopts = ["1t","2t","3t","5t","8t"]
        forklift_capacity = st.selectbox("Forklift capacity", fopts,
                                         index=(fopts.index(forklift_capacity) if forklift_capacity in fopts else 2),
                                         disabled=locked, key=f"pp_fl_cap_{job['id']}_{PP_VER}")
    else:
        forklift_capacity = ""

    scaffolding_required = st.checkbox("Scaffolding required", value=bool(job.get("lifting",{}).get("scaffolding_required", False)),
                                       disabled=locked, key=f"pp_scaff_req_{job['id']}_{PP_VER}")

    st.markdown("**C3. Logistics**")
    lg1, lg2 = st.columns(2)
    pickup_point = lg1.text_input("Stores pick-up point", value=job.get("logistics",{}).get("pickup_point",""),
                                  disabled=locked, key=f"pp_pickup_{job['id']}_{PP_VER}")
    delivery_loc = lg1.text_input("Delivery location (bay/area)", value=job.get("logistics",{}).get("delivery_location",""),
                                  disabled=locked, key=f"pp_delivery_{job['id']}_{PP_VER}")
    transport_needed = lg2.checkbox("Transport needed", value=bool(job.get("logistics",{}).get("transport_needed", False)),
                                    disabled=locked, key=f"pp_trans_needed_{job['id']}_{PP_VER}")
    transport_type = lg2.text_input("Transport type", value=job.get("logistics",{}).get("transport_type",""),
                                    disabled=(locked or not transport_needed), key=f"pp_trans_type_{job['id']}_{PP_VER}")
    permits = st.multiselect("Permits required", options=st.session_state.config.get("permit_presets", []),
                             default=job.get("logistics",{}).get("permits", []), disabled=locked, key=f"pp_permits_{job['id']}_{PP_VER}")
    names2 = [c["name"] for c in st.session_state.config.get("address_book", []) if "name" in c and c["name"]]
    resp_person = st.selectbox("Responsible person", options=[""]+names2,
                               index=(names2.index(job.get("logistics",{}).get("responsible_person",""))+1
                                      if job.get("logistics",{}).get("responsible_person","") in names2 else 0),
                               disabled=locked, key=f"pp_resp_{job['id']}_{PP_VER}")
    new_person = st.text_input("Add new person (name only, add email in Send panel)", disabled=locked, key=f"pp_add_person_{job['id']}_{PP_VER}")
    if new_person and not locked:
        if not any(c.get("name")==new_person for c in st.session_state.config["address_book"]):
            st.session_state.config["address_book"].append({"name": new_person, "email": ""})
            save_config(st.session_state.config)
            st.success("Person added. Re-open dropdown to pick.")

    if st.button("💾 Save Resources to WIP", key="pp_save_resources_wip_"+PP_VER, disabled=locked):
        st.session_state.pp_wip["labour"] = []
        for _, r in edited_labour.iterrows():
            if str(r.get("craft","")).strip():
                st.session_state.pp_wip["labour"].append({
                    "craft": str(r.get("craft","")),
                    "qty": int(to_num(r.get("qty",0))),
                    "hours": float(to_num(r.get("hours",0.0))),
                    "dept": str(r.get("dept","")),
                    "names": str(r.get("names","")),
                    "notes": str(r.get("notes",""))
                })
        st.session_state.pp_wip["lifting"] = {
            "cranage_required": bool(cranage_required),
            "crane_type": crane_type,
            "mobile_capacity": mobile_capacity,
            "overhead_spec": overhead_spec,
            "forklift_required": bool(forklift_required),
            "forklift_capacity": forklift_capacity,
            "scaffolding_required": bool(scaffolding_required)
        }
        st.session_state.pp_wip["logistics"] = {
            "pickup_point": pickup_point,
            "delivery_location": delivery_loc,
            "transport_needed": bool(transport_needed),
            "transport_type": transport_type,
            "permits": permits,
            "responsible_person": resp_person
        }
        st.success("Resources saved to WIP (not persisted).")

    st.markdown("---")

    # ---------- D) Schedule & Capacity ----------
    st.markdown(f"### D) Schedule & Capacity{lock_note}")
    sc1, sc2, sc3 = st.columns(3)
    st.session_state.pp_wip["planned_duration_hours"] = float(sc1.number_input(
        "Planned Duration (hours)", min_value=0.0, step=0.5, value=float(job.get("planned_duration_hours",0.0)),
        disabled=locked, key=f"pp_planned_dur_{job['id']}_{PP_VER}"
    ))
    total_mh = sum(float(to_num(r.get("qty",0))) * float(to_num(r.get("hours",0.0))) for r in st.session_state.pp_wip.get("labour", []))
    sc2.metric("Planned Man-hours", f"{total_mh:.1f}")

    start_default = _pdate(job.get("planned_start","")) or date.today()
    end_default   = _pdate(job.get("planned_end",""))   or (start_default + timedelta(days=6))
    sdate = sc3.date_input("Planned Start", value=start_default, disabled=locked, key=f"pp_planned_start_{job['id']}_{PP_VER}")
    edate = sc3.date_input("Planned End",   value=end_default,   disabled=locked, key=f"pp_planned_end_{job['id']}_{PP_VER}")
    st.session_state.pp_wip["planned_start"] = sdate.strftime("%Y-%m-%d")
    st.session_state.pp_wip["planned_end"]   = edate.strftime("%Y-%m-%d")

    def auto_slot(job_like, capacities: dict, start_date: date, end_date: date):
        from collections import defaultdict
        n_days = max((end_date - start_date).days, 0)
        dates = [start_date + timedelta(days=i) for i in range(n_days + 1)]
        if not dates: dates = [start_date]
        demand = {r["craft"]: float(to_num(r["qty"])) * float(to_num(r["hours"]))
                  for r in job_like.get("labour", []) if to_num(r.get("qty",0))>0 and to_num(r.get("hours",0.0))>0}
        per_day = {d: defaultdict(float) for d in dates}
        for craft, need in demand.items():
            cap = float(st.session_state.config.get("capacity_by_craft", {}).get(craft, 0))
            if cap <= 0:
                per_day[dates[0]][craft] += need
                continue
            remaining = need
            for d in dates:
                room = max(cap - per_day[d][craft], 0.0)
                take = min(room, remaining)
                per_day[d][craft] += take
                remaining -= take
                if remaining <= 1e-6: break
            if remaining > 1e-6:
                per_day[dates[-1]][craft] += remaining
        sched = []
        for d in dates:
            for craft, hrs in per_day[d].items():
                if hrs > 0:
                    sched.append({"date": d.strftime("%Y-%m-%d"), "craft": craft, "hours": float(hrs)})
        return sched, dates

    capacities = st.session_state.config.get("capacity_by_craft", {})
    schedule, dates_list = auto_slot(st.session_state.pp_wip, capacities, sdate, edate)
    st.session_state.pp_wip["capacity_fit"] = schedule

    date_errs = _validate_schedule_now(st.session_state.pp_wip)
    if date_errs:
        for e in date_errs: st.error(e)

    if schedule:
        df_sched = pd.DataFrame(schedule)
        pivot = df_sched.pivot_table(index="date", columns="craft", values="hours", aggfunc="sum").fillna(0.0)
        fig = go.Figure()
        for craft in pivot.columns:
            fig.add_trace(go.Bar(name=craft, x=pivot.index.tolist(), y=pivot[craft].tolist()))
        fig.update_layout(
            barmode="stack", title="This Job vs Daily Craft Capacities",
            height=360, margin=dict(l=10, r=10, t=40, b=10),
            xaxis_title="Date", yaxis_title="Hours",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Add labour rows to see schedule fit.")

    if st.button("💾 Save Schedule to WIP", key="pp_save_schedule_wip_"+PP_VER, disabled=(locked or bool(date_errs))):
        st.success("Schedule saved to WIP (not persisted).")

    st.markdown("---")

    # ---------- E) Attachments ----------
    st.markdown(f"### E) Attachments{lock_note}")
    wip_folder = os.path.join(ATTACH_DIR, "plans", job["id"])
    os.makedirs(wip_folder, exist_ok=True)

    files = st.file_uploader(
        "Upload job attachments (quotes, scope, RA, vendor PDFs)",
        type=["pdf","doc","docx","xls","xlsx","png","jpg","jpeg"], accept_multiple_files=True, key=f"pp_attach_uploader_{job['id']}_{PP_VER}"
    )
    if files and not locked:
        for f in files:
            path = os.path.join(wip_folder, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
    att_list = []
    if os.path.isdir(wip_folder):
        for nm in sorted(os.listdir(wip_folder)):
            att_list.append(os.path.join(wip_folder, nm))
    st.session_state.pp_wip["attachments"] = att_list
    if att_list:
        for pth in att_list:
            st.write("📎", os.path.basename(pth))

    st.markdown("---")

    # ---------- F) Finalize ----------
    st.markdown("### Finalize")
    f1, f2, f3 = st.columns(3)

    if f1.button("💾 Save Job Draft", key="pp_save_draft_final_"+PP_VER, disabled=locked or bool(date_errs)):
        ok, jid = _persist_wip_to_job()
        if ok:
            set_status(jid, "Draft")
            st.success(f"Saved draft {jid}")
            st.session_state.pp_after_save_select_id = jid
            st.rerun()
        else:
            st.error("Save failed.")

    if "pp_show_send_panel" not in st.session_state:
        st.session_state.pp_show_send_panel = False
    if "pp_last_submitted_id" not in st.session_state:
        st.session_state.pp_last_submitted_id = ""

    if f2.button("📤 Submit (Issue W/O)", key="pp_submit_final_"+PP_VER, disabled=locked or bool(date_errs)):
        if not job.get("persist_id"):
            ok, jid = _persist_wip_to_job()
            if not ok:
                st.error("Could not save before submit.")
                st.stop()
        ok = set_status(st.session_state.pp_wip["persist_id"], "Submitted")
        if ok:
            persisted = load_job(st.session_state.pp_wip["persist_id"]) or {}
            ok_pdf, pdf_or_html_path, msg = generate_job_card_pdf(
                persisted, st.session_state.assets,
                currency=st.session_state.pp_wip.get("currency_symbol","R"),
                fx_rate=1.0
            )
            persisted["pdf_path"] = pdf_or_html_path
            persisted["currency_symbol"] = st.session_state.pp_wip.get("currency_symbol","R")
            save_job(persisted, persist_id=persisted.get("persist_id") or persisted.get("id"))
            st.success(f"Job submitted. WO: {persisted.get('wo_number','')}. {msg}")
            st.session_state.pp_show_send_panel = True
            st.session_state.pp_last_submitted_id = persisted.get("persist_id") or persisted.get("id")
            st.session_state.pp_after_save_select_id = "— WIP (unsaved) —"
            st.rerun()
        else:
            st.warning("Cannot submit (maybe Completed).")

    if f3.button("✅ Complete W/O", key="pp_confirm_complete_"+PP_VER):
        if not job.get("persist_id"):
            ok, jid = _persist_wip_to_job()
            if not ok:
                st.error("Could not save before opening close-out.")
                st.stop()
        st.session_state.pp_show_closeout = True
        st.session_state.pp_closeout_for = job["persist_id"]

    # --- Optional: Send Job Card (scoped to this subtab) ---
    if st.session_state.get("pp_show_send_panel") and st.session_state.get("pp_last_submitted_id"):
        persisted_for_send = load_job(st.session_state["pp_last_submitted_id"])
        if persisted_for_send:
            st.markdown("### Send Job Card (optional)")
            send_now = st.checkbox("Send job card to recipients now", value=False, key="pp_send_checkbox_"+PP_VER)
            if send_now:
                email_re = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"

                api_key_present = bool(_get_sendgrid_key())
                from_default = st.session_state.config.get("sendgrid_from","planner@em7701.maraksreliability.com")
                st.session_state.config["sendgrid_from"] = st.text_input("From (verified domain)", value=from_default, key="pp_from_email_"+PP_VER)
                st.session_state.config["sendgrid_reply_to"] = st.text_input("Reply-To (optional)", value=st.session_state.config.get("sendgrid_reply_to",""), key="pp_reply_to_"+PP_VER)
                save_config(st.session_state.config)
                from_email = st.session_state.config["sendgrid_from"]
                reply_to  = st.session_state.config["sendgrid_reply_to"]

                with st.expander("Add / update address book contact"):
                    ab_name = st.text_input("Contact name", key="pp_ab_name_"+PP_VER)
                    ab_email = st.text_input("Contact email", key="pp_ab_email_"+PP_VER, placeholder="name@company.com")
                    if st.button("Save to Address Book", key="pp_ab_save_"+PP_VER):
                        if not ab_email or not re.match(email_re, ab_email.strip()):
                            st.error("Enter a valid email.")
                        else:
                            ab = st.session_state.config.get("address_book", []) or []
                            updated = False
                            for c in ab:
                                if c.get("email","").strip().lower() == ab_email.strip().lower() or (
                                   ab_name and c.get("name","").strip().lower() == ab_name.strip().lower()):
                                    c["name"]  = ab_name or c.get("name","")
                                    c["email"] = ab_email.strip()
                                    updated = True
                                    break
                            if not updated:
                                ab.append({"name": ab_name, "email": ab_email.strip()})
                            st.session_state.config["address_book"] = ab
                            save_config(st.session_state.config)
                            st.success("Contact saved.")

                ab = st.session_state.config.get("address_book", []) or []
                ab_labels = []
                for c in ab:
                    nm = (c.get("name","") or "").strip()
                    em = (c.get("email","") or "").strip()
                    if em and nm:
                        ab_labels.append(f"{nm} <{em}>")
                    elif em:
                        ab_labels.append(em)
                    elif nm:
                        ab_labels.append(nm)

                pick = st.multiselect("Recipients (Address Book)", options=ab_labels, default=[], key="pp_recipients_"+PP_VER)
                adhoc = st.text_input("Additional recipients (comma-separated emails)", key="pp_recipients_adhoc_"+PP_VER, placeholder="alice@x.com, bob@y.org")

                subj = f"Works Order: {persisted_for_send.get('wo_number','')} • {persisted_for_send.get('asset_tag','')} • Due {persisted_for_send.get('required_end_date','')}"
                body_default = f"""<p>Hi team,</p>
<p>Please find the PM Maintenance Order attached for <b>{persisted_for_send.get('asset_tag','')}</b> (WO: <b>{persisted_for_send.get('wo_number','')}</b>), due <b>{persisted_for_send.get('required_end_date','')}</b>.</p>
<p>Regards,<br/>Reliability App</p>"""
                body = st.text_area("Message", value=body_default, height=140, key="pp_email_body_"+PP_VER)
                attach_pdf = st.checkbox("Attach Job Card PDF", value=True, key="pp_attach_pdf_"+PP_VER)

                csv_data = build_job_csv(persisted_for_send, st.session_state.assets)
                ics_data = build_ics(persisted_for_send)
                html_blob = render_jobcard_html(persisted_for_send, st.session_state.assets,
                                                currency=persisted_for_send.get("currency_symbol","R"),
                                                fx_rate=1.0).encode("utf-8")
                attachments = [
                    {"filename": f"{(persisted_for_send.get('persist_id') or persisted_for_send.get('id','job'))}.csv", "content": _b64(csv_data.encode("utf-8")), "type": "text/csv"},
                    {"filename": f"{(persisted_for_send.get('persist_id') or persisted_for_send.get('id','job'))}.ics", "content": _b64(ics_data.encode("utf-8")), "type": "text/calendar"},
                    {"filename": f"{(persisted_for_send.get('persist_id') or persisted_for_send.get('id','job'))}.html", "content": _b64(html_blob), "type": "text/html"},
                ]
                pdf_path = persisted_for_send.get("pdf_path","")
                if attach_pdf and pdf_path and os.path.isfile(pdf_path):
                    with open(pdf_path, "rb") as fpdf:
                        attachments.append({"filename": os.path.basename(pdf_path), "content": _b64(fpdf.read()), "type": "application/pdf"})

                def _parse_label(lbl: str):
                    s = (lbl or "").strip()
                    m = re.match(r"^(.*)\s*<([^>]+)>$", s)
                    if m:
                        return {"name": m.group(1).strip(), "email": m.group(2).strip()}
                    if re.match(email_re, s):
                        return {"name": "", "email": s}
                    return {"name": s, "email": ""}

                to_list = [_parse_label(x) for x in (pick or []) if x]
                if adhoc:
                    for e in [p.strip() for p in adhoc.split(",") if p.strip()]:
                        if re.match(email_re, e):
                            to_list.append({"name": "", "email": e})

                seen=set(); uniq_to=[]; removed=[]
                for t in (to_list or []):
                    em=(t.get("email") or "").strip().lower()
                    if not em: continue
                    if em in seen:
                        removed.append(em); continue
                    seen.add(em)
                    uniq_to.append({"name": t.get("name",""), "email": (t.get("email") or "").strip()})
                if removed:
                    st.info("Removed duplicate recipients: " + ", ".join(sorted(set(removed))))

                vdoms = [d.lower() for d in st.session_state.config.get("verified_domains", [])]
                from_dom = (from_email.split("@",1)[-1] or "").lower()

                if st.button("📧 Send Email", key="pp_send_email_v7"):
                    if not api_key_present:
                        st.error("SENDGRID_API_KEY missing.")
                    elif not [t for t in uniq_to if t.get("email")]:
                        st.error("Pick at least one recipient with an email or type ad-hoc emails.")
                    elif st.session_state.config.get("require_verified_from", True) and (from_dom not in vdoms):
                        st.error(f"From '{from_email}' is not on verified domains {vdoms}.")
                    else:
                        ok, msg, headers = send_email_sendgrid(
                            from_email, [t for t in uniq_to if t.get("email")], subj, body,
                            attachments=attachments, reply_to=reply_to
                        )
                        if ok:
                            st.success("Sent. Check SendGrid activity if not received.")
                            mid = (headers or {}).get("X-Message-Id") or (headers or {}).get("x-message-id")
                            if mid:
                                st.info(f"SendGrid X-Message-Id: `{mid}`")
                        else:
                            st.error(f"Email failed: {msg}")

    # ---------- Close-out panel (scoped to this subtab) ----------
    if st.session_state.get("pp_show_closeout") and st.session_state.get("pp_closeout_for"):
        jid_for = st.session_state.get("pp_closeout_for")
        job = load_job(jid_for) or {}
        if job:
            st.info("Complete the close-out details below (appears only after **Complete W/O**). Then press **Finalize Completion**.")
            with st.container(border=True):
                outcome_done = st.checkbox("Job Done", value=True, key=f"co_done_{jid_for}_{PP_VER}")
                if not outcome_done:
                    reason = st.selectbox("Job not done — reason", options=["Job Cancelled","Duplicate Job","Other (specify)"], key=f"co_reason_{jid_for}_{PP_VER}")
                    reason_notes = st.text_area("Specify reason", height=80, key=f"co_reason_notes_{jid_for}_{PP_VER}") if reason == "Other (specify)" else ""
                    if st.button("🟥 Finalize — Not Done", key=f"co_finalize_notdone_{jid_for}_{PP_VER}"):
                        persisted = load_job(jid_for) or {}
                        persisted["closeout"] = {"done": False, "reason": reason, "reason_notes": reason_notes, "finished_at": _today_str()}
                        save_job(persisted, persist_id=jid_for)
                        set_status(jid_for, "Closed – Not Done")
                        st.success("Job closed as Not Done.")
                        st.session_state.pp_show_closeout = False
                        st.session_state.pp_closeout_for = ""
                        st.session_state.pp_wip = _new_wip_from_scratch()
                        st.rerun()
                    st.stop()

                reqd = _pdate(job.get("required_end_date", "")) or date.today()
                start_default = _pdate(job.get("planned_start", "")) or reqd
                finish_default = reqd

                row1 = st.columns(3)
                co_start = row1[0].date_input("Start date", value=start_default, key=f"co_start_{jid_for}_{PP_VER}")
                co_finish = row1[1].date_input("Finish/closed date (Date of Maintenance)", value=finish_default, key=f"co_finish_{jid_for}_{PP_VER}")
                maint_code = row1[2].text_input("Maintenance Code", value=job.get("pm_code",""), key=f"co_mcode_{jid_for}_{PP_VER}")

                row2 = st.columns(4)
                labour_hours = row2[0].number_input("Labour Hours (actual)", min_value=0.0, step=0.5, value=0.0, key=f"co_labhrs_{jid_for}_{PP_VER}")
                downtime_hours = row2[1].number_input("Asset Downtime Hours", min_value=0.0, step=0.5, value=0.0, key=f"co_dthrs_{jid_for}_{PP_VER}")
                hours_since = row2[2].number_input("Hours Run Since Last Repair", min_value=0.0, step=1.0, value=0.0, key=f"co_hrssince_{jid_for}_{PP_VER}")
                reset_asset_major = row2[3].checkbox("Reset Asset ‘Last Major Maintenance’", value=False, key=f"co_reset_asset_{jid_for}_{PP_VER}")

                st.markdown("**Spares actually used**")
                picked = job.get("spares", [])
                if not picked:
                    st.caption("No spares were picked. You can still close-out with no spares.")
                    used_df = pd.DataFrame(columns=["cmms","oem","desc","picked_qty","used_qty","replaced"])
                else:
                    base_df = pd.DataFrame(picked)
                    base_df["picked_qty"] = pd.to_numeric(base_df["qty"], errors="coerce").fillna(0).astype(int)
                    used_df = base_df[["cmms","oem","desc","picked_qty"]].copy()
                    used_df["used_qty"] = used_df["picked_qty"]
                    used_df["replaced"] = False
                used_edit = st.data_editor(
                    used_df,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "used_qty": st.column_config.NumberColumn("Used QTY", min_value=0, step=1),
                        "replaced": st.column_config.CheckboxColumn("Replaced (reset life)", default=False),
                        "picked_qty": st.column_config.NumberColumn("Picked QTY", disabled=True),
                        "cmms": st.column_config.TextColumn("CMMS", disabled=True),
                        "oem": st.column_config.TextColumn("OEM/Part", disabled=True),
                        "desc": st.column_config.TextColumn("Description", disabled=True)
                    },
                    key=f"co_used_editor_{jid_for}_{PP_VER}",
                    height=_auto_table_height(used_df)
                )

                notes = st.text_area("Notes & Findings", height=120, key=f"co_notes_{jid_for}_{PP_VER}")

                co_dir = os.path.join(ATTACH_DIR, "closeout", jid_for)
                os.makedirs(co_dir, exist_ok=True)
                co_files = st.file_uploader(
                    "Upload close-out attachments (scanned job card, RA, pictures)",
                    type=["pdf","doc","docx","xls","xlsx","png","jpg","jpeg"],
                    accept_multiple_files=True,
                    key=f"co_uploader_{jid_for}_{PP_VER}"
                )
                saved_co_paths = []
                if co_files:
                    for f in co_files:
                        p = os.path.join(co_dir, f.name)
                        with open(p, "wb") as out:
                            out.write(f.read())
                if os.path.isdir(co_dir):
                    for nm in sorted(os.listdir(co_dir)):
                        saved_co_paths.append(os.path.join(co_dir, nm))
                if saved_co_paths:
                    st.caption("Close-out attachments:")
                    for p in saved_co_paths:
                        st.write("📎", os.path.basename(p))

                if st.button("🟩 Finalize Completion", key=f"co_finalize_{jid_for}_{PP_VER}"):
                    persisted = load_job(jid_for) or {}
                    closeout = {
                        "done": True,
                        "start_date": co_start.strftime("%Y-%m-%d"),
                        "finish_date": co_finish.strftime("%Y-%m-%d"),
                        "maintenance_code": maint_code,
                        "labour_hours": float(labour_hours),
                        "downtime_hours": float(downtime_hours),
                        "hours_run_since": float(hours_since),
                        "notes": notes,
                        "attachments": saved_co_paths
                    }
                    persisted["closeout"] = closeout
                    ok, _ = save_job(persisted, persist_id=jid_for)
                    if not ok:
                        st.error("Failed to persist close-out before completing.")
                        st.stop()

                    used_lines = []
                    if isinstance(used_edit, pd.DataFrame) and not used_edit.empty:
                        for _, r in used_edit.iterrows():
                            used_lines.append({
                                "cmms": str(r.get("cmms","")),
                                "desc": str(r.get("desc","")),
                                "used_qty": int(to_num(r.get("used_qty", 0))),
                                "replaced": bool(r.get("replaced", False))
                            })

                    _append_history_rows_csv_and_sqlite(persisted, closeout, used_lines)

                    if bool(reset_asset_major) and job.get("asset_tag"):
                        _reset_runtime_asset(job["asset_tag"], closeout["finish_date"])
                    if job.get("asset_tag") and used_lines:
                        _reset_runtime_components(job["asset_tag"], used_lines, closeout["finish_date"])

                    ok, msg = mark_completed(jid_for, completed_by=job.get("creator",""))
                    if ok:
                        st.success("Job marked Completed and locked. Maintenance History updated. Runtime synced.")
                        st.session_state.pp_show_closeout = False
                        st.session_state.pp_closeout_for = ""
                        st.session_state.pp_wip = _new_wip_from_scratch()
                        st.rerun()
                    else:
                        st.error(msg)

# ======================= End Planning Pack (sub_pack) — FULL BLOCK =======================

# ======================= Jobs Manager (sub_manager, SIMPLE + EMAIL) — FULL DROP-IN =======================
# Replace your current Subtab 3 block with this whole code.

# ======================= Jobs Manager (Subtab 3) — SIMPLE + EMAIL (close-on-send) =======================
# Replace your entire Subtab 3 block (the whole `with sub_manager:` section) with this.

with sub_jobs:
    import os, json, math, base64 as _b64mod
    from datetime import datetime, date, timedelta
    from collections import defaultdict
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    JM_VER = "v4"  # bump keys

    # ---------- session helpers ----------
    if "config" not in st.session_state:
        st.session_state.config = {}
    if "jm_email_job_id" not in st.session_state:
        st.session_state.jm_email_job_id = ""

    assets = st.session_state.get("assets", {}) or {}

    # ---------- small utils ----------
    def jm_pdate(s):
        if "_pdate" in globals():  # reuse if available
            return _pdate(s)
        if not s: return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try: return datetime.strptime(str(s).replace("Z",""), fmt).date()
            except Exception: pass
        try: return datetime.fromisoformat(str(s).replace("Z","")).date()
        except Exception: return None

    def jm_today() -> date:
        return date.today()

    def jm_iso_week_key(d: date):
        if not d: return "—"
        y, w, _ = d.isocalendar()
        return f"{y}-W{w:02d}"

    def jm_safe_list(x):
        if isinstance(x, list): return x
        if isinstance(x, str):
            try: return json.loads(x)
            except Exception: return []
        return []

    def jm_age_badge(req_end):
        d = jm_pdate(req_end)
        if not d: return "—"
        delta = (jm_today() - d).days
        return "D-0" if delta == 0 else f"D{('+' if delta>0 else '')}{delta}"

    def _b64enc(b: bytes) -> str:
        return _b64mod.b64encode(b).decode("ascii")

    # ---------- fresh loader (JSON wins; fixes Completed showing under Submitted) ----------
    def jm_load_all_jobs() -> list[dict]:
        if "_read_index" not in globals():
            return []
        try:
            idx = _read_index()
        except Exception:
            return []
        jobs_meta = idx.get("jobs", {}) if isinstance(idx.get("jobs", {}), dict) else {}
        out: list[dict] = []
        for jid, meta in (jobs_meta or {}).items():
            try:
                job = load_job(jid) if "load_job" in globals() else None
            except Exception:
                job = None
            job = job or {}
            merged = dict(meta or {})
            merged.update(job or {})  # JSON wins
            merged["id"] = merged.get("persist_id") or merged.get("id") or jid
            merged["archived"] = bool(merged.get("archived", False))
            merged["trashed"]  = bool(merged.get("trashed", False))
            merged["status"] = (job.get("status") or meta.get("status") or "Draft")
            out.append(merged)
        return out

    # ---------- per-bucket ----------
    def jm_rows_for_bucket(bucket: str, rows: list[dict]):
        out=[]
        for j in rows:
            s = j.get("status","")
            t = bool(j.get("trashed", False))
            if bucket=="Drafts"    and s=="Draft" and not t: out.append(j)
            if bucket=="Submitted" and s=="Submitted" and not t: out.append(j)
            if bucket=="Completed" and s in ("Completed","Closed – Date Passed") and not t: out.append(j)
            if bucket=="Trash"     and t: out.append(j)
        return out

    # ---------- filters (simple) ----------
    def jm_apply_simple_filters(rows: list[dict], search_text: str, assets_sel: list[str]):
        q = (search_text or "").strip().lower()
        aset = set(assets_sel or [])
        out=[]
        for r in rows:
            if aset and r.get("asset_tag") not in aset:
                continue
            if q:
                hay = " ".join([
                    str(r.get("id","")), str(r.get("wo_number","")), str(r.get("asset_tag","")),
                    str(r.get("pm_code","")), str(r.get("creator","")), str(r.get("task_description",""))
                ]).lower()
                if q not in hay:
                    continue
            out.append(r)
        return out

    # ---------- dataframes ----------
    def jm_kitting_pct(job: dict) -> float:
        kr = job.get("kitting_readiness", {}) or {}
        if isinstance(kr, str):
            try: kr = json.loads(kr)
            except Exception: kr = {}
        try: return float(kr.get("pct_ok", 0.0) or 0.0)
        except Exception: return 0.0

    def jm_build_df(bucket: str, rows: list[dict]):
        if bucket == "Drafts":
            cols = ["Select","Job ID","WO #","Asset","PM","Priority","Planned Start","Required End","Age","Creator","Last Updated"]
            data=[]
            for r in rows:
                data.append({
                    "Select": False,
                    "Job ID": r.get("id",""),
                    "WO #": r.get("wo_number",""),
                    "Asset": r.get("asset_tag",""),
                    "PM": r.get("pm_code",""),
                    "Priority": r.get("priority",""),
                    "Planned Start": r.get("planned_start","") or "—",
                    "Required End": r.get("required_end_date","") or "—",
                    "Age": jm_age_badge(r.get("required_end_date")),
                    "Creator": r.get("creator","") or "—",
                    "Last Updated": (str(r.get("last_updated",""))[:19].replace("T"," ") or "—"),
                })
            return cols, pd.DataFrame(data)

        if bucket == "Submitted":
            cols = ["Select","Job ID","WO #","Asset","PM","Priority","Planned Start","Required End","Age","Permits","Kitting %","Creator"]
            data=[]
            for r in rows:
                permits = (r.get("logistics",{}) or {}).get("permits",[])
                if isinstance(permits, str):
                    try: permits = json.loads(permits)
                    except Exception: permits = []
                data.append({
                    "Select": False,
                    "Job ID": r.get("id",""),
                    "WO #": r.get("wo_number",""),
                    "Asset": r.get("asset_tag",""),
                    "PM": r.get("pm_code",""),
                    "Priority": r.get("priority",""),
                    "Planned Start": r.get("planned_start","") or "—",
                    "Required End": r.get("required_end_date","") or "—",
                    "Age": jm_age_badge(r.get("required_end_date")),
                    "Permits": ", ".join(permits) if permits else "—",
                    "Kitting %": round(jm_kitting_pct(r), 1),
                    "Creator": r.get("creator","") or "—",
                })
            return cols, pd.DataFrame(data)

        if bucket == "Completed":
            cols = ["Select","Job ID","WO #","Asset","PM","Priority","Planned Start","Finish Date","On-time?","Creator","Last Updated"]
            data=[]
            for r in rows:
                co = r.get("closeout",{}) or {}
                if isinstance(co, str):
                    try: co = json.loads(co)
                    except Exception: co = {}
                fd = co.get("finish_date","") or r.get("finish_date","")
                ps = jm_pdate(r.get("planned_start"))
                ontime = "—"
                if fd and ps:
                    ws = ps - timedelta(days=ps.weekday())
                    we = ws + timedelta(days=6)
                    dfd = jm_pdate(fd)
                    ontime = "Yes" if (dfd and dfd <= we) else "No"
                data.append({
                    "Select": False,
                    "Job ID": r.get("id",""),
                    "WO #": r.get("wo_number",""),
                    "Asset": r.get("asset_tag",""),
                    "PM": r.get("pm_code",""),
                    "Priority": r.get("priority",""),
                    "Planned Start": r.get("planned_start","") or "—",
                    "Finish Date": fd or "—",
                    "On-time?": ontime,
                    "Creator": r.get("creator","") or "—",
                    "Last Updated": (str(r.get("last_updated",""))[:19].replace("T"," ") or "—"),
                })
            return cols, pd.DataFrame(data)

        if bucket == "Trash":
            cols = ["Select","Job ID","WO #","Asset","PM","Priority","Planned Start","Finish Date","Reason","Creator"]
            data=[]
            for r in rows:
                co = r.get("closeout",{}) or {}
                if isinstance(co, str):
                    try: co = json.loads(co)
                    except Exception: co = {}
                fd = co.get("finish_date","") or r.get("finish_date","")
                data.append({
                    "Select": False,
                    "Job ID": r.get("id",""),
                    "WO #": r.get("wo_number",""),
                    "Asset": r.get("asset_tag",""),
                    "PM": r.get("pm_code",""),
                    "Priority": r.get("priority",""),
                    "Planned Start": r.get("planned_start","") or "—",
                    "Finish Date": fd or "—",
                    "Reason": r.get("close_reason","") or "—",
                    "Creator": r.get("creator","") or "—",
                })
            return cols, pd.DataFrame(data)

        return [], pd.DataFrame()

    # ---------- actions ----------
    def jm_submit_job(jid: str):
        if "set_status" in globals():
            ok = set_status(jid, "Submitted")
            return ok, ("Submitted." if ok else "Submit failed.")
        return False, "set_status() not available."

    def jm_duplicate_to_wip(jid: str):
        base = load_job(jid) if "load_job" in globals() else None
        if not base: return False, "Job not found."
        if "_new_wip_from_scratch" in globals():
            st.session_state.pp_wip = _new_wip_from_scratch()
            for k, v in base.items():
                if k == "id": continue
                st.session_state.pp_wip[k] = v
            st.session_state.pp_wip["persist_id"] = ""
            return True, "Duplicated into WIP."
        return False, "Planner WIP function not available."

    def jm_delete_job_permanently(jid: str):
        try:
            if "_job_path" in globals():
                path = _job_path(jid)
                if path and os.path.exists(path):
                    os.remove(path)
            if "_read_index" in globals() and "_write_index" in globals():
                idx = _read_index()
                if "jobs" in idx and jid in idx["jobs"]:
                    idx["jobs"].pop(jid, None)
                    _write_index(idx)
            return True
        except Exception:
            return False

    def jm_set_trashed(jid: str, flag: bool):
        job = load_job(jid) if "load_job" in globals() else None
        if not job: return False, "Job not found."
        job["trashed"] = bool(flag)
        if "save_job" in globals():
            ok, _ = save_job(job, persist_id=jid)
            return ok, ("Moved to Trash." if flag else "Restored.")
        return False, "save_job() not available."

    def jm_open_in_planner(jid: str):
        if "_wip_from_persisted" in globals():
            st.session_state.pp_wip = _wip_from_persisted(jid)
        else:
            st.session_state.pp_wip = load_job(jid)
        st.success("Opened in Planner Pack (Sub-tab 2).")

    def jm_open_closeout(jid: str):
        st.session_state["pp_show_closeout"] = True
        st.session_state["pp_closeout_for"] = jid
        st.success("Close-out opened in Planner (Sub-tab 2).")

    def jm_generate_jobcard(jid: str):
        jobx = load_job(jid) if "load_job" in globals() else None
        if not jobx:
            st.error("Job not found."); return
        html_str = (
            render_jobcard_html(jobx, assets, currency=jobx.get("currency_symbol","R"), fx_rate=1.0)
            if "render_jobcard_html" in globals() else "<html><body><p>No template.</p></body></html>"
        )
        fname = (jobx.get("wo_number") or jobx.get("id") or "job_card")
        st.download_button("⬇️ Job Card (HTML)", data=html_str.encode("utf-8"),
                           file_name=f"{fname}.html", mime="text/html", key=f"jm_dl_html_{jid}_{JM_VER}")
        if "generate_job_card_pdf" in globals():
            ok_pdf, path, msg = generate_job_card_pdf(jobx, assets, currency=jobx.get("currency_symbol","R"), fx_rate=1.0)
            if os.path.isfile(path):
                try:
                    with open(path, "rb") as fp:
                        st.download_button("⬇️ Job Card (PDF)" if ok_pdf else "⬇️ Job Card (HTML file)",
                                           data=fp.read(),
                                           file_name=os.path.basename(path),
                                           mime="application/pdf" if ok_pdf and path.lower().endswith(".pdf") else "text/html",
                                           key=f"jm_dl_pdf_{jid}_{JM_VER}")
                except Exception: pass
            st.caption(msg)

    # ---------- EMAIL panel (closes on send/cancel; ensures PDF attach) ----------
    def jm_email_panel(jid: str):
        import re as _re
        jobx = load_job(jid) if "load_job" in globals() else None
        if not jobx:
            st.error("Job not found.")
            return

        st.markdown("### 📧 Send Job Card")

        # From / Reply-To
        api_key_present = bool(_get_sendgrid_key()) if "_get_sendgrid_key" in globals() else False
        from_default = st.session_state.config.get("sendgrid_from", "planner@em7701.maraksreliability.com")
        from_email_input = st.text_input("From (verified domain)", value=from_default, key=f"jm_from_{jid}_{JM_VER}")
        reply_to_input = st.text_input("Reply-To (optional)", value=st.session_state.config.get("sendgrid_reply_to", ""), key=f"jm_reply_{jid}_{JM_VER}")
        st.session_state.config["sendgrid_from"] = from_email_input
        st.session_state.config["sendgrid_reply_to"] = reply_to_input
        if "save_config" in globals():
            save_config(st.session_state.config)

        # Recipients
        ab = st.session_state.config.get("address_book", []) or []
        if not isinstance(ab, list): ab = []
        ab_labels = []
        for c in ab:
            nm = (c.get("name","") or "").strip()
            em = (c.get("email","") or "").strip()
            if em and nm: ab_labels.append(f"{nm} <{em}>")
            elif em:      ab_labels.append(em)
            elif nm:      ab_labels.append(nm)

        pick = st.multiselect("Recipients (Address Book)", options=ab_labels, key=f"jm_mail_pick_{jid}_{JM_VER}")
        adhoc = st.text_input("Additional recipients (comma-separated emails)", key=f"jm_mail_adhoc_{jid}_{JM_VER}", placeholder="alice@x.com, bob@y.org")

        # Subject + Body
        subj = f"Works Order: {jobx.get('wo_number','')} • {jobx.get('asset_tag','')} • Due {jobx.get('required_end_date','')}"
        body_default = (
            f"<p>Hi team,</p>"
            f"<p>Please find the Job Card attached for <b>{jobx.get('asset_tag','')}</b>"
            f" (WO: <b>{jobx.get('wo_number','')}</b>).</p>"
            f"<p>Regards,<br/>Reliability App</p>"
        )
        body = st.text_area("Message", value=body_default, height=140, key=f"jm_mail_body_{jid}_{JM_VER}")

        # HTML blob (always)
        try:
            html_blob = (
                render_jobcard_html(
                    jobx, assets, currency=jobx.get("currency_symbol","R"), fx_rate=1.0
                ).encode("utf-8")
                if "render_jobcard_html" in globals()
                else f"<html><body><h3>WO {jobx.get('wo_number','')}</h3></body></html>".encode("utf-8")
            )
        except Exception:
            html_blob = f"<html><body><h3>WO {jobx.get('wo_number','')}</h3></body></html>".encode("utf-8")

        # PDF: use existing if valid; else generate once and persist
        candidate_pdf = str(jobx.get("pdf_path","") or "")
        pdf_available = bool(candidate_pdf and candidate_pdf.lower().endswith(".pdf") and os.path.isfile(candidate_pdf))
        if not pdf_available and "generate_job_card_pdf" in globals():
            try:
                ok_pdf, gen_path, _msg = generate_job_card_pdf(
                    jobx, assets, currency=jobx.get("currency_symbol","R"), fx_rate=1.0
                )
                if ok_pdf and gen_path and gen_path.lower().endswith(".pdf") and os.path.isfile(gen_path):
                    candidate_pdf = gen_path
                    pdf_available = True
                    jobx["pdf_path"] = gen_path
                    if "save_job" in globals():
                        save_job(jobx, persist_id=jid)
            except Exception:
                pdf_available = False

        # Attach toggles
        colA, colB = st.columns(2)
        with colA:
            include_html = st.checkbox("Attach Job Card (HTML)", value=True, key=f"jm_att_html_{jid}_{JM_VER}")
        with colB:
            include_pdf = st.checkbox("Attach Job Card (PDF)", value=pdf_available, key=f"jm_att_pdf_{jid}_{JM_VER}")

        # Additional attachments
        att_candidates = []
        for p in (jobx.get("attachments") or []):
            if p and os.path.isfile(p):
                att_candidates.append(p)
        if att_candidates:
            sel_atts = st.multiselect(
                "Select additional attachments to include",
                options=[os.path.basename(p) for p in att_candidates],
                default=[],
                key=f"jm_extra_atts_{jid}_{JM_VER}"
            )
        else:
            sel_atts = []

        # Assemble attachments
        attachments = []
        base_name = (jobx.get("persist_id") or jobx.get("id") or "job")
        if include_html:
            attachments.append({
                "filename": f"{base_name}.html",
                "type": "text/html",
                "content": _b64enc(html_blob),
            })
        if include_pdf and pdf_available and candidate_pdf and os.path.isfile(candidate_pdf):
            try:
                with open(candidate_pdf, "rb") as fp:
                    attachments.append({
                        "filename": os.path.basename(candidate_pdf),
                        "type": "application/pdf",
                        "content": _b64enc(fp.read()),
                    })
            except Exception:
                st.warning("Could not read PDF file; skipping.")
        for nm in (sel_atts or []):
            p = next((x for x in att_candidates if os.path.basename(x) == nm), "")
            if not p or not os.path.isfile(p): continue
            mime = "application/octet-stream"
            pl = p.lower()
            if   pl.endswith(".pdf"):  mime = "application/pdf"
            elif pl.endswith(".csv"):  mime = "text/csv"
            elif pl.endswith(".xlsx"): mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif pl.endswith(".xls"):  mime = "application/vnd.ms-excel"
            elif pl.endswith(".png"):  mime = "image/png"
            elif pl.endswith(".jpg") or pl.endswith(".jpeg"): mime = "image/jpeg"
            try:
                with open(p, "rb") as fh:
                    attachments.append({
                        "filename": os.path.basename(p),
                        "type": mime,
                        "content": _b64enc(fh.read()),
                    })
            except Exception:
                pass

        # Recipients parsing
        email_re = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        def _parse_label(lbl: str):
            s = (lbl or "").strip()
            m = _re.match(r"^(.*)\s*<([^>]+)>$", s)
            if m: return {"name": m.group(1).strip(), "email": m.group(2).strip()}
            if _re.match(email_re, s): return {"name": "", "email": s}
            return {"name": s, "email": ""}

        to_list = [_parse_label(x) for x in (pick or []) if x]
        if adhoc:
            for e in [p.strip() for p in adhoc.split(",") if p.strip()]:
                if _re.match(email_re, e):
                    to_list.append({"name": "", "email": e})

        seen = set(); uniq = []
        for t in to_list:
            em = (t.get("email") or "").strip().lower()
            if not em or em in seen: continue
            seen.add(em); uniq.append({"name": t.get("name",""), "email": em})

        # Verified-domain guard
        from_email = st.session_state.config.get("sendgrid_from", "planner@em7701.maraksreliability.com")
        reply_to   = st.session_state.config.get("sendgrid_reply_to", "")
        vdoms = [d.lower() for d in st.session_state.config.get("verified_domains", [])]
        from_dom = (from_email.split("@",1)[-1] or "").lower()
        require_verified = bool(st.session_state.config.get("require_verified_from", True))

        # Send / Cancel
        sc1, sc2, sc3 = st.columns([1,2,2])
        with sc1:
            do_send = st.button("📨 Send Email", key=f"jm_send_{jid}_{JM_VER}")
        with sc2:
            st.caption(f"Attachments queued: {len(attachments)}")
        with sc3:
            if st.button("✖ Cancel", key=f"jm_cancel_{jid}_{JM_VER}"):
                st.session_state.jm_email_job_id = ""   # close panel
                st.rerun()

        if do_send:
            if not api_key_present:
                st.error("SENDGRID_API_KEY missing in environment/config."); return
            if not uniq:
                st.error("Pick at least one valid recipient."); return
            if require_verified and vdoms and (from_dom not in vdoms):
                st.error(f"From '{from_email}' is not in verified domains {vdoms}."); return
            if "send_email_sendgrid" not in globals():
                st.error("send_email_sendgrid() not available."); return

            ok, msg, headers = send_email_sendgrid(
                from_email=from_email,
                to_list=uniq,
                subject=subj,
                html_body=body,
                attachments=attachments,
                reply_to=reply_to
            )
            if ok:
                st.success("Sent.")
                st.session_state.jm_email_job_id = ""   # close panel after success
                st.rerun()
            else:
                st.error(f"Email failed: {msg}")

    # ---------- compact viewer ----------
    def jm_render_viewer(job_id: str):
        job = load_job(job_id) if "load_job" in globals() else None
        if not job:
            st.warning("Job not found."); return
        st.markdown(f"#### 🗂️ {job.get('id', job_id)}")
        cols = st.columns(3)
        hdr = [
            ("Status", job.get("status","")), ("WO #", job.get("wo_number","")), ("Priority", job.get("priority","")),
            ("Asset", job.get("asset_tag","")), ("PM", job.get("pm_code","")), ("Required End", job.get("required_end_date","")),
        ]
        for i,(k,v) in enumerate(hdr):
            with cols[i%3]: st.metric(k, v if v else "—")
        st.caption("Task"); st.write(job.get("task_description","—"))

    # ---------- minimal analytics ----------
    def jm_analytics_block(rows: list[dict], key_base: str):
        if not rows:
            st.info("No data for analytics."); return
        r1 = st.columns(2)
        with r1[0]:
            sc = defaultdict(int)
            for r in rows: sc[r.get("status","")] += 1
            labels = list(sc.keys()) if sc else ["Draft","Submitted","Completed"]
            values = [sc.get(k,0) for k in labels]
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.45, textinfo="label+percent")])
            fig.update_layout(title="Statuses", height=300, margin=dict(l=10,r=10,t=36,b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig, use_container_width=True, key=f"{key_base}_status")
        with r1[1]:
            wk = defaultdict(lambda: defaultdict(int))
            for r in rows:
                re = jm_pdate(r.get("required_end_date"))
                if re: wk[jm_iso_week_key(re)][r.get("priority","") or "—"] += 1
            weeks = sorted(wk.keys())
            fig2 = go.Figure()
            prios = sorted({p for d in wk.values() for p in d.keys()})
            for p in prios:
                fig2.add_trace(go.Bar(name=p, x=weeks, y=[wk[w].get(p,0) for w in weeks]))
            fig2.update_layout(barmode="stack", title="Priority by Required Week", height=300,
                               margin=dict(l=10,r=10,t=36,b=10),
                               legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_base}_prio_week")

    # ---------- bucket UI ----------
    def jm_bucket_ui(bucket_name: str, container):
        with container:
            # Filters (expander): Search + Asset Tags
            with st.expander("Filters", expanded=False):
                f1, f2 = st.columns([2,2])
                with f1:
                    search = st.text_input("Search (Job/WO/Asset/PM/Creator/Desc)", key=f"jm_q_{bucket_name}_{JM_VER}")
                with f2:
                    all_assets = sorted({r.get("asset_tag","") for r in jm_load_all_jobs() if r.get("asset_tag")})
                    filt_assets = st.multiselect("Asset Tags", options=all_assets, key=f"jm_assets_{bucket_name}_{JM_VER}")

            # Data
            rows_bucket = jm_rows_for_bucket(bucket_name, jm_load_all_jobs())
            rows = jm_apply_simple_filters(rows_bucket, search, filt_assets)
            cols, df = jm_build_df(bucket_name, rows)

            # Pagination
            left, right = st.columns([1,1])
            with left:
                page_size = st.selectbox("Rows per page", [20,50,100,200], index=1, key=f"{bucket_name}_pagesize_{JM_VER}")
            total = len(df)
            pages = max(1, math.ceil(total / max(page_size,1)))
            with right:
                page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=f"{bucket_name}_page_{JM_VER}")
            start = (page_idx-1) * max(page_size,1)
            end = start + max(page_size,1)
            df_page = df.iloc[start:end].copy() if not df.empty else df

            # Table
            ed = st.data_editor(
                df_page, hide_index=True, use_container_width=True,
                column_config={"Select": st.column_config.CheckboxColumn(required=False)},
                key=f"{bucket_name}_table_{page_idx}_{JM_VER}"
            )

            # Select one job + actions
            ids_on_page = ed["Job ID"].tolist() if not ed.empty else []
            sel = st.selectbox("Select one job", options=[""] + ids_on_page, key=f"{bucket_name}_pick_{page_idx}_{JM_VER}")

            a = st.columns(5)
            if bucket_name=="Drafts":
                if a[0].button("👁 View", disabled=(sel==""), key=f"v_{bucket_name}_{page_idx}_{JM_VER}"): jm_render_viewer(sel)
                if a[1].button("✏ Edit in Planner", disabled=(sel==""), key=f"e_{bucket_name}_{page_idx}_{JM_VER}"): jm_open_in_planner(sel)
                if a[2].button("📤 Submit", disabled=(sel==""), key=f"s_{bucket_name}_{page_idx}_{JM_VER}"):
                    ok,msg = jm_submit_job(sel); (st.success if ok else st.error)(msg); st.rerun()
                if a[3].button("🧬 Duplicate WIP", disabled=(sel==""), key=f"d_{bucket_name}_{page_idx}_{JM_VER}"):
                    ok,msg = jm_duplicate_to_wip(sel); (st.success if ok else st.error)(msg)
                if a[4].button("🗑 Delete Permanently", disabled=(sel==""), key=f"del_{bucket_name}_{page_idx}_{JM_VER}"):
                    job = load_job(sel) if "load_job" in globals() else {}
                    if job.get("status")!="Draft": st.error("Only Drafts can be deleted permanently.")
                    else:
                        ok = jm_delete_job_permanently(sel)
                        (st.success if ok else st.error)("Deleted." if ok else "Delete failed.")
                        st.rerun()

            elif bucket_name=="Submitted":
                if a[0].button("👁 View", disabled=(sel==""), key=f"v_{bucket_name}_{page_idx}_{JM_VER}"): jm_render_viewer(sel)
                if a[1].button("✏ Edit in Planner", disabled=(sel==""), key=f"e_{bucket_name}_{page_idx}_{JM_VER}"): jm_open_in_planner(sel)
                if a[2].button("✅ Open Close-Out", disabled=(sel==""), key=f"co_{bucket_name}_{page_idx}_{JM_VER}"): jm_open_closeout(sel)
                if a[3].button("🖨 Generate Job Card", disabled=(sel==""), key=f"pdf_{bucket_name}_{page_idx}_{JM_VER}"): jm_generate_jobcard(sel)
                if a[4].button("📧 Email Job Card", disabled=(sel==""), key=f"mail_{bucket_name}_{page_idx}_{JM_VER}"):
                    st.session_state.jm_email_job_id = sel
                if st.session_state.get("jm_email_job_id") and st.session_state["jm_email_job_id"] in ids_on_page:
                    with st.container(border=True):
                        jm_email_panel(st.session_state["jm_email_job_id"])

            elif bucket_name=="Completed":
                if a[0].button("👁 View", disabled=(sel==""), key=f"v_{bucket_name}_{page_idx}_{JM_VER}"): jm_render_viewer(sel)
                if a[1].button("🧬 Duplicate WIP", disabled=(sel==""), key=f"d_{bucket_name}_{page_idx}_{JM_VER}"):
                    ok,msg = jm_duplicate_to_wip(sel); (st.success if ok else st.error)(msg)
                if a[2].button("🗃 Move to Trash", disabled=(sel==""), key=f"trash_{bucket_name}_{page_idx}_{JM_VER}"):
                    ok,msg = jm_set_trashed(sel, True); (st.success if ok else st.error)(msg); st.rerun()
                if a[3].button("🖨 Generate Job Card", disabled=(sel==""), key=f"pdf_{bucket_name}_{page_idx}_{JM_VER}"): jm_generate_jobcard(sel)
                if a[4].button("📧 Email Job Card", disabled=(sel==""), key=f"mail_{bucket_name}_{page_idx}_{JM_VER}"):
                    st.session_state.jm_email_job_id = sel
                if st.session_state.get("jm_email_job_id") and st.session_state["jm_email_job_id"] in ids_on_page:
                    with st.container(border=True):
                        jm_email_panel(st.session_state["jm_email_job_id"])

            elif bucket_name=="Trash":
                if a[0].button("👁 View", disabled=(sel==""), key=f"v_{bucket_name}_{page_idx}_{JM_VER}"): jm_render_viewer(sel)
                if a[1].button("↩ Restore", disabled=(sel==""), key=f"restore_{bucket_name}_{page_idx}_{JM_VER}"):
                    ok,msg = jm_set_trashed(sel, False); (st.success if ok else st.error)(msg); st.rerun()
                a[2].button("—", disabled=True, key=f"noop3_{bucket_name}_{page_idx}_{JM_VER}")
                a[3].button("—", disabled=True, key=f"noop4_{bucket_name}_{page_idx}_{JM_VER}")
                a[4].button("—", disabled=True, key=f"noop5_{bucket_name}_{page_idx}_{JM_VER}")

            # Analytics
            st.markdown("---")
            st.markdown("### Analytics")
            jm_analytics_block(rows, key_base=f"an_{bucket_name.lower()}_{page_idx}_{JM_VER}")

    # ---------- tabs ----------
    tab_d, tab_s, tab_c, tab_t = st.tabs(["📝 Drafts", "📤 Submitted", "✅ Completed", "🗑 Trash"])
    jm_bucket_ui("Drafts", tab_d)
    jm_bucket_ui("Submitted", tab_s)
    jm_bucket_ui("Completed", tab_c)
    jm_bucket_ui("Trash", tab_t)

# =========================
# TAB 3 — Maintenance Records (scaffold)
# =========================

with tab3:
    import os, json, sqlite3, math, io, base64 as _b64
    from datetime import date, datetime, timedelta
    from collections import defaultdict
    import pandas as pd
    import streamlit as st
    import plotly.graph_objects as go

    MR_VER = "v1"

    # ---------- Paths (use existing if defined) ----------
    DATA_DIR   = globals().get("DATA_DIR", "data")
    ATTACH_DIR = globals().get("ATTACH_DIR", os.path.join(DATA_DIR, "attachments"))
    HIST_CSV   = globals().get("HIST_CSV", os.path.join(DATA_DIR, "history.csv"))
    RELIAB_DB  = globals().get("RELIAB_DB", os.path.join(DATA_DIR, "reliability.db"))
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(ATTACH_DIR, exist_ok=True)

    # ---------- Helpers ----------
    def _pdate(s):
        if not s: return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try: return datetime.strptime(str(s).replace("Z",""), fmt).date()
            except Exception: pass
        try: return datetime.fromisoformat(str(s).replace("Z","")).date()
        except Exception: return None

    def _today(): return date.today()

    def _sanitize(name: str) -> str:
        s = "".join(ch if ch.isalnum() or ch in ("-","_",".") else "_" for ch in str(name))
        return s[:120] if len(s) > 120 else s

    def _attachments_count(sval):
        if not sval: return 0
        if isinstance(sval, list): return len([x for x in sval if x])
        parts = [p.strip() for p in str(sval).split(";") if p.strip()]
        return len(parts)

    def _auto_table_height(df: pd.DataFrame, min_h=140, row_h=32, max_h=480):
        try: rows = int(df.shape[0])
        except Exception: rows = 0
        h = min_h + rows * row_h
        return int(max(min(h, max_h), min_h))

    # Use provided BOM builder if available
    if "build_bom_table" not in globals():
        def build_bom_table(asset: dict) -> pd.DataFrame:
            return pd.DataFrame(columns=["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE"])

    # ---------- SQLite adapters ----------
    def _mh_init():
        with sqlite3.connect(RELIAB_DB) as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_history(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_tag TEXT,
                functional_location TEXT,
                wo_number TEXT,
                date_of_maintenance TEXT,
                maintenance_code TEXT,
                spare_cmms TEXT,
                spare_desc TEXT,
                qty INTEGER,
                hours_run_since REAL,
                labour_hours REAL,
                downtime_hours REAL,
                notes TEXT,
                attachments_json TEXT,
                created_at TEXT
            )
            """)
            conn.commit()

    def _mh_all(where_sql="", params=()):
        _mh_init()
        with sqlite3.connect(RELIAB_DB) as conn:
            conn.row_factory = sqlite3.Row
            q = "SELECT * FROM maintenance_history"
            if where_sql: q += " WHERE " + where_sql
            q += " ORDER BY date_of_maintenance DESC, id DESC"
            return [dict(r) for r in conn.execute(q, params).fetchall()]

    def _mh_exec(q, p=()):
        _mh_init()
        with sqlite3.connect(RELIAB_DB) as conn:
            conn.execute(q, p)
            conn.commit()

    def _mh_insert_rows(rows: list[dict]):
        """Rows keys: Asset Tag, Functional Location, WO Number, Date of Maintenance, Maintenance Code,
                      Spares Used (cmms or desc), QTY, Hours Run Since Last Repair, Labour Hours,
                      Asset Downtime Hours, Notes and Findings, Attachments (semicolon string or list)"""
        if not rows: return
        _mh_init()
        with sqlite3.connect(RELIAB_DB) as conn:
            cur = conn.cursor()
            for r in rows:
                at  = r.get("Asset Tag","")
                fl  = r.get("Functional Location","")
                wo  = r.get("WO Number","")
                dtm = r.get("Date of Maintenance","")
                mcd = r.get("Maintenance Code","")
                sp  = r.get("Spares Used","") or ""
                sp_cmms = sp if sp and len(str(sp))<=64 and not any(ch in str(sp) for ch in " /") else ""
                sp_desc = r.get("Spares Desc","") or ("" if sp_cmms else str(sp))
                qty = int(float(r.get("QTY", 0) or 0))
                hrs = float(r.get("Hours Run Since Last Repair", 0) or 0.0)
                lab = float(r.get("Labour Hours", 0) or 0.0)
                dwn = float(r.get("Asset Downtime Hours", 0) or 0.0)
                notes = r.get("Notes and Findings","") or ""
                atts = r.get("Attachments","") or []
                if isinstance(atts, str):
                    att_list = [p.strip() for p in atts.split(";") if p.strip()]
                else:
                    att_list = [p for p in atts if p]
                cur.execute("""
                INSERT INTO maintenance_history(
                    asset_tag,functional_location,wo_number,date_of_maintenance,maintenance_code,
                    spare_cmms,spare_desc,qty,hours_run_since,labour_hours,downtime_hours,notes,
                    attachments_json,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    at, fl, wo, dtm, mcd, sp_cmms, sp_desc, qty, hrs, lab, dwn, notes,
                    json.dumps(att_list, ensure_ascii=False), datetime.utcnow().isoformat()+"Z"
                ))
            conn.commit()

    # ---------- CSV mirror helpers ----------
    def _load_csv_df() -> pd.DataFrame:
        if os.path.exists(HIST_CSV):
            try: return pd.read_csv(HIST_CSV)
            except Exception: pass
        cols = ["Number","Asset Tag","Functional Location","WO Number","Date of Maintenance",
                "Maintenance Code","Spares Used","QTY","Hours Run Since Last Repair",
                "Labour Hours","Asset Downtime Hours","Notes and Findings","Attachments"]
        return pd.DataFrame(columns=cols)

    def _save_csv_df(df: pd.DataFrame):
        try: df.to_csv(HIST_CSV, index=False)
        except Exception: pass

    if "history_df" not in st.session_state:
        st.session_state.history_df = _load_csv_df()

    # ---------- Filters ----------
    assets = st.session_state.get("assets", {}) or {}
    all_asset_tags = sorted([k for k in assets.keys() if k])
    code_chips = ["PM00","PM01","PM02","PM03","PM04"]

    if "mr_first_sync_done" not in st.session_state:
        st.session_state.mr_first_sync_done = False

    st.markdown("### Maintenance Records")

    flt1, flt2, flt3, flt4, flt5 = st.columns([1.2, 1.8, 2.2, 1.8, 1.2])
    with flt1:
        date_to = _today()
        date_from = date_to - timedelta(days=90)
        mr_from = st.date_input("From", value=date_from, key=f"mr_from_{MR_VER}")
    with flt2:
        mr_to = st.date_input("To", value=date_to, key=f"mr_to_{MR_VER}")
    with flt3:
        mr_search = st.text_input("Search (WO/Asset/Code/Notes)", key=f"mr_search_{MR_VER}")
    with flt4:
        mr_assets = st.multiselect("Asset Tags", options=all_asset_tags, key=f"mr_assets_{MR_VER}")
    with flt5:
        mr_codes = st.multiselect("Maint Code", options=code_chips, key=f"mr_codes_{MR_VER}")

    # Resync button (CSV -> SQLite)
    with st.expander("Recovery: Resync DB from CSV", expanded=False):
        colr1, colr2 = st.columns([1,3])
        with colr1:
            ok_go = st.checkbox("I understand this will import from CSV", key=f"mr_resync_ck_{MR_VER}")
        with colr2:
            if st.button("Resync now", disabled=not ok_go, key=f"mr_resync_btn_{MR_VER}"):
                dfcsv = _load_csv_df().copy()
                rows=[]
                for _, r in dfcsv.iterrows():
                    rows.append({
                        "Asset Tag": r.get("Asset Tag",""),
                        "Functional Location": r.get("Functional Location",""),
                        "WO Number": r.get("WO Number",""),
                        "Date of Maintenance": r.get("Date of Maintenance",""),
                        "Maintenance Code": r.get("Maintenance Code",""),
                        "Spares Used": r.get("Spares Used",""),
                        "Spares Desc": r.get("Spares Used",""),
                        "QTY": r.get("QTY",0),
                        "Hours Run Since Last Repair": r.get("Hours Run Since Last Repair",0.0),
                        "Labour Hours": r.get("Labour Hours",0.0),
                        "Asset Downtime Hours": r.get("Asset Downtime Hours",0.0),
                        "Notes and Findings": r.get("Notes and Findings",""),
                        "Attachments": r.get("Attachments",""),
                    })
                _mh_insert_rows(rows)
                st.success("Resynced DB from CSV. Reloading…")
                st.session_state.mr_first_sync_done = True
                st.rerun()

    # ---------- Logger (visible only when exactly one asset selected) ----------
    def _logger_key(asset_tag: str) -> str:
        return f"mr_logger_buffer_{asset_tag}"

    def _logger_bom_maps(asset_tag: str):
        asset_data = assets.get(asset_tag, {})
        bom = build_bom_table(asset_data)
        sp_map = {}
        if isinstance(bom, pd.DataFrame) and not bom.empty and "CMMS MATERIAL CODE" in bom.columns:
            for _, row in bom.iterrows():
                cmms = str(row.get("CMMS MATERIAL CODE","")).strip()
                desc = str(row.get("DESCRIPTION","")).strip()
                if cmms:
                    sp_map[cmms] = f"{cmms} - {desc}" if desc else cmms
        rev_map = {v:k for k,v in sp_map.items()}
        return sp_map, rev_map, (bom if isinstance(bom, pd.DataFrame) else pd.DataFrame())

    def _logger_ensure_buffer(asset_tag: str):
        key = _logger_key(asset_tag)
        if key in st.session_state and isinstance(st.session_state[key], pd.DataFrame):
            return
        cols = ["WO Number","Date of Maintenance","Maintenance Code","Spares Used","QTY",
                "Hours Run Since Last Repair","Labour Hours","Asset Downtime Hours","Notes and Findings","Attachments","Select"]
        st.session_state[key] = pd.DataFrame(columns=cols)

    def _logger_save_to_history(asset_tag: str, func_loc: str, df_in: pd.DataFrame):
        df = df_in.copy()
        for col in ["QTY","Hours Run Since Last Repair","Labour Hours","Asset Downtime Hours"]:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0.0)
        df["Date of Maintenance"] = pd.to_datetime(df["Date of Maintenance"], errors="coerce").dt.strftime("%Y-%m-%d")

        mask = df["WO Number"].astype(str).ne("") & df["Spares Used"].astype(str).ne("")
        dups = df[mask].groupby(["WO Number","Spares Used"]).size().reset_index(name="n")
        bad = dups[dups["n"] > 1]
        if not bad.empty:
            errs = "\n".join(f"WO {r['WO Number']} → Spare {r['Spares Used']} (x{int(r['n'])})" for _, r in bad.iterrows())
            st.error("Duplicate spares within the same Work Order:\n" + errs)
            return False

        rows = []
        for _, r in df.iterrows():
            rows.append({
                "Asset Tag": asset_tag,
                "Functional Location": func_loc,
                "WO Number": str(r.get("WO Number","")),
                "Date of Maintenance": str(r.get("Date of Maintenance","")),
                "Maintenance Code": str(r.get("Maintenance Code","")),
                "Spares Used": str(r.get("Spares Used","")),
                "Spares Desc": str(r.get("Spares Used","")),
                "QTY": float(r.get("QTY",0.0)),
                "Hours Run Since Last Repair": float(r.get("Hours Run Since Last Repair",0.0)),
                "Labour Hours": float(r.get("Labour Hours",0.0)),
                "Asset Downtime Hours": float(r.get("Asset Downtime Hours",0.0)),
                "Notes and Findings": str(r.get("Notes and Findings","")),
                "Attachments": str(r.get("Attachments","")),
            })
        _mh_insert_rows(rows)

        csv_full = _load_csv_df()
        add_rows = []
        for r in rows:
            add_rows.append({
                "Number": "", "Asset Tag": r["Asset Tag"], "Functional Location": r["Functional Location"],
                "WO Number": r["WO Number"], "Date of Maintenance": r["Date of Maintenance"],
                "Maintenance Code": r["Maintenance Code"], "Spares Used": r["Spares Used"], "QTY": r["QTY"],
                "Hours Run Since Last Repair": r["Hours Run Since Last Repair"], "Labour Hours": r["Labour Hours"],
                "Asset Downtime Hours": r["Asset Downtime Hours"], "Notes and Findings": r["Notes and Findings"],
                "Attachments": r["Attachments"],
            })
        csv_full = pd.concat([csv_full, pd.DataFrame(add_rows)], ignore_index=True)
        _save_csv_df(csv_full)
        st.session_state.history_df = csv_full
        return True

    if len(mr_assets) == 1:
        log_asset = mr_assets[0]
        asset_data = assets.get(log_asset, {})
        func_loc = asset_data.get("Functional Location","")
        sp_map, rev_map, _bom = _logger_bom_maps(log_asset)
        disp_options = list(sp_map.values())

        _logger_ensure_buffer(log_asset)
        key = _logger_key(log_asset)
        df_log = st.session_state[key].copy()

        base_cols = ["WO Number","Date of Maintenance","Maintenance Code","Spares Used","QTY",
                     "Hours Run Since Last Repair","Labour Hours","Asset Downtime Hours","Notes and Findings","Attachments","Select"]
        for c in base_cols:
            if c not in df_log.columns:
                df_log[c] = "" if c not in {"QTY","Hours Run Since Last Repair","Labour Hours","Asset Downtime Hours","Select"} else (0.0 if c!="Select" else False)

        # FIX: DateColumn requires datetime64[ns]; coerce to avoid STRING/Date mismatch.
        if "Date of Maintenance" in df_log.columns:
            df_log["Date of Maintenance"] = pd.to_datetime(df_log["Date of Maintenance"], errors="coerce")

        st.markdown("### Maintenance Logger (for selected asset)")
        edited = st.data_editor(
            df_log,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "WO Number": st.column_config.TextColumn(required=True),
                "Date of Maintenance": st.column_config.DateColumn(format="YYYY-MM-DD"),
                "Maintenance Code": st.column_config.SelectboxColumn(options=[""]+code_chips),
                "Spares Used": st.column_config.SelectboxColumn(options=disp_options, required=False, help="From BOM: CMMS – DESCRIPTION"),
                "QTY": st.column_config.NumberColumn(min_value=0, step=1),
                "Hours Run Since Last Repair": st.column_config.NumberColumn(min_value=0.0, step=1.0),
                "Labour Hours": st.column_config.NumberColumn(min_value=0.0, step=0.5),
                "Asset Downtime Hours": st.column_config.NumberColumn(min_value=0.0, step=0.5),
                "Notes and Findings": st.column_config.TextColumn(),
                "Attachments": st.column_config.TextColumn(help="Filenames after upload, ';' separated"),
                "Select": st.column_config.CheckboxColumn(help="Mark to remove from buffer"),
            },
            key=f"mr_logger_tbl_{log_asset}_{MR_VER}",
            height=_auto_table_height(df_log)
        )

        if not edited.empty:
            for idx in range(len(edited)):
                with st.expander(f"Add attachments for row {idx+1}", expanded=False):
                    ups = st.file_uploader(
                        "Add files", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True,
                        key=f"mr_att_upl_{log_asset}_{idx}_{MR_VER}"
                    )
                    if ups:
                        names=[]
                        for up in ups:
                            safe = _sanitize(f"{log_asset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{up.name}")
                            with open(os.path.join(ATTACH_DIR, safe), "wb") as f:
                                f.write(up.getbuffer())
                            names.append(safe)
                        prev = str(edited.at[idx, "Attachments"]).strip()
                        edited.at[idx, "Attachments"] = (prev + "; " if prev else "") + "; ".join(names)

        st.session_state[key] = edited.copy()

        la, lb, lc = st.columns(3)
        with la:
            if st.button("💾 Save Maintenance History", use_container_width=True, key=f"mr_save_btn_{MR_VER}"):
                df_to_save = edited.copy()
                df_to_save["Spares Used"] = df_to_save["Spares Used"].map(lambda x: rev_map.get(x, x) if x else "")
                ok = _logger_save_to_history(log_asset, func_loc, df_to_save)
                if ok:
                    st.success("Saved to history (DB + CSV).")
                    st.rerun()
        with lb:
            if st.button("🗑 Remove Selected from Logger", use_container_width=True, key=f"mr_del_sel_{MR_VER}"):
                idxs = edited[edited["Select"]==True].index
                if len(idxs)==0:
                    st.info("Nothing selected.")
                else:
                    st.session_state[key] = edited.drop(index=idxs).reset_index(drop=True)
                    st.success(f"Removed {len(idxs)} row(s) from buffer.")
                    st.rerun()
        with lc:
            if st.button("🧹 Clear Logger Buffer", use_container_width=True, key=f"mr_clear_buf_{MR_VER}"):
                st.session_state[key] = pd.DataFrame(columns=base_cols)
                st.success("Cleared buffer.")
                st.rerun()
    else:
        st.info("Pick exactly one Asset Tag to use the Logger (or leave multiple for History-only view).")

    st.markdown("---")

    # ---------- History (SQLite-first) ----------
    def _history_where_params():
        wh, ps = [], []
        if mr_from:
            wh.append("date(date_of_maintenance) >= date(?)")
            ps.append(mr_from.strftime("%Y-%m-%d"))
        if mr_to:
            wh.append("date(date_of_maintenance) <= date(?)")
            ps.append(mr_to.strftime("%Y-%m-%d"))
        if mr_assets:
            qs = ",".join(["?"]*len(mr_assets))
            wh.append(f"asset_tag IN ({qs})")
            ps.extend(mr_assets)
        if mr_codes:
            qs = ",".join(["?"]*len(mr_codes))
            wh.append(f"maintenance_code IN ({qs})")
            ps.extend(mr_codes)
        if mr_search:
            like = f"%{mr_search.strip().lower()}%"
            wh.append("(lower(coalesce(wo_number,'')) LIKE ? OR lower(coalesce(asset_tag,'')) LIKE ? OR lower(coalesce(maintenance_code,'')) LIKE ? OR lower(coalesce(notes,'')) LIKE ?)")
            ps.extend([like, like, like, like])
        return " AND ".join(wh), tuple(ps)

    def _load_history_records():
        where, params = _history_where_params()
        rows = _mh_all(where, params)
        if rows: return rows
        df = _load_csv_df()
        if df.empty: return []
        df["Date of Maintenance"] = pd.to_datetime(df["Date of Maintenance"], errors="coerce")
        mask = pd.Series([True]*len(df))
        if mr_from: mask &= (df["Date of Maintenance"] >= pd.to_datetime(mr_from))
        if mr_to:   mask &= (df["Date of Maintenance"] <= pd.to_datetime(mr_to))
        if mr_assets: mask &= df["Asset Tag"].isin(mr_assets)
        if mr_codes:  mask &= df["Maintenance Code"].isin(mr_codes)
        if mr_search:
            s = mr_search.strip().lower()
            def _hit(row):
                return any(s in str(row.get(k,"")).lower() for k in ["WO Number","Asset Tag","Maintenance Code","Notes and Findings"])
            mask &= df.apply(_hit, axis=1)
        df = df[mask].copy()
        out=[]
        for _, r in df.iterrows():
            atts = [p.strip() for p in str(r.get("Attachments","")).split(";") if p.strip()]
            out.append({
                "asset_tag": r.get("Asset Tag",""),
                "functional_location": r.get("Functional Location",""),
                "wo_number": r.get("WO Number",""),
                "date_of_maintenance": (r.get("Date of Maintenance") if isinstance(r.get("Date of Maintenance"), str) else r.get("Date of Maintenance").strftime("%Y-%m-%d")),
                "maintenance_code": r.get("Maintenance Code",""),
                "spare_cmms": r.get("Spares Used",""),
                "spare_desc": r.get("Spares Used",""),
                "qty": r.get("QTY",0),
                "hours_run_since": r.get("Hours Run Since Last Repair",0.0),
                "labour_hours": r.get("Labour Hours",0.0),
                "downtime_hours": r.get("Asset Downtime Hours",0.0),
                "notes": r.get("Notes and Findings",""),
                "attachments_json": json.dumps(atts),
                "created_at": "",
            })
        return out

    rows = _load_history_records()

    # KPIs
    total = len(rows)
    lab_sum = sum(float(r.get("labour_hours") or 0.0) for r in rows)
    down_sum = sum(float(r.get("downtime_hours") or 0.0) for r in rows)
    hrs_since_avg = (sum(float(r.get("hours_run_since") or 0.0) for r in rows)/total) if total else 0.0
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Records", total)
    k2.metric("Labour Hours", f"{lab_sum:.1f}")
    k3.metric("Downtime Hours", f"{down_sum:.1f}")
   
    # History table
    def _compact_spares(r):
        cm = (r.get("spare_cmms") or "").strip()
        ds = (r.get("spare_desc") or "").strip()
        if cm and ds: return f"{cm} — {ds}"
        return cm or ds or ""

    tbl = []
    for r in rows:
        tbl.append({
            "Date": r.get("date_of_maintenance","") or "—",
            "WO #": r.get("wo_number","") or "—",
            "Asset Tag": r.get("asset_tag","") or "—",
            "Maint Code": r.get("maintenance_code","") or "—",
            "Labour Hrs": float(r.get("labour_hours") or 0.0),
            "Downtime Hrs": float(r.get("downtime_hours") or 0.0),
            "Hours Since": float(r.get("hours_run_since") or 0.0),
            "Spare": _compact_spares(r),
            "Notes": (r.get("notes","") or "")[:60] + ("…" if len(r.get("notes","") or "")>60 else ""),
            "Attachments #": _attachments_count(json.loads(r.get("attachments_json") or "[]"))
        })
    df_hist = pd.DataFrame(tbl)

    left, right = st.columns([1,1])
    with left:
        page_size = st.selectbox("Rows per page", [20,50,100,200], index=1, key=f"mr_pagesize_{MR_VER}")
    with right:
        pages = max(1, math.ceil((len(df_hist) or 1)/max(page_size,1)))
        page_idx = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=f"mr_page_{MR_VER}")
    start = (page_idx-1)*max(page_size,1)
    end = start + max(page_size,1)
    df_page = df_hist.iloc[start:end].copy() if not df_hist.empty else df_hist

    ed = st.data_editor(
        df_page, hide_index=True, use_container_width=True,
        key=f"mr_hist_tbl_{page_idx}_{MR_VER}",
        height=_auto_table_height(df_page)
    )

    # Select one record (by absolute index)
    ids_on_page = list(range(start, min(end, len(df_hist))))
    sel_idx = st.selectbox(
        "Select one record",
        options=[""] + [str(i) for i in ids_on_page],
        key=f"mr_pick_{page_idx}_{MR_VER}"
    )

    if sel_idx:
        picked_i = int(sel_idx)
        picked = rows[picked_i] if 0 <= picked_i < len(rows) else None
        if picked:
            st.markdown("#### 👁 Record Details")
            c = st.columns(3)
            c[0].metric("Date", picked.get("date_of_maintenance","—"))
            c[1].metric("WO #", picked.get("wo_number","—"))
            c[2].metric("Asset", picked.get("asset_tag","—"))
            st.caption(f"Maint Code: {picked.get('maintenance_code','—')} | Labour: {picked.get('labour_hours',0)} | Downtime: {picked.get('downtime_hours',0)} | Hours Since: {picked.get('hours_run_since',0)}")
            if picked.get("notes"): st.write("**Notes**"); st.write(picked.get("notes",""))

            st.write("**Spares**")
            st.table(pd.DataFrame([{
                "CMMS": picked.get("spare_cmms",""),
                "Description": picked.get("spare_desc",""),
                "Qty": picked.get("qty",0)
            }]))

            atts = []
            try:
                atts = json.loads(picked.get("attachments_json") or "[]")
            except Exception:
                atts = []
            if atts:
                st.write("**Attachments**")
                for nm in atts:
                    p = os.path.join(ATTACH_DIR, os.path.basename(nm))
                    exists = os.path.isfile(p)
                    st.write(("📎 " if exists else "⚠️ Missing: ") + os.path.basename(nm))
            else:
                st.caption("No attachments.")

            if st.button("✏ Open Job in Planner", disabled=(not picked.get("wo_number")), key=f"mr_to_planner_{picked_i}_{MR_VER}"):
                try:
                    idx = _read_index()
                    found = None
                    for jid, meta in (idx.get("jobs", {}) or {}).items():
                        if str((meta or {}).get("wo_number","")).strip() == str(picked.get("wo_number","")).strip():
                            found = jid; break
                    if found:
                        if "_wip_from_persisted" in globals():
                            st.session_state.pp_wip = _wip_from_persisted(found)
                        else:
                            st.session_state.pp_wip = load_job(found) if "load_job" in globals() else {}
                        st.success(f"Opened {found} in Planner.")
                    else:
                        st.warning("No matching job found for that WO.")
                except Exception:
                    st.warning("Could not search jobs index.")

    # Export current filtered set
    exp1, exp2 = st.columns(2)
    with exp1:
        if st.button("⬇ Export filtered (CSV)", key=f"mr_exp_csv_{MR_VER}"):
            out = pd.DataFrame(rows)
            st.download_button("Download CSV", data=out.to_csv(index=False), file_name="maintenance_filtered.csv", key=f"mr_dl_csv_{MR_VER}")
    with exp2:
        if st.button("⬇ Export filtered (Excel)", key=f"mr_exp_xls_{MR_VER}"):
            out = pd.DataFrame(rows)
            try:
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                    out.to_excel(xw, index=False, sheet_name="Maintenance")
                st.download_button("Download Excel", data=buf.getvalue(), file_name="maintenance_filtered.xlsx", key=f"mr_dl_xls_{MR_VER}")
            except Exception:
                st.warning("xlsxwriter not available; install to enable Excel export.")

    st.markdown("---")

    # ---------- Analytics (light) ----------
    if rows:
        wk_map = defaultdict(lambda: defaultdict(int))
        for r in rows:
            d = _pdate(r.get("date_of_maintenance"))
            if not d: continue
            y,w,_ = d.isocalendar()
            key = f"{y}-W{w:02d}"
            wk_map[key][r.get("maintenance_code","") or "—"] += 1
        weeks = sorted(wk_map.keys())
        fig1 = go.Figure()
        codes = sorted({c for dct in wk_map.values() for c in dct})
        for cde in codes:
            fig1.add_trace(go.Bar(name=cde, x=weeks, y=[wk_map[w].get(cde,0) for w in weeks]))
        fig1.update_layout(barmode="stack", title="Records by Week & Maintenance Code", height=340,
                           margin=dict(l=10,r=10,t=36,b=10), legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
        st.plotly_chart(fig1, use_container_width=True, key=f"mr_an_week_{MR_VER}")

        by_asset = defaultdict(float)
        for r in rows:
            by_asset[r.get("asset_tag","") or "—"] += float(r.get("downtime_hours") or 0.0)
        top = sorted(by_asset.items(), key=lambda kv: kv[1], reverse=True)[:10]
        if top:
            fig2 = go.Figure(data=[go.Bar(x=[k for k,_ in top], y=[v for _,v in top])])
            fig2.update_layout(title="Top Assets by Downtime (filtered)", height=320,
                               margin=dict(l=10,r=10,t=36,b=10))
            st.plotly_chart(fig2, use_container_width=True, key=f"mr_an_dt_{MR_VER}")

        xs, ys = [], []
        for r in rows:
            xs.append(float(r.get("labour_hours") or 0.0))
            ys.append(float(r.get("downtime_hours") or 0.0))
        fig3 = go.Figure(data=[go.Scatter(x=xs, y=ys, mode="markers")])
        fig3.update_layout(title="Labour vs Downtime", height=320, margin=dict(l=10,r=10,t=36,b=10),
                           xaxis_title="Labour Hours", yaxis_title="Downtime Hours")
        st.plotly_chart(fig3, use_container_width=True, key=f"mr_an_sc_{MR_VER}")
    else:
        st.info("No records in the selected range/filters.")


# =========================
# TAB 4 — Reliability Visualisation (scaffold)
# =========================
# file: reliability_app.py  (Tab 4 — Reliability Visualisation)

with tab4:
    import re
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st
    from math import ceil, isfinite
    from collections import defaultdict

    # -----------------------------
    # Global filters (top bar)
    # -----------------------------
    st.subheader("📉 Reliability Visualisation")

    # Base data (safe copies)
    assets_dict = st.session_state.get("assets", {}) or {}
    runtime_df0 = st.session_state.get("runtime_df", pd.DataFrame()).copy()
    history_df0 = st.session_state.get("history_df", pd.DataFrame()).copy()
    comp_store  = st.session_state.get("component_life", {}) or {}

    # Ensure essential columns exist in runtime_df
    need_rt = ["Asset Tag","Functional Location","Asset Model","Criticality","MTBF (Hours)",
               "Last Overhaul","Running Hours Since Last Major Maintenance","Remaining Hours","STATUS"]
    for c in need_rt:
        if c not in runtime_df0.columns:
            runtime_df0[c] = (0.0 if c in ("MTBF (Hours)","Running Hours Since Last Major Maintenance","Remaining Hours")
                              else "" if c!="STATUS" else "🟢 Healthy")

    # Coerce history types
    if not history_df0.empty:
        # Dates
        history_df0["Date of Maintenance"] = pd.to_datetime(history_df0.get("Date of Maintenance"), errors="coerce")
        # Numerics
        for c in ["QTY","Hours Run Since Last Repair","Labour Hours","Asset Downtime Hours"]:
            history_df0[c] = pd.to_numeric(history_df0.get(c, 0), errors="coerce").fillna(0.0)

    # Filter selectors
    all_asset_tags = sorted([k for k in assets_dict.keys() if k])
    crit_opts = ["Low","Medium","High"]
    code_opts = sorted(list({str(x) for x in history_df0.get("Maintenance Code", pd.Series([], dtype=str)).dropna().unique()})) if not history_df0.empty else []
    today = datetime.today().date()
    default_from = today - timedelta(days=90)

    f_col1, f_col2, f_col3, f_col4 = st.columns([1.3, 1.7, 1.7, 2.3])
    with f_col1:
        f_from = st.date_input("From", value=default_from, key="rv_from")
    with f_col2:
        f_to   = st.date_input("To", value=today, key="rv_to")
    with f_col3:
        f_assets = st.multiselect("Assets", options=all_asset_tags, default=[], key="rv_assets")
    with f_col4:
        f_crit   = st.multiselect("Criticality", options=crit_opts, default=[], key="rv_crit")
    f_codes = st.multiselect("Maintenance Codes", options=code_opts, default=[], key="rv_codes")

    # -----------------------------
    # Shared helper functions
    # -----------------------------
    def rv_section_from_tag(tag: str) -> str:
        try:
            m = re.match(r"^(\d{3})", str(tag))
            return m.group(1) if m else "Other"
        except Exception:
            return "Other"

    def rv_filter_runtime(rt: pd.DataFrame) -> pd.DataFrame:
        df = rt.copy()
        if f_assets:
            df = df[df["Asset Tag"].astype(str).isin(f_assets)]
        if f_crit:
            df = df[df["Criticality"].astype(str).isin(f_crit)]
        return df.reset_index(drop=True)

    def rv_filter_history(h: pd.DataFrame) -> pd.DataFrame:
        if h.empty: return h.copy()
        df = h.copy()
        if f_from: df = df[df["Date of Maintenance"] >= pd.to_datetime(f_from)]
        if f_to:   df = df[df["Date of Maintenance"] <= pd.to_datetime(f_to) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)]
        if f_assets: df = df[df["Asset Tag"].astype(str).isin(f_assets)]
        if f_codes:  df = df[df["Maintenance Code"].astype(str).isin(f_codes)]
        return df.reset_index(drop=True)

    def rv_status_counts(rt: pd.DataFrame) -> dict:
        s = rt.get("STATUS", pd.Series([], dtype=str)).value_counts(dropna=False)
        return {"healthy": int(s.get("🟢 Healthy", 0)),
                "plan":    int(s.get("🟠 Plan for maintenance", 0)),
                "overdue": int(s.get("🔴 Overdue for maintenance", 0)),
                "total":   int(len(rt))}

    def rv_weekly_counts(df: pd.DataFrame, date_col: str, value_col: str = None) -> pd.DataFrame:
        """Weekly aggregation by ISO week. If value_col is None -> count rows; else sum of value_col."""
        if df.empty or date_col not in df.columns: 
            return pd.DataFrame({"Week":[], "Value":[]})
        d = df.copy()
        d = d.dropna(subset=[date_col])
        if d.empty: return pd.DataFrame({"Week":[], "Value":[]})
        d["Week"] = d[date_col].dt.to_period("W").apply(lambda p: p.start_time.date().isoformat())
        if value_col is None:
            out = d.groupby("Week", as_index=False).size().rename(columns={"size":"Value"})
        else:
            out = d.groupby("Week", as_index=False)[value_col].sum().rename(columns={value_col:"Value"})
        return out.sort_values("Week")

    def rv_compute_mtbf_series(df: pd.DataFrame, date_col: str, asset_col: str) -> pd.DataFrame:
        """Return per-asset time-between-failures (hours) tagged by the later event's month."""
        rows = []
        if df.empty: return pd.DataFrame(columns=["Asset", "Month", "TBF_hrs"])
        d = df.dropna(subset=[date_col]).sort_values([asset_col, date_col]).copy()
        for asset, g in d.groupby(asset_col):
            times = list(g[date_col].sort_values())
            for i in range(1, len(times)):
                delta_h = (times[i] - times[i-1]).total_seconds()/3600.0
                if delta_h >= 0:
                    rows.append({"Asset": asset,
                                 "Month": times[i].strftime("%Y-%m"),
                                 "TBF_hrs": delta_h})
        return pd.DataFrame(rows)

    def rv_overall_mtbf(df_tbf: pd.DataFrame) -> float:
        try:
            if df_tbf.empty: return float("nan")
            return float(np.nanmean(pd.to_numeric(df_tbf["TBF_hrs"], errors="coerce")))
        except Exception:
            return float("nan")

    def rv_overall_mttr(h: pd.DataFrame) -> float:
        if h.empty: return float("nan")
        vals = pd.to_numeric(h.get("Asset Downtime Hours", 0), errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else float("nan")

    def rv_make_donut(labels, values, title=None, key=None, height=280):
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, textinfo="label+percent")])
        fig.update_layout(showlegend=True, height=height, margin=dict(t=10,b=10,l=10,r=10), title=title)
        st.plotly_chart(fig, use_container_width=True, key=key)

    def rv_safe_metric(label: str, value, help_text: str = None):
        try:
            st.metric(label, value, help=help_text)
        except Exception:
            st.metric(label, value)

    # Build filtered datasets
    rv_hist = rv_filter_history(history_df0)
    rv_rt   = rv_filter_runtime(runtime_df0)

# -----------------------------
# Sub-tabs  (REPLACE THIS LINE ONLY)
# -----------------------------
    t_exec, t_health, t_fail, t_work, t_cm, t_comp, t_quality = st.tabs([
        "📊 Executive Overview", "🧭 Asset Health & Risk", "🔁 Failure & Repair", 
        "🧱 Workload & Impact", "🛰 Condition Monitoring", "⚙️ Component Forecasts", "🧪 Data Quality"
    ])

    # =========================================================
    # 1) Executive Overview
    # =========================================================
    with t_exec:
        st.markdown("#### Executive Overview")

        # KPIs
        tbf_df = rv_compute_mtbf_series(rv_hist, date_col="Date of Maintenance", asset_col="Asset Tag")
        k_assets = len(rv_rt)
        k_fail   = len(rv_hist)
        k_mtbf   = rv_overall_mtbf(tbf_df)  # hrs
        k_mttr   = rv_overall_mttr(rv_hist) # hrs
        k_downt  = float(rv_hist["Asset Downtime Hours"].sum()) if not rv_hist.empty else 0.0
        st_cts   = rv_status_counts(rv_rt)
        overdue_pct = (100.0 * st_cts["overdue"] / max(st_cts["total"], 1))

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        rv_safe_metric("Assets", k_assets)
        c2.metric("Failures (period)", k_fail)
        c3.metric("MTBF (hrs)", f"{k_mtbf:.1f}" if isfinite(k_mtbf) else "—")
        c4.metric("MTTR (hrs)", f"{k_mttr:.1f}" if isfinite(k_mttr) else "—")
        c5.metric("Downtime (hrs)", f"{k_downt:.1f}")
        c6.metric("% Overdue", f"{overdue_pct:.0f}%")

        # Status donut
        st.markdown("#### Status Mix")
        rv_make_donut(
            labels=[f"🟢 Healthy ({st_cts['healthy']})", f"🟠 Plan ({st_cts['plan']})", f"🔴 Overdue ({st_cts['overdue']})"],
            values=[st_cts["healthy"], st_cts["plan"], st_cts["overdue"]],
            key="rv_exec_status"
        )

        # Trends: failures/week and downtime/week
        st.markdown("#### Trends (weekly)")
        w1, w2 = st.columns(2)
        with w1:
            wk_fail = rv_weekly_counts(rv_hist, "Date of Maintenance")
            fig = go.Figure(data=[go.Bar(x=wk_fail["Week"], y=wk_fail["Value"])])
            fig.update_layout(title="Failures per Week", height=300, margin=dict(t=34,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True, key="rv_exec_failwk")
        with w2:
            wk_dt = rv_weekly_counts(rv_hist, "Date of Maintenance", "Asset Downtime Hours")
            fig = go.Figure(data=[go.Bar(x=wk_dt["Week"], y=wk_dt["Value"])])
            fig.update_layout(title="Downtime Hours per Week", height=300, margin=dict(t=34,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True, key="rv_exec_dtwk")

        # Pareto: downtime by asset (top 10)
        st.markdown("#### Pareto — Downtime by Asset")
        if not rv_hist.empty:
            by_asset = rv_hist.groupby("Asset Tag", as_index=False)["Asset Downtime Hours"].sum()
            by_asset = by_asset.sort_values("Asset Downtime Hours", ascending=False).head(10)
            fig = go.Figure(data=[go.Bar(x=by_asset["Asset Tag"], y=by_asset["Asset Downtime Hours"])])
            fig.update_layout(height=320, margin=dict(t=20,b=30,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True, key="rv_exec_pareto")
        else:
            st.info("No downtime in the selected range.")

    # =========================================================
    # 2) Asset Health & Risk
    # =========================================================
    with t_health:
        st.markdown("#### Asset Health & Risk")

        if rv_rt.empty:
            st.info("No runtime rows match the current filters.")
        else:
            df = rv_rt.copy()
            # Compute life used & risk score
            df["MTBF (Hours)"] = pd.to_numeric(df["MTBF (Hours)"], errors="coerce").fillna(0.0)
            df["Run Since"] = pd.to_numeric(df["Running Hours Since Last Major Maintenance"], errors="coerce").fillna(0.0)
            df["Remaining"] = pd.to_numeric(df["Remaining Hours"], errors="coerce").fillna(0.0)
            df["Life Used %"] = np.where(df["MTBF (Hours)"]>0, (df["Run Since"]/df["MTBF (Hours)"])*100.0, 0.0).clip(0, 999)
            # Recent failures count per asset in window
            fail_counts = rv_hist["Asset Tag"].value_counts() if not rv_hist.empty else pd.Series(dtype=int)
            df["Failures (period)"] = df["Asset Tag"].map(fail_counts).fillna(0).astype(int)
            # Criticality weight
            crit_w = {"Low":1, "Medium":2, "High":3}
            df["Crit W"] = df["Criticality"].map(crit_w).fillna(1)
            df["Risk Score"] = (df["Crit W"] * (1.0 + df["Failures (period)"]) * (df["Life Used %"]/100.0)).round(2)

            # Bubble: Life Used vs Remaining
            fig = go.Figure(data=[go.Scatter(
                x=df["Life Used %"], y=df["Remaining"], mode="markers",
                marker=dict(size=np.sqrt(np.maximum(1.0, df["Failures (period)"]+1))*10),
                text=df["Asset Tag"], hovertemplate="Asset: %{text}<br>Life Used: %{x:.1f}%<br>Remaining: %{y:.0f}h<br>Fails: %{marker.size}<extra></extra>"
            )])
            fig.update_layout(title="Life Used (%) vs Remaining Hours (bubble size ~ failures)", height=360, xaxis_title="% Life Used", yaxis_title="Remaining (hrs)")
            st.plotly_chart(fig, use_container_width=True, key="rv_hr_bubble")

            # Top at-risk table
            risk_cols = ["Asset Tag","Criticality","STATUS","Run Since","MTBF (Hours)","Remaining","Life Used %","Failures (period)","Risk Score"]
            st.markdown("#### Top At-Risk Assets")
            st.dataframe(
                df.sort_values(["Risk Score","Remaining"], ascending=[False, True])[risk_cols].head(20),
                use_container_width=True,
                height=auto_table_height(min(20, len(df)), min_rows=5)
            )

            # Section × Status heatmap (counts)
            st.markdown("#### Heatmap — Section × Status")
            df["Section"] = df["Asset Tag"].map(rv_section_from_tag)
            pivot = df.pivot_table(index="Section", columns="STATUS", values="Asset Tag", aggfunc="count", fill_value=0)
            heat_vals = pivot.values
            fig_h = go.Figure(data=go.Heatmap(
                z=heat_vals, x=list(pivot.columns), y=list(pivot.index),
                hoverongaps=False
            ))
            fig_h.update_layout(height=320, margin=dict(t=30,b=10,l=10,r=10))
            st.plotly_chart(fig_h, use_container_width=True, key="rv_hr_heat")

    # =========================================================
    # 3) Failure & Repair Dynamics
    # =========================================================
    with t_fail:
        st.markdown("#### Failure & Repair Dynamics")

        if rv_hist.empty:
            st.info("No maintenance history in the selected range.")
        else:
            # Monthly MTBF (from TBF series)
            tbf_df = rv_compute_mtbf_series(rv_hist, "Date of Maintenance", "Asset Tag")
            if tbf_df.empty:
                st.info("Not enough consecutive events to compute MTBF.")
            else:
                mtbf_month = tbf_df.groupby("Month", as_index=False)["TBF_hrs"].mean()
                fig1 = go.Figure(data=[
                    go.Scatter(
                        x=mtbf_month["Month"],
                        y=mtbf_month["TBF_hrs"],
                        mode="lines+markers",
                        name="MTBF",
                        hovertemplate="Month: %{x}<br>MTBF: %{y:.1f} h<extra></extra>",
                    )
                ])
                fig1.update_layout(
                    title="MTBF by Month",
                    height=320,
                    margin=dict(t=36, b=10, l=10, r=10),
                    hovermode="x unified",
                    xaxis=dict(
                        title="Month (YYYY-MM)",
                        showgrid=True, gridcolor="rgba(0,0,0,0.1)",
                        tickangle=-30
                    ),
                    yaxis=dict(
                        title="MTBF (hours)",
                        showgrid=True, gridcolor="rgba(0,0,0,0.1)",
                        zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.25)"
                    ),
                )
                st.plotly_chart(fig1, use_container_width=True, key="rv_fail_mtbf")

            # Monthly MTTR (avg downtime per event, proxy)
            mttr_month = rv_hist.copy()
            mttr_month["Month"] = mttr_month["Date of Maintenance"].dt.strftime("%Y-%m")
            mttr_month = mttr_month.groupby("Month", as_index=False)["Asset Downtime Hours"].mean()

            if not mttr_month.empty:
                fig2 = go.Figure(data=[
                    go.Scatter(
                        x=mttr_month["Month"],
                        y=mttr_month["Asset Downtime Hours"],
                        mode="lines+markers",
                        name="MTTR (proxy)",
                        hovertemplate="Month: %{x}<br>MTTR: %{y:.1f} h<extra></extra>",
                    )
                ])
                fig2.update_layout(
                    title="MTTR by Month (Proxy from Downtime Hours)",
                    height=320,
                    margin=dict(t=36, b=10, l=10, r=10),
                    hovermode="x unified",
                    xaxis=dict(
                        title="Month (YYYY-MM)",
                        showgrid=True, gridcolor="rgba(0,0,0,0.1)",
                        tickangle=-30
                    ),
                    yaxis=dict(
                        title="MTTR (hours)",
                        showgrid=True, gridcolor="rgba(0,0,0,0.1)",
                        zeroline=True, zerolinewidth=1, zerolinecolor="rgba(0,0,0,0.25)"
                    ),
                )
                st.plotly_chart(fig2, use_container_width=True, key="rv_fail_mttr")
                st.caption("Note: MTTR is proxied by average **Asset Downtime Hours** per event until explicit repair start/finish is captured.")

            # Histogram: Time-Between-Failures
            if not tbf_df.empty:
                fig3 = go.Figure(data=[
                    go.Histogram(
                        x=tbf_df["TBF_hrs"],
                        nbinsx=30,
                        name="TBF",
                        hovertemplate="TBF (hrs): %{x:.1f}<br>Count: %{y}<extra></extra>",
                    )
                ])
                fig3.update_layout(
                    title="Histogram — Time Between Failures",
                    height=320,
                    margin=dict(t=36, b=10, l=10, r=10),
                    xaxis=dict(
                        title="Time Between Failures (hours)",
                        showgrid=True, gridcolor="rgba(0,0,0,0.1)"
                    ),
                    yaxis=dict(
                        title="Event count",
                        showgrid=True, gridcolor="rgba(0,0,0,0.1)"
                    ),
                    bargap=0.05
                )
                st.plotly_chart(fig3, use_container_width=True, key="rv_fail_hist")

            # Reliability curve for selected asset
            aset_opts = sorted(rv_hist["Asset Tag"].dropna().unique().tolist())
            sel_asset = st.selectbox("Reliability curve for asset", options=([""] + aset_opts), key="rv_fail_asset")
            if sel_asset:
                g = rv_hist[rv_hist["Asset Tag"] == sel_asset] \
                        .dropna(subset=["Date of Maintenance"]) \
                        .sort_values("Date of Maintenance")
                if len(g) < 2:
                    st.info("Need at least 2 failure events to estimate MTBF for this asset.")
                else:
                    times = list(g["Date of Maintenance"])
                    deltas = [(times[i] - times[i-1]).total_seconds()/3600.0 for i in range(1, len(times))]
                    mtbf = float(np.mean([x for x in deltas if x >= 0])) if deltas else float("nan")
                    if not np.isfinite(mtbf) or mtbf <= 0:
                        st.info("Unable to compute MTBF for selected asset.")
                    else:
                        # R(t)=exp(-t/MTBF)
                        t_hours = np.linspace(0, max(mtbf*2, 1), 100)
                        R = np.exp(-t_hours/mtbf)
                        figR = go.Figure(data=[
                            go.Scatter(
                                x=t_hours,
                                y=R,
                                mode="lines",
                                name="Reliability R(t)",
                                hovertemplate="t: %{x:.0f} h<br>R(t): %{y:.3f}<extra></extra>",
                            )
                        ])
                        figR.update_layout(
                            title=f"Reliability Curve R(t) — {sel_asset} (MTBF ≈ {mtbf:.1f} h)",
                            height=340,
                            hovermode="x unified",
                            xaxis=dict(
                                title="Time t (hours)",
                                showgrid=True, gridcolor="rgba(0,0,0,0.1)"
                            ),
                            yaxis=dict(
                                title="Reliability R(t)",
                                showgrid=True, gridcolor="rgba(0,0,0,0.1)",
                                range=[0,1]
                            ),
                        )
                        st.plotly_chart(figR, use_container_width=True, key="rv_fail_Rt")

    # =========================================================
    # 4) Workload & Impact (non-financial)
    # =========================================================
    with t_work:
        st.markdown("#### Workload & Impact (hours only)")

        if rv_hist.empty:
            st.info("No history rows in the selected range.")
        else:
            # Weekly labour and downtime hours
            wk_lab = rv_weekly_counts(rv_hist, "Date of Maintenance", "Labour Hours")
            wk_dow = rv_weekly_counts(rv_hist, "Date of Maintenance", "Asset Downtime Hours")

            # Align weeks
            weeks = sorted(set(wk_lab["Week"]).union(set(wk_dow["Week"])))
            d_lab = {w:0.0 for w in weeks}; d_lab.update(dict(zip(wk_lab["Week"], wk_lab["Value"])))
            d_dow = {w:0.0 for w in weeks}; d_dow.update(dict(zip(wk_dow["Week"], wk_dow["Value"])))
            weeks_sorted = weeks

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Labour Hours", x=weeks_sorted, y=[d_lab[w] for w in weeks_sorted]))
            fig.add_trace(go.Bar(name="Downtime Hours", x=weeks_sorted, y=[d_dow[w] for w in weeks_sorted]))
            fig.update_layout(barmode="stack", title="Weekly Labour vs Downtime Hours", height=340, margin=dict(t=36,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True, key="rv_work_stack")

            # Top components by usage count (via 'Spares Used')
            st.markdown("#### Top Components by Usage (counts)")
            if "Spares Used" in rv_hist.columns and not rv_hist["Spares Used"].dropna().empty:
                counts = rv_hist["Spares Used"].astype(str).str.strip().value_counts().head(10)
                figc = go.Figure(data=[go.Bar(x=list(counts.index), y=list(counts.values))])
                figc.update_layout(height=300, margin=dict(t=20,b=60,l=10,r=10))
                st.plotly_chart(figc, use_container_width=True, key="rv_work_comp")
            else:
                st.caption("No 'Spares Used' data.")

            # Treemap: Maintenance Code → Asset by hours (downtime)
            st.markdown("#### Treemap — Code → Asset by Downtime Hours")
            tre = rv_hist.copy()
            tre["Maintenance Code"] = tre["Maintenance Code"].fillna("—").astype(str)
            tre["Asset Tag"] = tre["Asset Tag"].fillna("—").astype(str)
            grp = tre.groupby(["Maintenance Code","Asset Tag"], as_index=False)["Asset Downtime Hours"].sum()
            if grp.empty or grp["Asset Downtime Hours"].sum() <= 0:
                st.caption("No downtime to display.")
            else:
                figt = go.Figure(go.Treemap(
                    labels=grp["Asset Tag"],
                    parents=grp["Maintenance Code"],
                    values=grp["Asset Downtime Hours"],
                    branchvalues="total"
                ))
                figt.update_layout(height=360, margin=dict(t=20,b=10,l=10,r=10))
                st.plotly_chart(figt, use_container_width=True, key="rv_work_tree")

# =========================================================
# 5) 🛰 Condition Monitoring (REGENERATED — paste as the 5th sub-tab)
# =========================================================
with t_cm:
    import os, io, re, json, hashlib, difflib, sqlite3, platform
    from typing import List, Dict, Tuple, Optional
    from datetime import datetime, timedelta, date as _date
    import pandas as pd
    import streamlit as st

    st.markdown("#### 🛰 Condition Monitoring Centre")

    # ---------- Paths & DB ----------
    DATA_DIR = globals().get("DATA_DIR", "data")
    CM_DIR   = os.path.join(DATA_DIR, "cm_attachments")
    os.makedirs(CM_DIR, exist_ok=True)
    RELIAB_DB = globals().get("RELIAB_DB", os.path.join(DATA_DIR, "reliability.db"))

    def _cm_init_db():
        with sqlite3.connect(RELIAB_DB) as conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cm_reports(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                assets_json TEXT,
                technique TEXT,
                severity TEXT,
                vendor TEXT,
                analyst TEXT,
                diagnosis TEXT,
                recommendation TEXT,
                next_review_date TEXT,
                attachments_json TEXT,
                status TEXT,
                file_hash TEXT,
                filename TEXT,
                created_at TEXT
            )""")
            cur.execute("""
            CREATE TABLE IF NOT EXISTS cm_alias(
                alias TEXT PRIMARY KEY,
                canonical TEXT,
                created_at TEXT
            )""")
            conn.commit()

    def _cm_load_alias_map() -> Dict[str, str]:
        _cm_init_db()
        try:
            with sqlite3.connect(RELIAB_DB) as conn:
                rows = conn.execute("SELECT alias, canonical FROM cm_alias").fetchall()
                return {a: c for a, c in rows}
        except Exception:
            return {}

    def _cm_upsert_alias(alias: str, canonical: str):
        if not alias or not canonical: return
        _cm_init_db()
        with sqlite3.connect(RELIAB_DB) as conn:
            conn.execute("INSERT OR REPLACE INTO cm_alias(alias, canonical, created_at) VALUES(?,?,?)",
                         (alias, canonical, datetime.utcnow().isoformat()+"Z"))
            conn.commit()

    def _cm_save_file(up) -> str:
        raw = up.read(); up.seek(0)
        h = hashlib.sha256(raw).hexdigest()[:16]
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", up.name)
        name = f"{h}__{base}"
        with open(os.path.join(CM_DIR, name), "wb") as f:
            f.write(raw)
        return name

    # ---------- Tag normalization & patterns ----------
    def _cm_norm_tag(s: str) -> str:
        if not s: return ""
        t = str(s).upper()
        t = t.replace("—","-").replace("–","-").replace("−","-").replace("_","-")
        t = re.sub(r"\s+", " ", t)
        t = t.replace(" -","-").replace("- ","-").replace(" ","-")
        t = re.sub(r"-{2,}", "-", t).strip("- ")
        return t

    def _cm_soft_norm(s: str) -> str:
        swaps = str.maketrans({"O":"0","o":"0","I":"1","l":"1","S":"5","B":"8","Z":"2","G":"6"})
        return _cm_norm_tag(s).translate(swaps)

    # split-class support: e.g., "408 P P 001"
    _PAT_SPLITCLS = re.compile(r"\b(\d{3})\s*[-\s]?\s*([A-Z](?:\s*[A-Z]){1,2})\s*[-\s]?\s*(\d{3,4})\b")

    # ---------- OCR / text extraction ----------
    def _cm_get_overrides() -> Dict[str, Optional[str]]:
        """Read user-provided overrides from session_state."""
        return {
            "tesseract": st.session_state.get("cm_tesseract_exe"),
            "poppler": st.session_state.get("cm_poppler_bin"),
        }

    def _cm_detect_env() -> dict:
        """Use overrides if set; else try common Windows locations."""
        info = {"tesseract": None, "poppler": None}
        ov = _cm_get_overrides()
        if ov.get("tesseract") and os.path.isfile(ov["tesseract"]):
            info["tesseract"] = ov["tesseract"]
        if ov.get("poppler") and os.path.isdir(ov["poppler"]):
            info["poppler"] = ov["poppler"]

        # Auto-detect only if not overridden/found
        if info["tesseract"] is None:
            try:
                import pytesseract  # noqa
                for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                          r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
                    if os.path.isfile(p):
                        info["tesseract"] = p; break
            except Exception:
                pass

        if info["poppler"] is None:
            for base in (r"C:\Program Files", r"C:\Program Files (x86)", r"C:\poppler", r"C:\tools\poppler"):
                try:
                    if not os.path.isdir(base): continue
                    # common layout
                    for name in os.listdir(base):
                        if "poppler" in name.lower():
                            cand = os.path.join(base, name, "Library", "bin")
                            if os.path.isdir(cand):
                                info["poppler"] = cand
                                raise StopIteration
                    cand = os.path.join(base, "poppler", "bin")
                    if os.path.isdir(cand):
                        info["poppler"] = cand
                except StopIteration:
                    break
                except Exception:
                    pass
        return info

    def _cm_extract_text(up) -> Tuple[str, List[str], List[str]]:
        """
        Returns (full_text, page_texts[], page_notes[]).
        PDF: PyPDF2 → pdfminer → OCR. Images: OCR.
        """
        page_notes: List[str] = []
        page_texts: List[str] = []
        raw = up.read(); up.seek(0)
        mime = (getattr(up, "type", "") or "").lower()

        # Images → OCR
        if mime.startswith("image/"):
            try:
                from PIL import Image, ImageOps
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                img = ImageOps.exif_transpose(img)
                try:
                    import pytesseract
                    env = _cm_detect_env()
                    if env["tesseract"]:
                        pytesseract.pytesseract.tesseract_cmd = env["tesseract"]
                    t = pytesseract.image_to_string(img)
                except Exception:
                    t = ""
                    page_notes.append("image:ocr-not-available")
                page_texts.append(t or "")
                page_notes.append("image:ocr" if (t or "").strip() else "image:blank")
            except Exception:
                page_texts.append("")
                page_notes.append("image:open-failed")
            return ("\n".join(page_texts), page_texts, page_notes)

        # PDFs
        any_text = False
        # 1) Embedded text (PyPDF2)
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            for i, p in enumerate(reader.pages):
                try:
                    t = p.extract_text() or ""
                except Exception:
                    t = ""
                page_texts.append(t)
                page_notes.append(f"p{i+1}:{'text' if t.strip() else 'no-text'}")
                if t.strip(): any_text = True
        except Exception:
            page_notes.append("PyPDF2-failed")

        # 2) pdfminer (single blob)
        if not any_text:
            try:
                from pdfminer.high_level import extract_text
                t = extract_text(io.BytesIO(raw)) or ""
                if t.strip():
                    page_texts = [t]
                    page_notes.append("pdfminer:text")
                    any_text = True
            except Exception:
                page_notes.append("pdfminer-failed")

        # 3) OCR (pdf2image + tesseract)
        if not any_text:
            try:
                import pytesseract
                from pdf2image import convert_from_bytes
                from PIL import ImageOps
                env = _cm_detect_env()
                if env["tesseract"]:
                    pytesseract.pytesseract.tesseract_cmd = env["tesseract"]
                pages = convert_from_bytes(raw, dpi=200, poppler_path=env["poppler"])
                page_texts = []
                for i, img in enumerate(pages):
                    try:
                        # Transpose to fix rotated scans
                        img = ImageOps.exif_transpose(img)
                        t = pytesseract.image_to_string(img)
                    except Exception:
                        t = ""
                    page_texts.append(t)
                    page_notes.append(f"p{i+1}:{'ocr' if t.strip() else 'blank'}")
                if not any(s.strip() for s in page_texts):
                    page_notes.append("ocr-no-text")
            except Exception:
                page_texts = [""]
                page_notes.append("ocr-not-available")

        full_text = "\n".join([s for s in page_texts if s is not None])
        return (full_text, page_texts, page_notes)

    # ---------- Candidate extraction (with page hits) ----------
    def _cm_tag_candidates_with_pages(text: str, filename: str, page_texts: List[str]) -> Tuple[List[str], Dict[str, List[int]]]:
        cands: set[str] = set()
        hits: Dict[str, set] = {}

        def _add_candidate(raw_tag: str, page_i: Optional[int]):
            tag = _cm_norm_tag(raw_tag)
            if not tag: return
            cands.add(tag)
            if page_i is not None:
                hits.setdefault(tag, set()).add(page_i + 1)

        def _scan(s: str, page_i: Optional[int]):
            if not s: return
            U = _cm_norm_tag(s)
            # strict
            for m in re.finditer(r"\b\d{3}-[A-Z]{2,3}-\d{3,4}\b", U):
                _add_candidate(m.group(0), page_i)
            # tolerant
            for m in re.finditer(r"\b\d{3}[-\s]?[A-Z]{2,3}[-\s]?\d{3,4}\b", U):
                _add_candidate(m.group(0), page_i)
            # split letters (P P)
            for m in _PAT_SPLITCLS.finditer(U):
                sec = m.group(1)
                cls = re.sub(r"\s+", "", m.group(2))
                num = m.group(3)
                _add_candidate(f"{sec}-{cls}-{num}", page_i)

        _scan(filename or "", None)
        if page_texts:
            for i, pg in enumerate(page_texts):
                _scan(pg or "", i)
        else:
            _scan(text or "", None)

        final_cands = sorted(cands)
        page_hits = {k: sorted(list(v)) for k, v in hits.items()}
        return final_cands, page_hits

    # ---------- Filename hints ----------
    _TECH_TOKENS = {
        "VA": "Vibration", "VIBRATION": "Vibration",
        "IR": "Thermography", "THERMO": "Thermography",
        "OIL": "Oil", "LUBE": "Oil",
        "US": "Ultrasound", "ULTRA": "Ultrasound"
    }
    def _cm_guess_from_filename(name: str) -> Tuple[Optional[str], Optional[str]]:
        up = os.path.basename(name).upper()
        tech = None
        for k, v in _TECH_TOKENS.items():
            if f"-{k}-" in up or k in up.split("-") or up.endswith(f"_{k}.PDF"):
                tech = v; break
        m = re.search(r"(?<!\d)(\d{6}|\d{8})(?!\d)", up)
        dt = None
        if m:
            s = m.group(1)
            try:
                dt = (datetime.strptime(s, "%Y%m%d") if len(s)==8 else datetime.strptime(s, "%y%m%d")).date().isoformat()
            except Exception:
                pass
        return tech, dt

    # ---------- Matching ----------
    def _cm_match_against_registry(cands: List[str], registry: List[str], alias_map: Dict[str,str]) -> List[dict]:
        reg_set = set([_cm_norm_tag(x) for x in registry])
        out = []
        for c in cands:
            cn = _cm_norm_tag(c)
            ali = alias_map.get(cn) or alias_map.get(_cm_soft_norm(cn))
            if ali and ali in reg_set:
                out.append({"candidate": cn, "canonical": ali, "confidence": 1.0, "reason": "alias"}); continue
            if cn in reg_set:
                out.append({"candidate": cn, "canonical": cn, "confidence": 1.0, "reason": "exact"}); continue
            soft = _cm_soft_norm(cn)
            if soft in reg_set:
                out.append({"candidate": cn, "canonical": soft, "confidence": 0.97, "reason": "normalized"}); continue
            match = difflib.get_close_matches(cn, list(reg_set), n=1, cutoff=0.86)
            if match:
                out.append({"candidate": cn, "canonical": match[0], "confidence": 0.90, "reason": "fuzzy"})
            else:
                out.append({"candidate": cn, "canonical": "", "confidence": 0.0, "reason": "unmatched"})
        return out

    # ---------- Persistence ----------
    def _cm_insert_report(date_s: str, assets: List[str], technique: str, severity: str,
                          vendor: str, analyst: str, diagnosis: str, recommendation: str,
                          next_review: str, attachments: List[str], file_hash: str, filename: str):
        _cm_init_db()
        with sqlite3.connect(RELIAB_DB) as conn:
            conn.execute("""
                INSERT INTO cm_reports(date, assets_json, technique, severity, vendor, analyst,
                                       diagnosis, recommendation, next_review_date,
                                       attachments_json, status, file_hash, filename, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                date_s or "",
                json.dumps(assets, ensure_ascii=False),
                technique or "",
                severity or "",
                vendor or "",
                analyst or "",
                diagnosis or "",
                recommendation or "",
                next_review or "",
                json.dumps(attachments, ensure_ascii=False),
                "Open",
                file_hash,
                filename,
                datetime.utcnow().isoformat()+"Z"
            ))
            conn.commit()

    def _cm_list_reports(filter_from: Optional[_date], filter_to: Optional[_date], techs: List[str], severities: List[str]) -> pd.DataFrame:
        _cm_init_db()
        with sqlite3.connect(RELIAB_DB) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM cm_reports ORDER BY date DESC, id DESC").fetchall()
            df = pd.DataFrame([dict(r) for r in rows])
        if df.empty: return df
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if filter_from: df = df[df["date"] >= pd.to_datetime(filter_from)]
        if filter_to:   df = df[df["date"] <= pd.to_datetime(filter_to)]
        if techs:       df = df[df["technique"].isin(techs)]
        if severities:  df = df[df["severity"].isin(severities)]
        df["assets"]      = df["assets_json"].apply(lambda s: ", ".join(json.loads(s or "[]")))
        df["attachments"] = df["attachments_json"].apply(lambda s: len(json.loads(s or "[]")))
        return df[["date","technique","severity","assets","vendor","analyst","diagnosis","recommendation","next_review_date","attachments","filename"]].copy()

    # ---------- Settings & Diagnostics ----------
    with st.expander("OCR / Text Extraction Settings & Diagnostics", expanded=False):
        env_detected = _cm_detect_env()
        colA, colB = st.columns(2)
        with colA:
            t_path = st.text_input(
                "Tesseract executable (tesseract.exe)",
                value=st.session_state.get("cm_tesseract_exe", env_detected.get("tesseract") or ""),
                help="Set if you installed Tesseract. Example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                key="cm_tesseract_exe_input"
            )
            if t_path:
                st.session_state["cm_tesseract_exe"] = t_path
        with colB:
            p_path = st.text_input(
                "Poppler bin folder",
                value=st.session_state.get("cm_poppler_bin", env_detected.get("poppler") or ""),
                help="Set if you installed Poppler. Example: C:\\Program Files\\poppler-xx\\Library\\bin",
                key="cm_poppler_bin_input"
            )
            if p_path:
                st.session_state["cm_poppler_bin"] = p_path

        st.caption(f"Auto-detected → Tesseract: {'found' if env_detected.get('tesseract') else 'not found'} | Poppler: {'found' if env_detected.get('poppler') else 'not found'}")

        if platform.system() == "Windows":
            st.markdown("**Install commands (Windows)**")
            st.code(
                "winget install UB-Mannheim.Tesseract-OCR\n"
                "choco install poppler\n"
                "pip install pytesseract pdf2image pillow pdfminer.six PyPDF2",
                language="bash"
            )
        else:
            st.markdown("**Install packages**")
            st.code("sudo apt-get install tesseract-ocr poppler-utils\npip install pytesseract pdf2image pillow pdfminer.six PyPDF2", language="bash")

        if st.button("Run OCR diagnostics", key="cm_diag_btn"):
            msgs = []
            # imports
            for mod in ("PyPDF2", "pdfminer.six", "pytesseract", "pdf2image", "Pillow"):
                try:
                    __import__(mod.split(".")[0])
                    msgs.append(f"✅ import {mod}")
                except Exception as e:
                    msgs.append(f"❌ import {mod}: {e}")
            # env
            env_now = _cm_detect_env()
            msgs.append(f"Tesseract path: {env_now.get('tesseract') or 'not set'}")
            msgs.append(f"Poppler path: {env_now.get('poppler') or 'not set'}")
            # version + simple OCR test
            try:
                import pytesseract
                if env_now.get("tesseract"):
                    pytesseract.pytesseract.tesseract_cmd = env_now["tesseract"]
                v = pytesseract.get_tesseract_version()
                msgs.append(f"Tesseract version: {v}")
                try:
                    from PIL import Image, ImageDraw
                    img = Image.new("RGB", (600, 120), "white")
                    d = ImageDraw.Draw(img)
                    d.text((10, 40), "408-PP-001 TEST", fill="black")
                    txt = pytesseract.image_to_string(img)
                    msgs.append("OCR sample: " + (txt.strip() or "<empty>"))
                except Exception as e:
                    msgs.append(f"OCR sample failed: {e}")
            except Exception as e:
                msgs.append(f"Tesseract not usable: {e}")
            st.write("\n".join(msgs))

    # ---------- Filters ----------
    today = _date.today()
    f_from_cm = st.date_input("From", value=today - timedelta(days=90), key="cm_from")
    f_to_cm   = st.date_input("To", value=today, key="cm_to")
    tech_opts = ["Vibration","Thermography","Oil","Ultrasound","Other"]
    sev_opts  = ["Green","Amber","Red"]
    f_techs   = st.multiselect("Technique(s)", options=tech_opts, key="cm_ftechs")
    f_sevs    = st.multiselect("Severity", options=sev_opts, key="cm_fsevs")

    st.markdown("---")

    # ---------- Upload & Analyze ----------
    up_files = st.file_uploader("Upload CM report(s): PDF / PNG / JPG", type=["pdf","png","jpg","jpeg"], accept_multiple_files=True, key="cm_upl")
    alias_map = _cm_load_alias_map()
    known_assets = sorted(list(st.session_state.get("assets", {}).keys()))

    if up_files:
        for idx, up in enumerate(up_files):
            with st.expander(f"📄 {up.name}", expanded=False):
                raw = up.read(); up.seek(0)
                fhash = hashlib.sha256(raw).hexdigest()[:16]
                tech_hint, date_hint = _cm_guess_from_filename(up.name)

                with st.spinner("Analyzing document…"):
                    full_text, page_texts, notes = _cm_extract_text(up)
                st.caption("Analysis notes: " + ", ".join(notes))

                cands, page_hits = _cm_tag_candidates_with_pages(full_text, up.name, page_texts)
                matches = _cm_match_against_registry(cands, known_assets, alias_map)

                st.write("**Detected asset tag candidates**")
                hdr = st.columns([3,3,2,2,2,2])
                hdr[0].markdown("**Candidate**")
                hdr[1].markdown("**Resolved Asset**")
                hdr[2].markdown("**Confidence**")
                hdr[3].markdown("**Reason**")
                hdr[4].markdown("**Pages**")
                hdr[5].markdown("**Save alias**")

                resolved, alias_flags = [], []
                if matches:
                    for i, m in enumerate(matches):
                        cand, canon, conf, reason = m["candidate"], m["canonical"], m["confidence"], m["reason"]
                        c1, c2, c3, c4, c5, c6 = st.columns([3,3,2,2,2,2])
                        c1.write(cand)
                        options = [""] + known_assets
                        default_ix = options.index(canon) if canon in options else 0
                        choice = c2.selectbox(f"res_{idx}_{i}", options=options, index=default_ix, key=f"cm_resolve_{idx}_{i}")
                        c3.write(f"{conf:.2f}")
                        c4.write(reason)
                        c5.write(", ".join([f"p{p}" for p in page_hits.get(cand, [])]) or "—")
                        save_alias = c6.checkbox(" ", key=f"cm_alias_{idx}_{i}", value=(reason in ("normalized","fuzzy")) and bool(choice))
                        resolved.append(choice); alias_flags.append((cand, choice, save_alias))
                else:
                    st.info("No tags detected. Select the asset manually below.")

                extra_asset = st.selectbox("Add asset (manual)", options=[""]+known_assets, key=f"cm_manual_{idx}")
                if extra_asset: resolved.append(extra_asset)

                st.markdown("**Report metadata**")
                m1, m2, m3 = st.columns(3)
                with m1:
                    date_val = st.date_input("Date", value=(pd.to_datetime(date_hint).date() if date_hint else today), key=f"cm_date_{idx}")
                with m2:
                    technique = st.selectbox("Technique", options=["","Vibration","Thermography","Oil","Ultrasound","Other"],
                                             index=(["","Vibration","Thermography","Oil","Ultrasound","Other"].index(tech_hint) if tech_hint in ["Vibration","Thermography","Oil","Ultrasound","Other"] else 0),
                                             key=f"cm_tech_{idx}")
                with m3:
                    severity = st.selectbox("Severity", options=["","Green","Amber","Red"], key=f"cm_sev_{idx}")

                m4, m5 = st.columns(2)
                with m4:
                    vendor  = st.text_input("Vendor", "", key=f"cm_vendor_{idx}")
                    analyst = st.text_input("Analyst", "", key=f"cm_analyst_{idx}")
                with m5:
                    next_rev = st.date_input("Next review date", value=None, key=f"cm_next_{idx}")
                diagnosis      = st.text_area("Diagnosis / Findings", key=f"cm_diag_{idx}", height=100)
                recommendation = st.text_area("Recommendation / Action", key=f"cm_rec_{idx}", height=100)

                if st.button("💾 Save report", key=f"cm_save_{idx}"):
                    final_assets = sorted(list({a for a in resolved if a}))
                    if not final_assets:
                        st.error("Select at least one asset to link this report.")
                    else:
                        up.seek(0)
                        fname = _cm_save_file(up)
                        for cand, chosen, flag in alias_flags:
                            if flag and cand and chosen:
                                _cm_upsert_alias(_cm_norm_tag(cand), _cm_norm_tag(chosen))
                        _cm_insert_report(
                            date_s=(date_val.isoformat() if isinstance(date_val, (_date, datetime)) else ""),
                            assets=final_assets,
                            technique=technique,
                            severity=severity,
                            vendor=vendor,
                            analyst=analyst,
                            diagnosis=diagnosis,
                            recommendation=recommendation,
                            next_review=(next_rev.isoformat() if next_rev else ""),
                            attachments=[fname],
                            file_hash=fhash,
                            filename=up.name
                        )
                        st.success(f"Saved. Linked assets: {', '.join(final_assets)}")
                        if "open_job_in_planner_from_cm" in globals():
                            try:
                                open_job_in_planner_from_cm(final_assets, diagnosis, recommendation, fname)
                                st.info("Planner job created from CM context.")
                            except Exception:
                                st.caption("Planner hook not available.")

    st.markdown("---")
    # ---------- KPIs + Saved list ----------
    df_reports = _cm_list_reports(f_from_cm, f_to_cm, f_techs, f_sevs)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Reports", len(df_reports))
    if not df_reports.empty:
        sev_vc = df_reports["severity"].value_counts()
        c2.metric("Red", int(sev_vc.get("Red", 0)))
        c3.metric("Amber", int(sev_vc.get("Amber", 0)))
        c4.metric("Green", int(sev_vc.get("Green", 0)))
    else:
        c2.metric("Red", 0); c3.metric("Amber", 0); c4.metric("Green", 0)

    if df_reports.empty:
        st.info("No CM reports saved for the selected filters.")
    else:
        st.markdown("#### Saved Reports")
        height = min(420, 120 + 28*len(df_reports))
        st.dataframe(df_reports, use_container_width=True, height=height)

 # =========================================================
 # 6) Component Life & Forecasts
 # =========================================================
 # --- replace only the contents of the "with t_comp:" block in Tab 4 with this ---
with t_comp:
    st.markdown("#### Component Life & Forecasts")

    tag_opts = sorted(assets_dict.keys())
    if not tag_opts:
        st.info("No assets available.")
    else:
        csel1, csel2 = st.columns([2,1])
        with csel1:
            comp_asset = st.selectbox("Select Asset", options=tag_opts, key="rv_comp_asset")
        with csel2:
            horizon_days = st.number_input("Horizon (days)", min_value=1, value=14, step=1, key="rv_comp_horizon")

        # avg daily hours from Tech Details (if present)
        def _avg_daily_hours_for(tag: str, default_hours: float = 16.0) -> float:
            try:
                a = assets_dict.get(tag, {})
                for row in a.get("Technical Details", []) or []:
                    if str(row.get("Parameter","")).strip().lower() == "avg daily (h)":
                        val = float(pd.to_numeric(row.get("Value", default_hours), errors="coerce"))
                        return val if val > 0 else default_hours
            except Exception:
                pass
            return float(default_hours)

        avg_daily_h = _avg_daily_hours_for(comp_asset, 16.0)
        st.caption(f"Avg Daily Hours (from Technical Details if present): {avg_daily_h:.1f} h/d")

        # ----- components source: session → sync from BOM → ephemeral BOM map -----
        comp_store = st.session_state.get("component_life", {}) or {}
        comp_map = (comp_store.get(comp_asset, {}) or {}).get("components", {}) or {}

        if not comp_map:
            # try to populate session store using your existing sync helper
            if "build_bom_table" in globals():
                bom_df = build_bom_table(assets_dict.get(comp_asset, {}))
            else:
                # safe fallback if helper isn't present
                bom_df = pd.DataFrame((assets_dict.get(comp_asset, {}) or {}).get("BOM Table", []))

            if "sync_components_runtime_from_bom" in globals():
                try:
                    sync_components_runtime_from_bom(comp_asset, bom_df, strict=False)  # why: populate once, keep state consistent
                    comp_store = st.session_state.get("component_life", {}) or {}
                    comp_map = (comp_store.get(comp_asset, {}) or {}).get("components", {}) or {}
                except Exception:
                    pass

            # if still empty, build an ephemeral mapping directly from BOM (no state mutation)
            if not comp_map and isinstance(bom_df, pd.DataFrame) and not bom_df.empty:
                comp_map = {}
                for _, r in bom_df.iterrows():
                    cmms = str(r.get("CMMS MATERIAL CODE","") or "").strip()
                    name = str(r.get("DESCRIPTION","") or cmms or "Component").strip()
                    crit = str(r.get("CRITICALITY","Medium") or "Medium")
                    mtbf = float(pd.to_numeric(r.get("MTBF (H)", 0.0), errors="coerce") or 0.0)
                    key = cmms if cmms else (str(r.get("OEM PART NUMBER","") or "").strip() or name)
                    comp_map[key] = {"name": name, "cmms": cmms, "criticality": crit, "mtbf_h": mtbf}

        # ----- build forecast table -----
        if not comp_map:
            st.info("No components found for the selected asset. Add BOM rows or sync via Runtime Tracker.")
        else:
            rows = []
            for key, node in comp_map.items():
                mtbf = float(pd.to_numeric((node or {}).get("mtbf_h", 0.0), errors="coerce") or 0.0)
                run_since = 0.0  # if you later persist component runtime, map it here
                rem_now = max(mtbf - run_since, 0.0)
                proj = avg_daily_h * float(horizon_days)
                rem_after = max(rem_now - proj, 0.0)
                ratio_used = (run_since/mtbf) if mtbf > 0 else 0.0
                status = "🟢 Healthy" if ratio_used < 0.80 else ("🟠 Plan for maintenance" if ratio_used < 1.0 else "🔴 Overdue for maintenance")
                rows.append({
                    "Component": (node or {}).get("name", key),
                    "CMMS": (node or {}).get("cmms",""),
                    "Criticality": (node or {}).get("criticality","Medium"),
                    "MTBF (H)": mtbf,
                    "Run Since": run_since,
                    "Remaining Now": rem_now,
                    f"Remaining After {int(horizon_days)}d": rem_after,
                    "Status": status
                })
            comp_df = pd.DataFrame(rows)
            comp_df_sorted = comp_df.sort_values(["Status","Remaining Now"]).reset_index(drop=True)
            st.dataframe(comp_df_sorted, use_container_width=True, height=auto_table_height(len(comp_df_sorted), min_rows=6))

            # status donut
            vs = comp_df["Status"].value_counts(dropna=False)
            fig = go.Figure(data=[go.Pie(labels=[f"{k} ({int(v)})" for k,v in vs.items()],
                                         values=[int(v) for v in vs.values], hole=0.5, textinfo="label+percent")])
            fig.update_layout(showlegend=True, height=280, margin=dict(t=10,b=10,l=10,r=10), title="Component Status Mix")
            st.plotly_chart(fig, use_container_width=True, key="rv_comp_donut")

            st.caption("Note: 'Run Since' defaults to 0 here. Populate per-component runtime in Tab 2 to refine forecasts.")

# =========================================================
# 7) Data Quality & Assumptions
# =========================================================
    with t_quality:
        st.markdown("#### Data Quality & Assumptions (no costs)")

        issues = []

        # Runtime checks
        if not rv_rt.empty:
            miss_mtbf = rv_rt[pd.to_numeric(rv_rt["MTBF (Hours)"], errors="coerce").fillna(0.0) <= 0]
            if not miss_mtbf.empty:
                issues.append({"Area":"Runtime","Issue":"MTBF ≤ 0","Count": len(miss_mtbf), "Example": miss_mtbf.iloc[0]["Asset Tag"]})
        else:
            st.caption("No runtime rows after filter.")

        # History checks
        if not history_df0.empty:
            bad_dates = history_df0[history_df0["Date of Maintenance"].isna()]
            if not bad_dates.empty:
                issues.append({"Area":"History","Issue":"Unparseable dates","Count": len(bad_dates),
                               "Example": bad_dates.iloc[0].get("WO Number","")})
            neg_hours_cols = ["Hours Run Since Last Repair","Labour Hours","Asset Downtime Hours"]
            for c in neg_hours_cols:
                bad = history_df0[pd.to_numeric(history_df0[c], errors="coerce").fillna(0.0) < 0]
                if not bad.empty:
                    issues.append({"Area":"History","Issue":f"Negative {c}","Count": len(bad), "Example": bad.iloc[0].get("WO Number","")})
            # Duplicate WO+Spare
            df = history_df0.copy()
            mask = df["WO Number"].astype(str).ne("") & df["Spares Used"].astype(str).ne("")
            dups = df[mask].groupby(["WO Number","Spares Used"]).size().reset_index(name="n")
            bad = dups[dups["n"]>1]
            if not bad.empty:
                issues.append({"Area":"History","Issue":"Duplicate WO+Spare pairs","Count": int(bad["n"].sum()),
                               "Example": f"{bad.iloc[0]['WO Number']} / {bad.iloc[0]['Spares Used']}"})
        else:
            st.caption("No history rows loaded.")

        # Unknown CMMS (not present in BOM of selected assets)
        if f_assets:
            known_cmms = set()
            for t in f_assets:
                bom = pd.DataFrame((assets_dict.get(t, {}) or {}).get("BOM Table", []))
                if not bom.empty and "CMMS MATERIAL CODE" in bom.columns:
                    known_cmms.update([str(x).strip() for x in bom["CMMS MATERIAL CODE"].dropna().unique()])
            if known_cmms and not rv_hist.empty:
                used = set([str(x).strip() for x in rv_hist.get("Spares Used", pd.Series([], dtype=str)).dropna().unique()])
                unknown = sorted(list(used - known_cmms))
                if unknown:
                    issues.append({"Area":"Mapping","Issue":"Spares Used not found in BOM (current assets filter)","Count": len(unknown),
                                   "Example": unknown[0]})

        # Render issues
        if issues:
            st.markdown("##### Issues found")
            st.dataframe(pd.DataFrame(issues), use_container_width=True, height=auto_table_height(len(issues), min_rows=4))
        else:
            st.success("No data quality issues detected with current filters.")

        st.markdown("##### Assumptions")
        st.write("- MTTR uses **Asset Downtime Hours** average as proxy.")
        st.write("- Reliability curve R(t)=exp(-t/MTBF) estimated from **consecutive failure gaps**.")
        st.write("- Component 'Run Since' defaults to 0 here; refine via Tab 2 if available.")
        st.write("- No financial metrics on this tab by design.")

# =========================
# TAB 5 — Financial Analysis (Fast, Robust, All Subtabs Included)
# =========================
with tab5:
    # ---------------- Imports ----------------
    import os, re, json, math
    from datetime import datetime, timedelta, date as _date
    from collections import defaultdict
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import streamlit as st

    # ---------------- Lazy heavy libs ----------------
    def _get_sm():
        """Lazy statsmodels import; fallback to tiny OLS with add_constant."""
        try:
            import statsmodels.api as _sm
            return _sm
        except Exception:
            import numpy as _np
            class _OLSModel:
                def __init__(self, y, X):
                    self.y=_np.asarray(y,float); self.X=_np.asarray(X,float); self.beta=None
                def fit(self):
                    XtX=self.X.T@self.X
                    try: self.beta=_np.linalg.solve(XtX,self.X.T@self.y)
                    except _np.linalg.LinAlgError: self.beta=_np.linalg.pinv(XtX)@(self.X.T@self.y)
                    return self
                def predict(self,Xn): return _np.asarray(Xn,float)@self.beta
            class _sm:
                @staticmethod
                def add_constant(X):
                    X=_np.asarray(X,float); X=X.reshape(-1,1) if X.ndim==1 else X
                    return _np.hstack([_np.ones((X.shape[0],1)), X])
                @staticmethod
                def OLS(y,X): return _OLSModel(y,X)
            return _sm()

    def _get_kmeans():
        """Lazy KMeans import; fallback to small NumPy k-means."""
        try:
            from sklearn.cluster import KMeans as _SKKMeans
            return _SKKMeans
        except Exception:
            import numpy as _np
            class _KMeans:
                def __init__(self, n_clusters=3, n_init=5, max_iter=50, random_state=0):
                    self.k=n_clusters; self.n_init=n_init; self.max_iter=max_iter
                    self.rng=_np.random.default_rng(random_state)
                def fit_predict(self, X):
                    X=_np.asarray(X,float)
                    def _init_pp(X,k):
                        n=X.shape[0]; C=_np.empty((k,X.shape[1])); i0=self.rng.integers(0,n); C[0]=X[i0]
                        for i in range(1,k):
                            d2=_np.min(_np.sum((X[:,None,:]-C[None,:i,:])**2,axis=2),axis=1)
                            p=d2/(d2.sum() if d2.sum()>0 else 1.0)
                            C[i]=X[self.rng.choice(n,p=p)]
                        return C
                    best_lbl,best_inertia=None,float("inf")
                    for _ in range(self.n_init):
                        C=_init_pp(X,self.k); lbl=_np.zeros(X.shape[0],int)
                        for _ in range(self.max_iter):
                            D=_np.sum((X[:,None,:]-C[None,:,:])**2,axis=2)
                            new=_np.argmin(D,axis=1)
                            if _np.all(new==lbl): break
                            lbl=new
                            for j in range(self.k):
                                pts=X[lbl==j]
                                if len(pts): C[j]=pts.mean(axis=0)
                        inertia=float(_np.sum((X-C[lbl])**2))
                        if inertia<best_inertia: best_inertia,best_lbl=inertia,lbl.copy()
                    return best_lbl
            return _KMeans

    # ---------------- Helpers (shared) ----------------
    def to_num(x, default: float = 0.0) -> float:
        try:
            if x is None or (isinstance(x,str) and not x.strip()): return float(default)
            if isinstance(x,(int,float,np.floating)): return float(x)
            s=str(x).strip().lower().replace(",","")
            if s in {"","-","—","none","nan","n/a","na"}: return float(default)
            return float(s)
        except Exception: return float(default)

    def build_bom_table(asset: dict) -> pd.DataFrame:
        return pd.DataFrame(columns=["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE"])

    def auto_table_height(n_rows: int, min_rows: int = 3, max_rows: int = 999, row_px: int = 38, header_px: int = 42, padding_px: int = 16) -> int:
        rows = min(max(n_rows, min_rows), max_rows)
        return int(header_px + padding_px + rows * row_px)

    def _avg_daily_hours_for(tag: str, default_hours: float = 16.0) -> float:
        """Reads avg daily (h) from assets_dict → Technical Details; default if missing."""
        try:
            a = assets_dict.get(tag, {})
            for row in a.get("Technical Details", []) or []:
                if str(row.get("Parameter","")).strip().lower() == "avg daily (h)":
                    val = to_num(row.get("Value", default_hours), default_hours)
                    return float(val) if val > 0 else default_hours
        except Exception:
            pass
        return float(default_hours)

    def fin_section(tag: str) -> str:
        m = re.match(r"^(\d{3})", str(tag or ""))
        return m.group(1) if m else "Other"

    def fin_get_bom_price(asset_tag: str, cmms: str) -> float:
        a = assets_dict.get(asset_tag, {}) or {}
        if not a: return 0.0
        df = build_bom_table(a)
        if df.empty: return 0.0
        for col in ["CMMS MATERIAL CODE","PRICE"]:
            if col not in df.columns: return 0.0
        row = df[df["CMMS MATERIAL CODE"].astype(str).str.strip() == str(cmms).strip()]
        if row.empty: return 0.0
        return float(to_num(row.iloc[0]["PRICE"], 0.0))

    def fin_get_unit_price(cmms: str, asset_tag: str) -> float:
        cmms = str(cmms or "").strip()
        if not cmms: return 0.0
        if cmms in _price_map: return float(_price_map[cmms])
        return fin_get_bom_price(asset_tag, cmms)

    def fin_is_callout(row: pd.Series, rules: dict) -> bool:
        try:
            code = str(row.get("Maintenance Code","")).upper().strip()
            dt = row.get("Date of Maintenance")
            is_weekend = False
            if isinstance(dt, pd.Timestamp):
                dow = int(dt.dayofweek)  # 0=Mon ... 6=Sun
                is_weekend = (dow >= 5) and bool(rules.get("callout_on_weekend", True))
            if rules.get("treat_pm00_as_callout", True) and code == "PM00": return True
            return bool(is_weekend)
        except Exception:
            return False

    # ---------------- Data sources from other tabs ----------------
    assets_dict = st.session_state.get("assets", {}) or {}
    runtime_df0 = st.session_state.get("runtime_df", pd.DataFrame()).copy()
    history_df0 = st.session_state.get("history_df", pd.DataFrame()).copy()
    comp_store  = st.session_state.get("component_life", {}) or {}

    # Normalize history df
    if not history_df0.empty:
        history_df0["Date of Maintenance"] = pd.to_datetime(history_df0.get("Date of Maintenance"), errors="coerce")
        for c in ["QTY","Hours Run Since Last Repair","Labour Hours","Asset Downtime Hours"]:
            history_df0[c] = pd.to_numeric(history_df0.get(c, 0), errors="coerce").fillna(0.0)
        for c in ["Asset Tag","Maintenance Code","Spares Used","WO Number","Functional Location"]:
            if c in history_df0.columns: history_df0[c] = history_df0[c].astype(str)

    # ---------------- Rates & Assumptions store ----------------
    DATA_DIR = globals().get("DATA_DIR","data")
    FIN_RATES_PATH = os.path.join(DATA_DIR, "financial_rates.json")
    os.makedirs(os.path.dirname(FIN_RATES_PATH), exist_ok=True)

    FIN_DEFAULTS = {
        "labour": {"regular_rate": 400.0, "callout_rate": 600.0, "callout_min_hours": 4.0, "callout_uplift_pct": 0.15},
        "price_list": [],  # [{"cmms":"AM0525666","unit_price":150.0,"vendor":"Weir"}]
        "tariffs": {"energy_kwh_rate": 1.80, "water_m3_rate": 25.00},
        "working_hours": {
            "weekdays": [1,2,3,4,5], "day_start": "07:00", "day_end":"17:00",
            "callout_on_weekend": True, "treat_pm00_as_callout": True
        },
        "meters": [],  # [{"asset_tag":"408-PP-001","kwh_per_hour":120.0,"m3_per_hour":6.5}]
        "assumptions": {"avg_daily_hours_default": 16.0},
        "duty_groups": []  # [{"group":"PumpSet A","members":[{"asset_tag":"A","util_share":0.75}, ...]}]
    }

    def fin_load_rates() -> dict:
        try:
            d = json.load(open(FIN_RATES_PATH,"r",encoding="utf-8")) if os.path.exists(FIN_RATES_PATH) else {}
        except Exception:
            d = {}
        out = json.loads(json.dumps(FIN_DEFAULTS))
        out.update(d if isinstance(d,dict) else {})
        return out

    def fin_save_rates(d: dict):
        try:
            with open(FIN_RATES_PATH, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            st.success("Saved Financial Rates & Assumptions.")
            # Optional: clear caches after save
            fin_filter_history_cached.clear()
            fin_compute_cost_rows_cached.clear()
        except Exception as e:
            st.error(f"Could not save rates: {e}")

    fin_rates = fin_load_rates()

    # Lookups from rates
    _price_map  = {str(r.get("cmms","")).strip(): float(to_num(r.get("unit_price",0.0),0.0)) for r in fin_rates.get("price_list", []) if str(r.get("cmms","")).strip()}
    _vendor_map = {str(r.get("cmms","")).strip(): str(r.get("vendor","") or "Unknown") for r in fin_rates.get("price_list", []) if str(r.get("cmms","")).strip()}
    _meters_map = {str(r.get("asset_tag","")).strip(): (float(to_num(r.get("kwh_per_hour",0.0),0.0)), float(to_num(r.get("m3_per_hour",0.0),0.0))) for r in fin_rates.get("meters", []) if str(r.get("asset_tag","")).strip()}

    # ---------------- Cache heavy transforms ----------------
    @st.cache_data(ttl=180, show_spinner=False)
    def fin_filter_history_cached(h: pd.DataFrame, _from, _to, _assets_tuple):
        if h.empty: return h.copy()
        d = h.copy()
        if _from: d = d[d["Date of Maintenance"] >= pd.to_datetime(_from)]
        if _to:   d = d[d["Date of Maintenance"] <= pd.to_datetime(_to)]
        if _assets_tuple: d = d[d["Asset Tag"].astype(str).isin(list(_assets_tuple))]
        return d.reset_index(drop=True)

    def fin_filter_history(h, _from, _to, _assets):
        return fin_filter_history_cached(h, _from, _to, tuple(_assets) if _assets else ())

    @st.cache_data(ttl=180, show_spinner=False)
    def fin_compute_cost_rows_cached(df_hist: pd.DataFrame, period_days: int, fin_rates: dict, meters_map: dict) -> pd.DataFrame:
        if df_hist.empty:
            return pd.DataFrame(columns=[
                "Date of Maintenance","Asset Tag","Maintenance Code","WO Number","Spares Used","QTY",
                "spares_cost","labour_cost_regular","labour_cost_callout","energy_cost","water_cost","callout_flag"
            ])
        lab = fin_rates.get("labour", {})
        lab_reg = float(to_num(lab.get("regular_rate", 0.0), 0.0))
        lab_co  = float(to_num(lab.get("callout_rate", 0.0), 0.0))
        co_min  = float(to_num(lab.get("callout_min_hours", 0.0), 0.0))
        co_up   = float(to_num(lab.get("callout_uplift_pct", 0.0), 0.0))
        en_rate = float(to_num(fin_rates.get("tariffs",{}).get("energy_kwh_rate", 0.0), 0.0))
        wa_rate = float(to_num(fin_rates.get("tariffs",{}).get("water_m3_rate", 0.0), 0.0))

        out = []
        for _, r in df_hist.iterrows():
            at = str(r.get("Asset Tag",""))
            qty = float(to_num(r.get("QTY",0.0),0.0))
            cm  = str(r.get("Spares Used","")).strip()
            # spares from price list first; else 0 (keep lightweight in cache)
            sp_cost = float(qty) * float(_price_map.get(cm, 0.0)) if cm else 0.0

            is_co = fin_is_callout(r, fin_rates.get("working_hours", {}))
            lab_hrs = float(to_num(r.get("Labour Hours", 0.0), 0.0))
            if is_co:
                billable = max(lab_hrs, co_min)
                lab_cost_regular, lab_cost_callout = 0.0, billable * lab_co * (1.0 + co_up)
            else:
                lab_cost_regular, lab_cost_callout = lab_hrs * lab_reg, 0.0

            kwh_h, m3_h = meters_map.get(at, (0.0, 0.0))
            avg_daily = _avg_daily_hours_for(at, fin_rates.get("assumptions",{}).get("avg_daily_hours_default", 16.0))
            run_h_est = float(avg_daily) * float(max(period_days, 0))
            energy_cost = (kwh_h * run_h_est * en_rate) if kwh_h > 0 else 0.0
            water_cost  = (m3_h  * run_h_est * wa_rate) if m3_h  > 0 else 0.0

            out.append({
                "Date of Maintenance": r.get("Date of Maintenance"),
                "Asset Tag": at,
                "Maintenance Code": str(r.get("Maintenance Code","")),
                "WO Number": str(r.get("WO Number","")),
                "Spares Used": cm, "QTY": qty,
                "spares_cost": sp_cost,
                "labour_cost_regular": lab_cost_regular,
                "labour_cost_callout": lab_cost_callout,
                "energy_cost": energy_cost, "water_cost": water_cost,
                "callout_flag": bool(is_co),
            })
        return pd.DataFrame(out)

    # ---------------- Shared filters ----------------
    st.subheader("💡 Financial Analysis")

    all_tags = sorted([k for k in assets_dict.keys() if k])
    today = _date.today()
    default_from = today - timedelta(days=180)

    colF1, colF2, colF3 = st.columns([1.2, 1.2, 2.4])
    with colF1:
        fa_from = st.date_input("From", value=default_from, key="fa_from")
    with colF2:
        fa_to = st.date_input("To", value=today, key="fa_to")
    with colF3:
        fa_assets = st.multiselect("Filter Assets", options=all_tags, default=[], key="fa_assets")

    fa_hist = fin_filter_history(history_df0, fa_from, fa_to, fa_assets)
    period_days = (pd.to_datetime(fa_to) - pd.to_datetime(fa_from)).days + 1
    fa_cost_rows = fin_compute_cost_rows_cached(fa_hist, period_days, fin_rates, _meters_map)

    # ---------------- Subtabs ----------------
    sub_exec, sub_rates, sub_break, sub_perf, sub_budget = st.tabs([
        "💼 Executive Cost Summary", "⚙️ Rates & Assumptions", "🛠 Maintenance Cost Breakdown", "📈 Performance Cost & Efficiency", "🗓 Budget Planner"
    ])

    # =========================================================
    # 1) 💼 Executive Cost Summary
    # =========================================================
    with sub_exec:
        st.markdown("### Executive Cost Summary")
        if fa_cost_rows.empty:
            st.info("No history rows in selected period. Load Maintenance Records to unlock Financial Analysis.")
        else:
            # Sensitivities
            st.markdown("#### Sensitivity Analysis (What-If)")
            sens_col1, sens_col2 = st.columns(2)
            with sens_col1:
                sens_runtime = st.slider("Runtime Adjustment (%)", -50, 200, 0, step=5, key="fa_exec_runtime_pct") / 100.0
            with sens_col2:
                sens_avail = st.slider("Availability Adjustment (%)", -20, 20, 0, step=5, key="fa_exec_avail_pct") / 100.0

            d = fa_cost_rows.copy()
            d[["energy_cost", "water_cost"]] *= (1 + sens_runtime) * (1 + sens_avail)
            d[["spares_cost", "labour_cost_regular", "labour_cost_callout"]] *= (1 + sens_runtime)

            gcol1, gcol2, gcol3 = st.columns([1.2, 1.2, 1.6])
            with gcol1:
                gran = st.radio("Granularity", ["Month","Quarter"], horizontal=True, key="fa_gran")
            with gcol2:
                include_cats = st.multiselect(
                    "Include Categories",
                    options=["Spares","Labour (Regular)","Call-out","Energy","Water"],
                    default=["Spares","Labour (Regular)","Call-out","Energy","Water"],
                    key="fa_cats"
                )
            with gcol3:
                plan_toggle = st.radio(
                    "Planned vs Breakdown",
                    ["All","Planned only (PM01–PM04)","Breakdown only (PM00)"],
                    key="fa_plan_toggle"
                )

            def _is_planned(code: str) -> bool:
                return str(code or "").upper().strip() in {"PM01","PM02","PM03","PM04"}

            if plan_toggle == "Planned only (PM01–PM04)":
                d = d[d["Maintenance Code"].apply(_is_planned)]
            elif plan_toggle == "Breakdown only (PM00)":
                d = d[d["Maintenance Code"].astype(str).str.upper().eq("PM00")]

            d["period"] = pd.to_datetime(d["Date of Maintenance"]).dt.to_period("M" if gran=="Month" else "Q").astype(str)

            parts = []
            if "Spares" in include_cats: parts.append("spares_cost")
            if "Labour (Regular)" in include_cats: parts.append("labour_cost_regular")
            if "Call-out" in include_cats: parts.append("labour_cost_callout")
            if "Energy" in include_cats: parts.append("energy_cost")
            if "Water" in include_cats: parts.append("water_cost")
            if not parts: parts = ["spares_cost","labour_cost_regular","labour_cost_callout","energy_cost","water_cost"]

            d["total_cost"] = d[parts].sum(axis=1)

            # Cumulative line
            byp = d.groupby("period", as_index=False)["total_cost"].sum().sort_values("period")
            byp["cumulative"] = byp["total_cost"].cumsum()
            figL = go.Figure([go.Scatter(x=byp["period"], y=byp["cumulative"], mode="lines+markers", name="Cumulative Spend")])
            figL.update_layout(title="Cumulative Spend over Time", height=340, margin=dict(t=36,b=10,l=10,r=10),
                               xaxis_title=("Month" if gran=="Month" else "Quarter"), yaxis_title="Cost (R)")
            st.plotly_chart(figL, use_container_width=True, key="fa_exec_cumline")

            # MTBF quick alerts
            st.markdown("#### MTBF Risk Alerts (Quick Scan)")
            rt = runtime_df0[runtime_df0["Asset Tag"].isin(fa_assets)] if fa_assets else runtime_df0
            if not rt.empty:
                risks = rt[rt["STATUS"].str.contains("Overdue|Plan", na=False)]
                if not risks.empty:
                    st.warning(f"{len(risks)} assets at risk based on MTBF/remaining hours.")
                    st.dataframe(risks[["Asset Tag", "Remaining Hours", "STATUS"]], use_container_width=True)
                else:
                    st.success("No immediate MTBF risks in selected assets.")
            else:
                st.info("No runtime data for MTBF alerts.")

            # Variance vs prior window
            try:
                pd_to   = pd.to_datetime(fa_to)
                pd_from = pd.to_datetime(fa_from)
                span    = (pd_to - pd_from).days + 1
                prev_to = pd_from - timedelta(days=1)
                prev_from = prev_to - timedelta(days=span-1)
                prev_hist = fin_filter_history(history_df0, prev_from, prev_to, fa_assets)
                prev_rows = fin_compute_cost_rows_cached(prev_hist, span, fin_rates, _meters_map) if not prev_hist.empty else pd.DataFrame(columns=fa_cost_rows.columns)
            except Exception:
                prev_rows = pd.DataFrame(columns=fa_cost_rows.columns)

            def _sum_parts(df: pd.DataFrame) -> dict:
                if df.empty: return {"spares":0.0,"lab_reg":0.0,"lab_co":0.0,"energy":0.0,"water":0.0}
                return {
                    "spares": float(df["spares_cost"].sum()),
                    "lab_reg": float(df["labour_cost_regular"].sum()),
                    "lab_co": float(df["labour_cost_callout"].sum()),
                    "energy": float(df["energy_cost"].sum()),
                    "water": float(df["water_cost"].sum()),
                }

            cur = _sum_parts(d); prv = _sum_parts(prev_rows)
            diff = {k: cur[k]-prv[k] for k in cur}
            wf_labels = ["Start (Prev)","Spares","Labour (Regular)","Call-out","Energy","Water","End (Current)"]
            wf_values = [prv["spares"]+prv["lab_reg"]+prv["lab_co"]+prv["energy"]+prv["water"],
                         diff["spares"], diff["lab_reg"], diff["lab_co"], diff["energy"], diff["water"],
                         cur["spares"]+cur["lab_reg"]+cur["lab_co"]+cur["energy"]+cur["water"]]
            figW = go.Figure(go.Waterfall(x=wf_labels,
                                          measure=["absolute","relative","relative","relative","relative","relative","total"],
                                          y=wf_values))
            figW.update_layout(title="Variance vs Prior Period (Waterfall)", height=320, margin=dict(t=36,b=10,l=10,r=10),
                               yaxis_title="Δ Cost (R)")
            st.plotly_chart(figW, use_container_width=True, key="fa_exec_waterfall")

            # Sunburst (heavy) toggle
            if st.checkbox("Build Sunburst (heavy)", value=False, key="fa_exec_sunburst_toggle"):
                d["bucket"]  = np.where(d["Maintenance Code"].astype(str).str.upper().eq("PM00"), "Breakdown", "Planned")
                d["section"] = d["Asset Tag"].map(fin_section)
                sb = d.copy(); sb["amount"] = sb["total_cost"]
                labels, parents, values, index = [], [], [], {}
                def _add_node(name, parent):
                    key=(name,parent)
                    if key in index: return index[key]
                    idx=len(labels); labels.append(name); parents.append(parent); values.append(0.0); index[key]=idx; return idx
                _add_node("Spend", "")
                for (b, c, s, a), g in sb.groupby(["bucket","Maintenance Code","section","Asset Tag"]):
                    _add_node(b, "Spend"); _add_node(str(c), b); _add_node(str(s), str(c))
                    na = _add_node(str(a), str(s)); values[na] += float(g["amount"].sum())
                figSB = go.Figure(go.Sunburst(labels=labels, parents=parents, values=values, branchvalues="total", maxdepth=4))
                figSB.update_layout(title="Spend Composition — Planned/Breakdown → Code → Section → Asset", height=420, margin=dict(t=36,b=10,l=10,r=10))
                st.plotly_chart(figSB, use_container_width=True, key="fa_exec_sunburst")

            # Pareto
            pa = d.groupby("Asset Tag", as_index=False)["total_cost"].sum().sort_values("total_cost", ascending=False).head(10)
            pa["cum_pct"] = 100.0 * pa["total_cost"].cumsum() / max(pa["total_cost"].sum(), 1.0)
            figP = go.Figure()
            figP.add_bar(x=pa["Asset Tag"], y=pa["total_cost"], name="Spend")
            figP.add_trace(go.Scatter(x=pa["Asset Tag"], y=pa["cum_pct"], yaxis="y2", mode="lines+markers", name="Cumulative %"))
            figP.update_layout(
                title="Pareto — Top Assets by Spend (80/20)",
                height=360, margin=dict(t=36,b=10,l=10,r=10),
                xaxis_title="Asset Tag",
                yaxis=dict(title="Cost (R)"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0,100]),
                shapes=[dict(type="line", x0=-0.5, x1=len(pa)-0.5, y0=80, y1=80, yref="y2", line=dict(dash="dash"))]
            )
            st.plotly_chart(figP, use_container_width=True, key="fa_exec_pareto")

    # =========================================================
    # 2) ⚙️ Rates & Assumptions (BOM Price List, Duty/Standby)
    # =========================================================
    with sub_rates:
        st.markdown("### Rates & Assumptions")

        # Labour
        st.markdown("#### Labour Rate Card")
        l = fin_rates.get("labour", {}).copy()
        c1,c2,c3,c4 = st.columns(4)
        with c1: l["regular_rate"] = st.number_input("Regular Rate (R/h)", min_value=0.0, value=float(l.get("regular_rate",400.0)), step=10.0, key="fa_lr_regular")
        with c2: l["callout_rate"] = st.number_input("Call-out Rate (R/h)", min_value=0.0, value=float(l.get("callout_rate",600.0)), step=10.0, key="fa_lr_callout")
        with c3: l["callout_min_hours"] = st.number_input("Call-out Min Hours", min_value=0.0, value=float(l.get("callout_min_hours",4.0)), step=0.5, key="fa_lr_min")
        with c4: l["callout_uplift_pct"] = st.number_input("Call-out Uplift (%)", min_value=0.0, value=float(100*float(l.get("callout_uplift_pct",0.15))), step=1.0, key="fa_lr_uplift")/100.0
        fin_rates["labour"] = l

        # Tariffs
        st.markdown("#### Tariffs")
        t = fin_rates.get("tariffs", {}).copy()
        t1,t2 = st.columns(2)
        with t1: t["energy_kwh_rate"] = st.number_input("Energy Tariff (R/kWh)", min_value=0.0, value=float(t.get("energy_kwh_rate",1.8)), step=0.05, key="fa_tar_energy")
        with t2: t["water_m3_rate"]   = st.number_input("Water Tariff (R/m³)",   min_value=0.0, value=float(t.get("water_m3_rate",25.0)), step=0.5,  key="fa_tar_water")
        fin_rates["tariffs"] = t

        # Working Hours & Call-out Rules
        st.markdown("#### Working Hours & Call-out Rules")
        wh = fin_rates.get("working_hours", {}).copy()
        w1, w2, w3, w4 = st.columns([1,1,1,1])
        with w1:
            sel_days = st.multiselect(
                "Workdays",
                options=[1,2,3,4,5,6,7],
                default=wh.get("weekdays",[1,2,3,4,5]),
                format_func=lambda d: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d-1],
                key="fa_wh_days"
            )
            wh["weekdays"] = sel_days
        with w2: wh["day_start"] = st.text_input("Day Start (HH:MM)", value=wh.get("day_start","07:00"), key="fa_wh_start")
        with w3: wh["day_end"]   = st.text_input("Day End (HH:MM)",   value=wh.get("day_end","17:00"), key="fa_wh_end")
        with w4: wh["callout_on_weekend"] = st.checkbox("Call-out on weekends", value=bool(wh.get("callout_on_weekend", True)), key="fa_wh_weekend")
        wh["treat_pm00_as_callout"] = st.checkbox("Treat PM00 as call-out", value=bool(wh.get("treat_pm00_as_callout", True)), key="fa_wh_pm00")
        fin_rates["working_hours"] = wh

        # Meters
        st.markdown("#### Meters (per asset)")
        mt_cols = ["asset_tag","kwh_per_hour","m3_per_hour"]
        df_mt = pd.DataFrame(fin_rates.get("meters", [])) if isinstance(fin_rates.get("meters"), list) else pd.DataFrame(columns=mt_cols)
        if df_mt.empty and assets_dict:
            df_mt = pd.DataFrame([{"asset_tag": t, "kwh_per_hour": 0.0, "m3_per_hour": 0.0} for t in sorted(assets_dict.keys())])
        cfg_mt = {
            "asset_tag": st.column_config.SelectboxColumn(options=[""]+sorted(assets_dict.keys())),
            "kwh_per_hour": st.column_config.NumberColumn(min_value=0.0, step=0.1),
            "m3_per_hour":  st.column_config.NumberColumn(min_value=0.0, step=0.1),
        }
        df_mt_edit = st.data_editor(df_mt, column_config=cfg_mt, num_rows="dynamic", use_container_width=True, key="fa_mt_editor",
                                    height=auto_table_height(df_mt.shape[0], min_rows=6))
        fin_rates["meters"] = df_mt_edit.to_dict("records") if not df_mt_edit.empty else []

        # Assumptions
        st.markdown("#### Assumptions")
        asum = fin_rates.get("assumptions", {}).copy()
        asum["avg_daily_hours_default"] = st.number_input("Avg Daily Hours (default)", min_value=0.0, value=float(asum.get("avg_daily_hours_default",16.0)), step=0.5, key="fa_assum_avgd")
        fin_rates["assumptions"] = asum

        # Currency selector (display only)
        st.markdown("#### Currency Settings")
        currencies = ["ZAR", "USD", "EUR"]
        curr_rates = {"ZAR": 1.0, "USD": 0.055, "EUR": 0.05}  # Example display multipliers
        selected_curr = st.selectbox("Display Currency", currencies, index=0, key="fa_curr_select")
        curr_mult = curr_rates.get(selected_curr, 1.0)
        st.caption(f"All displays scaled by {selected_curr} (multiplier: {curr_mult:.3f})")

        # BOM-driven Price List per Asset
        st.markdown("#### Price List (from Asset BOM)")
        pl_asset = st.selectbox("Select Asset to load BOM", options=[""]+sorted(assets_dict.keys()), key="fa_pl_asset")
        df_bom_pl = pd.DataFrame(columns=["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","UNIT PRICE","EXTENDED","VENDOR"])
        if pl_asset:
            bom_df = build_bom_table(assets_dict.get(pl_asset, {}))
            if isinstance(bom_df, pd.DataFrame) and not bom_df.empty:
                df_bom_pl = pd.DataFrame({
                    "CMMS MATERIAL CODE": bom_df.get("CMMS MATERIAL CODE", pd.Series([],dtype=str)).astype(str),
                    "OEM PART NUMBER":    bom_df.get("OEM PART NUMBER", pd.Series([],dtype=str)).astype(str),
                    "DESCRIPTION":        bom_df.get("DESCRIPTION", pd.Series([],dtype=str)).astype(str),
                    "QUANTITY":           pd.to_numeric(bom_df.get("QUANTITY", pd.Series([],dtype=float)), errors="coerce").fillna(0.0),
                })
                unit_prices, vendors = [], []
                for _, r in bom_df.iterrows():
                    cmms = str(r.get("CMMS MATERIAL CODE","")).strip()
                    unit = _price_map.get(cmms, float(to_num(r.get("PRICE",0.0),0.0)))
                    unit_prices.append(unit)
                    vendors.append(_vendor_map.get(cmms, ""))
                df_bom_pl["UNIT PRICE"] = unit_prices
                df_bom_pl["EXTENDED"]   = pd.to_numeric(df_bom_pl["QUANTITY"], errors="coerce").fillna(0.0) * \
                                          pd.to_numeric(df_bom_pl["UNIT PRICE"], errors="coerce").fillna(0.0)
                df_bom_pl["VENDOR"]     = vendors

        cfg_pl = {
            "CMMS MATERIAL CODE": st.column_config.TextColumn(disabled=True),
            "OEM PART NUMBER": st.column_config.TextColumn(disabled=True),
            "DESCRIPTION": st.column_config.TextColumn(disabled=True),
            "QUANTITY": st.column_config.NumberColumn(format="%.2f", disabled=True),
            "UNIT PRICE": st.column_config.NumberColumn(min_value=0.0, step=1.0),
            "EXTENDED": st.column_config.NumberColumn(format="%.2f", disabled=True),
            "VENDOR": st.column_config.TextColumn(),
        }
        df_bom_pl_edit = st.data_editor(df_bom_pl, column_config=cfg_pl, num_rows="dynamic", use_container_width=True, key="fa_pl_bom_editor",
                                        height=auto_table_height(df_bom_pl.shape[0], min_rows=6))
        if not df_bom_pl_edit.empty:
            df_bom_pl_edit["EXTENDED"] = pd.to_numeric(df_bom_pl_edit["QUANTITY"], errors="coerce").fillna(0.0) * \
                                         pd.to_numeric(df_bom_pl_edit["UNIT PRICE"], errors="coerce").fillna(0.0)

        col_pl1, col_pl2 = st.columns(2)
        with col_pl1:
            if st.button("💾 Save to Price List", key="fa_pl_save"):
                existing = {str(r.get("cmms","")).strip(): r for r in fin_rates.get("price_list", []) if str(r.get("cmms","")).strip()}
                for _, r in df_bom_pl_edit.iterrows():
                    cmms = str(r.get("CMMS MATERIAL CODE","")).strip()
                    if not cmms: continue
                    existing[cmms] = {"cmms": cmms,
                                      "unit_price": float(to_num(r.get("UNIT PRICE",0.0),0.0)),
                                      "vendor": str(r.get("VENDOR","") or "")}
                fin_rates["price_list"] = list(existing.values())
                _price_map.clear(); _vendor_map.clear()
                _price_map.update({str(x["cmms"]): float(to_num(x["unit_price"],0.0)) for x in fin_rates["price_list"]})
                _vendor_map.update({str(x["cmms"]): str(x.get("vendor","") or "Unknown") for x in fin_rates["price_list"]})
                fin_save_rates(fin_rates)
        with col_pl2:
            st.download_button("⬇ Export Price List (CSV)",
                               data=pd.DataFrame(fin_rates.get("price_list",[])).to_csv(index=False).encode("utf-8"),
                               file_name="price_list.csv", key="fa_pl_export")

        # Duty & Standby Groups
        st.markdown("#### Duty & Standby Groups")
        duty_groups = fin_rates.get("duty_groups", []) or []
        plant_avail = st.slider("Plant Availability Target (%)", min_value=0, max_value=100, value=85, step=1, key="fa_rates_avail") / 100.0

        if duty_groups:
            rows = []
            for g in duty_groups:
                members = g.get("members", [])
                total_share = sum(float(to_num(m.get("util_share", 0.0), 0.0)) for m in members)
                rows.append({
                    "Group": g.get("group", ""),
                    "Members": ", ".join([m.get("asset_tag", "") for m in members]),
                    "Sum of Shares": round(total_share, 3),
                    "Status": "✅" if abs(total_share - 1.0) < 1e-6 else "❌ (Adjust)"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=auto_table_height(len(rows), min_rows=3))

        with st.expander("Add / Edit Group", expanded=False):
            g_name = st.text_input("Group Name", key="fa_bp_grp_name_unique")
            members = st.multiselect("Members (Assets)", options=all_tags, key="fa_bp_grp_members_unique")

            util_inputs = {}
            if members:
                util_cols = st.columns(max(1, min(4, len(members))))
                default_share = 1.0 / len(members)
                for i, at in enumerate(members):
                    with util_cols[i % len(util_cols)]:
                        util_inputs[at] = st.number_input(
                            f"Share for {at}",
                            min_value=0.0, max_value=1.0,
                            value=default_share,
                            step=0.05,
                            key=f"fa_util_share_{g_name}_{at}"
                        )
                current_sum = sum(util_inputs.values())
                status = "✅" if abs(current_sum - 1.0) < 1e-6 else "❌"
                st.metric("Sum of Shares", f"{current_sum:.3f} {status}")

                if st.button("Auto-Normalize", key=f"auto_norm_{g_name}"):
                    total = sum(util_inputs.values()) or 1.0
                    for at in members:
                        st.session_state[f"fa_util_share_{g_name}_{at}"] = util_inputs[at] / total

                # Preview scaled by availability
                st.markdown("##### Projected Monthly Hours Preview (Scaled by Availability)")
                preview_rows = []
                for at in members:
                    share = st.session_state.get(f"fa_util_share_{g_name}_{at}", util_inputs.get(at, 0.0))
                    avg_daily = _avg_daily_hours_for(at)
                    proj_hours = avg_daily * 30 * plant_avail * share
                    preview_rows.append({"Asset": at, "Util Share": share, "Proj Hours/Month": round(proj_hours, 1)})
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

            if st.button("➕ Add/Update Group", key="fa_bp_addgrp_unique"):
                if g_name and members:
                    new_members = [{"asset_tag": at, "util_share": st.session_state.get(f"fa_util_share_{g_name}_{at}", 1.0/len(members))} for at in members]
                    fin_groups = [g for g in duty_groups if g.get("group") != g_name]
                    fin_groups.append({"group": g_name, "members": new_members})
                    fin_rates["duty_groups"] = fin_groups
                    fin_save_rates(fin_rates)
                    duty_groups = fin_groups
                    st.success(f"Group '{g_name}' saved with shares summing to ~1.")

        if st.button("💾 Save All Rates & Assumptions", key="fa_rates_save_all"):
            fin_save_rates(fin_rates)

    # =========================================================
    # 3) 🛠 Maintenance Cost Breakdown
    # =========================================================
    with sub_break:
        st.markdown("### Maintenance Cost Breakdown")
        if fa_cost_rows.empty:
            st.info("No history rows in selected period.")
        else:
            st.markdown("#### Filters and Summary")
            currencies = ["ZAR", "USD", "EUR"]
            curr_rates = {"ZAR": 1.0, "USD": 0.055, "EUR": 0.05}
            selected_curr = st.selectbox("Display Currency", currencies, index=0, key="fa_break_curr")
            curr_mult = curr_rates.get(selected_curr, 1.0)
            st.caption(f"Costs displayed in {selected_curr} (multiplier: {curr_mult:.3f})")

            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                metric = st.radio("Metric", ["Total","Spares","Labour"], horizontal=True, key="fa_break_metric")
            with filter_col2:
                sens_runtime = st.slider("Runtime Adjustment (%)", -50, 200, 0, step=5, key="fa_break_sens_runtime") / 100.0

            # Adjust & currency
            d = fa_cost_rows.copy()
            d[["spares_cost", "labour_cost_regular", "labour_cost_callout", "energy_cost", "water_cost"]] *= (1 + sens_runtime) * curr_mult

            d["period"] = pd.to_datetime(d["Date of Maintenance"]).dt.to_period("M").astype(str)
            d["is_planned"] = ~d["Maintenance Code"].astype(str).str.upper().eq("PM00")
            if metric == "Total":
                d["v"] = d[["spares_cost","labour_cost_regular","labour_cost_callout"]].sum(axis=1)
            elif metric == "Spares":
                d["v"] = d["spares_cost"]
            else:
                d["v"] = d[["labour_cost_regular","labour_cost_callout"]].sum(axis=1)

            # KPIs
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            total_spend = d["v"].sum()
            avg_per_asset = total_spend / d["Asset Tag"].nunique() if d["Asset Tag"].nunique() > 0 else 0.0
            breakdown_pct = (d[~d["is_planned"]]["v"].sum() / total_spend * 100) if total_spend > 0 else 0.0
            with kpi_col1: st.metric("Total Spend", f"${total_spend:,.2f}")
            with kpi_col2: st.metric("Avg per Asset", f"${avg_per_asset:,.2f}")
            with kpi_col3: st.metric("Breakdown %", f"{breakdown_pct:.1f}%")

            # Visuals
            st.markdown("#### Cost Visualizations")
            toggle_advanced = st.checkbox("Show Advanced Features (Forecasts & Risks)", value=False, key="fa_break_adv")

            grid_col1, grid_col2 = st.columns(2)
            with grid_col1:
                # Stacked area + simple forecast (lightweight)
                area = d.groupby(["period","is_planned"], as_index=False)["v"].sum().sort_values("period")
                tab = area.pivot(index="period", columns="is_planned", values="v").fillna(0.0)
                tab = tab.rename(columns={True:"Planned", False:"Breakdown"}).reset_index()
                figA = go.Figure()
                figA.add_trace(go.Scatter(x=tab["period"], y=tab.get("Planned", pd.Series([])), stackgroup="one", name="Planned"))
                figA.add_trace(go.Scatter(x=tab["period"], y=tab.get("Breakdown", pd.Series([])), stackgroup="one", name="Breakdown"))
                if toggle_advanced and len(tab) > 1:
                    total = tab.get("Planned",0) + tab.get("Breakdown",0)
                    slope = (total.iloc[-1] - total.iloc[0]) / max(len(total)-1,1)
                    next_periods = [str(pd.Period(tab["period"].iloc[-1]) + i + 1) for i in range(3)]
                    forecast = [total.iloc[-1] + slope * (i+1) for i in range(3)]
                    figA.add_trace(go.Scatter(x=next_periods, y=forecast, mode="lines", line=dict(dash="dash"), name="Forecast"))
                figA.update_layout(title=f"Monthly {metric} Trends", height=300, xaxis_title="Period", yaxis_title="Cost ($)")
                st.plotly_chart(figA, use_container_width=True)

                # Pareto top assets
                pa = d.groupby("Asset Tag", as_index=False)["v"].sum().sort_values("v", ascending=False).head(10)
                pa["cum_pct"] = 100.0 * pa["v"].cumsum() / max(pa["v"].sum(), 1.0)
                figPA = go.Figure()
                figPA.add_bar(x=pa["Asset Tag"], y=pa["v"], name="Spend")
                figPA.add_trace(go.Scatter(x=pa["Asset Tag"], y=pa["cum_pct"], yaxis="y2", mode="lines+markers", name="Cum %"))
                figPA.update_layout(title=f"Top Assets by {metric}", height=300, yaxis_title="Cost ($)",
                                    yaxis2={"title":"Cum %", "overlaying":"y", "side":"right"},
                                    shapes=[{"type":"line","x0":-0.5,"x1":len(pa)-0.5,"y0":80,"y1":80,"yref":"y2","line":{"dash":"dash"}}])
                st.plotly_chart(figPA, use_container_width=True)

            with grid_col2:
                # Heatmap
                hm = d.groupby(["Maintenance Code","Asset Tag"], as_index=False)["v"].sum()
                topN = hm.groupby("Asset Tag")["v"].sum().sort_values(ascending=False).head(12).index.tolist()
                hm = hm[hm["Asset Tag"].isin(topN)]
                piv = hm.pivot(index="Maintenance Code", columns="Asset Tag", values="v").fillna(0.0)
                figH = go.Figure(data=go.Heatmap(
                    z=piv.values, x=list(piv.columns), y=list(piv.index),
                    hovertemplate="Code: %{y}<br>Asset: %{x}<br>Cost: $%{z:.2f}<extra></extra>"
                ))
                figH.update_layout(title=f"Code × Asset Heatmap ({metric})", height=300, xaxis_title="Asset", yaxis_title="Code")
                st.plotly_chart(figH, use_container_width=True)

                # Treemap for spares
                sp = d[d["spares_cost"] > 0].copy()
                if not sp.empty:
                    sp["vendor"] = sp["Spares Used"].map(lambda x: _vendor_map.get(str(x).strip(), "Unknown"))
                    agg = sp.groupby(["vendor","Spares Used","Asset Tag"], as_index=False)["spares_cost"].sum()
                    labels, parents, values = [], [], []
                    v_tot = agg.groupby("vendor")["spares_cost"].sum().to_dict()
                    for v, vt in v_tot.items():
                        labels.append(v); parents.append(""); values.append(vt)
                    for (v, cm), g in agg.groupby(["vendor","Spares Used"]):
                        labels.append(str(cm)); parents.append(v); values.append(float(g["spares_cost"].sum()))
                    for (v, cm, at), g in agg.groupby(["vendor","Spares Used","Asset Tag"]):
                        labels.append(at); parents.append(str(cm)); values.append(float(g["spares_cost"].sum()))
                    figT = go.Figure(go.Treemap(labels=labels, parents=parents, values=values, branchvalues="total",
                                                hovertemplate="%{label}<br>Cost: $%{value:.2f}<extra></extra>"))
                    figT.update_layout(title="Vendor → Spares → Asset (Spares)", height=300)
                    st.plotly_chart(figT, use_container_width=True)
                else:
                    st.caption("No spares spend in period.")

            # Call-out bar
            co = d[d["labour_cost_callout"] > 0].copy()
            if not co.empty:
                co["dow"] = pd.to_datetime(co["Date of Maintenance"]).dt.dayofweek
                dw = co.groupby("dow", as_index=False)["labour_cost_callout"].sum()
                dw["name"] = dw["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
                figC = go.Figure([go.Bar(x=dw["name"], y=dw["labour_cost_callout"],
                                         hovertemplate="Day: %{x}<br>Cost: $%{y:.2f}<extra></extra>")])
                figC.update_layout(title="Call-out Costs by Day", height=300, xaxis_title="Day", yaxis_title="Cost ($)")
                st.plotly_chart(figC, use_container_width=True)

            # Scatter anomalies
            if "Asset Downtime Hours" in fa_hist.columns:
                scatter = fa_hist.groupby("Asset Tag", as_index=False)["Asset Downtime Hours"].sum()
                scatter = scatter.merge(d.groupby("Asset Tag")["v"].sum().reset_index(), on="Asset Tag", how="left")
                figS = go.Figure([go.Scatter(
                    x=scatter["Asset Downtime Hours"], y=scatter["v"], mode="markers", text=scatter["Asset Tag"],
                    hovertemplate="Asset: %{text}<br>Downtime: %{x}h<br>Cost: $%{y:.2f}<extra></extra>"
                )])
                figS.update_layout(title="Cost vs Downtime Anomalies", height=300, xaxis_title="Downtime (h)", yaxis_title="Cost ($)")
                st.plotly_chart(figS, use_container_width=True)

            # Sidebar details
            st.sidebar.markdown("#### Details Panel")
            selected_asset = st.sidebar.selectbox("Select Asset for Breakdown", options=sorted(d["Asset Tag"].unique()), key="fa_break_sidebar_asset")
            if selected_asset:
                asset_data = d[d["Asset Tag"] == selected_asset]
                st.sidebar.dataframe(asset_data[["Maintenance Code", "v", "Date of Maintenance"]], use_container_width=True)
                st.sidebar.caption("Navigate to Runtime Tracker for details.")

            # Footer actions
            st.markdown("#### Actions")
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                st.download_button("⬇ Export All Data (CSV)", data=d.to_csv(index=False).encode("utf-8"), file_name="cost_breakdown.csv", key="fa_break_dl")
            with action_col2:
                if st.button("Refresh Visuals", key="fa_break_refresh"):
                    st.rerun()

            # Optional risks
            if toggle_advanced:
                st.markdown("#### Top Risks Based on MTBF")
                rt2 = runtime_df0[runtime_df0["Asset Tag"].isin(d["Asset Tag"].unique())]
                risks = rt2[rt2["Remaining Hours"] < rt2["MTBF (Hours)"] * 0.2] if not rt2.empty else pd.DataFrame()
                if not risks.empty:
                    st.warning(f"{len(risks)} risks detected.")
                    st.dataframe(risks[["Asset Tag", "Remaining Hours", "MTBF (Hours)", "STATUS"]], use_container_width=True)
                else:
                    st.success("No high risks.")

    # =========================================================
    # 4) 📈 Performance Cost & Efficiency
    # =========================================================
    # Namespaced keys
    WIDGET_NS_PERF = "perf"
    def skey(name: str) -> str: return f"{WIDGET_NS_PERF}:{name}"

    with sub_perf:
        st.markdown("### Performance Cost & Efficiency")
        if fa_cost_rows.empty:
            st.info("No data—check history or filters.")
        else:
            st.markdown("#### Filters")
            filter_col1, filter_col2, filter_col3 = st.columns([1,1,1])
            with filter_col1:
                selected_assets = st.multiselect(
                    "Assets",
                    options=sorted(list(fa_assets or assets_dict.keys())),
                    default=fa_assets,
                    key=skey("assets")
                )
            with filter_col2:
                sens_avail = st.slider("Availability Adjustment (%)", -20, 20, 0, step=5, key=skey("availability_adjustment")) / 100.0
            with filter_col3:
                enable_forecast = st.checkbox("Enable Forecast", value=False, key=skey("fc_toggle"))
                enable_ai       = st.checkbox("Enable AI Scoring", value=False, key=skey("ai_toggle"))

            df = fa_cost_rows[fa_cost_rows["Asset Tag"].isin(selected_assets)] if selected_assets else fa_cost_rows
            df = df.copy()
            df["run_hours"] = df["Asset Tag"].map(lambda at: _avg_daily_hours_for(at) * period_days * (1 + sens_avail))

            # KPIs
            st.markdown("#### Key Performance Indicators")
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            unique_assets = df["Asset Tag"].nunique()
            total_run_hours = float(df["run_hours"].sum())

            oee = ((total_run_hours / (period_days * 24 * unique_assets)) * 100.0) if unique_assets > 0 else 0.0
            cost_cols = ["spares_cost", "labour_cost_regular", "labour_cost_callout", "energy_cost", "water_cost"]
            total_cost = float(df[cost_cols].sum().sum()) if not df.empty else 0.0
            cost_per_hour = (total_cost / total_run_hours) if total_run_hours > 0 else 0.0
            energy_total = float(df["energy_cost"].sum()) if "energy_cost" in df else 0.0

            with kpi_col1: st.metric("OEE (Proxy)", f"{oee:.1f}%", help="Availability-based estimate")
            with kpi_col2: st.metric("Cost per Run Hour", f"${cost_per_hour:.2f}")
            with kpi_col3: st.metric("Total Energy Cost", f"${energy_total:.2f}")

            # Visuals
            st.markdown("#### Interactive Visuals")
            grid_col1, grid_col2 = st.columns(2)
            with grid_col1:
                if "Date of Maintenance" in df.columns:
                    df["period"] = pd.to_datetime(df["Date of Maintenance"], errors="coerce").dt.to_period("M").astype(str)
                else:
                    st.warning("Missing 'Date of Maintenance' column—trends unavailable.")
                    df["period"] = "N/A"
                trend = df.groupby("period", as_index=False)[["energy_cost", "water_cost"]].sum()

                figT = go.Figure()
                figT.add_trace(go.Scatter(x=trend["period"], y=trend["energy_cost"], name="Energy"))
                figT.add_trace(go.Scatter(x=trend["period"], y=trend["water_cost"], name="Water"))

                if enable_forecast and len(trend) > 1:
                    sm = _get_sm()
                    X = np.arange(len(trend)).reshape(-1, 1)
                    model = sm.OLS(trend["energy_cost"], sm.add_constant(X)).fit()
                    future_x = np.arange(len(trend), len(trend) + 3).reshape(-1, 1)
                    forecast = model.predict(sm.add_constant(future_x))
                    last_period = pd.Period(trend["period"].iloc[-1], freq="M")
                    future_periods = [str(last_period + i + 1) for i in range(3)]
                    figT.add_trace(go.Scatter(x=future_periods, y=forecast, mode="lines", line=dict(dash="dash"), name="Energy Forecast"))

                figT.update_layout(title="Utility Cost Trends", height=300)
                st.plotly_chart(figT, use_container_width=True)

            with grid_col2:
                scatter = df.groupby("Asset Tag", as_index=False).agg({"run_hours": "first", "energy_cost": "sum", "water_cost": "sum"})
                figS = go.Figure()
                figS.add_trace(go.Scatter(x=scatter["run_hours"], y=scatter["energy_cost"], mode="markers", text=scatter["Asset Tag"], name="Energy"))
                figS.add_trace(go.Scatter(x=scatter["run_hours"], y=scatter["water_cost"], mode="markers", text=scatter["Asset Tag"], name="Water"))
                figS.update_layout(title="Runtime vs Utility Cost", height=300)
                st.plotly_chart(figS, use_container_width=True)

            # AI Efficiency (toggle)
            if enable_ai and len(df) > 1:
                features = df.groupby("Asset Tag")[["run_hours", "energy_cost", "water_cost"]].mean().fillna(0.0)
                n_assets = len(features)
                if n_assets > 0:
                    KMeans = _get_kmeans()
                    k = min(3, n_assets)  # avoids k > n
                    labels = KMeans(n_clusters=k, n_init=10).fit_predict(features.values)
                    features = features.copy()
                    features["cluster"] = labels
                    order = features.groupby("cluster")["run_hours"].mean().sort_values(ascending=False).index.tolist()
                    label_to_grade = {cl: g for cl, g in zip(order, ["A (High Efficiency)", "B (Moderate)", "C (Improvement Needed)"])}
                    features["Grade"] = features["cluster"].map(label_to_grade)
                    st.markdown("#### AI Efficiency Scores")
                    st.dataframe(features[["Grade"]], use_container_width=True)

            # Sidebar details
            st.sidebar.markdown("#### Asset Details")
            selected = st.sidebar.selectbox("Select Asset", options=df["Asset Tag"].unique(), key=skey("asset_select"))
            if selected:
                asset_df = df[df["Asset Tag"] == selected]
                st.sidebar.dataframe(asset_df[["energy_cost", "water_cost", "run_hours"]])

            # Actions
            st.markdown("#### Actions")
            colA, colB = st.columns(2)
            with colA:
                if st.button("Refresh", key=skey("refresh")): st.rerun()
            with colB:
                st.download_button("Export Data", data=df.to_csv(index=False).encode("utf-8"), file_name="performance.csv", mime="text/csv", key=skey("export"))

# =========================================================
# END OF PERFORMANCE & COST EFFICIENCY SUBTAB
# =========================================================

# -------------------------------------------
# Robust BOM extraction (override any earlier stub)
# -------------------------------------------
import re
import pandas as pd
import numpy as np

def _ci_get(d: dict, key: str, default=None):
    """Case-insensitive key getter for dicts."""
    if not isinstance(d, dict): return default
    for k, v in d.items():
        if str(k).strip().lower() == key.strip().lower():
            return v
    return default

def _ensure_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return float(default)
        return float(str(x).replace(",", "").strip())
    except Exception:
        return float(default)

def _ensure_int(x, default=0):
    try:
        if x is None or (isinstance(x, str) and not x.strip()):
            return int(default)
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return int(default)

def build_bom_table(asset: dict) -> pd.DataFrame:
    """
    Parse BOM data from multiple known shapes in assets_dict:
      1) "BOM Table": list[dict] with the target columns already (best path)
      2) "BOM": list[str] like "OEM • DESCRIPTION • CMMS"
      3) "CRR": list[dict] using columns A..H (A=CMMS, B=OEM, C=DESC, D=QTY)
    Returns DataFrame with columns:
      ["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE"]
    """
    # Target schema
    cols = ["CMMS MATERIAL CODE","OEM PART NUMBER","DESCRIPTION","QUANTITY","PRICE"]
    out = pd.DataFrame(columns=cols)

    if not isinstance(asset, dict) or not asset:
        return out

    # 1) Best case: "BOM Table" (case-insensitive)
    bom_table = _ci_get(asset, "BOM Table")
    if isinstance(bom_table, list) and len(bom_table) > 0:
        df = pd.DataFrame(bom_table)
        # Try to map a few possible header variants → target names
        col_map = {}
        for c in df.columns:
            cl = str(c).strip().lower()
            if cl in {"cmms material code", "cmms", "material code", "cmms_code"}:
                col_map[c] = "CMMS MATERIAL CODE"
            elif cl in {"oem part number", "oem", "part number", "oem_part"}:
                col_map[c] = "OEM PART NUMBER"
            elif cl in {"description", "desc"}:
                col_map[c] = "DESCRIPTION"
            elif cl in {"quantity", "qty"}:
                col_map[c] = "QUANTITY"
            elif cl in {"price", "unit price", "unit_price", "cost"}:
                col_map[c] = "PRICE"
        df = df.rename(columns=col_map)

        # Create missing columns with defaults
        for c in cols:
            if c not in df.columns:
                df[c] = 0 if c in {"QUANTITY","PRICE"} else ""

        # Fill from price list if PRICE is zero/missing
        def _fill_price(cmms, current):
            cmms = str(cmms or "").strip()
            if (current is None) or (float(_ensure_float(current)) <= 0.0):
                if ' _price_map' in globals() or True:
                    try:
                        return float(_price_map.get(cmms, 0.0))
                    except Exception:
                        return 0.0
            return float(_ensure_float(current))

        df["CMMS MATERIAL CODE"] = df["CMMS MATERIAL CODE"].astype(str).str.strip()
        df["OEM PART NUMBER"] = df["OEM PART NUMBER"].astype(str).str.strip()
        df["DESCRIPTION"] = df["DESCRIPTION"].astype(str).str.strip()
        df["QUANTITY"] = [max(1, _ensure_int(x, 1)) for x in df["QUANTITY"]]
        df["PRICE"] = [ _fill_price(cm, pr) for cm, pr in zip(df["CMMS MATERIAL CODE"], df["PRICE"]) ]

        return df[cols].copy()

    # 2) "BOM": list of strings like "OEM • DESC • CMMS"
    bom_list = _ci_get(asset, "BOM")
    if isinstance(bom_list, list) and len(bom_list) > 0 and all(isinstance(s, str) for s in bom_list):
        rows = []
        for s in bom_list:
            # Split by bullet or middle dot or hyphen patterns
            parts = re.split(r"\s*[•·|-]\s*", s)
            parts = [p.strip() for p in parts if p and p.strip()]
            oem, desc, cmms = "", "", ""
            if len(parts) >= 3:
                # Heuristic: last token often CMMS
                cmms = parts[-1]
                oem = parts[0]
                desc = " • ".join(parts[1:-1]) if len(parts) > 2 else ""
            else:
                # Fallback: try to extract CMMS code pattern like AM\d+
                m = re.search(r"(AM\d+)", s, flags=re.I)
                cmms = m.group(1) if m else ""
                oem = s if not cmms else s.replace(cmms, "").strip(" -•")
                desc = ""
            price = 0.0
            try:
                price = float(_price_map.get(cmms, 0.0))
            except Exception:
                price = 0.0
            rows.append({
                "CMMS MATERIAL CODE": str(cmms).strip(),
                "OEM PART NUMBER": str(oem).strip(),
                "DESCRIPTION": str(desc).strip(),
                "QUANTITY": 1,
                "PRICE": float(price),
            })
        df = pd.DataFrame(rows)
        # Filter out missing cmms lines
        df = df[df["CMMS MATERIAL CODE"].astype(str).str.len() > 0]
        return df[cols].reset_index(drop=True)

    # 3) "CRR": list of dicts (A,B,C,D…)
    crr = _ci_get(asset, "CRR")
    if isinstance(crr, list) and len(crr) > 0 and all(isinstance(x, dict) for x in crr):
        rows = []
        for r in crr:
            cmms = str(_ci_get(r, "A", "") or "").strip()
            oem  = str(_ci_get(r, "B", "") or "").strip()
            desc = str(_ci_get(r, "C", "") or "").strip()
            qty  = max(1, _ensure_int(_ci_get(r, "D", 1), 1))
            price = 0.0
            try:
                price = float(_price_map.get(cmms, 0.0))
            except Exception:
                price = 0.0
            rows.append({
                "CMMS MATERIAL CODE": cmms,
                "OEM PART NUMBER": oem,
                "DESCRIPTION": desc,
                "QUANTITY": qty,
                "PRICE": price,
            })
        df = pd.DataFrame(rows)
        return df[cols].reset_index(drop=True)

    # Nothing found → empty
    return out

# -------------------------------------------
# Budget Planner (fixed to use robust BOM)
# -------------------------------------------
with sub_budget:
    import pandas as pd
    import plotly.graph_objects as go
    from datetime import datetime, timedelta
    import numpy as np

    # Safe linregress fallback (avoid SciPy requirement)
    try:
        from scipy.stats import linregress as _scipy_linregress
        def linregress(x, y):
            return _scipy_linregress(x, y)
    except Exception:
        def linregress(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if x.ndim != 1: x = x.ravel()
            if y.ndim != 1: y = y.ravel()
            b, a = np.polyfit(x, y, 1)  # y = b*x + a
            # Return tuple compatible with scipy: slope, intercept, rvalue, pvalue, stderr (we fake last 3)
            return b, a, 0.0, 1.0, 0.0  # we only use slope/intercept

    st.markdown("### Predictive Budget Planner with Insights")

    # Date range
    today = datetime.today().date()
    default_start = today + timedelta(days=1)
    default_end = default_start + timedelta(days=365)
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        plan_start = st.date_input("Planning Start", value=default_start, key="fa_bp_start")
    with col_in2:
        plan_end = st.date_input("Planning End", value=default_end, key="fa_bp_end")
    if plan_start >= plan_end:
        st.error("Start date must be before end date.")
        st.stop()

    # Months & horizon
    months = pd.date_range(plan_start, plan_end, freq="MS").strftime("%Y-%m").tolist()
    horizon_m = len(months)

    # Inputs
    infl_cols = st.columns(4)
    with infl_cols[0]: infl_spares = st.number_input("Spares Inflation %", min_value=0.0, value=5.0, step=0.5, key="fa_bp_infl_sp")/100.0
    with infl_cols[1]: infl_labour = st.number_input("Labour Escalation %", min_value=0.0, value=5.0, step=0.5, key="fa_bp_infl_lab")/100.0
    with infl_cols[2]: infl_tariff = st.number_input("Tariff Growth % (Energy/Water)", min_value=0.0, value=8.0, step=0.5, key="fa_bp_infl_trf")/100.0
    with infl_cols[3]: contingency_pct = st.number_input("Contingency Buffer %", min_value=0.0, value=10.0, step=1.0, key="fa_bp_conting")/100.0

    col_seed, col_mode = st.columns(2)
    with col_seed: seed = st.checkbox("Seed baseline from last 12-36 months actuals + trends", value=True, key="fa_bp_seed")
    with col_mode: insights_mode = st.selectbox("Insights & Forecast Mode", ["Basic (Averages)", "Advanced (Trends/AI)"], key="fa_bp_mode")

    # Duty/Standby overview
    st.markdown("#### Duty & Standby Groups")
    duty_groups = fin_rates.get("duty_groups", []) or []
    if duty_groups:
        rows=[]
        for g in duty_groups:
            total = sum([_ensure_float(m.get("util_share",0.0), 0.0) for m in g.get("members",[])])
            rows.append({"Group": g.get("group",""), "Members": ", ".join([m.get("asset_tag","") for m in g.get("members",[])]), "Sum of shares": round(total,3)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=auto_table_height(len(rows), min_rows=3))
    plant_avail = st.slider("Plant Availability Target (%)", min_value=0, max_value=100, value=85, step=1, key="fa_bp_avail")/100.0

    scope = st.radio("Scope", ["Single Asset", "Multi Asset", "All Assets"], horizontal=True, key="fa_bp_scope")

    # Asset selection
    if scope == "Single Asset":
        bp_assets = [st.selectbox("Select Asset", options=(fa_assets if fa_assets else all_tags), key="fa_bp_one_asset")]
    elif scope == "Multi Asset":
        bp_assets = st.multiselect("Select Assets", options=(fa_assets if fa_assets else all_tags), default=(fa_assets if fa_assets else []), key="fa_bp_multi_assets")
    else:
        bp_assets = fa_assets if fa_assets else all_tags

    # Overrides
    st.markdown("#### Asset Overrides & Manual Adjustments")
    ov_cols = ["Asset Tag","Runtime %","MTBF Uplift %","Call-out Reduction %","Manual Spares Adj %","Manual Labour Adj %"]
    df_ov = pd.DataFrame([{"Asset Tag": t, "Runtime %": 0.0, "MTBF Uplift %": 0.0, "Call-out Reduction %": 0.0,
                           "Manual Spares Adj %": 0.0, "Manual Labour Adj %": 0.0} for t in bp_assets])
    cfg_ov = {
        "Asset Tag": st.column_config.TextColumn(disabled=True),
        "Runtime %": st.column_config.NumberColumn(min_value=-50.0, max_value=200.0, step=5.0),
        "MTBF Uplift %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=5.0),
        "Call-out Reduction %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=5.0),
        "Manual Spares Adj %": st.column_config.NumberColumn(min_value=-50.0, max_value=100.0, step=5.0),
        "Manual Labour Adj %": st.column_config.NumberColumn(min_value=-50.0, max_value=100.0, step=5.0),
    }
    df_ov_edit = st.data_editor(df_ov, column_config=cfg_ov, num_rows="fixed", use_container_width=True, key="fa_bp_ov",
                                height=auto_table_height(df_ov.shape[0], min_rows=4))
    ov = df_ov_edit.set_index("Asset Tag") if not df_ov_edit.empty else pd.DataFrame(columns=ov_cols[1:])

    # Enhanced baseline (24 months default)
    def _enhanced_baseline(asset_list: list, seed_months=24) -> dict:
        if history_df0.empty or not asset_list: return {}
        end = pd.to_datetime(fa_to)
        start = end - pd.DateOffset(months=seed_months) + pd.DateOffset(days=1)
        base_hist = history_df0[(history_df0["Date of Maintenance"] >= start) & (history_df0["Date of Maintenance"] <= end)]
        base_hist = base_hist[base_hist["Asset Tag"].isin(asset_list)]
        if base_hist.empty: return {}

        base_hist = base_hist.copy()
        base_hist["Date of Maintenance"] = pd.to_datetime(base_hist["Date of Maintenance"], errors="coerce")

        days = (end.date() - start.date()).days + 1
        rows = fin_compute_cost_rows_cached(base_hist, days, fin_rates, _meters_map) if not base_hist.empty else pd.DataFrame(columns=fa_cost_rows.columns)
        out = {}
        for at, g in rows.groupby("Asset Tag"):
            out[at] = {
                "spares": float(g["spares_cost"].sum()),
                "labour": float(g[["labour_cost_regular","labour_cost_callout"]].sum(axis=1).sum()),
                "energy": float(g["energy_cost"].sum()),
                "water": float(g["water_cost"].sum()),
                "downtime": float(g.get("downtime_cost", pd.Series(0)).sum()),
            }

        # Add simple insights
        for at in asset_list:
            hist_at = base_hist[base_hist["Asset Tag"] == at]
            out.setdefault(at, {})
            if not hist_at.empty:
                out[at]["mtbf"] = float(hist_at["Hours Run Since Last Repair"].mean()) if "Hours Run Since Last Repair" in hist_at.columns else 0.0
                out[at]["downtime_avg"] = float(hist_at["Asset Downtime Hours"].mean()) if "Asset Downtime Hours" in hist_at.columns else 0.0

                monthly = hist_at.resample("M", on="Date of Maintenance")["Labour Hours"].sum().reset_index(drop=True)
                if len(monthly) > 1 and insights_mode == "Advanced (Trends/AI)":
                    x = np.arange(len(monthly))
                    y = monthly.values
                    slope, intercept, *_ = linregress(x, y)
                    out[at]["labour_trend_slope"] = float(slope)
                else:
                    out[at]["labour_trend_slope"] = 0.0

                # Average spares consumption per CMMS from history
                if "QTY" in hist_at.columns and "Spares Used" in hist_at.columns:
                    out[at]["spares_consump"] = hist_at.groupby("Spares Used")["QTY"].mean().to_dict()
                else:
                    out[at]["spares_consump"] = {}
            else:
                out[at].update({"mtbf":0.0,"downtime_avg":0.0,"labour_trend_slope":0.0,"spares_consump":{}})
        return out

    baseline = _enhanced_baseline(bp_assets) if seed else {
        at: {"spares":0,"labour":0,"energy":0,"water":0,"downtime":0,"mtbf":0,"labour_trend_slope":0,"spares_consump":{},"downtime_avg":0}
        for at in bp_assets
    }

    forecast_tab, insights_tab = st.tabs(["📊 Forecasts & Budgets", "🔍 Insights & Analysis"])

    # --------- Forecasts & Budgets ---------
    with forecast_tab:
        # Single-asset BOM planner with correct BOM extraction
        if scope == "Single Asset" and bp_assets and bp_assets[0]:
            sel_asset = bp_assets[0]
            st.markdown("#### Single-Asset BOM Forecast Planner (monthly picks with manual edits)")
            bom_df = build_bom_table(assets_dict.get(sel_asset, {}))
            if bom_df.empty:
                st.warning("No structured BOM rows parsed for this asset. Check that the asset has 'BOM Table', 'BOM', or 'CRR' data.")
                st.caption("Tip: Open Rates → Price List (from Asset BOM) to import/normalize BOM for this asset.")
            else:
                # Prepare planning table
                consump = baseline.get(sel_asset, {}).get("spares_consump", {}) or {}
                rows=[]
                for _, r in bom_df.iterrows():
                    cmms = str(r["CMMS MATERIAL CODE"]).strip()
                    oem  = str(r["OEM PART NUMBER"]).strip()
                    desc = str(r["DESCRIPTION"]).strip()
                    unit = float(_ensure_float(r.get("PRICE", 0.0)))
                    if unit <= 0.0:
                        try:
                            unit = float(_price_map.get(cmms, 0.0))
                        except Exception:
                            unit = 0.0
                    # Auto suggest per-month qty
                    avg_qty = float(consump.get(cmms, 0.0)) / max(horizon_m, 1)
                    base = baseline.get(sel_asset, {})
                    if insights_mode == "Advanced (Trends/AI)" and base.get("mtbf", 0) > 0:
                        adj_factor = base.get("labour_trend_slope", 0.0) / max(base["mtbf"], 1.0)
                        avg_qty *= (1 + adj_factor * horizon_m)
                    row = {"CMMS": cmms, "OEM": oem, "DESCRIPTION": desc, "UNIT PRICE": unit}
                    for m in months: row[m] = max(0.0, round(avg_qty, 2))
                    rows.append(row)
                df_plan = pd.DataFrame(rows)

                cfg_plan = {
                    "CMMS": st.column_config.TextColumn(disabled=True),
                    "OEM": st.column_config.TextColumn(disabled=True),
                    "DESCRIPTION": st.column_config.TextColumn(disabled=True),
                    "UNIT PRICE": st.column_config.NumberColumn(min_value=0.0, step=1.0),
                }
                for m in months:
                    cfg_plan[m] = st.column_config.NumberColumn(min_value=0.0, step=1.0, help="Edit quantity per month")

                df_plan_edit = st.data_editor(
                    df_plan, column_config=cfg_plan, use_container_width=True,
                    key="fa_bp_single_editor", height=auto_table_height(df_plan.shape[0], min_rows=6)
                )

                # Monthly totals + inflations + manual adj
                month_totals = []
                ov_row = ov.loc[sel_asset] if sel_asset in ov.index else pd.Series({"Manual Spares Adj %":0, "Manual Labour Adj %":0})
                for m in months:
                    qty = pd.to_numeric(df_plan_edit[m], errors="coerce").fillna(0.0)
                    unit = pd.to_numeric(df_plan_edit["UNIT PRICE"], errors="coerce").fillna(0.0)
                    sp = float((qty * unit).sum()) * (1 + infl_spares) * (1 + float(ov_row.get("Manual Spares Adj %", 0))/100.0)
                    month_totals.append(sp)
                df_month = pd.DataFrame({"Month": months, "Spares": month_totals})

                base = baseline.get(sel_asset, {"labour":0,"energy":0,"water":0,"downtime":0,"mtbf":0,"labour_trend_slope":0})
                lab_base = (base["labour"] / max(horizon_m,1)) * (1 + infl_labour) * (1 + float(ov_row.get("Manual Labour Adj %", 0))/100.0)
                en_base = (base["energy"] / max(horizon_m,1)) * (1 + infl_tariff)
                wa_base = (base["water"] / max(horizon_m,1)) * (1 + infl_tariff)
                dt_base = (base["downtime"] / max(horizon_m,1))

                if insights_mode == "Advanced (Trends/AI)":
                    # Simple trend application across months
                    trend = base.get("labour_trend_slope", 0.0) / max(base.get("mtbf", 1.0), 1.0)
                    df_month["Spares"] = df_month["Spares"] * (1 + 0.5 * trend * np.arange(horizon_m))

                df_month["Labour"] = lab_base
                df_month["Energy"] = en_base
                df_month["Water"]  = wa_base
                df_month["Downtime"] = dt_base
                df_month["Total"] = df_month[["Spares","Labour","Energy","Water","Downtime"]].sum(axis=1) * (1 + contingency_pct)

                figB1 = go.Figure([go.Scatter(x=df_month["Month"], y=df_month["Total"].cumsum(), mode="lines+markers", name="Cumulative Forecast")])
                figB1.update_layout(title=f"Forecasted Cumulative Spend — {sel_asset}", height=300, margin=dict(t=30,b=10,l=10,r=10),
                                    xaxis_title="Month", yaxis_title="Cost (R)")
                st.plotly_chart(figB1, use_container_width=True, key="fa_bp_single_cum")

                st.markdown("#### Monthly Forecast Table (Single Asset)")
                st.dataframe(df_month, use_container_width=True, height=auto_table_height(df_month.shape[0], min_rows=6))
                st.download_button("⬇ Download Single-Asset Forecast (CSV)", data=df_month.to_csv(index=False).encode("utf-8"),
                                   file_name=f"budget_{sel_asset}.csv", key="fa_bp_single_dl")

        # Portfolio phase
        if scope in {"Multi Asset","All Assets"} and bp_assets:
            st.markdown("#### Portfolio Budget Forecast (phased with manuals)")
            def asset_utilization(tag: str, duty_groups: list) -> float:
                try:
                    for g in duty_groups or []:
                        for m in g.get("members", []):
                            if str(m.get("asset_tag","")).strip() == str(tag).strip():
                                share = _ensure_float(m.get("util_share",1.0), 1.0)
                                return max(0.0, min(1.0, share))
                except Exception:
                    pass
                return 1.0
            util_map = {t: asset_utilization(t, fin_rates.get("duty_groups", [])) for t in bp_assets}
            loc_map = {t: assets_dict.get(t, {}).get("Functional Location", "Unknown") for t in bp_assets}

            rows=[]
            for at in bp_assets:
                base = baseline.get(at, {"spares":0,"labour":0,"energy":0,"water":0,"downtime":0,"mtbf":0,"labour_trend_slope":0})
                runtime_factor = 1.0 + _ensure_float(ov.loc[at]["Runtime %"] if at in ov.index else 0.0,0.0)/100.0
                mtbf_uplift = _ensure_float(ov.loc[at]["MTBF Uplift %"] if at in ov.index else 0.0,0.0)/100.0
                callout_reduction = _ensure_float(ov.loc[at]["Call-out Reduction %"] if at in ov.index else 0.0,0.0)/100.0
                sp_adj = _ensure_float(ov.loc[at]["Manual Spares Adj %"] if at in ov.index else 0.0,0.0)/100.0
                lab_adj = _ensure_float(ov.loc[at]["Manual Labour Adj %"] if at in ov.index else 0.0,0.0)/100.0
                util_share = util_map.get(at, 1.0)
                for i, m in enumerate(months):
                    sp_base = (base["spares"] / max(horizon_m,1)) * (1 + infl_spares) * (1 + sp_adj)
                    lab_base = (base["labour"] / max(horizon_m,1)) * (1 + infl_labour) * (1 - callout_reduction) * (1 + lab_adj)
                    if insights_mode == "Advanced (Trends/AI)" and base["mtbf"] > 0:
                        trend_adj = base["labour_trend_slope"] * i / max(base["mtbf"],1)
                        lab_base *= (1 + trend_adj - mtbf_uplift)
                        sp_base  *= (1 + 0.5*trend_adj - mtbf_uplift)
                    en = (base["energy"] / max(horizon_m,1)) * (1 + infl_tariff) * plant_avail * util_share * runtime_factor
                    wa = (base["water"]  / max(horizon_m,1)) * (1 + infl_tariff) * plant_avail * util_share * runtime_factor
                    dt = (base["downtime"] / max(horizon_m,1))
                    rows.append({"Month": m, "Asset Tag": at, "Location": loc_map.get(at, "Unknown"),
                                 "Spares": sp_base, "Labour": lab_base, "Energy": en, "Water": wa, "Downtime": dt})

            df_phase = pd.DataFrame(rows)
            if df_phase.empty:
                st.info("No baseline to forecast. Load history or disable 'Seed baseline'.")
            else:
                df_phase["Total"] = df_phase[["Spares","Labour","Energy","Water","Downtime"]].sum(axis=1) * (1 + contingency_pct)
                agg_month = df_phase.groupby("Month", as_index=False)[["Spares","Labour","Energy","Water","Downtime","Total"]].sum()
                agg_loc = df_phase.groupby(["Location","Month"], as_index=False)[["Spares","Labour","Energy","Water","Downtime","Total"]].sum()

                figB = go.Figure([go.Scatter(x=agg_month["Month"], y=agg_month["Total"].cumsum(), mode="lines+markers", name="Cumulative Forecast")])
                figB.update_layout(title="Forecasted Cumulative Spend (Portfolio)", height=320, margin=dict(t=36,b=10,l=10,r=10),
                                   xaxis_title="Month", yaxis_title="Cost (R)")
                st.plotly_chart(figB, use_container_width=True, key="fa_bp_portfolio_cum")

                st.markdown("#### Monthly Forecast Table (Plant-Wide)")
                st.dataframe(agg_month, use_container_width=True, height=auto_table_height(agg_month.shape[0], min_rows=6))

                with st.expander("Location-Level Forecasts"):
                    for loc in agg_loc["Location"].unique():
                        st.markdown(f"**{loc}**")
                        st.dataframe(agg_loc[agg_loc["Location"] == loc].drop("Location", axis=1))

                st.download_button("⬇ Download Portfolio Forecast (CSV)", data=agg_month.to_csv(index=False).encode("utf-8"),
                                   file_name="budget_summary.csv", key="fa_bp_portfolio_dl")

    # --------- Insights ---------
    with insights_tab:
        if not baseline:
            st.info("No data for insights. Enable seeding or select assets with history.")
        else:
            st.markdown("#### Key Insights & Analysis")
            all_mtbf = np.mean([b["mtbf"] for b in baseline.values() if b.get("mtbf",0) > 0]) or 1
            all_trend = np.mean([b.get("labour_trend_slope",0.0) for b in baseline.values()])
            high_risk_assets = [at for at, b in baseline.items() if b.get("mtbf",0) < all_mtbf * 0.5]
            savings_pot = sum([baseline.get(at, {"labour":0})["labour"] * (_ensure_float(ov.loc[at]["Call-out Reduction %"],0)/100.0 if at in ov.index else 0.0) for at in bp_assets]) / max(horizon_m,1)

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg MTBF (Hours)", f"{all_mtbf:.1f}")
            k2.metric("Labour Trend Slope", f"{all_trend:.2f}", help="Positive = increasing costs")
            k3.metric("High-Risk Assets", len(high_risk_assets))
            k4.metric("Projected Savings from Overrides", f"{savings_pot:.2f} R")

            if scope in {"Multi Asset","All Assets"} and 'df_phase' in locals() and not df_phase.empty:
                pivot = df_phase.pivot_table(index="Location", columns="Month", values="Total", aggfunc="sum").fillna(0.0)
                fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index))
                fig_heat.update_layout(title="Budget Risk Heatmap (Higher = Riskier)", height=400)
                st.plotly_chart(fig_heat, use_container_width=True)

            insights = []
            if all_trend > 0: insights.append("Increasing labour trends: Consider MTBF uplift to mitigate.")
            if high_risk_assets: insights.append(f"High-risk assets (low MTBF): {', '.join(high_risk_assets[:5])}…")
            if contingency_pct > 0: insights.append(f"Applied {contingency_pct*100:.0f}% contingency buffer.")
            st.markdown("**Key Insights:**")
            for ins in insights: st.markdown(f"- {ins}")

            if len(bp_assets) <= 10:
                for at in bp_assets:
                    with st.expander(f"Insights for {at}"):
                        base = baseline.get(at, {"mtbf":0,"downtime_avg":0,"spares_consump":{}, "labour_trend_slope":0})
                        st.write(f"MTBF: {base['mtbf']:.1f} h")
                        st.write(f"Downtime Avg: {base.get('downtime_avg', 0):.1f} h")
                        st.write(f"Spares Consumption (avg QTY per code): {base.get('spares_consump', {})}")
                        if base.get("labour_trend_slope",0) > 0: st.warning("Increasing labour trend detected.")

    st.markdown("---")
    if st.button("Export Insights to Runtime Tracker (Tab 2)"):
        st.session_state.bp_insights = {"high_risk": high_risk_assets, "mtbf_avg": all_mtbf}
        st.success("Insights exported to session for use in Runtime Tracker.")

# =========================
# TAB 6 — Engineering Tool
# =========================
with tab6:
    import streamlit as st
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import linregress
    import sympy as sp
    import plotly.express as px

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

    # ---- Reusable blocks ----
    def process_inputs(prefix="hyd"):
        c1, c2, c3, c4 = st.columns(4)
        elev = c1.number_input(f"Site Elevation ({u_len_label()})", min_value=0.0, value=0.0, step=10.0, key=f"{prefix}_elev")
        tempC = c2.number_input("Fluid Temperature (°C)", min_value=-10.0, value=20.0, step=1.0, key=f"{prefix}_tempC")
        fluid = c3.selectbox("Fluid", list(FLUID_PROPS.keys()), key=f"{prefix}_fluid")
        props = FLUID_PROPS[fluid]
        dens_kgL = c3.number_input("Density (kg/L)", min_value=0.2, value=props["density_kgL"], step=0.01, key=f"{prefix}_rho", disabled=(fluid!="Custom"))
        mu_cP = c4.number_input("Viscosity (cP)", min_value=0.05, value=props["viscosity_cP"], step=0.05, key=f"{prefix}_mu_cP", disabled=(fluid!="Custom"))
        vap = c4.number_input(f"Vapor Pressure ({u_press_label()})", min_value=0.0, value=2.3, step=0.01, key=f"{prefix}_vpk")
        qmin = st.number_input(f"Curve Minimum Flow ({u_flow_label()})", min_value=0.0, value=0.0, step=10.0, key=f"{prefix}_qmin")
        qmax = st.number_input(f"Curve Maximum Flow ({u_flow_label()})", min_value=1.0, value=1500.0, step=50.0, key=f"{prefix}_qmax")
        rho = dens_kgL * 1000.0
        mu = mu_cP / 1000.0
        patm_kpa = patm_kpa_from_elevation(to_si_len(elev))
        return qmin, qmax, rho, mu, patm_kpa, to_si_press(vap), tempC

    def suction_block(prefix="hyd_suc", design_flow=None):
        s1, s2, s3, s4 = st.columns(4)
        Hs = s1.number_input(f"Suction static head (+flooded, −lift) ({u_head_label()})", value=0.0, key=f"{prefix}_Hs")
        Ds = s2.number_input(f"Suction Pipe ID ({u_diam_label()})", min_value=0.1, value=300.0, step=1.0, key=f"{prefix}_Ds")
        Ls = s3.number_input(f"Suction Pipe Length ({u_len_label()})", min_value=0.0, value=10.0, step=1.0, key=f"{prefix}_Ls")
        Ps = s4.number_input(f"Suction Residual Pressure ({u_press_label()})", min_value=0.0, value=0.0, step=0.01, key=f"{prefix}_Ps")
        t1, t2, t3 = st.columns(3)
        lining = t1.selectbox("Suction lining / material", list(ROUGHNESS_MM.keys()), index=11, key=f"{prefix}_lining")
        auto_r = t2.checkbox("Use preset roughness", value=True, key=f"{prefix}_auto_r")
        rough = t3.number_input(f"Override roughness ({u_rough_label()})", value=ROUGHNESS_MM[lining], step=0.001, key=f"{prefix}_rough", disabled=auto_r)
        K_s = fittings_panel("Suction Fittings", prefix)
        Dm = to_si_diam(Ds)
        Am = math.pi * (Dm**2) / 4.0 if Dm > 0 else 1e9
        V_suc = from_si_vel((to_si_flow(design_flow))/Am) if (design_flow and Am>0) else 0.0
        st.caption(f"Suction velocity @ design flow: {V_suc:.2f} {u_vel_label()}")
        eps_m = to_si_rough(rough) if not auto_r else ROUGHNESS_MM[lining]/1000.0
        return {"Hs_m": to_si_head(Hs), "Ds_m": Dm, "Ls_m": to_si_len(Ls), "Ps_kpa": to_si_press(Ps), "eps_m": eps_m, "Ksum": float(K_s)}

    def discharge_block(prefix="hyd_dis", design_flow=None):
        d1, d2, d3, d4 = st.columns(4)
        Hd = d1.number_input(f"Discharge static head ({u_head_label()})", value=10.0, key=f"{prefix}_Hd")
        Dd = d2.number_input(f"Discharge Pipe ID ({u_diam_label()})", min_value=0.1, value=300.0, step=1.0, key=f"{prefix}_Dd")
        Ld = d3.number_input(f"Discharge Pipe Length ({u_len_label()})", min_value=0.0, value=200.0, step=1.0, key=f"{prefix}_Ld")
        Pd = d4.number_input(f"Discharge Residual Pressure ({u_press_label()})", min_value=0.0, value=0.0, step=0.01, key=f"{prefix}_Pd")
        t1, t2, t3 = st.columns(3)
        lining = t1.selectbox("Discharge lining / material", list(ROUGHNESS_MM.keys()), index=4, key=f"{prefix}_lining")
        auto_r = t2.checkbox("Use preset roughness", value=True, key=f"{prefix}_auto_r")
        rough = t3.number_input(f"Override roughness ({u_rough_label()})", value=ROUGHNESS_MM[lining], step=0.001, key=f"{prefix}_rough", disabled=auto_r)
        K_d = fittings_panel("Discharge Fittings", prefix)
        Dm = to_si_diam(Dd)
        Am = math.pi * (Dm**2) / 4.0 if Dm > 0 else 1e9
        V_dis = from_si_vel((to_si_flow(design_flow))/Am) if (design_flow and Am>0) else 0.0
        st.caption(f"Discharge velocity @ design flow: {V_dis:.2f} {u_vel_label()}")
        eps_m = to_si_rough(rough) if not auto_r else ROUGHNESS_MM[lining]/1000.0
        return {"Hd_m": to_si_head(Hd), "Dd_m": Dm, "Ld_m": to_si_len(Ld), "Pd_kpa": to_si_press(Pd), "eps_m": eps_m, "Ksum": float(K_d)}

    def pump_curve_inputs(prefix="pump"):
        st.markdown("#### Pump curve (manual quadratic)")
        c1, c2, c3 = st.columns(3)
        H0 = c1.number_input(f"Shutoff head ({u_head_label()}) at Q=0", min_value=0.0, value=60.0, step=1.0, key=f"{prefix}_H0")
        Qb = c2.number_input(f"BEP Flow ({u_flow_label()})", min_value=1.0, value=1800.0, step=10.0, key=f"{prefix}_Qb")
        Hb = c3.number_input(f"Head at BEP ({u_head_label()})", min_value=0.0, value=45.0, step=1.0, key=f"{prefix}_Hb")
        return H0, Qb, Hb

    def pump_head_quadratic(Q_axis, H0, Qb, Hb):
        k = (H0 - Hb) / max(Qb**2, 1e-6)
        return H0 - k * (Q_axis**2)

    def system_simple_curve(Q_axis, suc, dis, rho, mu, vap_kpa, patm_kpa):
        Qs = [to_si_flow(q) for q in Q_axis]
        Ds, Ls, epss, Ks = suc["Ds_m"], suc["Ls_m"], suc["eps_m"], suc["Ksum"]
        Dd, Ld, epsd, Kd = dis["Dd_m"], dis["Ld_m"], dis["eps_m"], dis["Ksum"]
        Hs_stat, Hd_stat = suc["Hs_m"], dis["Hd_m"]
        Ps_kpa, Pd_kpa = suc["Ps_kpa"], dis["Pd_kpa"]
        A_s = math.pi * (Ds**2) / 4.0 if Ds > 0 else 1e9
        A_d = math.pi * (Dd**2) / 4.0 if Dd > 0 else 1e9
        H_totals, NPSHa_vals, Hs_loss, Hd_loss = [], [], [], []
        for Q in Qs:
            V_s = Q / A_s
            V_d = Q / A_d
            Re_s = (rho * V_s * Ds) / mu if Ds > 0 else 0.0
            Re_d = (rho * V_d * Dd) / mu if Dd > 0 else 0.0
            f_s = swamee_jain_f(Re_s, epss, Ds) if Ds > 0 else 0.0
            f_d = swamee_jain_f(Re_d, epsd, Dd) if Dd > 0 else 0.0
            h_s_pipe = (f_s * (Ls / max(Ds, 1e-6))) * (V_s**2) / (2 * g)
            h_d_pipe = (f_d * (Ld / max(Dd, 1e-6))) * (V_d**2) / (2 * g)
            h_s_fit = Ks * (V_s**2) / (2 * g)
            h_d_fit = Kd * (V_d**2) / (2 * g)
            H_static = Hd_stat - Hs_stat
            H_press = ((Pd_kpa - Ps_kpa) * 1000.0) / (rho * g)
            H_totals.append(H_static + H_press + h_s_pipe + h_s_fit + h_d_pipe + h_d_fit)
            patm_m = (patm_kpa - vap_kpa) * 1000.0 / (rho * g)
            NPSHa = patm_m + Hs_stat - (h_s_pipe + h_s_fit) - (V_s**2) / (2 * g)
            NPSHa_vals.append(NPSHa)
            Hs_loss.append(h_s_pipe + h_s_fit)
            Hd_loss.append(h_d_pipe + h_d_fit)
        return np.array(H_totals), np.array(NPSHa_vals), np.array(Hs_loss), np.array(Hd_loss)

    # ===== Sub-tabs (immediately under units)
    sub_hyd, sub_mech, sub_elec, sub_unit, sub_sci = st.tabs([
        "💧  Hydraulics Toolbox", "🔧 Mechanical", "⚡ Electrical/Thermal", "🔄 Unit Converter", "🧮 Scientific Calculator"
    ])

# ============ HYDRAULICS ============
with sub_hyd:
    st.markdown("### 💧 Hydraulics Toolbox")
    st.caption("Industrial-grade tools for fluid systems.")

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
                    from_si_head_m(H_sys_m), from_si_head_m(H_pump_scaled_m),
                    np.array([from_si_head_m(H_static_total_m)])
                ])
                ypad = max(0.05*(yvals.max()-yvals.min()+1e-6), 0.5)
                fig_sys.update_yaxes(range=[float(yvals.min()-ypad), float(yvals.max()+ypad)])
                fig_sys.update_traces(hovertemplate=f"{flow_lab}: %{{x:.2f}}<br>{head_lab}: %{{y:.2f}}<br>%{{fullData.name}}<extra></extra>")
                fig_sys.update_layout(height=360, dragmode="pan", xaxis_title=flow_lab, yaxis_title=head_lab, legend_title_text="")
                st.plotly_chart(fig_sys, use_container_width=True)

            with p2:
                st.subheader("NPSHa Curve")
                df_npsh = pd.DataFrame({"Flow": Q_axis_disp, "NPSHa (m)": NPSHa_m})
                fig_npsh = px.line(df_npsh, x="Flow", y="NPSHa (m)")
                vapor_margin = st.slider("Vapor Margin (m)", 0.0, 10.0, 3.0, key="hc_vapor_margin")
                fig_npsh.add_hrect(y0=vapor_margin, y1=max(float(np.nanmax(NPSHa_m)), vapor_margin + 1.0), opacity=0.2, annotation_text="Safe")
                fig_npsh.update_traces(hovertemplate=f"{flow_lab}: %{{x:.2f}}<br>NPSHa (m): %{{y:.2f}}<extra></extra>")
                fig_npsh.update_layout(height=360, xaxis_title=flow_lab, yaxis_title="NPSHa (m)")
                st.plotly_chart(fig_npsh, use_container_width=True)

            with p3:
                st.subheader("Power vs. Flow")
                df_power = pd.DataFrame({
                    "Flow": Q_axis_disp,
                    f"Hydraulic Power ({u_power_label()})": from_si_power_kw(P_h_kw),
                    f"Shaft Power ({u_power_label()})": from_si_power_kw(P_sh_kw),
                })
                fig_power = px.line(df_power, x="Flow", y=[f"Hydraulic Power ({u_power_label()})", f"Shaft Power ({u_power_label()})"])
                fig_power.update_traces(hovertemplate=f"{flow_lab}: %{{x:.2f}}<br>{power_lab}: %{{y:.2f}}<br>%{{fullData.name}}<extra></extra>")
                fig_power.update_layout(height=360, xaxis_title=flow_lab, yaxis_title=power_lab, legend_title_text="")
                st.plotly_chart(fig_power, use_container_width=True)

            # Deposition Velocity Analysis (compact)
            st.subheader("Deposition Velocity Analysis")
            rA1, rA2, rA3, rA4 = st.columns(4)
            flow_sf = rA1.number_input(f"Flow Rate ({u_flow_label()})", min_value=0.0, value=float(Q_design), key="sf_flow_hsc_rowA")
            Cv_sf   = rA2.number_input("Volumetric Concentration (%vol)", min_value=0.0, value=20.0, key="sf_cv_hsc_rowA")
            Sg_sf   = rA3.number_input("Specific Gravity of Solids", min_value=1.0, value=2.65, key="sf_sg_hsc_rowA")
            D_sf    = to_si_diam(rA4.number_input(f"Pipe ID ({u_diam_label()})", min_value=50.0, value=300.0, key="sf_pipe_id_hsc_rowA"))
            FL_sf   = 1.0
            Vc_sf   = FL_sf * math.sqrt(2 * g * max(D_sf, 1e-6) * (Sg_sf - 1)) * (Cv_sf / 100.0)**0.125
            pill(
                f"Critical Velocity ({u_vel_label()})",
                f"{from_si_vel(Vc_sf):.2f}",
                "Estimated minimum velocity to avoid solids deposition (Durand–Condolios). Increases with pipe diameter and solids SG."
            )

            rB1, rB2, rB3 = st.columns(3)
            tau_y_sf   = rB1.number_input("Yield Stress (Pa)", min_value=0.0, value=5.0, key="sf_tauy_hsc_rowB")
            mu_p_sf    = rB2.number_input("Plastic Viscosity (Pa.s)", min_value=0.001, value=0.01, key="sf_mup_hsc_rowB")
            rho_mix_sf = rB3.number_input("Mixture Density (kg/m³)", min_value=500.0, value=1200.0, key="sf_rhomix_hsc_rowB")
            He_sf      = (rho_mix_sf * tau_y_sf * D_sf**2) / (mu_p_sf**2) if mu_p_sf > 0 else 0.0
            pill(
                "Hedstrom Number",
                f"{He_sf:.2e}",
                "Non-dimensional group for yield/pseudo-plastic slurries: He = (ρ_mix · τ_y · D²) / μ_p². Larger He → stronger yield-stress effects."
            )

            rC1, rC2, rC3 = st.columns(3)
            particle_size_sf = rC1.number_input("Avg Particle Size (mm)", min_value=0.0, value=0.1, key="sf_psize_hsc_rowC")
            cv_range_sf      = rC2.slider("Concentration Range (%vol)", min_value=0, max_value=50, value=(0, 20), key="sf_cv_range_hsc_rowC")
            particle_slider_sf = rC3.slider("Vary Particle Size (mm)", min_value=0.01, max_value=5.0, value=0.1, key="sf_particle_slider_hsc_rowC")

            dep_left, dep_right = st.columns(2)
            with dep_left:
                cv_axis = np.linspace(cv_range_sf[0], cv_range_sf[1], 50)
                vc_axis = [FL_sf * math.sqrt(2 * g * max(D_sf, 1e-6) * (Sg_sf - 1)) * (c / 100.0)**0.125 for c in cv_axis]
                df_slurry = pd.DataFrame({
                    "Concentration (%vol)": cv_axis,
                    f"Velocity ({u_vel_label()})": [from_si_vel(v) for v in vc_axis]
                })
                fig_slurry = px.line(df_slurry, x="Concentration (%vol)", y=f"Velocity ({u_vel_label()})")
                v_set_rec_local = 2.0
                fig_slurry.add_hline(y=from_si_vel(v_set_rec_local), annotation_text="Recommended Min")
                fig_slurry.update_traces(hovertemplate=f"Concentration: %{{x:.2f}} %<br>Velocity: %{{y:.2f}} {u_vel_label()}<extra></extra>")
                fig_slurry.update_layout(height=320, xaxis_title="Concentration (%vol)", yaxis_title=f"Velocity ({u_vel_label()})")
                st.plotly_chart(fig_slurry, use_container_width=True)

            # Action row ends tool
            if st.button("Simulate Flow", key="hc_sim_flow"):
                pass

    # ---------- B) PIPE SIZING ASSISTANT ----------
    elif tool_option == "Pipe Sizing Assistant":
        st.markdown("### Pipe Sizing Assistant")
        advanced_mode = st.checkbox("Advanced/Industrial Mode", key="ps_adv")
        flow_in = st.number_input(f"Flow ({u_flow_label()})", min_value=0.0, value=1000.0, key="ps_flow_in")
        L_m = to_si_len(st.number_input(f"Straight length L ({u_len_label()})", min_value=0.0, value=100.0, key="ps_L"))
        rho_kgL = st.number_input("Fluid density (kg/L)", min_value=0.2, value=1.00, key="ps_rho_kgL")
        mu_cP = st.number_input("Dynamic viscosity (cP)", min_value=0.05, value=1.00, key="ps_mu_cP")
        lining_name = st.selectbox("Pipe lining / material", list(ROUGHNESS_MM.keys()), index=4, key="ps_lining")
        auto_eps = st.checkbox("Use preset roughness", value=True, key="ps_auto_eps")
        rough_inp = st.number_input(f"Override roughness ({u_rough_label()})", value=ROUGHNESS_MM[lining_name], step=0.001, disabled=auto_eps, key="ps_rough_inp")
        K_total = fittings_panel("Fittings", "ps")
        fluid_type = st.selectbox("Fluid type", ["Water / clear", "Slurry (fine)", "Slurry (coarse)"], key="ps_fluid_type")
        vmax_lim = st.number_input(f"Max velocity limit ({u_vel_label()})", min_value=0.5, value=4.0, key="ps_vmax")
        vset_user = st.number_input(f"Min settling velocity target ({u_vel_label()})", min_value=0.0, value=2.0, key="ps_vset_user")
        pipe_mawp = st.number_input(f"Pipe MAWP ({u_press_label()})", min_value=0.0, value=2000.0, key="ps_mawp")
        H_shut = st.number_input(f"Max discharge head (static + shutoff) ({u_head_label()})", min_value=0.0, value=100.0, key="ps_H_shut")
        raw_diams = st.text_input(f"Candidate pipe IDs ({u_diam_label()}, comma-separated)", value="150, 200, 250, 300, 350, 400", key="ps_raw_diams")
        cand = [float(x.strip()) for x in raw_diams.split(",") if x.strip()]
        if advanced_mode:
            fouling_factor = st.slider("Fouling Factor (scale roughness)", 1.0, 2.0, 1.0, key="ps_ff")
            cost_per_m = st.number_input("Cost per m ($)", 0.0, 1000.0, 50.0, key="ps_costpm")

        if not cand:
            st.warning("Provide at least one candidate diameter.")
        else:
            rho = rho_kgL * 1000.0
            mu = mu_cP / 1000.0
            eps_m = to_si_rough(rough_inp) if not auto_eps else ROUGHNESS_MM[lining_name]/1000.0
            if advanced_mode:
                eps_m *= fouling_factor
            Q = to_si_flow(flow_in)
            v_set_rec = 3.0 if fluid_type == "Water / clear" else (1.5 if fluid_type == "Slurry (fine)" else 2.5)
            v_set_limit = vset_user or v_set_rec
            rows = []
            for Dui in cand:
                D = to_si_diam(Dui)
                if D <= 0:
                    continue
                A = math.pi * (D**2) / 4.0
                V = Q / A if A > 0 else 0.0
                Re = (rho * V * D) / mu if mu > 0 else 0.0
                f = swamee_jain_f(Re, eps_m, D)
                hl_pipe = f * (L_m / max(D, 1e-9)) * (V**2) / (2 * g)
                hl_minor = K_total * (V**2) / (2 * g)
                hl_total = hl_pipe + hl_minor
                hl_100m = f * (100.0 / max(D, 1e-9)) * (V**2) / (2 * g)
                V_disp = from_si_vel(V)
                v_warn = []
                if V_disp < v_set_limit: v_warn.append("below settling target")
                if V_disp > vmax_lim: v_warn.append("over max limit")
                row = {
                    f"ID ({u_diam_label()})": Dui,
                    f"Velocity ({u_vel_label()})": float(V_disp),
                    "Re": float(Re),
                    "f": float(f),
                    f"HL per 100 {u_len_label()} ({u_head_label()})": float(from_si_head_m(hl_100m)),
                    f"Pipe HL ({u_head_label()})": float(from_si_head_m(hl_pipe)),
                    f"Minor HL ({u_head_label()})": float(from_si_head_m(hl_minor)),
                    f"Total HL ({u_head_label()})": float(from_si_head_m(hl_total)),
                    "Warnings": "; ".join(v_warn) if v_warn else "",
                    "OK": "Yes" if not v_warn else "No",
                }
                if advanced_mode:
                    row["Cost ($)"] = cost_per_m * from_si_len_m(L_m)
                rows.append(row)

            df_res = pd.DataFrame(rows)
            if df_res.empty:
                st.warning("No calculations produced (check inputs).")
            else:
                df_res = df_res.sort_values(
                    by=["OK", f"Total HL ({u_head_label()})"],
                    ascending=[False, True]
                ).reset_index(drop=True)
                st.dataframe(df_res, use_container_width=True)

                rec = None
                for _, r in df_res.iterrows():
                    if str(r["OK"]).strip() == "Yes":
                        rec = r
                        break
                if rec is None and not df_res.empty:
                    v_mid = 0.5 * (v_set_limit + vmax_lim)
                    rec = min(
                        df_res.to_dict("records"),
                        key=lambda rr: abs(float(rr[f"Velocity ({u_vel_label()})"]) - v_mid)
                    )
                if rec is not None:
                    st.success(
                        f"Recommended ID: {rec[f'ID ({u_diam_label()})']} {u_diam_label()} | "
                        f"Velocity ~ {float(rec[f'Velocity ({u_vel_label()})']):.2f} {u_vel_label()} | "
                        f"Total HL ~ {float(rec[f'Total HL ({u_head_label()})']):.1f} {u_head_label()} | "
                        f"{'No warnings' if str(rec['Warnings']).strip()=='' else rec['Warnings']}"
                    )

                P_max_kpa = (rho * g * to_si_head(H_shut)) / 1000.0
                P_max_disp = from_si_press_kpa(P_max_kpa)
                if P_max_disp > pipe_mawp:
                    st.error(f"⚠️ Max discharge pressure at shutoff ≈ {P_max_disp:.1f} {u_press_label()} exceeds pipe MAWP!")
                else:
                    st.info(f"Max discharge pressure at shutoff ≈ {P_max_disp:.1f} {u_press_label()} (MAWP OK)")

            st.subheader("Velocity vs. Head Loss")
            fig_pipe = px.scatter(
                df_res,
                x=f"Velocity ({u_vel_label()})",
                y=f"Total HL ({u_head_label()})",
                color="OK",
                hover_data=["Re", "f", "Warnings"],
                size=f"ID ({u_diam_label()})"
            )
            fig_pipe.add_vline(x=v_set_limit, annotation_text="Settling Limit")
            fig_pipe.add_vline(x=vmax_lim, annotation_text="Max Velocity")
            fig_pipe.update_layout(
                height=300,
                xaxis_title=f"Velocity ({u_vel_label()})",
                yaxis_title=f"Total Head Loss ({u_head_label()})"
            )
            flow_var = st.slider("Vary Flow (%)", 50, 150, 100, key="ps_flow_var")
            st.plotly_chart(fig_pipe, use_container_width=True)

    # ---------- C) VALVE SIZING ----------
    elif tool_option == "Valve Sizing":
        st.markdown("### Valve Sizing")
        # keep imports local as in your snippet (safe)
        import streamlit as st
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import math

        # Placeholder for unit conversion functions (assumed from original context)
        def u_len_label(): return "m" if not st.session_state.get("is_us", False) else "ft"
        def u_press_label(): return "kPa" if not st.session_state.get("is_us", False) else "psi"
        def u_flow_label(): return "m³/h" if not st.session_state.get("is_us", False) else "GPM"
        def u_vel_label(): return "m/s" if not st.session_state.get("is_us", False) else "ft/s"
        def u_diam_label(): return "mm" if not st.session_state.get("is_us", False) else "in"
        def to_si_len(x): return x / 3.28084 if st.session_state.get("is_us", False) else x
        def to_si_press(x): return x * 6.89476 if st.session_state.get("is_us", False) else x
        def to_si_flow(x): return x / 3600.0 if not st.session_state.get("is_us", False) else x / 15850.323141489
        def to_si_diam(x): return x / 1000.0 if not st.session_state.get("is_us", False) else x * 0.0254
        def from_si_flow_m3h(x): return x * 3600.0 if not st.session_state.get("is_us", False) else x * 15850.323141489
        def from_si_vel(x): return x * 3.28084 if st.session_state.get("is_us", False) else x
        def from_si_press_kpa(x): return x / 6.89476 if st.session_state.get("is_us", False) else x
        g = 9.80665  # gravity (m/s²)

        # Atmospheric pressure from elevation (kPa)
        def patm_kpa_from_elevation(elev_m: float) -> float:
            return 101.325 * (1 - 2.255e-5 * elev_m) ** 5.2559

        # Helper for compact pill with tooltip
        def vs_pill(label: str, value_text: str, tooltip: str):
            st.markdown(
                f"""
                <span title="{tooltip}"
                      style="
                        display:inline-block;
                        padding:4px 10px;
                        border-radius:9999px;
                        background:#eef3ff;
                        color:#0b2950;
                        font-size:0.92rem;
                        border:1px solid #d9e3ff;
                        margin:2px 6px 8px 0;
                      ">
                    <strong style="margin-right:6px;">{label}:</strong>{value_text}
                </span>
                """,
                unsafe_allow_html=True,
            )

        # Unit conversion helpers
        def m3s_to_gpm(q_m3s: float) -> float:
            return float(q_m3s * 15850.323141489)

        def kpa_to_psi(p_kpa: float) -> float:
            return float(p_kpa / 6.89476)

        def psi_to_kpa(p_psi: float) -> float:
            return float(p_psi * 6.89476)

        # Valve data (indicative; replace with vendor-specific data for production)
        inch_sizes = [4, 6, 8, 10, 12, 14, 16, 18, 20]
        data_families = {
            "Globe": {
                "FL": 0.90, "Ff": 0.96, "xT": 0.70, "char": "Linear",
                "Cv_table": {4: 110, 6: 240, 8: 420, 10: 650, 12: 1000, 14: 1400, 16: 1800, 18: 2200, 20: 2600}
            },
            "Ball (V-port / Segment)": {
                "FL": 0.80, "Ff": 0.96, "xT": 0.65, "char": "Equal % (R≈50)",
                "Cv_table": {4: 180, 6: 380, 8: 700, 10: 1100, 12: 1600, 14: 2200, 16: 3000, 18: 3800, 20: 4700}
            },
            "Butterfly": {
                "FL": 0.65, "Ff": 0.96, "xT": 0.60, "char": "Equal % (geom.)",
                "Cv_table": {4: 500, 6: 1200, 8: 2200, 10: 3600, 12: 5200, 14: 7200, 16: 9400, 18: 14000, 20: 16500}
            },
            "Plug": {
                "FL": 0.80, "Ff": 0.96, "xT": 0.65, "char": "Linear",
                "Cv_table": {4: 150, 6: 320, 8: 600, 10: 950, 12: 1400, 14: 1900, 16: 2500, 18: 3200, 20: 4000}
            },
            "Gate (High-Pressure / Slurry)": {
                "FL": 0.95, "Ff": 0.98, "xT": 0.80, "char": "Quick-open (Isolation)",
                "Cv_table": {4: 900, 6: 2200, 8: 4000, 10: 6500, 12: 9300, 14: 12000, 16: 15000, 18: 18500, 20: 22000},
                "K_open_default": 0.8
            }
        }

        # Main valve sizing function
        def valve_sizing():
            # Initialize df_sizes to avoid UnboundLocalError
            df_sizes = pd.DataFrame()
            st.caption("Calculate valve size for control or isolation, with cavitation, velocity, and actuator checks.")

            # --- Step 1: Process Data ---
            with st.expander("Process Conditions", expanded=True):
                svc_col1, svc_col2, svc_col3, svc_col4 = st.columns(4)
                service_type = svc_col1.selectbox("Service Type", ["Control (Throttling)", "Isolation (On/Off)"], index=0, key="vs_service_type")
                fluid_state = svc_col2.selectbox("Fluid State", ["Liquid", "Slurry", "Gas (Compressible)"], index=0, key="vs_fluid_state")
                elev_vs = svc_col3.number_input(f"Site Elevation ({u_len_label()})", min_value=0.0, value=0.0, step=10.0, key="vs_elev")
                use_gauge = svc_col4.checkbox("Pressures are Gauge?", value=True, key="vs_is_gauge")
                patm_kpa_vs = patm_kpa_from_elevation(to_si_len(elev_vs))

                # Fluid properties
                fp1, fp2, fp3 = st.columns(3)
                rho_liq_kgL = fp1.number_input("Fluid Density (kg/L)", min_value=0.2, value=1.00, step=0.01, key="vs_rho_liq")
                sg_liq = fp2.number_input("Specific Gravity (SG)", min_value=0.2, value=1.00, step=0.01, key="vs_sg_liq")
                vap_vs_kpa = fp3.number_input(f"Vapor Pressure ({u_press_label()})", min_value=0.0, value=2.3, step=0.1, key="vs_vap")
                rho_liq = rho_liq_kgL * 1000.0

                # Slurry-specific inputs
                if fluid_state == "Slurry":
                    sl1, sl2, sl3, sl4 = st.columns(4)
                    Cv_pct = sl1.number_input("Solids Concentration (%vol)", min_value=0.0, value=20.0, key="vs_sl_cv")
                    Sg_s = sl2.number_input("Solids SG", min_value=1.0, value=2.65, step=0.01, key="vs_sl_sg")
                    d_p_mm = sl3.number_input("Particle Size (mm)", min_value=0.0, value=0.5, step=0.1, key="vs_sl_psize")
                    slurry_derate = sl4.slider("Cv Derating (Slurry)", 0.3, 1.0, 0.75, 0.05, key="vs_sl_derate")
                else:
                    Cv_pct, Sg_s, d_p_mm, slurry_derate = 0.0, 1.0, 0.0, 1.0

                # Gas-specific inputs
                if fluid_state == "Gas (Compressible)":
                    g1, g2 = st.columns(2)
                    gamma = g1.number_input("Gas Specific Heat Ratio (γ)", min_value=1.0, value=1.4, step=0.01, key="vs_gamma")
                    mol_weight = g2.number_input("Molecular Weight (g/mol)", min_value=1.0, value=29.0, step=1.0, key="vs_mol_weight")
                else:
                    gamma, mol_weight = 1.4, 29.0

            # --- Step 2: Operating Scenarios ---
            with st.expander("Operating Scenarios (Min / Normal / Max)", expanded=True):
                st.caption("Enter flow and pressures for each scenario. Gauge pressures are converted to absolute using site elevation.")
                s1a, s1b, s1c = st.columns(3)
                # Min Scenario
                with s1a:
                    st.markdown("**Min**")
                    q_min_disp = st.number_input(f"Flow ({u_flow_label()}) [Min]", min_value=0.0, value=50.0, key="vs_q_min")
                    p1_min = st.number_input(f"P1 ({u_press_label()}) [Min]", min_value=0.0, value=300.0, key="vs_p1_min")
                    p2_min = st.number_input(f"P2 ({u_press_label()}) [Min]", min_value=0.0, value=100.0, key="vs_p2_min")
                # Normal Scenario
                with s1b:
                    st.markdown("**Normal**")
                    q_nor_disp = st.number_input(f"Flow ({u_flow_label()}) [Normal]", min_value=0.0, value=100.0, key="vs_q_nor")
                    p1_nor = st.number_input(f"P1 ({u_press_label()}) [Normal]", min_value=0.0, value=600.0, key="vs_p1_nor")
                    p2_nor = st.number_input(f"P2 ({u_press_label()}) [Normal]", min_value=0.0, value=200.0, key="vs_p2_nor")
                # Max Scenario
                with s1c:
                    st.markdown("**Max**")
                    q_max_disp = st.number_input(f"Flow ({u_flow_label()}) [Max]", min_value=0.0, value=150.0, key="vs_q_max")
                    p1_max = st.number_input(f"P1 ({u_press_label()}) [Max]", min_value=0.0, value=900.0, key="vs_p1_max")
                    p2_max = st.number_input(f"P2 ({u_press_label()}) [Max]", min_value=0.0, value=250.0, key="vs_p2_max")

                # Convert to SI and absolute pressures
                def to_abs_kpa(p_disp: float) -> float:
                    p_kpa = to_si_press(p_disp)
                    return p_kpa + patm_kpa_vs if use_gauge else p_kpa
                q_min_si = to_si_flow(q_min_disp)
                q_nor_si = to_si_flow(q_nor_disp)
                q_max_si = to_si_flow(q_max_disp)
                P1_min_abs = to_abs_kpa(p1_min); P2_min_abs = to_abs_kpa(p2_min)
                P1_nor_abs = to_abs_kpa(p1_nor); P2_nor_abs = to_abs_kpa(p2_nor)
                P1_max_abs = to_abs_kpa(p1_max); P2_max_abs = to_abs_kpa(p2_max)

            # --- Step 3: Piping Context ---
            with st.expander("Piping Near Valve", expanded=True):
                lp1, lp2, lp3, lp4 = st.columns(4)
                line_id_disp = lp1.number_input(f"Line ID at Valve ({u_diam_label()})", min_value=1.0, value=450.0, key="vs_line_id")
                K_up = lp2.number_input("Upstream K (dimensionless)", min_value=0.0, value=2.0, step=0.1, key="vs_k_up")
                K_dn = lp3.number_input("Downstream K (dimensionless)", min_value=0.0, value=3.0, step=0.1, key="vs_k_dn")
                show_k_helper = lp4.checkbox("Quick K Helper", value=False, key="vs_show_k_helper")
                if show_k_helper:
                    with st.expander("Fittings K Helper (Sum K Near Valve)"):
                        st.write("Add fittings logic here (e.g., elbows, tees).")
                D_line_m = to_si_diam(line_id_disp)
                A_line = math.pi * (D_line_m ** 2) / 4.0 if D_line_m > 0 else 1e9

            # --- Step 4: Valve Family & Behavior ---
            with st.expander("Valve Family & Behavior", expanded=True):
                fam = st.selectbox(
                    "Valve Family",
                    ["Globe", "Ball (V-port / Segment)", "Butterfly", "Plug", "Gate (High-Pressure / Slurry)"],
                    index=4 if "Slurry" in fluid_state else 0,
                    key="vs_family"
                )
                if "Gate" in fam and service_type.startswith("Control"):
                    st.warning("Gate valves are for **isolation**. Throttling in slurry service causes erosion. Use Globe, V-ball, Butterfly, or Plug for control.")
                FL = data_families[fam]["FL"]
                Ff = data_families[fam]["Ff"]
                xT = data_families[fam]["xT"]
                Cv_table = data_families[fam]["Cv_table"]
                inherent_char = data_families[fam]["char"]
                cand_sizes_in = st.multiselect("Candidate Sizes (in)", inch_sizes, default=[int(round(D_line_m / 0.0254))], key="vs_cand_sizes")

            # --- Calculation Helpers ---
            # Inherent Cv vs opening
            def cv_inherent(Cv_max: float, opening_pct: float, family: str) -> float:
                x = max(0.0, min(100.0, opening_pct)) / 100.0
                if "Equal %" in inherent_char:
                    R = 50.0
                    Cv_min = max(Cv_max / R, 1e-6)
                    return float(Cv_min * (R ** x))
                elif "Quick-open" in inherent_char:
                    return float(Cv_max * (1 - (1 - x) ** 2))
                else:  # Linear
                    return float(Cv_max * x)

            # Cavitation allowable ΔP (ISA RP 75.23)
            def dp_valve_allow_kpa(P1_abs_kpa: float, Pv_kpa: float) -> float:
                return max(0.0, (FL ** 2) * (P1_abs_kpa - Ff * Pv_kpa))

            # Piping pressure drop from K losses
            def dp_k_kpa(Q_m3s: float) -> float:
                V = Q_m3s / max(A_line, 1e-9)
                dP_Pa = (K_up + K_dn) * 0.5 * rho_liq * (V ** 2)
                return dP_Pa / 1000.0

            # Required Cv for liquid flow
            def required_cv_for(Q_si: float, P1_abs_kpa: float, P2_abs_kpa: float) -> float:
                dP_total = max(0.0, P1_abs_kpa - P2_abs_kpa)
                dP_pipe = dp_k_kpa(Q_si)
                dP_valve = max(1e-6, dP_total - dP_pipe)
                Q_gpm = m3s_to_gpm(Q_si)
                Cv_req = Q_gpm * math.sqrt(max(sg_liq, 1e-6) / kpa_to_psi(dP_valve))
                return float(Cv_req)

            # Gas flow Cv (simplified ISA 75.01)
            def required_cv_gas(Q_si: float, P1_abs_kpa: float, P2_abs_kpa: float, T_K: float = 293.15) -> float:
                P1_psi = kpa_to_psi(P1_abs_kpa)
                P2_psi = kpa_to_psi(P2_abs_kpa)
                dP_psi = P1_psi - P2_psi
                x = min(dP_psi / P1_psi, xT)
                Y = 1 - x / (3 * xT)
                Q_scfh = Q_si * 3600 * 101.325 / (mol_weight * T_K) * 1.185e6  # Approximate conversion to SCFH
                Cv_req = Q_scfh / (1360 * Y * math.sqrt(x * P1_psi / max(sg_liq, 1e-6)))
                return float(Cv_req)

            # Solve installed flow for control valve
            def solve_installed_Q_for_opening(Cv_max: float, opening_pct: float, P1_abs_kpa: float, P2_abs_kpa: float, Pv_kpa: float):
                Cv_eff = slurry_derate * cv_inherent(Cv_max, opening_pct, fam)
                if Cv_eff <= 1e-9:
                    return 0.0, False
                Q_hi = max(q_max_si, q_nor_si * 1.5 + 1e-6)
                cav = False
                for _ in range(40):
                    Q_mid = 0.5 * Q_hi
                    dP_pipe = dp_k_kpa(Q_mid)
                    dP_total = max(0.0, P1_abs_kpa - P2_abs_kpa)
                    dP_valve_kpa_max = max(0.0, dP_total - dP_pipe)
                    Q_gpm = m3s_to_gpm(Q_mid)
                    dP_valve_psi = (Q_gpm / max(Cv_eff, 1e-9)) ** 2 * max(sg_liq, 1e-6)
                    dP_valve_kpa = psi_to_kpa(dP_valve_psi)
                    dP_allow = dp_valve_allow_kpa(P1_abs_kpa, Pv_kpa)
                    if dP_valve_kpa > dP_allow:
                        dP_valve_kpa = dP_allow
                        cav = True
                    if dP_valve_kpa > dP_valve_kpa_max + 1e-6:
                        Q_hi *= 0.5
                    else:
                        Q_hi *= 1.2
                        if Q_hi > 5.0 * max(q_max_si, q_nor_si + 1e-6):
                            break
                return Q_mid, cav

            # Deposition velocity for slurry (Durand–Condolios)
            def deposition_velocity() -> float:
                if fluid_state != "Slurry":
                    return 0.0
                FL_dc = 1.0
                Vc_dep = FL_dc * math.sqrt(2 * g * max(D_line_m, 1e-6) * (Sg_s - 1)) * (Cv_pct / 100.0) ** 0.125
                return Vc_dep

            # --- Step 5: Calculations & Results ---
            if st.button("Calculate Valve Sizing"):
                # Input validation
                if q_max_si <= 0 or P1_max_abs <= P2_max_abs:
                    st.error("Invalid inputs: Ensure Max flow > 0 and P1 > P2 for all scenarios.")
                    return

                # Calculate Cv requirements
                Cv_req_min = required_cv_gas(q_min_si, P1_min_abs, P2_min_abs) if fluid_state == "Gas (Compressible)" else required_cv_for(q_min_si, P1_min_abs, P2_min_abs)
                Cv_req_nor = required_cv_gas(q_nor_si, P1_nor_abs, P2_nor_abs) if fluid_state == "Gas (Compressible)" else required_cv_for(q_nor_si, P1_nor_abs, P2_nor_abs)
                Cv_req_max = required_cv_gas(q_max_si, P1_max_abs, P2_max_abs) if fluid_state == "Gas (Compressible)" else required_cv_for(q_max_si, P1_max_abs, P2_max_abs)

                # Rank candidate sizes
                rows = []
                opening_grid = np.linspace(10, 90, 41)
                for sz in cand_sizes_in:
                    Cv_max_base = Cv_table.get(sz, None)
                    if Cv_max_base is None:
                        continue
                    Cv_max_eff = slurry_derate * Cv_max_base
                    best_theta, best_err, cav_flag = None, 1e9, False
                    for th in opening_grid:
                        Q_mid, cav_mid = solve_installed_Q_for_opening(Cv_max_base, th, P1_nor_abs, P2_nor_abs, vap_vs_kpa)
                        err = abs(Q_mid - q_nor_si)
                        if err < best_err:
                            best_err, best_theta, cav_flag = err, th, cav_mid
                    # Valve authority
                    V_n = q_nor_si / max(A_line, 1e-9)
                    dP_pipe_n = (K_up + K_dn) * 0.5 * rho_liq * (V_n ** 2) / 1000.0
                    dP_total_n = max(0.0, P1_nor_abs - P2_nor_abs)
                    dP_valve_n_kpa = max(1e-6, dP_total_n - dP_pipe_n)
                    Cv_req_n = Cv_req_nor
                    dP_valve_only_n_psi = (m3s_to_gpm(q_nor_si) / max(Cv_req_n, 1e-6)) ** 2 * max(sg_liq, 1e-6)
                    dP_valve_only_n_kpa = psi_to_kpa(dP_valve_only_n_psi)
                    authority = dP_valve_only_n_kpa / max(dP_total_n, 1e-6)
                    rows.append({
                        "Size (in)": sz,
                        "Cv_max (eff)": round(Cv_max_eff, 1),
                        "%Open @Normal": round(best_theta if best_theta is not None else float('nan'), 1),
                        "Authority (N)": round(authority, 3),
                        "Cavitation @Normal": "Yes" if cav_flag and fluid_state != "Gas (Compressible)" else "No",
                        "Cv_req Min": round(Cv_req_min, 1),
                        "Cv_req Normal": round(Cv_req_nor, 1),
                        "Cv_req Max": round(Cv_req_max, 1),
                        "OK (capacity)": "Yes" if Cv_max_eff >= Cv_req_max else "No",
                    })

                df_sizes = pd.DataFrame(rows)
                if not df_sizes.empty:
                    # Recommend best size
                    df_ok = df_sizes[df_sizes["OK (capacity)"] == "Yes"].copy()
                    if not df_ok.empty:
                        df_ok["open_score"] = (df_ok["%Open @Normal"] - 55.0).abs()
                        rec_row = df_ok.sort_values(by=["open_score", "Authority (N)"], ascending=[True, False]).iloc[0]
                    else:
                        rec_row = df_sizes.sort_values(by=["Cv_req Max"], ascending=True).iloc[0]

                    # Display results
                    st.markdown("#### Results")
                    vs_pill("Recommended Size / Family",
                            f"{int(rec_row['Size (in)'])} in — {fam}",
                            "Selected for Max flow with Cv while keeping %Open near 40–70% at Normal.")
                    vs_pill("% Open @ Normal",
                            f"{rec_row['%Open @Normal']:.1f} %",
                            "Target 40–70% for good controllability.")
                    vs_pill("Valve Authority (N)",
                            f"{rec_row['Authority (N)']:.3f}",
                            "N = ΔP_valve / (ΔP_valve + ΔP_piping) at Normal. ≥0.3 recommended.")
                    vs_pill("Cavitation @ Normal",
                            f"{rec_row['Cavitation @Normal']}",
                            "If 'Yes', valve drop reaches cavitation limit; consider anti-cavitation trim or staging.")

                    # Plots
                    cplt1, cplt2, cplt3 = st.columns(3)
                    rec_size = int(rec_row["Size (in)"])
                    Cv_max_base = Cv_table.get(rec_size, list(Cv_table.values())[-1])
                    thetas = np.linspace(5, 95, 61)
                    Q_inst = []
                    cav_marks = []
                    for th in thetas:
                        q_th, cflag = solve_installed_Q_for_opening(Cv_max_base, th, P1_nor_abs, P2_nor_abs, vap_vs_kpa)
                        Q_inst.append(q_th)
                        cav_marks.append(cflag)
                    Q_inst = np.array(Q_inst)

                    with cplt1:
                        st.subheader("Installed Flow vs %Open")
                        df_inst = pd.DataFrame({
                            "%Open": thetas,
                            f"Flow ({u_flow_label()})": [from_si_flow_m3h(q) for q in Q_inst]
                        })
                        fig1 = px.line(df_inst, x="%Open", y=f"Flow ({u_flow_label()})")
                        fig1.add_hline(y=from_si_flow_m3h(q_nor_si), line_dash="dot", annotation_text="Normal flow")
                        fig1.add_hline(y=from_si_flow_m3h(q_min_si), line_dash="dot", annotation_text="Min flow")
                        fig1.add_hline(y=from_si_flow_m3h(q_max_si), line_dash="dot", annotation_text="Max flow")
                        fig1.update_layout(height=330, xaxis_title="% Open", yaxis_title=f"Flow ({u_flow_label()})")
                        st.plotly_chart(fig1, use_container_width=True)

                    with cplt2:
                        st.subheader("Inherent Cv vs %Open")
                        Cv_curve = [slurry_derate * cv_inherent(Cv_max_base, th, fam) for th in thetas]
                        df_cv = pd.DataFrame({"%Open": thetas, "Cv (effective)": Cv_curve})
                        fig2 = px.line(df_cv, x="%Open", y="Cv (effective)")
                        fig2.add_hline(y=Cv_req_min, line_dash="dot", annotation_text="Cv required (Min)")
                        fig2.add_hline(y=Cv_req_nor, line_dash="dot", annotation_text="Cv required (Normal)")
                        fig2.add_hline(y=Cv_req_max, line_dash="dot", annotation_text="Cv required (Max)")
                        fig2.update_layout(height=330, xaxis_title="% Open", yaxis_title="Cv (dimensionless)")
                        st.plotly_chart(fig2, use_container_width=True)

                    with cplt3:
                        st.subheader("Outlet Velocity vs %Open")
                        V_out = [from_si_vel(q / max(A_line, 1e-9)) for q in Q_inst]
                        df_v = pd.DataFrame({"%Open": thetas, f"Velocity ({u_vel_label()})": V_out})
                        fig3 = px.line(df_v, x="%Open", y=f"Velocity ({u_vel_label()})")
                        if fluid_state == "Slurry":
                            Vc_dep = deposition_velocity()
                            fig3.add_hline(y=from_si_vel(Vc_dep), line_dash="dot", annotation_text="Deposition velocity")
                        fig3.update_layout(height=330, xaxis_title="% Open", yaxis_title=f"Velocity ({u_vel_label()})")
                        st.plotly_chart(fig3, use_container_width=True)

                    st.dataframe(df_sizes, use_container_width=True)
                    st.info("Switch to Isolation mode for gate valve specifics or adjust candidate sizes above.")

                # Isolation Mode
                else:
                    st.markdown("#### Isolation Valve Sizing")
                    iso1, iso2, iso3, iso4 = st.columns(4)
                    gate_K_open = iso1.number_input("Gate Valve K (open)", min_value=0.0, value=data_families.get(fam, {}).get("K_open_default", 0.8), step=0.1, key="vs_gate_k")
                    design_pressure = iso2.number_input(f"Design Pressure ({u_press_label()})", min_value=0.0, value=3600.0, step=50.0, key="vs_design_press")
                    rating_pressure = iso3.number_input(f"Selected Valve Rating ({u_press_label()})", min_value=0.0, value=4000.0, step=50.0, key="vs_rating_press")
                    cand_iso_in = iso4.selectbox(
                        "Isolation Size (in)",
                        inch_sizes,
                        index=min(
                            len(inch_sizes) - 1,
                            max(0, inch_sizes.index(int(round(D_line_m / 0.0254))) if int(round(D_line_m / 0.0254)) in inch_sizes else 4)
                        ),
                        key="vs_iso_size"
                    )
                    V_n = q_nor_si / max(A_line, 1e-9)
                    dP_gate_open_kpa = (gate_K_open * 0.5 * rho_liq * V_n ** 2) / 1000.0
                    vel_line = from_si_vel(V_n)
                    ok_rating = (design_pressure <= rating_pressure)
                    bore_m = to_si_diam(line_id_disp)
                    seat_area = math.pi * (bore_m ** 2) / 4.0
                    deltaP_Pa = to_si_press(design_pressure) * 1000.0
                    thrust_N = 1.2 * deltaP_Pa * seat_area * 1.10  # 20% safety + 10% friction
                    vs_pill("ΔP across Gate (open)",
                            f"{from_si_press_kpa(dP_gate_open_kpa):.2f} {u_press_label()}",
                            "Pressure drop from open-geometry K: ΔP = K · ρV²/2.")
                    vs_pill("Line Velocity @ Normal",
                            f"{vel_line:.2f} {u_vel_label()}",
                            "High velocity in slurry increases erosion risk.")
                    vs_pill("Rating Check",
                            "PASS" if ok_rating else "FAIL",
                            "Compares design pressure to valve rating.")
                    vs_pill("Estimated Actuator Thrust",
                            f"{thrust_N / 1000.0:.1f} kN",
                            "Approximate closing thrust for hydraulic actuator (incl. margin).")

                    # Isolation plots
                    ciso1, ciso2, ciso3 = st.columns(3)
                    flows = np.linspace(max(q_min_si, 1e-6), max(q_max_si, q_nor_si * 1.2 + 1e-6), 50)
                    with ciso1:
                        st.subheader("Velocity vs Flow")
                        vel = [from_si_vel(q / max(A_line, 1e-9)) for q in flows]
                        df_vf = pd.DataFrame({f"Flow ({u_flow_label()})": [from_si_flow_m3h(q) for q in flows],
                                              f"Velocity ({u_vel_label()})": vel})
                        fig_vf = px.line(df_vf, x=f"Flow ({u_flow_label()})", y=f"Velocity ({u_vel_label()})")
                        fig_vf.update_layout(height=330, xaxis_title=f"Flow ({u_flow_label()})", yaxis_title=f"Velocity ({u_vel_label()})")
                        st.plotly_chart(fig_vf, use_container_width=True)
                    with ciso2:
                        st.subheader("ΔP(open) vs Flow")
                        dps = [(gate_K_open * 0.5 * rho_liq * (q / max(A_line, 1e-9)) ** 2) / 1000.0 for q in flows]
                        dp_key = f"ΔP ({'psi' if st.session_state.get('is_us', False) else 'kPa'})"
                        df_dp = pd.DataFrame({f"Flow ({u_flow_label()})": [from_si_flow_m3h(q) for q in flows],
                                              dp_key: [kpa_to_psi(dp) if st.session_state.get("is_us", False) else dp for dp in dps]})
                        fig_dp = px.line(df_dp, x=f"Flow ({u_flow_label()})", y=dp_key)
                        fig_dp.update_layout(height=330, xaxis_title=f"Flow ({u_flow_label()})", yaxis_title=dp_key)
                        st.plotly_chart(fig_dp, use_container_width=True)
                    with ciso3:
                        st.subheader("Capacity (Cv) by Size")
                        df_cv_iso = pd.DataFrame({"Size (in)": list(Cv_table.keys()),
                                                  "Cv_max": list(Cv_table.values())})
                        fig_cv_iso = px.line(df_cv_iso, x="Size (in)", y="Cv_max")
                        fig_cv_iso.update_layout(height=330, xaxis_title="Size (in)", yaxis_title="Cv (max)")
                        st.plotly_chart(fig_cv_iso, use_container_width=True)

        valve_sizing()

    # ---------- D) ORIFICE SIZING ----------
    elif tool_option == "Orifice Sizing":
        st.markdown("### Orifice Sizing")
        advanced_mode = st.checkbox("Advanced/Industrial Mode", key="os_adv")

        flow = st.number_input(f"Flow Rate ({u_flow_label()})", min_value=0.0, value=100.0, key="os_flow")
        deltaP = st.number_input(f"Pressure Drop ({u_press_label()})", min_value=0.0, value=10.0, key="os_dp")
        rho = st.number_input("Density (kg/m³)", min_value=100.0, value=1000.0, key="os_rho")
        Cd = st.number_input("Discharge Coefficient", min_value=0.1, value=0.62, key="os_cd")
        beta = st.number_input("Beta Ratio (d/D)", min_value=0.1, max_value=0.8, value=0.5, key="os_beta")
        D_pipe = to_si_diam(st.number_input(f"Pipe ID ({u_diam_label()})", min_value=10.0, value=100.0, key="os_pipe_id"))
        fluid_state = st.selectbox("Fluid State", ["Liquid", "Gas", "Two-Phase"], key="os_state")

        # Geometry
        d_orif = beta * max(D_pipe, 1e-9)
        A_orif = math.pi * (d_orif**2) / 4.0

        # Flow (SI)
        Q_si = to_si_flow(flow)

        # Basic calc: ΔP from orifice equation (kPa)
        calculated_deltaP = ((Q_si / max(Cd * max(A_orif, 1e-12), 1e-12))**2) * rho / 2.0 / 1000.0 if A_orif > 0 else 0.0
        st.metric("Calculated ΔP (kPa)", f"{calculated_deltaP:.2f}")

        # Solve required orifice ID for target ΔP
        if deltaP > 0:
            d_solved = math.sqrt(
                4.0 * Q_si / (max(Cd, 1e-9) * math.pi * math.sqrt(2.0 * to_si_press(deltaP) * 1000.0 / max(rho, 1e-9)))
            )
        else:
            d_solved = 0.0
        st.metric("Required Orifice ID (m)", f"{d_solved:.4f}")

        if advanced_mode:
            tap_loc = st.selectbox("Tap Location", ["Flange", "D and D/2"], key="os_tap")
            # Placeholder permanent loss estimate
            perm_loss = 0.5 * calculated_deltaP
            st.metric("Permanent Loss (kPa)", f"{perm_loss:.2f}")

        # Sensitivity: ΔP vs β at fixed Q, D, Cd, ρ
        betas = np.linspace(0.2, 0.8, 50)
        dPs = []
        for b in betas:
            d_b = b * max(D_pipe, 1e-9)
            A_b = math.pi * (d_b**2) / 4.0
            dp_b = ((Q_si / max(Cd * max(A_b, 1e-12), 1e-12))**2) * rho / 2.0 / 1000.0
            dPs.append(dp_b)

        fig_orif = px.line(x=betas, y=dPs, labels={"x": "Beta (d/D)", "y": "ΔP (kPa)"})
        fig_orif.add_scatter(x=[beta], y=[calculated_deltaP], mode="markers", name="Current")
        fig_orif.update_layout(height=300, xaxis_title="Beta (d/D)", yaxis_title="ΔP (kPa)")
        st.plotly_chart(fig_orif, use_container_width=True)

# ============ MECHANICAL ============
with sub_mech:
    st.markdown("### Mechanical Calculators")
    mech_tools = [
        "Beam Deflection & Stress", "Shaft Torque & Critical Speed", "Bearing Life (L10)",
        "Bolt Preload & Torque", "Fatigue Analysis (Goodman Diagram)", "Gear Ratio & Life",
        "Vibration Analysis"
    ]
    tool_search = st.session_state.get("tool_search", "")
    filtered_mech = [t for t in mech_tools if tool_search.lower() in t.lower()]
    mech_tool = st.selectbox("Select Mechanical Tool", filtered_mech or mech_tools, index=0)

    if mech_tool == "Beam Deflection & Stress":
        st.markdown("### Beam Deflection & Stress")
        beam_type = st.selectbox("Beam Type", ["Cantilever", "Simply Supported", "Fixed-Fixed", "Overhanging"])
        material = st.selectbox("Material", list(MATERIAL_PROPS.keys()))
        props = MATERIAL_PROPS[material]
        E = props["E_GPa"] * 1e9
        yield_strength = props["yield_MPa"]
        L = to_si_len(st.number_input(f"Span Length ({u_len_label()})", min_value=0.1, value=1.0))
        cross_section = st.selectbox("Cross Section", ["Rectangular", "Circular", "I-Beam"])
        # Section properties
        if cross_section == "Rectangular":
            b = st.number_input("Width (m)", min_value=0.01, value=0.1)
            h = st.number_input("Height (m)", min_value=0.01, value=0.2)
            I = (b * h**3) / 12
            c = h/2
        elif cross_section == "Circular":
            d = st.number_input("Diameter (m)", min_value=0.01, value=0.1)
            I = math.pi * (d**4) / 64
            c = d/2
        else:
            I = st.number_input("Moment of Inertia (m⁴)", min_value=1e-10, value=1e-6, format="%e")
            c = st.number_input("Outer-fiber distance c (m)", min_value=0.001, value=0.05)

        load_type = st.selectbox("Load Type", ["Point Load at Center", "Point Load at End", "Uniform Distributed Load", "Triangular Load"])
        if "Point" in load_type:
            P = to_si_force(st.number_input(f"Point Load ({u_force_label()})", min_value=0.0, value=1000.0))
            _ = st.number_input("Load Position from Left (fraction of L)", min_value=0.0, max_value=1.0, value=0.5)
        else:
            w = to_si_force(st.number_input(f"Distributed Load ({u_force_label()}/m)", min_value=0.0, value=100.0)) / max(L,1e-9)
            P = w * L

        # Closed-form examples
        if beam_type == "Simply Supported" and load_type == "Point Load at Center":
            deflection = (P * L**3) / (48 * E * I)
            moment = P * L / 4
        elif beam_type == "Cantilever" and load_type == "Point Load at End":
            deflection = (P * L**3) / (3 * E * I)
            moment = P * L
        elif beam_type == "Simply Supported" and load_type == "Uniform Distributed Load":
            deflection = (5 * (P/L) * L**4) / (384 * E * I)  # P/L = w
            moment = (P/L) * L**2 / 8
        else:
            deflection = 0.0
            moment = 0.0
        stress = (moment * c) / max(I,1e-12)

        st.metric("Max Deflection (m)", f"{deflection:.6f}")
        st.metric("Max Bending Stress (MPa)", f"{stress / 1e6:.2f}")
        if (stress / 1e6) > yield_strength:
            st.warning("Stress exceeds yield strength!")

        # Plot deflection curve (vectorized)
        fig, ax = plt.subplots()
        x = np.linspace(0, L, 400)
        if beam_type == "Simply Supported" and load_type == "Point Load at Center":
            x_mid = L/2
            y_left  = -(P * x**2 / (6 * E * I)) * (3*L - 4*x)
            y_right = -(P * (L - x)**2 / (6 * E * I)) * (3*L - 4*(L - x))
            y = np.where(x <= x_mid, y_left, y_right)
        elif beam_type == "Cantilever" and load_type == "Point Load at End":
            y = -(P * x**2 * (3*L - x)) / (6 * E * I)
        elif beam_type == "Simply Supported" and load_type == "Uniform Distributed Load":
            w_lin = P / max(L,1e-12)
            # y(x) = w x (L^3 - 2 L x^2 + x^3) / (24 E I)
            y = - (w_lin * x * (L**3 - 2*L*x**2 + x**3)) / (24 * E * I)
        else:
            y = np.zeros_like(x)
        ax.plot(x, y)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Deflection (m)")
        st.pyplot(fig)

    elif mech_tool == "Shaft Torque & Critical Speed":
        st.markdown("### Shaft Torque & Critical Speed")
        power = to_si_power_kw(st.number_input(f"Power ({u_power_label()})", min_value=0.0, value=10.0))
        rpm = st.number_input("Speed (rpm)", min_value=0.0, value=1500.0)
        torque = (power * 1000 * 60) / (2 * math.pi * rpm) if rpm > 0 else 0.0
        st.metric("Torque (N-m)", f"{torque:.2f}")
        material = st.selectbox("Material", list(MATERIAL_PROPS.keys()))
        props = MATERIAL_PROPS[material]
        E = props["E_GPa"] * 1e9
        density = props["density_kgm3"]
        L = to_si_len(st.number_input(f"Shaft Length ({u_len_label()})", min_value=0.1, value=1.0))
        d = to_si_diam(st.number_input(f"Diameter ({u_diam_label()})", min_value=10.0, value=50.0))
        I = math.pi * (d**4) / 64
        mass = density * math.pi * (d**2) / 4 * L
        delta = (mass * g * L**3) / (3 * E * I) if I>0 else 1e12
        N_cr = (60 / (2 * math.pi)) * math.sqrt(g / max(delta,1e-12))
        st.metric("Critical Speed (rpm)", f"{N_cr:.0f}")

    elif mech_tool == "Bearing Life (L10)":
        st.markdown("### Bearing Life (L10)")
        P = to_si_force(st.number_input(f"Equivalent Load ({u_force_label()})", min_value=0.0, value=1000.0))
        C = to_si_force(st.number_input(f"Dynamic Load Rating ({u_force_label()})", min_value=0.0, value=5000.0))
        rpm = st.number_input("Speed (rpm)", min_value=0.0, value=1500.0)
        a = 3
        L10 = (C / max(P,1e-12))**a * 1e6 / max(60 * max(rpm,1e-12),1e-12)
        st.metric("L10 Life (hours)", f"{L10:.2f}")

    elif mech_tool == "Bolt Preload & Torque":
        st.markdown("### Bolt Preload & Torque")
        d = to_si_diam(st.number_input(f"Bolt Diameter ({u_diam_label()})", min_value=1.0, value=10.0))
        K = st.number_input("Torque Coefficient", min_value=0.1, value=0.2)
        preload = to_si_force(st.number_input(f"Desired Preload ({u_force_label()})", min_value=0.0, value=5000.0))
        torque = K * preload * (d / 1000)
        st.metric("Required Torque (N-m)", f"{torque:.2f}")

    elif mech_tool == "Fatigue Analysis (Goodman Diagram)":
        st.markdown("### Fatigue Analysis (Goodman Diagram)")
        Su = to_si_stress(st.number_input(f"Ultimate Strength ({u_stress_label()})", min_value=0.0, value=500.0))
        Sy = to_si_stress(st.number_input(f"Yield Strength ({u_stress_label()})", min_value=0.0, value=300.0))
        sigma_a = to_si_stress(st.number_input(f"Alternating Stress ({u_stress_label()})", min_value=0.0, value=100.0))
        sigma_m = to_si_stress(st.number_input(f"Mean Stress ({u_stress_label()})", min_value=0.0, value=150.0))
        Se = 0.5 * Su
        denom = sigma_a + (Se / max(Su,1e-12)) * sigma_m
        safety = (Se / denom) if denom > 0 else float('inf')
        st.metric("Safety Factor", f"{safety:.2f}")
        fig, ax = plt.subplots()
        mean = np.linspace(0, Su, 100)
        alt = Se * (1 - mean / max(Su,1e-12))
        ax.plot(mean, alt, label="Goodman Line")
        ax.plot([0, Sy], [Sy, 0], label="Yield Line")
        ax.scatter([sigma_m], [sigma_a], label="Operating Point")
        ax.set_xlabel("Mean Stress (MPa)")
        ax.set_ylabel("Alternating Stress (MPa)")
        ax.legend()
        st.pyplot(fig)

    elif mech_tool == "Gear Ratio & Life":
        st.markdown("### Gear Ratio & Life")
        N1 = st.number_input("Input Speed (rpm)", min_value=0.0, value=1500.0)
        N2 = st.number_input("Output Speed (rpm)", min_value=0.0, value=500.0)
        ratio = (N1 / N2) if N2 > 0 else 0.0
        st.metric("Gear Ratio", f"{ratio:.2f}")
        P = to_si_power_kw(st.number_input(f"Power ({u_power_label()})", min_value=0.0, value=10.0))
        st.caption("Add detailed AGMA factors as needed.")

    elif mech_tool == "Vibration Analysis":
        st.markdown("### Vibration Analysis")
        mass = st.number_input("Mass (kg)", min_value=0.0, value=10.0)
        k = st.number_input("Stiffness (N/m)", min_value=0.0, value=1000.0)
        c = st.number_input("Damping (Ns/m)", min_value=0.0, value=20.0)
        fn = math.sqrt(k / max(mass,1e-12)) / (2 * math.pi) if mass > 0 else 0.0
        zeta = c / (2 * math.sqrt(max(k,1e-12) * max(mass,1e-12))) if mass > 0 else 0.0
        st.metric("Natural Frequency (Hz)", f"{fn:.2f}")
        st.metric("Damping Ratio", f"{zeta:.2f}")

# ============ ELECTRICAL / THERMAL ============
with sub_elec:
    st.markdown("### Electrical & Thermal Tools")
    elec_tools = [
        "Voltage Drop Calculator", "Cable Sizing", "Power Factor Correction",
        "Heat Transfer (Conduction/Convection)", "Pipe Insulation Thickness", "Thermal Expansion"
    ]
    tool_search = st.session_state.get("tool_search", "")
    filtered_elec = [t for t in elec_tools if tool_search.lower() in t.lower()]
    elec_tool = st.selectbox("Select Tool", filtered_elec or elec_tools, index=0)

    # common resistance table (ohm/km, copper) available to both tools
    resistance_ohm_km = {
        14: 8.286, 12: 5.211, 10: 3.277, 8: 2.061, 6: 1.296, 4: 0.815, 2: 0.512, 1: 0.406,
        "1/0": 0.322, "2/0": 0.255, "3/0": 0.202, "4/0": 0.160
    }

    if elec_tool == "Voltage Drop Calculator":
        st.markdown("### Voltage Drop Calculator")
        voltage = st.number_input("System Voltage (V)", min_value=1.0, value=480.0)
        current = st.number_input("Current (A)", min_value=0.0, value=100.0)
        length = to_si_len(st.number_input(f"Cable Length ({u_len_label()})", min_value=0.0, value=100.0))
        conductor_size_awg = st.selectbox("Conductor Size (AWG)", list(resistance_ohm_km.keys()))
        R = resistance_ohm_km.get(conductor_size_awg, 1.0) / 1000  # ohm/m
        vd = (2 * length * current * R) / voltage * 100 if voltage > 0 else 0.0
        st.metric("Voltage Drop (%)", f"{vd:.2f}")
        if vd > 3.0:
            st.warning("Exceeds 3% limit - upsize cable or reduce load.")

    elif elec_tool == "Cable Sizing":
        st.markdown("### Cable Sizing")
        current = st.number_input("Load Current (A)", min_value=0.0, value=100.0)
        voltage = st.number_input("Voltage (V)", min_value=1.0, value=400.0)
        length = to_si_len(st.number_input(f"Length ({u_len_label()})", min_value=0.0, value=100.0))
        vd_max = st.number_input("Max VD (%)", min_value=1.0, value=3.0)
        R_max = (vd_max / 100 * voltage) / (2 * max(length,1e-12) * max(current,1e-12))
        candidates = [awg for awg, r in resistance_ohm_km.items() if (r / 1000) <= R_max]
        if candidates:
            st.success(f"Recommended Cable: {candidates[0]} AWG (or better)")
        else:
            st.warning("Required voltage-drop limit is too strict for listed sizes.")

    elif elec_tool == "Power Factor Correction":
        st.markdown("### Power Factor Correction")
        P = to_si_power_kw(st.number_input(f"Active Power ({u_power_label()})", min_value=0.0, value=100.0))
        pf_current = st.number_input("Current PF", min_value=0.0, max_value=1.0, value=0.8)
        pf_target = st.number_input("Target PF", min_value=0.0, max_value=1.0, value=0.95)
        Qc = P * (math.tan(math.acos(pf_current)) - math.tan(math.acos(pf_target)))
        st.metric("Capacitor Bank (kVAR)", f"{Qc:.2f}")

    elif elec_tool == "Heat Transfer (Conduction/Convection)":
        st.markdown("### Heat Transfer")
        mode = st.selectbox("Mode", ["Conduction", "Convection", "Radiation"])
        if mode == "Conduction":
            k = MATERIAL_PROPS[st.selectbox("Material", list(MATERIAL_PROPS.keys()))]["thermal_cond_WmK"]
            A = st.number_input("Area (m²)", min_value=0.0, value=1.0)
            dx = st.number_input("Thickness (m)", min_value=0.001, value=0.01)
            dT = st.number_input("Temperature Difference (°C)", min_value=0.0, value=50.0)
            q = k * A * dT / max(dx,1e-12)
            st.metric("Heat Flux (W)", f"{q:.2f}")
        elif mode == "Convection":
            h = st.number_input("Conv Coefficient (W/m²K)", min_value=0.0, value=10.0)
            A = st.number_input("Area (m²)", min_value=0.0, value=1.0)
            dT = st.number_input("Temperature Difference (°C)", min_value=0.0, value=50.0)
            q = h * A * dT
            st.metric("Heat Transfer Rate (W)", f"{q:.2f}")
        else:
            sigma = 5.67e-8
            e = st.number_input("Emissivity", min_value=0.0, max_value=1.0, value=0.9)
            A = st.number_input("Area (m²)", min_value=0.0, value=1.0)
            T1 = to_si_temp(st.number_input("T1 (°C)", value=100.0), 'C') + 273.15
            T2 = to_si_temp(st.number_input("T2 (°C)", value=20.0), 'C') + 273.15
            q = sigma * e * A * (T1**4 - T2**4)
            st.metric("Radiation Heat (W)", f"{q:.2f}")

    elif elec_tool == "Pipe Insulation Thickness":
        st.markdown("### Pipe Insulation Thickness")
        D_pipe = to_si_diam(st.number_input(f"Pipe OD ({u_diam_label()})", min_value=10.0, value=100.0))
        k_ins = st.number_input("Insulation k (W/mK)", min_value=0.01, value=0.04)
        h_air = st.number_input("Air Conv Coefficient (W/m²K)", min_value=5.0, value=10.0)
        T_pipe = st.number_input("Pipe Temp (°C)", value=80.0)
        T_amb = st.number_input("Ambient Temp (°C)", value=25.0)
        q_max = st.number_input("Max Allowed Heat Loss (W/m)", min_value=0.0, value=50.0)
        t = 0.01
        for _ in range(80):
            R_cond = math.log((D_pipe + 2*t) / max(D_pipe,1e-9)) / (2 * math.pi * k_ins)
            R_conv = 1 / (h_air * math.pi * (D_pipe + 2*t))
            q = (T_pipe - T_amb) / (R_cond + R_conv)
            t += 0.001 * (q_max - q) / max(q_max,1e-9)
            t = max(t, 0.0)
        st.metric("Required Thickness (mm)", f"{t*1000:.1f}")

    elif elec_tool == "Thermal Expansion":
        st.markdown("### Thermal Expansion")
        material = st.selectbox("Material", list(MATERIAL_PROPS.keys()))
        # simple table; expand if needed
        alpha_tbl = {"Steel (mild)": 12e-6, "Aluminum": 23e-6, "Copper": 17e-6, "Concrete": 10e-6, "Wood (pine)": 5e-6, "Custom": 12e-6}
        alpha = alpha_tbl.get(material, 12e-6)
        L0 = to_si_len(st.number_input(f"Initial Length ({u_len_label()})", min_value=0.0, value=1.0))
        dT = st.number_input("Temperature Change (°C)", value=50.0)
        deltaL = alpha * L0 * dT
        st.metric("Expansion (m)", f"{deltaL:.6f}")

# ============ UNIT CONVERTER ============
with sub_unit:
    st.markdown("### Unit Converter")
    categories = ["Length", "Area", "Volume", "Mass", "Density", "Pressure", "Flow", "Velocity", "Power", "Temperature", "Viscosity", "Force", "Stress", "Torque"]
    cat = st.selectbox("Category", categories)
    units = {
        "Length": {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0, "ft": 0.3048, "in": 0.0254, "yd": 0.9144, "mi": 1609.34},
        "Area": {"m²": 1.0, "cm²": 1e-4, "mm²": 1e-6, "km²": 1e6, "ft²": 0.092903, "in²": 0.00064516},
        "Volume": {"m³": 1.0, "L": 0.001, "cm³": 1e-6, "gal (US)": 0.00378541, "ft³": 0.0283168, "bbl": 0.158987},
        "Mass": {"kg": 1.0, "g": 0.001, "lb": 0.453592, "oz": 0.0283495, "ton (metric)": 1000.0},
        "Density": {"kg/m³": 1.0, "g/cm³": 1000.0, "lb/ft³": 16.0185},
        "Pressure": {"Pa": 1.0, "kPa": 1000.0, "bar": 1e5, "psi": 6894.76, "atm": 101325.0, "mmHg": 133.322},
        "Flow": {"m³/s": 1.0, "m³/h": 1/3600, "L/min": 1/60000, "gpm (US)": 0.0000630902, "cfm": 0.000471947},
        "Velocity": {"m/s": 1.0, "km/h": 1/3.6, "ft/s": 0.3048, "mph": 0.44704},
        "Power": {"W": 1.0, "kW": 1000.0, "hp": 745.7, "BTU/h": 0.293071},
        "Temperature": {"C": lambda x: x, "F": lambda x: (x-32)*5/9, "K": lambda x: x-273.15},
        "Viscosity": {"Pa·s": 1.0, "cP": 0.001, "lb/(ft·s)": 1.48816},
        "Force": {"N": 1.0, "kN": 1000.0, "lb": 4.44822, "kgf": 9.80665},
        "Stress": {"Pa": 1.0, "MPa": 1e6, "kPa": 1000.0, "psi": 6894.76},
        "Torque": {"N-m": 1.0, "kN-m": 1000.0, "lb-ft": 1.35582, "kg-m": 9.80665},
    }
    if cat == "Temperature":
        from_unit = st.selectbox("From Unit", ["°C", "°F", "K"])
        to_unit = st.selectbox("To Unit", ["°C", "°F", "K"])
        value = st.number_input("Value")
        base = value if from_unit=="°C" else ((value - 32) * 5 / 9 if from_unit=="°F" else value - 273.15)
        result = base if to_unit=="°C" else (base * 9 / 5 + 32 if to_unit=="°F" else base + 273.15)
        st.metric("Result", f"{result:.2f} {to_unit}")
    else:
        u_dict = units[cat]
        from_unit = st.selectbox("From Unit", list(u_dict.keys()))
        to_unit = st.selectbox("To Unit", list(u_dict.keys()))
        value = st.number_input("Value", value=1.0)
        base = value * u_dict[from_unit]
        result = base / u_dict[to_unit]
        st.metric("Result", f"{result:.4f} {to_unit}")

    with st.expander("Batch Conversion"):
        df_batch = pd.DataFrame(columns=["Value", "From", "To"])
        edited_batch = st.data_editor(df_batch, num_rows="dynamic")
        if not edited_batch.empty:
            results = []
            for _, row in edited_batch.iterrows():
                try:
                    v = float(row["Value"]); fu = row["From"]; tu = row["To"]
                    if fu in u_dict and tu in u_dict:
                        base = v * u_dict[fu]
                        res = base / u_dict[tu]
                        results.append(res)
                    else:
                        results.append(np.nan)
                except Exception:
                    results.append(np.nan)
            edited_batch["Result"] = results
            st.dataframe(edited_batch, use_container_width=True)

# ============ SCIENTIFIC CALCULATOR ============
with sub_sci:
    st.markdown("### Scientific Calculator with Units")

    import streamlit as st
    from streamlit.components.v1 import html

    html(r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MRX-82 PRO – Scientific Calculator</title>
<style>
  :root{
    --bg:#e8ebf0; --bezel:#14181d; --rim:#2a3036; --screen1:#e8f4ea; --screen2:#cfe5d4;
    --txt:#0f172a; --accent:#3b82f6; --key:#f6f7fb; --key-d:#dfe3ee; --shadow:rgba(0,0,0,.45);
    --key-b:#eaf2ff; --key-o:#fff4e5; --key-r:#ffe5e5;
  }
  html,body{margin:0;height:100%;background:linear-gradient(180deg,var(--bg),#cdd4dd);font-family:Inter,system-ui,Segoe UI,Roboto,Arial,sans-serif}
  .wrap{display:flex;align-items:center;justify-content:center;padding:14px}
  .device{width:min(640px,96vw);border-radius:28px;background:var(--bezel);color:#e5e7eb;
    border:1px solid var(--rim);box-shadow:0 28px 56px var(--shadow), inset 0 2px 0 rgba(255,255,255,.05);
    padding:16px 16px 18px}
  .top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;color:#d1d5db}
  .brand{font-weight:800;letter-spacing:.35px}.model{color:var(--accent);font-weight:800}
  .screen{border-radius:16px;padding:12px 14px;background:linear-gradient(180deg,var(--screen1),var(--screen2));
    color:#0f172a;box-shadow:inset 0 1px 0 #fff, inset 0 -3px 12px rgba(0,0,0,.08)}
  .ind{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}
  .chip{padding:2px 10px;border-radius:999px;background:rgba(10,27,18,.06);border:1px solid rgba(10,27,18,.12);font-size:12px}
  .lcd-in{font-family:ui-monospace,Consolas,Menlo,monospace;font-size:16px;min-height:22px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .lcd-out{font-family:ui-monospace,Consolas,Menlo,monospace;font-size:28px;margin-top:2px;text-shadow:0 1px 0 rgba(255,255,255,.6)}
  .pills{display:flex;gap:8px;margin-top:8px}
  .pill{font-size:11px;color:#334155;padding:3px 8px;border-radius:999px;border:1px solid rgba(51,65,85,.25);background:rgba(255,255,255,.7)}
  .pill.on{background:#dbeafe;border-color:#93c5fd;color:#1d4ed8}

  .pad{margin-top:14px;border-radius:18px;padding:14px;background:linear-gradient(180deg,#0f1317,#13181d);
    box-shadow:inset 0 2px 0 rgba(255,255,255,.04)}
  .grid{display:grid;grid-template-columns:repeat(7,1fr);gap:10px}

  .key{user-select:none;cursor:pointer;border:none;border-radius:12px;padding:12px 10px;font-weight:800;letter-spacing:.2px;
    background:var(--key);color:#0b1220;box-shadow:0 6px 0 var(--key-d), 0 1px 0 rgba(255,255,255,.6) inset;transition:all .08s}
  .func{background:#f8fafc}.blue{background:var(--key-b);box-shadow:0 6px 0 #cfe3ff;color:#0f3a88}
  .orange{background:var(--key-o);box-shadow:0 6px 0 #ffd9a8;color:#7a3b00}
  .red{background:var(--key-r);box-shadow:0 6px 0 #ffc1c1;color:#7f1d1d}
  .key:active{transform:translateY(2px);box-shadow:0 3px 0 rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.5)}
  .small{padding:9px 8px;font-size:12px}
  .equal{grid-column:span 2;background:#e0f2fe;box-shadow:0 6px 0 #bfe0f7}

  .row{display:contents}
  .hint{margin-top:8px;color:#9aa3ad;font-size:12px;text-align:center}
</style>
</head>
<body>
<div class="wrap">
  <div class="device" id="calc">
    <div class="top">
      <div class="brand">MRX-82 <span class="model">PRO</span></div>
      <div style="width:10px;height:10px;border-radius:50%;background:#9ee3a3;box-shadow:0 0 8px #9ee3a3"></div>
    </div>

    <div class="screen">
      <div class="ind">
        <span class="chip" id="chip-angle">DEG</span>
        <span class="chip" id="chip-format">Norm2</span>
        <span class="chip" id="chip-io">MathIO</span>
        <span class="chip" id="chip-mode">COMP</span>
      </div>
      <div class="lcd-in" id="lcd-in"> </div>
      <div class="lcd-out" id="lcd-out">= 0</div>
      <div class="pills">
        <span class="pill" id="pill-shift">SHIFT</span>
        <span class="pill" id="pill-alpha">ALPHA</span>
        <span class="pill" id="pill-hyp">HYP</span>
        <span class="pill" id="pill-m">M</span>
      </div>
    </div>

    <div class="pad"><div class="grid" id="keys">
      <!-- Row 1 -->
      <button class="key orange small" data-k="SHIFT">SHIFT</button>
      <button class="key red small" data-k="ALPHA">ALPHA</button>
      <button class="key small func" data-k="MODE">MODE</button>
      <button class="key small func" data-k="SETUP">SETUP</button>
      <button class="key small func" data-k="DRG">DRG▶</button>
      <button class="key small func" data-k="AC">AC</button>
      <button class="key small func" data-k="DEL">DEL</button>

      <!-- Row 2 -->
      <button class="key func" data-k="(">(</button>
      <button class="key func" data-k=")">)</button>
      <button class="key blue" data-k="SIN">sin</button>
      <button class="key blue" data-k="COS">cos</button>
      <button class="key blue" data-k="TAN">tan</button>
      <button class="key func" data-k="FAC">x!</button>
      <button class="key func" data-k="RND">Rnd</button>

      <!-- Row 3 -->
      <button class="key blue" data-k="SQR">x²</button>
      <button class="key blue" data-k="SQRT">√</button>
      <button class="key blue" data-k="INV">1/x</button>
      <button class="key blue" data-k="LOG">log</button>
      <button class="key blue" data-k="LN">ln</button>
      <button class="key func" data-k="SD">S↔D</button>
      <button class="key func" data-k="ENG">ENG▶</button>

      <!-- Row 4 -->
      <button class="key func" data-k="GCD">GCD</button>
      <button class="key func" data-k="NCR">nCr</button>
      <button class="key func" data-k="STAT">STAT</button>
      <button class="key func" data-k="TABLE">TABLE</button>
      <button class="key func" data-k="BASE">BASE-N</button>
      <button class="key func" data-k="ABS">Abs</button>
      <button class="key func" data-k="ANS">Ans</button>

      <!-- Row 5 -->
      <button class="key" data-k="7">7</button>
      <button class="key" data-k="8">8</button>
      <button class="key" data-k="9">9</button>
      <button class="key func" data-k="DIV">÷</button>
      <button class="key func" data-k="PCT">%</button>
      <button class="key func" data-k="COMMA">,</button>
      <button class="key func" data-k="PI">π</button>

      <!-- Row 6 -->
      <button class="key" data-k="4">4</button>
      <button class="key" data-k="5">5</button>
      <button class="key" data-k="6">6</button>
      <button class="key func" data-k="MUL">×</button>
      <button class="key func" data-k="POW">^</button>
      <button class="key func" data-k="EXP">EXP</button>
      <button class="key func" data-k="MR">MR</button>

      <!-- Row 7 -->
      <button class="key" data-k="1">1</button>
      <button class="key" data-k="2">2</button>
      <button class="key" data-k="3">3</button>
      <button class="key func" data-k="SUB">−</button>
      <button class="key func" data-k="MPLUS">M+</button>
      <button class="key func" data-k="MMINUS">M−</button>
      <button class="key equal" data-k="EQ">=</button>

      <!-- Row 8 -->
      <button class="key" data-k="0">0</button>
      <button class="key" data-k="00">0</button>
      <button class="key" data-k="DOT">.</button>
      <button class="key func" data-k="ADD">+</button>
      <button class="key func" data-k="COPY">COPY</button>
      <button class="key func" data-k="LP"> ( </button>
      <button class="key func" data-k="RP"> ) </button>
    </div></div>

    <div class="hint">Fast, on-device math • SHIFT: inverse trig / 10^x / exp • DRG▶ cycles DEG/RAD/GRA • EXP inserts ×10^( ) • COPY copies result</div>
  </div>
</div>

<script>
(()=>{
// ---------- State ----------
let buf = "";           // input buffer
let ans = 0;            // last answer
let mem = 0;            // memory
let angle = "DEG";      // DEG/RAD/GRA
let format = "Norm2";   // Fix/Sci/Eng/Norm1/Norm2
let fixN = 2, sciN = 6;
let shift = false, alpha = false, hyp = false;
let showFraction = false;

// ---------- Elements ----------
const lcdIn = document.getElementById('lcd-in');
const lcdOut = document.getElementById('lcd-out');
const chipAngle = document.getElementById('chip-angle');
const chipFmt = document.getElementById('chip-format');
const pillShift = document.getElementById('pill-shift');
const pillAlpha = document.getElementById('pill-alpha');
const pillHyp = document.getElementById('pill-hyp');
const pillM = document.getElementById('pill-m');

// ---------- Utilities ----------
const toRad = x => angle==="DEG"? x*Math.PI/180 : (angle==="GRA"? x*Math.PI/200 : x);
const fromRad = x => angle==="DEG"? x*180/Math.PI : (angle==="GRA"? x*200/Math.PI : x);
const clampExp3 = e => Math.trunc(e/3)*3;

function fact(n){ n=Math.floor(n); if(n<0) return NaN; let r=1; for(let i=2;i<=n;i++) r*=i; return r; }
function gcd(a,b){ a=Math.trunc(a); b=Math.trunc(b); a=Math.abs(a); b=Math.abs(b); while(b){const t=b;b=a%b;b=t;} return a; }
function comb(n,r){ n=Math.trunc(n); r=Math.trunc(r); if(r<0||n<0||r>n) return NaN; return fact(n)/(fact(r)*fact(n-r)); }

function toFraction(x,maxDen=100000){
  if(!isFinite(x)) return null;
  let a = Math.floor(x), h1=1, k1=0, h= a, k=1, x1=x, iter=0;
  while(Math.abs(x - h/k) > 1e-12 && k<=maxDen && iter<128){
    x1 = 1/(x1 - a); a = Math.floor(x1);
    const h2=h1; h1=h; h=a*h + h2;
    const k2=k1; k1=k; k=a*k + k2;
    iter++;
  }
  return [h,k];
}

function fmtNumber(x){
  if(!isFinite(x)) return String(x);
  if(showFraction){
    const fr = toFraction(x);
    if(fr){ const [p,q]=fr; return q===1? String(p) : `${p}/${q}`; }
  }
  if(format==="Fix"){
    return x.toFixed(fixN);
  } else if(format==="Sci"){
    return x.toExponential(Math.max(1,sciN));
  } else if(format==="Eng"){
    if(x===0) return "0";
    const e = clampExp3(Math.floor(Math.log10(Math.abs(x))));
    const m = x / Math.pow(10,e);
    return `${m.toPrecision(6)}×10^${e}`;
  } else if(format==="Norm1"){
    const ax=Math.abs(x);
    return (ax!==0 && ax<1e-2)||ax>=1e10 ? x.toExponential(6) : x.toPrecision(10).replace(/\.?0+$/,'');
  } else { // Norm2
    const ax=Math.abs(x);
    return (ax!==0 && ax<1e-9)||ax>=1e10 ? x.toExponential(6) : x.toPrecision(10).replace(/\.?0+$/,'');
  }
}

function sanitize(expr){
  // token replacements
  expr = expr.replace(/×/g,"*").replace(/÷/g,"/").replace(/−/g,"-");
  expr = expr.replace(/\^/g,"**");
  expr = expr.replace(/π/g,"PI");
  expr = expr.replace(/Ans/g, String(ans));
  // factorials: number! or )!
  expr = expr.replace(/(\d+(?:\.\d+)?|\))!/g, (m,p)=>{
    if(p===")") return "fact(#)".replace("#", "__PAREN__"); // handled in loop below
    return `fact(${p})`;
  });
  // simple fix for )!  -> fact(<prev>)
  // Replace '__PAREN__' by capturing preceding parenthesis group via balancing (approx by inserting ) then wrap)
  // (basic heuristic)
  while(expr.includes("fact(__PAREN__)")){
    expr = expr.replace(/fact\(__PAREN__\)/, "fact"); // becomes prefix, will attach by following replace
    expr = expr.replace(/fact\(([^()]*\([^()]*\)[^()]*)\)/, "fact($1)");
  }
  // percent: treat % as /100
  expr = expr.replace(/%/g,"/100");
  return expr;
}

const fns = {
  PI: Math.PI, E: Math.E,
  abs: Math.abs, sqrt: Math.sqrt, log10: (x)=>Math.log10(x), ln: (x)=>Math.log(x), exp: (x)=>Math.exp(x),
  sin: (x)=>Math.sin(toRad(x)), cos:(x)=>Math.cos(toRad(x)), tan:(x)=>Math.tan(toRad(x)),
  asin: (x)=>fromRad(Math.asin(x)), acos:(x)=>fromRad(Math.acos(x)), atan:(x)=>fromRad(Math.atan(x)),
  sinh: (x)=>Math.sinh(x), cosh:(x)=>Math.cosh(x), tanh:(x)=>Math.tanh(x),
  fact, gcd, comb, pow: Math.pow
};

function evalExpr(expr){
  const js = sanitize(expr);
  try{
    // eslint-disable-next-line no-new-func
    const val = Function("f", "with(f){ return ("+js+"); }")(fns);
    return Number(val);
  }catch(e){ return NaN; }
}

function render(){
  lcdIn.textContent = buf || " ";
  const out = fmtNumber(ans);
  lcdOut.textContent = "= " + out;
  pillShift.classList.toggle("on", shift);
  pillAlpha.classList.toggle("on", alpha);
  pillHyp.classList.toggle("on", hyp);
  pillM.classList.toggle("on", Math.abs(mem) > 1e-12);
  chipAngle.textContent = angle;
  chipFmt.textContent = format;
}

function insert(s){ buf += s; render(); }
function back(){ buf = buf.slice(0,-1); render(); }
function clearAll(){ buf = ""; render(); }
function compute(){
  // if empty, compute Ans (noop)
  const expr = buf || String(ans);
  const val = evalExpr(expr);
  if(isFinite(val)){ ans = val; showFraction = false; }
  else { ans = NaN; }
  render();
}

function keyHandler(k){
  switch(k){
    case "SHIFT": shift = !shift; render(); break;
    case "ALPHA": alpha = !alpha; render(); break;
    case "DRG":
      angle = angle==="DEG"?"RAD":(angle==="RAD"?"GRA":"DEG"); render(); break;
    case "AC": clearAll(); break;
    case "DEL": back(); break;

    case "SIN": insert( shift ? "asin(" : (hyp ? "sinh(" : "sin(") ); shift=false; hyp=false; break;
    case "COS": insert( shift ? "acos(" : (hyp ? "cosh(" : "cos(") ); shift=false; hyp=false; break;
    case "TAN": insert( shift ? "atan(" : (hyp ? "tanh(" : "tan(") ); shift=false; hyp=false; break;
    case "FAC": insert("fact("); break;
    case "RND": insert("Math.random()"); break;
    case "SQR": insert("**2"); break;
    case "SQRT": insert("sqrt("); break;
    case "INV": insert("1/("); break;
    case "LOG": insert( shift ? "10**(" : "log10(" ); shift=false; break;
    case "LN":  insert( shift ? "exp(" : "ln(" ); shift=false; break;
    case "ABS": insert("abs("); break;
    case "NCR": insert("comb("); break;

    case "ENG": format="Eng"; render(); break;
    case "SD":  showFraction = !showFraction; render(); break;

    case "STAT": /* stub: open STAT in host app */ alert("STAT (visual)"); break;
    case "TABLE": alert("TABLE (visual)"); break;
    case "BASE": alert("BASE-N (visual)"); break;

    case "ANS": insert("Ans"); break;

    case "DIV": insert("÷"); break;
    case "MUL": insert("×"); break;
    case "SUB": insert("−"); break;
    case "ADD": insert("+"); break;
    case "POW": insert("^"); break;
    case "PCT": insert("%"); break;
    case "EXP": insert("×10^("); break;

    case "PI": insert("π"); break;
    case "COMMA": insert(","); break;
    case "LP": insert("("); break;
    case "RP": insert(")"); break;

    case "MR": insert(String(mem)); break;
    case "MPLUS": mem += ans; render(); break;
    case "MMINUS": mem -= ans; render(); break;

    case "MODE": /* simple cycle of formats */ 
      format = (format==="Norm2")?"Norm1":(format==="Norm1")?"Fix":(format==="Fix")?"Sci":(format==="Sci")?"Eng":"Norm2";
      render(); break;
    case "SETUP": /* toggle MathIO label only */ 
      document.getElementById('chip-io').textContent =
        document.getElementById('chip-io').textContent==="MathIO" ? "LineIO" : "MathIO";
      break;

    case "EQ": compute(); break;
    case "COPY":
      navigator.clipboard?.writeText(String(ans)).then(()=>{}); break;

    default:
      if(/^\d$/.test(k)) insert(k);
      else if(k==="00") insert("0");
      else if(k==="DOT") insert(".");
      else if(k==="(" || k===")") insert(k);
  }
}

document.getElementById('keys').addEventListener('click', e=>{
  const b = e.target.closest('button[data-k]'); if(!b) return;
  keyHandler(b.dataset.k);
});

// keyboard support
document.addEventListener('keydown', e=>{
  const m = {
    "+":"ADD","-":"SUB","*":"MUL","/":"DIV","^":"POW","%":"PCT","(":"LP",")":"RP",".":"DOT",",":"COMMA",
    "Enter":"EQ","=":"EQ","Backspace":"DEL","Delete":"AC"
  };
  if(/\d/.test(e.key)) keyHandler(e.key);
  else if(m[e.key]) keyHandler(m[e.key]);
});

render();
})();
</script>
</body>
</html>
''', height=900, scrolling=False)

# =========================
# TAB 7 — My Project Manager (Overview + Create/Manage + FMECA Hub)
# =========================
with tab7:
    # =========================================================
    # ============  TAB 7 — My Project Manager  ===============
    # Structure:
    #   📊 Overview (view-only)
    #   🛠 Management:
    #       📋 Project Details (Add or Edit)
    #       ✅ Tasks & Milestones
    #       📊 Analytics
    #       🔗 Linked Items
    #   🔍 FMECA Hub
    # =========================================================
    import os, json, re
    from datetime import datetime, timedelta, date
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    # ----------------------- Storage -------------------------
    if 'DATA_DIR' not in globals():
        DATA_DIR = "data"
    os.makedirs(DATA_DIR, exist_ok=True)
    PROJECTS_JSON = os.path.join(DATA_DIR, "projects.json")
    TASKS_JSON    = os.path.join(DATA_DIR, "tasks.json")
    FMECAS_JSON   = os.path.join(DATA_DIR, "fmecas.json")
    CONTACTS_JSON = os.path.join(DATA_DIR, "contacts.json")
    ATTACH_DIR    = os.path.join(DATA_DIR, "attachments")
    os.makedirs(ATTACH_DIR, exist_ok=True)

    # ----------------------- Helpers -------------------------
    def _read_json(path, fallback):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    return d if isinstance(d, list) else fallback
        except Exception:
            pass
        return fallback

    def _write_json(path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Save failed for {os.path.basename(path)}: {e}")

    def _load_contacts():
        return _read_json(CONTACTS_JSON, [])

    def _save_contacts(contacts):
        # Dedup by email (case-insensitive)
        seen = set()
        out  = []
        for c in contacts:
            email = str(c.get("email","")).strip().lower()
            if not email or email in seen:
                continue
            seen.add(email)
            out.append({
                "name": c.get("name","").strip(),
                "email": email,
                "role": c.get("role","").strip()
            })
        _write_json(CONTACTS_JSON, out)
        return out

    def _upsert_contacts(new_list):
        existing = _load_contacts()
        by_email = {str(c.get("email","")).lower(): c for c in existing}
        for n in new_list:
            em = str(n.get("email","")).lower().strip()
            if not em:
                continue
            by_email[em] = {
                "name": n.get("name","").strip(),
                "email": em,
                "role": n.get("role","").strip()
            }
        return _save_contacts(list(by_email.values()))

    def _rerun():
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    def _next_id(rows, key):
        try:
            return (max(int(r.get(key, 0) or 0) for r in rows) + 1) if rows else 1
        except Exception:
            return 1

    def _next_project_number(existing_numbers):
        prefix = "PRJ-"
        nums = []
        for s in existing_numbers:
            try:
                if isinstance(s, str) and s.startswith(prefix):
                    n = int(s.replace(prefix, ""))
                    nums.append(n)
            except Exception:
                continue
        nxt = (max(nums) + 1) if nums else 1
        return f"{prefix}{nxt:04d}"

    def _today():
        return pd.Timestamp(datetime.now().strftime("%Y-%m-%d"))

    def _pdate(s):
        if not s: return None
        try:
            return pd.to_datetime(str(s), errors="coerce")
        except Exception:
            return None

    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def _normalize_list(x):
        """Return list[str] from any shape."""
        if x is None or (isinstance(x, float) and pd.isna(x)): return []
        if isinstance(x, list):
            return [str(i).strip() for i in x if str(i).strip()]
        if isinstance(x, str):
            try:
                parsed = json.loads(x)
            except Exception:
                parts = [p.strip() for p in re.split(r"[,\n]+", x) if p.strip()]
                return parts
            else:
                return _normalize_list(parsed)
        s = str(x).strip()
        return [s] if s else []

    def _normalize_int_list(x):
        out=[]
        for v in _normalize_list(x):
            if str(v).isdigit():
                out.append(int(str(v)))
        return out

    def _normalize_comments(x):
        if x is None or (isinstance(x, float) and pd.isna(x)): return []
        if isinstance(x, str):
            try: parsed=json.loads(x)
            except Exception: return [{"user":"User","timestamp":"","text":x}]
            else: return _normalize_comments(parsed)
        if isinstance(x, dict):
            return [{"user":x.get("user","User"),"timestamp":x.get("timestamp",""),"text":x.get("text","")}]
        if isinstance(x, list):
            out=[]
            for it in x:
                if isinstance(it, dict):
                    out.append({"user":it.get("user","User"),"timestamp":it.get("timestamp",""),"text":it.get("text","")})
                else:
                    out.append({"user":"User","timestamp":"","text":str(it)})
            return out
        return []

    def _safe_select_index(options, val, default_label=None, default_index=0):
        if default_label is not None:
            try: default_index = options.index(default_label)
            except Exception: default_index = 0
        s=None
        if isinstance(val,str): s=val.strip()
        elif val is None or (isinstance(val,float) and pd.isna(val)): s=None
        else:
            try: s=str(val).strip()
            except Exception: s=None
        return options.index(s) if s in options else default_index

    def _task_status_color(s: str) -> str:
        pal={"To Do":"#94A3B8","In Progress":"#22C55E","Blocked":"#F59E0B","Review":"#8B5CF6","Done":"#10B981"}
        return pal.get(str(s or "").strip(), "#94A3B8")

    def _priority_color(p: str) -> str:
        pal={"Low":"#9CA3AF","Medium":"#0EA5E9","High":"#F59E0B"}
        return pal.get(str(p or "Medium"), "#0EA5E9")

    def _compute_project_progress(tasks_df: pd.DataFrame) -> float:
        if tasks_df is None or tasks_df.empty or "status" not in tasks_df.columns: return 0.0
        done = (tasks_df["status"].fillna("") == "Done").sum()
        total = len(tasks_df)
        return (done/total*100) if total>0 else 0.0

    def _compute_burndown(tasks_df: pd.DataFrame) -> pd.DataFrame:
        if tasks_df is None or tasks_df.empty: return pd.DataFrame()
        cal = tasks_df.copy()
        if "completed_at" not in cal.columns or "status" not in cal.columns: return pd.DataFrame()
        cal["completed_at"] = pd.to_datetime(cal["completed_at"], errors="coerce")
        cal = cal[cal["completed_at"].notna() & (cal["status"]=="Done")]
        if cal.empty: return pd.DataFrame()
        cnt = cal.groupby(cal["completed_at"].dt.date).size().cumsum().reset_index(name="cum_done")
        cnt["date"] = pd.to_datetime(cnt["completed_at"])
        total_tasks = len(tasks_df)
        cnt["remaining"] = total_tasks - cnt["cum_done"]
        return cnt[["date","remaining"]]

    def _date_to_str(d: date|None) -> str:
        if d is None: return ""
        try: return d.isoformat()
        except Exception: return str(d)

    # ------------------------ Load ----------------------------
    projects = _read_json(PROJECTS_JSON, [])
    tasks    = _read_json(TASKS_JSON, [])
    fmecas   = _read_json(FMECAS_JSON, [])
    contacts = _load_contacts()
    assets_map = st.session_state.get("assets", {}) or {}

    proj_df = pd.DataFrame(projects)
    task_df = pd.DataFrame(tasks)
    fmeca_df = pd.DataFrame(fmecas)

    # Ensure schema columns
    proj_cols = ["project_id","project_number","name","description","status","priority",
                 "start_date","end_date","linked_assets","linked_fmecas","team", # team = list of dicts {name,email,role}
                 "attachments","comments","created_at","updated_at"]
    for c in proj_cols:
        if c not in proj_df.columns: proj_df[c]=pd.Series(dtype="object")

    task_cols = ["task_id","project_id","parent_task_id","name","status","priority",
                 "assignees","due_date","dependencies","estimate_h","actual_h",
                 "description","milestones","comments","links","attachments","completed_at","assigned_to"]
    # note: assigned_to kept for backward-compat; we'll maintain 'assignees' going forward
    for c in task_cols:
        if c not in task_df.columns: task_df[c]=pd.Series(dtype="object")

    fm_cols = ["fmeca_id","linked_id","type","created_at","closed_at","failure_modes","version"]
    for c in fm_cols:
        if c not in fmeca_df.columns: fmeca_df[c]=pd.Series(dtype="object")

    # =================== Top Tabs =============================
    sub_overview, sub_manage, sub_fmeca = st.tabs([
        "📊 Overview (Dashboard - View Only)",
        "🛠 Management",
        "🔍 FMECA Hub (Risk Management Center)"
    ])

    # =================== 📊 OVERVIEW ==========================
    with sub_overview:
        st.subheader("Portfolio Overview")
        # KPIs
        total_projects = len(proj_df) if not proj_df.empty else 0
        active = len(proj_df[proj_df["status"].isin(["Planning","Execution"])]) if not proj_df.empty else 0
        done_count = len(proj_df[proj_df["status"]=="Done"]) if not proj_df.empty else 0
        total_tasks = len(task_df) if not task_df.empty else 0
        overdue = 0
        if not task_df.empty and "due_date" in task_df.columns and "status" in task_df.columns:
            due_ = pd.to_datetime(task_df["due_date"], errors="coerce")
            overdue = int(((due_ < _today()) & (task_df["status"] != "Done")).sum())

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Projects", total_projects)
        c2.metric("Active", active)
        c3.metric("Completed", done_count)
        c4.metric("Overdue Tasks", overdue)

        st.markdown("---")

        # Projects by Status (donut)
        if not proj_df.empty:
            counts = proj_df["status"].fillna("Unknown").value_counts().reset_index()
            counts.columns=["Status","Count"]
            fig_s = px.pie(counts, names="Status", values="Count", hole=0.55, title="Projects by Status")
            st.plotly_chart(fig_s, use_container_width=True)

        # Portfolio Timeline (Gantt-style)
        if not proj_df.empty:
            tl = []
            for _, r in proj_df.iterrows():
                s = r.get("start_date") or ""
                e = r.get("end_date") or ""
                if s or e:
                    s = s or str(_today().date())
                    e = e or (pd.to_datetime(s) + timedelta(days=14)).strftime("%Y-%m-%d")
                    tl.append({"Project": f"{r.get('project_number','PRJ-????')} — {r.get('name','(no name)')}",
                               "Start": s, "Finish": e, "Status": r.get("status","Planning")})
            if tl:
                fig_t = px.timeline(pd.DataFrame(tl), x_start="Start", x_end="Finish", y="Project", color="Status")
                fig_t.update_layout(height=380, title="Portfolio Timeline")
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.info("No valid dates to plot.")

        # Simple FMECA Tracker (Created vs Closed per month)
        if not fmeca_df.empty:
            df = fmeca_df.copy()
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
            df["closed_at"]  = pd.to_datetime(df.get("closed_at"), errors="coerce")
            created = df.dropna(subset=["created_at"]).groupby(df["created_at"].dt.to_period("M")).size().rename("Created")
            closed  = df.dropna(subset=["closed_at"]).groupby(df["closed_at"].dt.to_period("M")).size().rename("Closed")
            combo = pd.concat([created, closed], axis=1).fillna(0).reset_index()
            combo["index"] = combo["index"].astype(str)
            fig_f = go.Figure()
            fig_f.add_bar(x=combo["index"], y=combo["Created"], name="Created")
            fig_f.add_bar(x=combo["index"], y=combo["Closed"], name="Closed")
            fig_f.update_layout(barmode="group", title="FMECA Tracker (Monthly)")
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.info("No FMECAs yet.")

    # =================== 🛠 MANAGEMENT ========================
    with sub_manage:
        st.subheader("Project Management")

        # Remember mode & selected project
        if "t7_mgmt_mode" not in st.session_state:
            st.session_state["t7_mgmt_mode"] = "Add Project"
        if "t7_selected_pid" not in st.session_state:
            st.session_state["t7_selected_pid"] = None

        # Nested management tabs
        d_tab, t_tab, a_tab, l_tab = st.tabs([
            "📋 Project Details (Add or Edit)",
            "✅ Tasks & Milestones (Planning & Execution)",
            "📊 Analytics (Progress Insights)",
            "🔗 Linked Items (Connections)"
        ])

        # ---------- Utilities for Team UI ----------
        def _team_df_key_add(): return "t7_team_add_df"
        def _team_df_key_edit(pid): return f"t7_team_edit_df_{pid}"

        def _ensure_team_df(key, seed=None):
            if key not in st.session_state:
                if isinstance(seed, list):
                    df = pd.DataFrame(seed)
                else:
                    df = pd.DataFrame(columns=["name","email","role","save_to_contacts"])
                if "save_to_contacts" not in df.columns:
                    df["save_to_contacts"] = True
                st.session_state[key] = df

        def _collect_team_from_df(df: pd.DataFrame):
            if df is None or df.empty:
                return []
            out=[]
            for _, r in df.iterrows():
                name = str(r.get("name","")).strip()
                email= str(r.get("email","")).strip()
                role = str(r.get("role","")).strip()
                if email:
                    out.append({"name":name, "email":email, "role":role})
            return out

        def _import_contacts_to_df(df: pd.DataFrame, selected_emails: list[str]):
            emap = {c["email"].lower(): c for c in contacts}
            existing_emails = set([str(e).lower() for e in df.get("email", []).tolist()]) if "email" in df.columns else set()
            rows=[]
            for em in selected_emails:
                c = emap.get(str(em).lower())
                if c and c["email"].lower() not in existing_emails:
                    rows.append({"name":c.get("name",""),"email":c.get("email",""),"role":c.get("role",""),"save_to_contacts":False})
            if rows:
                df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
            return df

# =================== 📋 Project Details ===================
from datetime import date

with d_tab:
    # -------- Safe defaults for shared state --------
    if 'contacts' not in globals() or contacts is None:
        contacts = []
    if 'assets_map' not in globals():
        assets_map = st.session_state.get("assets", {}) or {}

    # -------- Small helpers (local, no side effects) --------
    def _ensure_team_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the 4 required columns exist and in the right order; keep existing values."""
        base = pd.DataFrame(columns=["name","email","role","save_to_contacts"])
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            out = base.copy()
            out.loc[0] = {"name":"", "email":"", "role":"", "save_to_contacts": True}
            return out
        out = df.copy()
        if "name" not in out.columns: out["name"] = ""
        if "email" not in out.columns: out["email"] = ""
        if "role" not in out.columns: out["role"] = ""
        if "save_to_contacts" not in out.columns: out["save_to_contacts"] = False
        # reorder
        out = out[["name","email","role","save_to_contacts"]]
        # normalize types
        out["name"]  = out["name"].astype(str)
        out["email"] = out["email"].astype(str)
        out["role"]  = out["role"].astype(str)
        out["save_to_contacts"] = out["save_to_contacts"].astype(bool)
        return out

    def _team_rows_from_df(df: pd.DataFrame) -> list:
        """Extract list[dict] from the editor df; keep row if it has any content."""
        df = _ensure_team_cols(df)
        rows = []
        for _, r in df.iterrows():
            nm = (r.get("name")  or "").strip()
            em = (r.get("email") or "").strip()
            rl = (r.get("role")  or "").strip()
            sv = bool(r.get("save_to_contacts", False))
            if any([nm, em, rl]):  # keep non-empty rows
                rows.append({"name": nm, "email": em, "role": rl, "save_to_contacts": sv})
        return rows

    def _dedupe_by_email(rows: list) -> list:
        """Dedupe by email (keep last). Rows without email are kept as-is."""
        seen = {}
        no_email = []
        for r in rows:
            em = (r.get("email") or "").strip()
            if em:
                seen[em] = r
            else:
                no_email.append(r)
        return list(seen.values()) + no_email

    def _contacts_options():
        return sorted({c.get("email","") for c in contacts if c.get("email")})

    def _fmt_contact(em: str) -> str:
        c = next((c for c in contacts if c.get("email")==em), None)
        return f"{c.get('name','(no name)')} <{em}>" if c else em

    def _rows_from_contact_emails(emails: list, save_flag=False) -> list:
        """Map selected contact emails to team-row dicts (save_to_contacts defaults False)."""
        out = []
        for em in emails or []:
            c = next((c for c in contacts if c.get("email")==em), None)
            if c:
                out.append({
                    "name": c.get("name",""),
                    "email": c.get("email",""),
                    "role":  c.get("role",""),
                    "save_to_contacts": bool(save_flag)
                })
        return out

    def _safe_select_index(options, value, default_idx=0):
        try:
            return options.index(value)
        except Exception:
            return default_idx

    # -------- Mode switch (you control it) --------
    mode = st.segmented_control("Mode", ["Add Project", "Edit Project"]) if hasattr(st, "segmented_control") \
        else st.radio("Mode", ["Add Project", "Edit Project"], horizontal=True)
    st.session_state["t7_mgmt_mode"] = mode

    # ============== ADD PROJECT ==============
    if mode == "Add Project":
        # Persist the team editor grid in session so it's available when submitting the form
        if "t7_team_add_df" not in st.session_state:
            st.session_state["t7_team_add_df"] = _ensure_team_cols(pd.DataFrame())

        with st.form("t7_add_project_form", clear_on_submit=False):
            # Project number preview
            pn_preview = _next_project_number([p.get("project_number") for p in projects])
            st.markdown(f"**Project Number:** `{pn_preview}`")

            c1, c2 = st.columns(2)
            with c1:
                p_name  = st.text_input("Project Name *")
                p_start = st.date_input("Start Date", value=date.today())
                st.text_input("Status", value="Planning", disabled=True)  # fixed on add
                p_pri   = st.selectbox("Priority", ["Low","Medium","High"], index=1)
            with c2:
                p_end    = st.date_input("End Date", value=date.today())
                asset_opts = sorted(map(str, (assets_map or {}).keys()))
                p_assets  = st.multiselect("Link Existing Assets (optional)", asset_opts, default=[])

            st.markdown("#### Team Members")
            # Show the identical 4-column table (dynamic rows)
            st.session_state["t7_team_add_df"] = _ensure_team_cols(st.session_state["t7_team_add_df"])
            team_df_in = st.data_editor(
                st.session_state["t7_team_add_df"],
                num_rows="dynamic",
                column_config={
                    "name":  st.column_config.TextColumn("Name"),
                    "email": st.column_config.TextColumn("Email"),
                    "role":  st.column_config.TextColumn("Role"),
                    "save_to_contacts": st.column_config.CheckboxColumn("Save to Contacts"),
                },
                use_container_width=True,
                key="t7_team_editor_add_form"
            )
            # Reflect edits back into session during form lifetime
            st.session_state["t7_team_add_df"] = _ensure_team_cols(team_df_in)

            # Add-from-contacts INSIDE the form; merged on submit
            st.caption("Add from Contacts")
            sel_contacts_add = st.multiselect(
                "Pick contacts to include in team (merged on submit)",
                options=_contacts_options(),
                format_func=_fmt_contact,
                key="t7_sel_contacts_add"
            )

            p_desc = st.text_area("Project Description", height=100)

            cbtn1, cbtn2 = st.columns(2)
            create    = cbtn1.form_submit_button("Create Project", use_container_width=True)
            create_an = cbtn2.form_submit_button("Create & Add Another", use_container_width=True)

            if create or create_an:
                # Validation
                if not p_name.strip():
                    st.error("Project name is required.")
                    st.stop()
                if p_end < p_start:
                    st.error("End Date cannot be earlier than Start Date.")
                    st.stop()

                # Merge team rows: manual grid + selected contacts (dedupe by email)
                grid_rows    = _team_rows_from_df(st.session_state["t7_team_add_df"])
                contact_rows = _rows_from_contact_emails(sel_contacts_add, save_flag=False)  # imported default: not saved to contacts unless user ticks
                merged_rows  = _dedupe_by_email(grid_rows + contact_rows)

                # Persist project
                pid = _next_id(projects, "project_id")
                pn  = pn_preview
                row = {
                    "project_id": pid,
                    "project_number": pn,
                    "name": p_name.strip(),
                    "description": p_desc.strip(),
                    "status": "Planning",
                    "priority": p_pri,
                    "start_date": _date_to_str(p_start),
                    "end_date": _date_to_str(p_end),
                    "linked_assets": list(_normalize_list(p_assets)),
                    "linked_fmecas": [],
                    "team": [{"name": r["name"], "email": r["email"], "role": r["role"]} for r in merged_rows],
                    "attachments": [],
                    "comments": [],
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
                projects.append(row)
                _write_json(PROJECTS_JSON, projects)

                # Save to contacts for any grid rows flagged
                to_save = []
                for r in grid_rows:
                    if r.get("save_to_contacts", False) and (r.get("email","").strip()):
                        to_save.append({"name": r["name"], "email": r["email"], "role": r["role"]})
                if to_save:
                    _upsert_contacts(to_save)

                st.success(f"Project **{pn}** created.")
                if create_an:
                    # Reset the add form state for a fresh entry
                    st.session_state["t7_team_add_df"] = _ensure_team_cols(pd.DataFrame())
                    st.session_state["t7_sel_contacts_add"] = []
                    st.rerun()
                else:
                    st.rerun()

    # ============== EDIT PROJECT ==============
    if mode == "Edit Project":
        if proj_df.empty:
            st.info("No projects yet. Switch to **Add Project** to create one.")
        else:
            options = (
                proj_df.assign(
                    label=lambda df: df.apply(
                        lambda r: f"{int(r['project_id'])} | {r.get('project_number','PRJ-????')} — {r.get('name','(no name)')} [{r.get('status','')}]",
                        axis=1
                    )
                )[["project_id","label"]].sort_values("label")
            )
            pick_lbl = st.selectbox("Select Project", options=options["label"].tolist(), index=0, key="t7_edit_pick")
            try:
                selected_pid = int(str(pick_lbl).split("|")[0].strip())
            except Exception:
                selected_pid = int(options.iloc[0]["project_id"])

            row = proj_df[proj_df["project_id"].astype("string").fillna("-1").astype(int) == selected_pid]
            if row.empty:
                st.warning("Project not found.")
                st.stop()

            r0 = row.iloc[0].to_dict()
            pn = r0.get("project_number", "PRJ-????")
            st.markdown(f"#### 📁 {pn} — {r0.get('name','(no name)')}")
            st.caption(f"Status: {r0.get('status','')} • Priority: {r0.get('priority','Medium')} • "
                       f"Start: {r0.get('start_date','—')} • End: {r0.get('end_date','—')}")

            # Keep a per-project team grid in session and ensure it has the SAME 4 columns
            team_session_key = f"t7_team_edit_df_{selected_pid}"
            if team_session_key not in st.session_state:
                seed_rows = []
                for trow in (r0.get("team", []) or []):
                    seed_rows.append({
                        "name": trow.get("name",""),
                        "email": trow.get("email",""),
                        "role": trow.get("role",""),
                        "save_to_contacts": False
                    })
                st.session_state[team_session_key] = _ensure_team_cols(pd.DataFrame(seed_rows))

            with st.form(f"t7_edit_project_form_{selected_pid}", clear_on_submit=False):
                c1, c2 = st.columns(2)
                with c1:
                    ename  = st.text_input("Name", value=str(r0.get("name","") or ""))
                    estart = st.date_input("Start Date",
                                           value=pd.to_datetime(r0.get("start_date"), errors="coerce").date() if r0.get("start_date") else date.today())
                    estatus = st.selectbox("Status", ["Planning","Execution","On Hold","Done","Cancelled"],
                                           index=_safe_select_index(["Planning","Execution","On Hold","Done","Cancelled"], r0.get("status"), default_idx=0))
                    epri   = st.selectbox("Priority", ["Low","Medium","High"],
                                          index=_safe_select_index(["Low","Medium","High"], r0.get("priority"), default_idx=1))
                with c2:
                    eend = st.date_input("End Date",
                                         value=pd.to_datetime(r0.get("end_date"), errors="coerce").date() if r0.get("end_date") else date.today())
                    asset_opts = sorted(map(str, (assets_map or {}).keys()))
                    eassets = st.multiselect("Linked Assets", options=asset_opts,
                                             default=list(r0.get("linked_assets", []) or []))

                st.markdown("#### Team Members")
                # IDENTICAL 4-COLUMN TABLE IN EDIT MODE
                st.session_state[team_session_key] = _ensure_team_cols(st.session_state[team_session_key])
                team_df_edit = st.data_editor(
                    st.session_state[team_session_key],
                    num_rows="dynamic",
                    column_config={
                        "name":  st.column_config.TextColumn("Name"),
                        "email": st.column_config.TextColumn("Email"),
                        "role":  st.column_config.TextColumn("Role"),
                        "save_to_contacts": st.column_config.CheckboxColumn("Save to Contacts"),
                    },
                    use_container_width=True,
                    key=f"t7_team_editor_edit_form_{selected_pid}"
                )
                # Reflect the latest editor state into session during the form
                st.session_state[team_session_key] = _ensure_team_cols(team_df_edit)

                # Add-from-contacts INSIDE the form; merged on save
                st.caption("Add from Contacts")
                sel_contacts_edit = st.multiselect(
                    "Pick contacts to include in team (merged on save)",
                    options=_contacts_options(),
                    format_func=_fmt_contact,
                    key=f"t7_sel_contacts_edit_{selected_pid}"
                )

                edesc = st.text_area("Description", value=str(r0.get("description","") or ""), height=100)

                save_changes = st.form_submit_button("Save Project Changes", use_container_width=True)

                if save_changes:
                    # validation
                    if not ename.strip():
                        st.error("Name is required.")
                        st.stop()
                    if eend < estart:
                        st.error("End Date cannot be earlier than Start Date.")
                        st.stop()

                    # Merge (team grid + selected contacts) and dedupe by email
                    grid_rows    = _team_rows_from_df(st.session_state[team_session_key])
                    contact_rows = _rows_from_contact_emails(sel_contacts_edit, save_flag=False)
                    merged_rows  = _dedupe_by_email(grid_rows + contact_rows)

                    # Persist edits
                    for p in projects:
                        if int(p.get("project_id",-1)) == selected_pid:
                            p["name"]          = ename.strip()
                            p["start_date"]    = _date_to_str(estart)
                            p["status"]        = estatus
                            p["priority"]      = epri
                            p["end_date"]      = _date_to_str(eend)
                            p["linked_assets"] = list(eassets)
                            p["team"]          = [{"name": r["name"], "email": r["email"], "role": r["role"]} for r in merged_rows]
                            p["description"]   = edesc.strip()
                            p["updated_at"]    = datetime.now().isoformat(timespec="seconds")
                            break
                    _write_json(PROJECTS_JSON, projects)

                    # Upsert contacts for rows flagged in the editor grid
                    to_save = []
                    for r in grid_rows:
                        if r.get("save_to_contacts", False) and (r.get("email","").strip()):
                            to_save.append({"name": r["name"], "email": r["email"], "role": r["role"]})
                    if to_save:
                        _upsert_contacts(to_save)

                    st.success("Project updated.")
                    st.rerun()

            # ---- Danger Zone (outside form) ----
            with st.expander("Danger Zone"):
                confirm = st.checkbox(
                    "I understand this will permanently delete the project and its tasks.",
                    key=f"t7_del_confirm_{selected_pid}"
                )
                if st.button("Delete Project", type="primary", disabled=not confirm, key=f"t7_delete_btn_{selected_pid}"):
                    # Delete project
                    projects[:] = [p for p in projects if int(p.get("project_id",-1)) != selected_pid]
                    _write_json(PROJECTS_JSON, projects)
                    # Delete tasks for this project
                    if isinstance(tasks, list):
                        tasks[:] = [t for t in tasks if int(t.get("project_id",-1)) != selected_pid]
                        _write_json(TASKS_JSON, tasks)
                    st.success("Project and its tasks deleted.")
                    st.rerun()


# =================== ✅ Tasks & Milestones ===================
with t_tab:
    import plotly.express as px  # ensure px is available
    import plotly.graph_objects as go

    # ---------- sanity: make sure 'tasks' exists ----------
    if "tasks" not in globals() or not isinstance(tasks, list):
        tasks = []

    # ---------- local helpers (no dependency on _safe_select_index) ----------
    def _safe_int(v, default=0):
        try:
            if v is None:
                return default
            if isinstance(v, (int,)):
                return v
            if isinstance(v, float):
                if pd.isna(v):
                    return default
                return int(v)
            s = str(v).strip()
            if s == "" or s.lower() == "none" or s.lower() == "nan":
                return default
            return int(float(s)) if any(c in s for c in [".","e","E"]) else int(s)
        except Exception:
            return default

    def _idx(options, value, fallback=0):
        try:
            return options.index(value)
        except Exception:
            return fallback

    def _people_union_for_project(proj_row):
        """Return unified people list (team + contacts), and fast maps."""
        team = proj_row.get("team", []) or []
        norm_team = []
        for t in team:
            norm_team.append({
                "name":  str(t.get("name","")).strip(),
                "email": str(t.get("email","")).strip(),
                "role":  str(t.get("role","")).strip(),
            })
        norm_contacts = []
        for c in (contacts or []):
            norm_contacts.append({
                "name":  str(c.get("name","")).strip(),
                "email": str(c.get("email","")).strip(),
                "role":  str(c.get("role","")).strip(),
            })

        labels = []
        label_to_email = {}
        seen_names = {}
        for person in norm_team + norm_contacts:
            nm = person["name"] or person["email"] or "(unknown)"
            em = person["email"]
            if not em:
                continue
            if nm in seen_names:
                lbl = f"{nm} <{em}>"
            else:
                lbl = nm
                seen_names[nm] = True
            if lbl not in labels:
                labels.append(lbl)
                label_to_email[lbl] = em

        email_to_label = {}
        for lbl, em in label_to_email.items():
            nm = lbl.split(" <")[0]
            email_to_label.setdefault(em, nm if nm in seen_names else lbl)

        return (norm_team, norm_contacts, labels, label_to_email, email_to_label)

    def _labels_from_emails(emails, email_to_label):
        out = []
        for em in _normalize_list(emails):
            lbl = email_to_label.get(str(em).strip())
            if lbl:
                out.append(lbl)
        return sorted(set(out))

    def _emails_from_labels(labels, label_to_email):
        out = []
        for lbl in _normalize_list(labels):
            em = label_to_email.get(str(lbl).strip())
            if em:
                out.append(em)
        return sorted(set(out))

    def _task_ids_for_project(pid):
        return sorted(set(_safe_int(t.get("task_id"), 0) for t in tasks if _safe_int(t.get("project_id"), -1) == _safe_int(pid, -1)))

    def _next_task_id_for_project(pid):
        """Return the smallest positive integer not used in this project."""
        used = _task_ids_for_project(pid)
        n = 1
        for u in used:
            if u == n:
                n += 1
            elif u > n:
                break
        return n

    def _renumber_tasks_for_project(pid):
        """
        Renumber this project's task_id to 1..N in a stable order:
        by start_date asc, then name asc, then old id.
        Also updates dependencies inside the project with the new mapping.
        """
        pid = _safe_int(pid, -1)
        proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"), -1) == pid]
        if not proj_tasks:
            return

        def _key(t):
            sd = _pdate(t.get("start_date"))
            return (
                sd if pd.notna(sd) else pd.Timestamp.max,
                str(t.get("name","")).lower(),
                _safe_int(t.get("task_id"), 1_000_000)
            )

        proj_tasks_sorted = sorted(proj_tasks, key=_key)

        mapping = {}
        for i, t in enumerate(proj_tasks_sorted, start=1):
            old = _safe_int(t.get("task_id"), i)
            mapping[old] = i

        for t in tasks:
            if _safe_int(t.get("project_id"), -1) != pid:
                continue
            old_id = _safe_int(t.get("task_id"), 0)
            if old_id in mapping:
                t["task_id"] = mapping[old_id]
            # fix dependencies
            deps = t.get("dependencies", [])
            if isinstance(deps, list):
                new_deps = []
                for d in deps:
                    d_int = _safe_int(d, None)
                    if d_int is not None and d_int in mapping:
                        new_deps.append(mapping[d_int])
                t["dependencies"] = sorted(set(new_deps))

    def _ensure_task_ids_for_project(pid):
        """Self-heal: assign missing/None/invalid task_id values, then renumber 1..N."""
        pid = _safe_int(pid, -1)
        changed = False
        used = set(_task_ids_for_project(pid))
        next_id = 1
        while next_id in used:
            next_id += 1
        for t in tasks:
            if _safe_int(t.get("project_id"), -1) != pid:
                continue
            tid = t.get("task_id")
            if _safe_int(tid, 0) <= 0:
                t["task_id"] = next_id
                changed = True
                used.add(next_id)
                next_id += 1
                while next_id in used:
                    next_id += 1
        if changed:
            _renumber_tasks_for_project(pid)
            try:
                _write_json(TASKS_JSON, tasks)
            except Exception:
                pass

    def _project_bounds(pid):
        row = proj_df[proj_df["project_id"].astype("string").fillna("-1").astype(int)==int(pid)]
        if row.empty:
            s = _today()
            e = s + pd.Timedelta(days=14)
            return s, e
        r0 = row.iloc[0]
        s = pd.to_datetime(r0.get("start_date"), errors="coerce")
        e = pd.to_datetime(r0.get("end_date"), errors="coerce")
        if pd.isna(s): s = _today()
        if pd.isna(e): e = s + pd.Timedelta(days=14)
        return s, e

    def _clamp_to_project(pid, ts):
        """Clamp timestamp ts into the project's start..end window."""
        if ts is None or pd.isna(ts):
            return ts
        s, e = _project_bounds(pid)
        ts = pd.to_datetime(ts, errors="coerce")
        if pd.isna(ts): return ts
        if ts < s: return s
        if ts > e: return e
        return ts

    # ---------- Select Project ----------
    if proj_df.empty:
        st.info("No projects available. Create one in Project Details.")
        st.stop()

    proj_rows = proj_df.copy().reset_index(drop=True)
    proj_rows["__ord"] = proj_rows.index + 1
    proj_rows["__label"] = proj_rows.apply(
        lambda r: f"{int(r['__ord'])} — {r.get('project_number','PRJ-????')} — {r.get('name','(no name)')} [{r.get('status','')}]",
        axis=1
    )
    pick_lbl = st.selectbox("Choose Project", options=proj_rows["__label"].tolist(), index=0, key="t7_tasks_pick")
    pid_t = int(proj_rows.loc[proj_rows["__label"]==pick_lbl, "project_id"].iloc[0])

    # ---- self-heal any missing/None task IDs for this project BEFORE any view logic ----
    _ensure_task_ids_for_project(pid_t)

    # Build selected tasks df for the project (optional table view)
    sel_tasks = pd.DataFrame(columns=task_df.columns)
    if not task_df.empty:
        mask = task_df["project_id"].astype("string").fillna("-1").astype(int) == pid_t
        sel_tasks = task_df[mask].copy()

    # People sources and maps
    rproj = proj_df[proj_df["project_id"].astype("string").fillna("-1").astype(int)==pid_t]
    team, book, assignee_options, label_to_email, email_to_label = _people_union_for_project(rproj.iloc[0] if not rproj.empty else {})

    # View switcher
    view_key = f"t7_view_{pid_t}"
    if view_key not in st.session_state:
        st.session_state[view_key] = "List"
    view_choices = ["List","Board","Calendar","Timeline"]
    vmode = st.segmented_control("View", view_choices, key=view_key) if hasattr(st, "segmented_control") else st.radio("View", view_choices, key=view_key, horizontal=True)

    # Only the List view has Add/Edit modes
    mode_key = f"t7_task_mode_{pid_t}"
    if vmode == "List":
        if mode_key not in st.session_state:
            st.session_state[mode_key] = "Add Task"
        mode = st.segmented_control("Mode", ["Add Task","Edit Task"], key=mode_key) if hasattr(st, "segmented_control") else st.radio("Mode", ["Add Task","Edit Task"], key=mode_key, horizontal=True)
    else:
        mode = None

    # ---------- LIST VIEW ----------
    if vmode == "List":

        # ======== Add Task mode ========
        if mode == "Add Task":
            st.markdown("#### Add Task")

            c1, c2, c3, c4, c5 = st.columns([1.6, 1.0, 1.0, 1.0, 1.4])
            q_name  = c1.text_input("Task Name", key=f"t7_q_name_{pid_t}")
            q_start = c2.date_input("Start Date", value=_project_bounds(pid_t)[0].date(), key=f"t7_q_start_{pid_t}")
            q_due   = c3.date_input("End Date",   value=_project_bounds(pid_t)[0].date(), key=f"t7_q_due_{pid_t}")
            q_pri   = c4.selectbox("Priority", ["Low","Medium","High"], index=1, key=f"t7_q_pri_{pid_t}")
            q_asg   = c5.multiselect("Assignees (names)", options=assignee_options,
                                     default=assignee_options[:1] if assignee_options else [],
                                     key=f"t7_q_asg_{pid_t}")

            if st.button("Add Task", key=f"t7_q_add_{pid_t}"):
                if not q_name.strip():
                    st.warning("Task name required.")
                else:
                    start_c = _clamp_to_project(pid_t, pd.to_datetime(q_start))
                    due_c   = _clamp_to_project(pid_t, pd.to_datetime(q_due))
                    if pd.notna(start_c) and pd.notna(due_c) and due_c < start_c:
                        due_c = start_c
                    asg_emails = _emails_from_labels(q_asg, label_to_email)
                    tid = _next_task_id_for_project(pid_t)
                    tasks.append({
                        "task_id":tid, "project_id":int(pid_t), "parent_task_id":None,
                        "name":q_name.strip(), "status":"To Do", "priority":q_pri,
                        "assignees": list(asg_emails),
                        "start_date": _date_to_str(start_c.date() if pd.notna(start_c) else _today().date()),
                        "due_date":   _date_to_str(due_c.date()   if pd.notna(due_c)   else _today().date()),
                        "dependencies": [], "estimate_h": 0.0, "actual_h": 0.0,
                        "description":"", "milestones": [], "comments": [], "links": [], "attachments": [], "completed_at":""
                    })
                    _renumber_tasks_for_project(pid_t)
                    _write_json(TASKS_JSON, tasks)
                    st.success("Task added.")
                    _rerun()

            st.markdown("---")

            # Tasks table (inline edits) + Save
            proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"), -1)==pid_t]
            if not proj_tasks:
                st.info("No tasks yet.")
            else:
                view = pd.DataFrame(proj_tasks)
                for c in ["task_id","name","status","priority","assignees","start_date","due_date","estimate_h","actual_h","dependencies"]:
                    if c not in view.columns:
                        view[c] = "" if c not in ("assignees","dependencies") else []
                view["assignees_csv"] = view["assignees"].apply(lambda v: ", ".join(_labels_from_emails(v, email_to_label)))
                view["dependencies_csv"] = view["dependencies"].apply(lambda v: ", ".join(str(_safe_int(x,0)) for x in (v or []) if str(x).strip().isdigit()))
                view["estimate_h"] = view["estimate_h"].apply(_safe_float)
                view["actual_h"]   = view["actual_h"].apply(_safe_float)

                grid = view[["task_id","name","status","priority","assignees_csv","start_date","due_date","estimate_h","actual_h","dependencies_csv"]].copy()
                edited = st.data_editor(
                    grid,
                    column_config={
                        "status": st.column_config.SelectboxColumn("Status", options=["To Do","In Progress","Blocked","Review","Done"]),
                        "priority": st.column_config.SelectboxColumn("Priority", options=["Low","Medium","High"]),
                        "assignees_csv": st.column_config.TextColumn("Assignees (names, comma)"),
                        "start_date": st.column_config.TextColumn("Start (YYYY-MM-DD)"),
                        "due_date":   st.column_config.TextColumn("End (YYYY-MM-DD)"),
                        "estimate_h": st.column_config.NumberColumn("Est H", min_value=0.0),
                        "actual_h":   st.column_config.NumberColumn("Act H", min_value=0.0),
                        "dependencies_csv": st.column_config.TextColumn("Deps (IDs comma)"),
                    },
                    hide_index=True, use_container_width=True,
                    key=f"t7_grid_add_{pid_t}"
                )

                if st.button("💾 Save Changes", key=f"t7_save_grid_add_{pid_t}"):
                    changed = 0
                    for _, r in edited.iterrows():
                        tid = _safe_int(r["task_id"], 0)
                        if tid <= 0:
                            continue
                        for t in tasks:
                            if _safe_int(t.get("project_id"),-1)==pid_t and _safe_int(t.get("task_id"),-1)==tid:
                                t["name"] = str(r["name"])
                                t["status"] = str(r["status"])
                                t["priority"] = str(r["priority"])
                                t["assignees"] = _emails_from_labels([x.strip() for x in _normalize_list(r["assignees_csv"])], label_to_email)
                                sd = _clamp_to_project(pid_t, _pdate(str(r["start_date"])))
                                dd = _clamp_to_project(pid_t, _pdate(str(r["due_date"])))
                                if pd.notna(sd) and pd.notna(dd) and dd < sd:
                                    dd = sd
                                t["start_date"] = _date_to_str(sd.date() if pd.notna(sd) else "")
                                t["due_date"]   = _date_to_str(dd.date() if pd.notna(dd) else "")
                                t["estimate_h"] = _safe_float(r["estimate_h"])
                                t["actual_h"]   = _safe_float(r["actual_h"])
                                t["dependencies"] = [_safe_int(x, None) for x in _normalize_list(r["dependencies_csv"]) if str(x).strip().isdigit()]
                                t["dependencies"] = sorted(set([d for d in t["dependencies"] if d is not None]))
                                if t["status"]=="Done" and not t.get("completed_at"):
                                    t["completed_at"]= datetime.now().strftime("%Y-%m-%d")
                                changed += 1
                                break
                    if changed:
                        _renumber_tasks_for_project(pid_t)
                        _write_json(TASKS_JSON, tasks)
                        st.success("Updates saved.")
                        _rerun()
                    else:
                        st.info("No changes to save.")

                with st.expander("➕ Add Person (for assignees / owners)"):
                    p1, p2, p3, p4 = st.columns([1.3,1.3,1.0,1.0])
                    ap_name = p1.text_input("Name", key=f"t7_addp_name_{pid_t}")
                    ap_email = p2.text_input("Email", key=f"t7_addp_email_{pid_t}")
                    ap_role = p3.text_input("Role", key=f"t7_addp_role_{pid_t}", placeholder="e.g., Engineer")
                    ap_save = p4.checkbox("Save to Contacts", value=True, key=f"t7_addp_save_{pid_t}")
                    if st.button("Add Person", key=f"t7_addp_btn_{pid_t}"):
                        if not ap_name.strip() or not ap_email.strip():
                            st.warning("Name and Email required.")
                        else:
                            for p in projects:
                                if _safe_int(p.get("project_id"),-1)==pid_t:
                                    tlist = p.get("team", []) or []
                                    tlist.append({"name":ap_name.strip(),"email":ap_email.strip(),"role":ap_role.strip()})
                                    p["team"] = tlist
                                    p["updated_at"] = datetime.now().isoformat(timespec="seconds")
                                    break
                            _write_json(PROJECTS_JSON, projects)
                            if ap_save:
                                _upsert_contacts([{"name":ap_name.strip(),"email":ap_email.strip(),"role":ap_role.strip()}])
                            st.success("Person added.")
                            _rerun()

            st.markdown("---")

            # ===== Add Task Milestones (optional) =====
            add_ms = st.checkbox("Add Task Milestones")
            if add_ms:
                proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"),-1)==pid_t]
                if not proj_tasks:
                    st.info("No tasks yet.")
                else:
                    proj_tasks = sorted(proj_tasks, key=lambda x: _safe_int(x.get("task_id"), 10**9))
                    task_labels = [f"{_safe_int(t.get('task_id'),0)} — {t.get('name','(no name)')}" for t in proj_tasks]
                    pick_task = st.selectbox("Select Task", options=["— Select —"]+task_labels, key=f"t7_addms_pick_{pid_t}")
                    if pick_task == "— Select —":
                        st.info("Pick a task to add milestones.")
                    else:
                        cur_tid = _safe_int(str(pick_task).split(" — ")[0], 0)
                        cur_task = next(t for t in tasks if _safe_int(t.get("project_id"),-1)==pid_t and _safe_int(t.get("task_id"),-1)==cur_tid)
                        if "milestones" not in cur_task or not isinstance(cur_task["milestones"], list):
                            cur_task["milestones"] = []
                        def _next_ms_id(tobj):
                            used = sorted(_safe_int(m.get("milestone_id"), 0) for m in (tobj.get("milestones") or []) if str(m.get("milestone_id","")).isdigit())
                            n = 1
                            for u in used:
                                if u == n: n += 1
                                elif u > n: break
                            return n

                        st.markdown("#### Add Milestone")
                        mc1, mc2, mc3, mc4, mc5 = st.columns([1.5,1,1,1,0.8])
                        m_name = mc1.text_input("Name", key=f"t7_ms_name_add_{pid_t}_{cur_tid}")
                        m_sd   = mc2.date_input("Start", value=_project_bounds(pid_t)[0].date(), key=f"t7_ms_sd_add_{pid_t}_{cur_tid}")
                        m_ed   = mc3.date_input("End",   value=_project_bounds(pid_t)[0].date(), key=f"t7_ms_ed_add_{pid_t}_{cur_tid}")
                        m_own  = mc4.multiselect("Owners (names)", options=assignee_options,
                                                 default=assignee_options[:1] if assignee_options else [],
                                                 key=f"t7_ms_own_add_{pid_t}_{cur_tid}")
                        m_pct  = mc5.slider("%", min_value=0, max_value=100, value=0, key=f"t7_ms_pct_add_{pid_t}_{cur_tid}")
                        if st.button("Add Milestone", key=f"t7_ms_btn_add_{pid_t}_{cur_tid}"):
                            if not m_name.strip():
                                st.warning("Milestone name required.")
                            else:
                                sdc = _clamp_to_project(pid_t, pd.to_datetime(m_sd))
                                edc = _clamp_to_project(pid_t, pd.to_datetime(m_ed))
                                if pd.notna(sdc) and pd.notna(edc) and edc < sdc:
                                    edc = sdc
                                owners_em = _emails_from_labels(m_own, label_to_email)
                                cur_task["milestones"].append({
                                    "milestone_id": _next_ms_id(cur_task),
                                    "name": m_name.strip(),
                                    "start_date": _date_to_str(sdc.date() if pd.notna(sdc) else _today().date()),
                                    "end_date":   _date_to_str(edc.date() if pd.notna(edc) else _today().date()),
                                    "owners": owners_em,
                                    "status": "To Do",
                                    "percent": int(_safe_float(m_pct)),
                                })
                                _write_json(TASKS_JSON, tasks)
                                st.success("Milestone added.")
                                _rerun()

                        ms_view = pd.DataFrame(cur_task.get("milestones", []))
                        if ms_view.empty:
                            st.info("No milestones yet for this task.")
                        else:
                            ms_view["owners_names"] = ms_view["owners"].apply(lambda v: ", ".join(_labels_from_emails(v, email_to_label)))
                            st.dataframe(
                                ms_view[["milestone_id","name","start_date","end_date","owners_names","percent"]],
                                use_container_width=True
                            )

                        if st.button("Save Task Details", key=f"t7_btn_save_task_details_{pid_t}_{cur_tid}"):
                            _write_json(TASKS_JSON, tasks)
                            st.success("Task details saved.")
                            _rerun()

        # ======== Edit Task mode ========
        elif mode == "Edit Task":
            st.markdown("#### Edit Tasks")

            proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"),-1)==pid_t]
            proj_tasks = sorted(proj_tasks, key=lambda x: _safe_int(x.get("task_id"), 10**9))
            if not proj_tasks:
                st.info("No tasks yet.")
            else:
                view = pd.DataFrame(proj_tasks)
                for c in ["task_id","name","status","priority","assignees","start_date","due_date","estimate_h","actual_h"]:
                    if c not in view.columns:
                        view[c] = "" if c not in ("assignees",) else []
                view["assignees_names"] = view["assignees"].apply(lambda v: ", ".join(_labels_from_emails(v, email_to_label)))
                st.dataframe(
                    view[["task_id","name","status","priority","assignees_names","start_date","due_date","estimate_h","actual_h"]],
                    use_container_width=True
                )

                task_labels = ["— Select —"] + [f"{_safe_int(t.get('task_id'),0)} — {t.get('name','(no name)')}" for t in proj_tasks]
                pick_task = st.selectbox("Select a Task", options=task_labels, key=f"t7_edit_pick_task_{pid_t}")
                if pick_task == "— Select —":
                    st.info("Pick a task to edit details & milestones.")
                else:
                    cur_tid = _safe_int(str(pick_task).split(" — ")[0], 0)
                    trow = next(t for t in proj_tasks if _safe_int(t.get("task_id"),-1)==cur_tid)

                    st.markdown("#### Task Details")
                    d1, d2, d3, d4 = st.columns([1.1,1,1,1.1])
                    stat = d1.selectbox("Status", ["To Do","In Progress","Blocked","Review","Done"],
                                        index=_idx(["To Do","In Progress","Blocked","Review","Done"], trow.get("status","To Do"), 0),
                                        key=f"t7_td_stat_{pid_t}_{cur_tid}")
                    pri  = d2.selectbox("Priority", ["Low","Medium","High"],
                                        index=_idx(["Low","Medium","High"], trow.get("priority","Medium"), 1),
                                        key=f"t7_td_pri_{pid_t}_{cur_tid}")
                    sdate = d3.date_input("Start Date", value=pd.to_datetime(trow.get("start_date"), errors="coerce").date() if trow.get("start_date") else _project_bounds(pid_t)[0].date(),
                                          key=f"t7_td_start_{pid_t}_{cur_tid}")
                    due   = d4.date_input("End Date",   value=pd.to_datetime(trow.get("due_date"),   errors="coerce").date() if trow.get("due_date")   else _project_bounds(pid_t)[0].date(),
                                          key=f"t7_td_due_{pid_t}_{cur_tid}")

                    with st.expander("➕ Add Person (for assignees / owners)"):
                        p1, p2, p3, p4 = st.columns([1.3,1.3,1.0,1.0])
                        ep_name = p1.text_input("Name", key=f"t7_edit_addp_name_{pid_t}_{cur_tid}")
                        ep_email = p2.text_input("Email", key=f"t7_edit_addp_email_{pid_t}_{cur_tid}")
                        ep_role = p3.text_input("Role", key=f"t7_edit_addp_role_{pid_t}_{cur_tid}", placeholder="e.g. Planner")
                        ep_save = p4.checkbox("Save to Contacts", value=True, key=f"t7_edit_addp_save_{pid_t}_{cur_tid}")
                        if st.button("Add Person", key=f"t7_edit_addp_btn_{pid_t}_{cur_tid}"):
                            if not ep_name.strip() or not ep_email.strip():
                                st.warning("Name and Email required.")
                            else:
                                for p in projects:
                                    if _safe_int(p.get("project_id"),-1)==pid_t:
                                        tlist = p.get("team", []) or []
                                        tlist.append({"name":ep_name.strip(),"email":ep_email.strip(),"role":ep_role.strip()})
                                        p["team"] = tlist
                                        p["updated_at"] = datetime.now().isoformat(timespec="seconds")
                                        break
                                _write_json(PROJECTS_JSON, projects)
                                if ep_save:
                                    _upsert_contacts([{"name":ep_name.strip(),"email":ep_email.strip(),"role":ep_role.strip()}])
                                st.success("Person added.")
                                _rerun()

                    cur_asg_labels = _labels_from_emails(trow.get("assignees", []), email_to_label)
                    cur_asg_labels = [x for x in cur_asg_labels if x in assignee_options]
                    asg = st.multiselect("Assignees (names)", options=assignee_options, default=cur_asg_labels, key=f"t7_td_asg_{pid_t}_{cur_tid}")

                    desc = st.text_area("Description", value=str(trow.get("description","") or ""), height=80, key=f"t7_td_desc_{pid_t}_{cur_tid}")

                    other_ids = [_safe_int(t["task_id"], 0) for t in proj_tasks if _safe_int(t["task_id"], 0) != cur_tid]
                    cur_deps = [_safe_int(x, None) for x in (trow.get("dependencies") or []) if str(x).strip().isdigit() and _safe_int(x, 0) in other_ids]
                    dep_sel = st.multiselect("Dependencies", options=sorted(other_ids), default=sorted([d for d in cur_deps if d is not None]), key=f"t7_td_deps_{pid_t}_{cur_tid}")

                    cbtn1, cbtn2, cbtn3 = st.columns([1,1,2])
                    if cbtn1.button("Save Task Changes", key=f"t7_td_save_{pid_t}_{cur_tid}"):
                        for t in tasks:
                            if _safe_int(t.get("project_id"),-1)==pid_t and _safe_int(t.get("task_id"),-1)==cur_tid:
                                t["status"] = stat
                                t["priority"] = pri
                                sdc = _clamp_to_project(pid_t, pd.to_datetime(sdate))
                                ddc = _clamp_to_project(pid_t, pd.to_datetime(due))
                                if pd.notna(sdc) and pd.notna(ddc) and ddc < sdc:
                                    ddc = sdc
                                t["start_date"] = _date_to_str(sdc.date() if pd.notna(sdc) else "")
                                t["due_date"]   = _date_to_str(ddc.date() if pd.notna(ddc) else "")
                                t["assignees"]  = _emails_from_labels(asg, label_to_email)
                                t["description"]= desc.strip()
                                t["dependencies"]= sorted(set([_safe_int(x, 0) for x in dep_sel if _safe_int(x,0) > 0]))
                                if t["status"]=="Done" and not t.get("completed_at"):
                                    t["completed_at"] = datetime.now().strftime("%Y-%m-%d")
                                break
                        _renumber_tasks_for_project(pid_t)
                        _write_json(TASKS_JSON, tasks)
                        st.success("Task updated.")
                        _rerun()

                    with cbtn2:
                        confirm = st.checkbox("Confirm delete", key=f"t7_td_del_confirm_{pid_t}_{cur_tid}")
                        if st.button("Delete Task", key=f"t7_td_del_{pid_t}_{cur_tid}", disabled=not confirm):
                            tasks[:] = [t for t in tasks if not (_safe_int(t.get("project_id"),-1)==pid_t and _safe_int(t.get("task_id"),-1)==cur_tid)]
                            for t in tasks:
                                if _safe_int(t.get("project_id"),-1)==pid_t:
                                    t["dependencies"] = [d for d in (t.get("dependencies") or []) if _safe_int(d,0)!=cur_tid]
                            _renumber_tasks_for_project(pid_t)
                            _write_json(TASKS_JSON, tasks)
                            st.success("Task deleted.")
                            _rerun()

                    st.markdown("---")

                    edit_ms = st.checkbox("Edit Milestones", key=f"t7_edit_ms_toggle_{pid_t}_{cur_tid}")
                    if edit_ms:
                        proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"),-1)==pid_t]
                        trow = next(t for t in proj_tasks if _safe_int(t.get("task_id"),-1)==cur_tid)
                        if "milestones" not in trow or not isinstance(trow["milestones"], list):
                            trow["milestones"] = []

                        ms_view = pd.DataFrame(trow["milestones"])
                        if ms_view.empty:
                            st.info("No milestones for this task yet.")
                        else:
                            ms_view["owners_names"] = ms_view["owners"].apply(lambda v: ", ".join(_labels_from_emails(v, email_to_label)))
                            st.dataframe(
                                ms_view[["milestone_id","name","start_date","end_date","owners_names","percent"]],
                                use_container_width=True
                            )

                        ms_ids = [_safe_int(m.get("milestone_id"), 0) for m in (trow["milestones"] or []) if str(m.get("milestone_id","")).isdigit()]
                        if not ms_ids:
                            st.info("No milestones to edit.")
                        else:
                            pick_ms = st.selectbox("Select Milestone", options=sorted(ms_ids), key=f"t7_pick_ms_{pid_t}_{cur_tid}")
                            cur_ms = next((m for m in trow["milestones"] if _safe_int(m.get("milestone_id"),-1)==_safe_int(pick_ms, -1)), None)
                            if cur_ms is None:
                                st.warning("Milestone not found.")
                            else:
                                st.markdown("#### Edit Milestone")
                                mc1, mc2, mc3, mc4, mc5 = st.columns([1.5,1,1,1,0.8])
                                m_name = mc1.text_input("Name", value=str(cur_ms.get("name","")), key=f"t7_ms_name_edit_{pid_t}_{cur_tid}_{pick_ms}")
                                m_sd   = mc2.date_input("Start", value=pd.to_datetime(cur_ms.get("start_date"), errors="coerce").date() if cur_ms.get("start_date") else _project_bounds(pid_t)[0].date(),
                                                        key=f"t7_ms_sd_edit_{pid_t}_{cur_tid}_{pick_ms}")
                                m_ed   = mc3.date_input("End",   value=pd.to_datetime(cur_ms.get("end_date"), errors="coerce").date() if cur_ms.get("end_date") else _project_bounds(pid_t)[0].date(),
                                                        key=f"t7_ms_ed_edit_{pid_t}_{cur_tid}_{pick_ms}")
                                cur_own_labels = _labels_from_emails(cur_ms.get("owners", []), email_to_label)
                                cur_own_labels = [x for x in cur_own_labels if x in assignee_options]
                                m_own  = mc4.multiselect("Owners (names)", options=assignee_options, default=cur_own_labels,
                                                         key=f"t7_ms_own_edit_{pid_t}_{cur_tid}_{pick_ms}")
                                m_pct  = mc5.slider("%", min_value=0, max_value=100, value=int(_safe_float(cur_ms.get("percent",0))),
                                                    key=f"t7_ms_pct_edit_{pid_t}_{cur_tid}_{pick_ms}")

                                u1, u2 = st.columns([1,1])
                                if u1.button("Update Task Milestone", key=f"t7_ms_update_{pid_t}_{cur_tid}_{pick_ms}"):
                                    sdc = _clamp_to_project(pid_t, pd.to_datetime(m_sd))
                                    edc = _clamp_to_project(pid_t, pd.to_datetime(m_ed))
                                    if pd.notna(sdc) and pd.notna(edc) and edc < sdc:
                                        edc = sdc
                                    owners_em = _emails_from_labels(m_own, email_to_label)
                                    for m in trow["milestones"]:
                                        if _safe_int(m.get("milestone_id"),-1)==_safe_int(pick_ms,-1):
                                            m["name"] = m_name.strip()
                                            m["start_date"] = _date_to_str(sdc.date() if pd.notna(sdc) else _today().date())
                                            m["end_date"]   = _date_to_str(edc.date() if pd.notna(edc) else _today().date())
                                            m["owners"] = owners_em
                                            m["percent"] = int(_safe_float(m_pct))
                                            break
                                    _write_json(TASKS_JSON, tasks)
                                    st.success("Milestone updated.")
                                    _rerun()

                                with u2:
                                    confirm_rm = st.checkbox("Confirm remove", key=f"t7_ms_rm_confirm_{pid_t}_{cur_tid}_{pick_ms}")
                                    if st.button("Remove Task Milestone", key=f"t7_ms_remove_{pid_t}_{cur_tid}_{pick_ms}", disabled=not confirm_rm):
                                        trow["milestones"] = [m for m in trow["milestones"] if _safe_int(m.get("milestone_id"),-1)!=_safe_int(pick_ms,-1)]
                                        _write_json(TASKS_JSON, tasks)
                                        st.success("Milestone removed.")
                                        _rerun()

    # ---------- BOARD VIEW ----------
    elif vmode == "Board":
        proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"),-1)==pid_t]
        if not proj_tasks:
            st.info("No tasks yet.")
        else:
            cols = ["To Do","In Progress","Blocked","Review","Done"]
            buckets = {k: [] for k in cols}
            for r in proj_tasks:
                top = (r.get("parent_task_id") is None) or pd.isna(r.get("parent_task_id"))
                if top:
                    buckets.get(r.get("status","To Do"), buckets["To Do"]).append(r)
            kcols = st.columns(len(cols))
            for i, colname in enumerate(cols):
                with kcols[i]:
                    st.markdown(f"**{colname}** ({len(buckets[colname])})")
                    for r in sorted(buckets[colname], key=lambda x: _pdate(x.get("start_date")) or pd.Timestamp.max):
                        tid=_safe_int(r.get("task_id"), 0)
                        due=str(r.get("due_date","") or "—")
                        pri=r.get("priority","Medium")
                        ass=", ".join(_labels_from_emails(r.get("assignees", []), email_to_label))
                        st.markdown(
                            f"<div style='border:1px solid #e5e7eb;border-radius:12px;padding:8px;margin-bottom:8px;background:{_task_status_color(colname)}22;'>"
                            f"<div style='font-weight:600;margin-bottom:4px;'>{r.get('name')}</div>"
                            f"<div style='font-size:12px;'>👥 {ass or '—'} &nbsp; | &nbsp; 🗓 {due} &nbsp; | &nbsp; "
                            f"<span style='color:{_priority_color(pri)}'>⚑ {pri}</span></div></div>",
                            unsafe_allow_html=True
                        )
                        new_s = st.selectbox("Move to", cols, index=cols.index(colname), key=f"t7_move_{pid_t}_{tid}")
                        if new_s != colname:
                            for t in tasks:
                                if _safe_int(t.get("project_id"),-1)==pid_t and _safe_int(t.get("task_id"),-1)==tid:
                                    t["status"]= new_s
                                    if new_s=="Done" and not t.get("completed_at"):
                                        t["completed_at"]= datetime.now().strftime("%Y-%m-%d")
                                    break
                            _write_json(TASKS_JSON, tasks); _rerun()

    # ---------- CALENDAR VIEW ----------
    elif vmode == "Calendar":
        proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"),-1)==pid_t]
        if not proj_tasks:
            st.info("No tasks yet.")
        else:
            cal = pd.DataFrame(proj_tasks)
            cal["__due"] = pd.to_datetime(cal["due_date"], errors="coerce").dt.date
            cal = cal[cal["__due"].notna()]
            if cal.empty:
                st.info("No valid due dates to display.")
            else:
                cnt = cal.groupby("__due").size().reset_index(name="count").sort_values("__due")
                days = pd.date_range(
                    start=min(pd.to_datetime(cnt["__due"]) - pd.Timedelta(days=2)),
                    end=max(pd.to_datetime(cnt["__due"]) + pd.Timedelta(days=2)),
                    freq="D"
                )
                m = {pd.to_datetime(d).date(): 0 for d in days}
                for _, r in cnt.iterrows(): m[pd.to_datetime(r["__due"]).date()] = int(r["count"])
                y = [m[d.date()] for d in days]
                fig_cal = go.Figure(go.Bar(
                    x=[d.strftime("%Y-%m-%d") for d in days], y=y,
                    marker=dict(color=y, colorscale="Blues"),
                    hovertemplate="Date %{x}<br>Tasks due: %{y}<extra></extra>"
                ))
                fig_cal.update_layout(height=260, margin=dict(t=16,b=8,l=8,r=8),
                                      xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=10)),
                                      yaxis=dict(title="Tasks due"))
                st.plotly_chart(fig_cal, use_container_width=True)

    # ---------- TIMELINE VIEW ----------
    elif vmode == "Timeline":
        proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"),-1)==pid_t]
        if not proj_tasks:
            st.info("No tasks yet.")
        else:
            pstart, pend = _project_bounds(pid_t)
            items=[]
            for r in proj_tasks:
                name=r.get("name","(no name)")
                s0=_pdate(r.get("start_date"))
                d0=_pdate(r.get("due_date"))
                ms = r.get("milestones", []) or []
                msd, med = None, None
                for m in ms:
                    sd=_pdate(m.get("start_date")); ed=_pdate(m.get("end_date"))
                    if pd.notna(sd): msd = sd if (msd is None or sd<msd) else msd
                    if pd.notna(ed): med = ed if (med is None or ed>med) else med
                s = s0 or msd or pstart
                e = d0 or med or s
                s=_clamp_to_project(pid_t, s); e=_clamp_to_project(pid_t, e)
                if pd.notna(s) and pd.notna(e) and e<s: e=s
                items.append({"Label":name,"Start":s,"Finish":e,"Status":r.get("status","To Do")})

            if items:
                df_g = pd.DataFrame(items)
                df_g = df_g.sort_values("Start", ascending=True).reset_index(drop=True)
                fig = px.timeline(df_g, x_start="Start", x_end="Finish", y="Label", color="Status")
                fig.add_vline(x=_today(), line_width=2, line_dash="dash")
                fig.update_layout(
                    height=380, title=f"Project Timeline (Tasks)",
                    xaxis=dict(range=[pstart, pend]),
                    yaxis=dict(categoryorder="array", categoryarray=list(df_g["Label"]))
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid dates to plot.")

# ---------- Unified Project Gantt (Tasks | Milestones) — Advanced ----------
with t_tab:
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("#### Project Gantt (Tasks & Milestones)")
    pstart, pend = _project_bounds(pid_t)
    proj_tasks = [t for t in tasks if _safe_int(t.get("project_id"), -1) == pid_t]

    def _safe_span(start_s, end_s, fallback_start):
        s = _clamp_to_project(pid_t, _pdate(start_s))
        e = _clamp_to_project(pid_t, _pdate(end_s))
        if pd.isna(s) and pd.isna(e):
            s = _clamp_to_project(pid_t, fallback_start or pstart); e = s
        if pd.isna(s) and pd.notna(e): s = e
        if pd.isna(e) and pd.notna(s): e = s
        if pd.notna(s) and pd.notna(e) and e < s: e = s
        return s, e

    def _names_for_emails(email_list):
        if not email_list: return []
        team_names = {}
        _prow = next((p for p in projects if _safe_int(p.get("project_id"), -1) == pid_t), None)
        if _prow:
            for tm in _prow.get("team", []) or []:
                em = (tm or {}).get("email", "")
                nm = (tm or {}).get("name", "") or em
                if em: team_names[em] = nm
        if 'contacts' in globals() and isinstance(contacts, list):
            for c in contacts:
                em = (c or {}).get("email", ""); nm = (c or {}).get("name", "") or em
                if em and em not in team_names: team_names[em] = nm
        base_names = [team_names.get(e, e) for e in email_list]
        name_counts = {}
        for n in base_names: name_counts[n] = name_counts.get(n, 0) + 1
        out = []
        for n, e in zip(base_names, email_list):
            out.append(f"{n} <{e}>" if name_counts.get(n,0) > 1 else n)
        return out

    def _task_percent(trow: dict) -> int:
        ms = trow.get("milestones", []) or []
        vals = []
        for m in ms:
            try: vals.append(float(m.get("percent", 0)))
            except Exception: pass
        if vals:
            return int(max(0, min(100, round(sum(vals)/len(vals)))))
        est = _safe_float(trow.get("estimate_h", 0.0))
        act = _safe_float(trow.get("actual_h", 0.0))
        if est > 0:
            return int(max(0, min(100, round((act/est)*100.0))))
        stat = str(trow.get("status", "To Do"))
        return {"Done": 100, "Review": 70, "In Progress": 50, "Blocked": 10}.get(stat, 0)

    gmode_key = f"t7_gmode_{pid_t}"
    gmode = st.segmented_control("Show", ["Tasks","Milestones"], key=gmode_key) if hasattr(st, "segmented_control") else st.radio("Show", ["Tasks","Milestones"], key=gmode_key, horizontal=True)

    if gmode == "Tasks":
        task_rows, overlay_ms_x, overlay_ms_y, overlay_ms_txt = [], [], [], []
        for r in proj_tasks:
            name = r.get("name","(no name)")
            assignees_nm = _names_for_emails(list(r.get("assignees", []) or []))
            who = ", ".join(assignees_nm) or "—"
            s0 = _pdate(r.get("start_date")); e0 = _pdate(r.get("due_date"))
            msd, med = None, None
            for m in (r.get("milestones", []) or []):
                sd = _pdate(m.get("start_date")); ed = _pdate(m.get("end_date"))
                if pd.notna(sd): msd = sd if (msd is None or sd < msd) else msd
                if pd.notna(ed): med = ed if (med is None or ed > med) else med
            s_fallback = msd or pstart
            s, e = _safe_span(s0, e0, s_fallback)
            label = f"{who} ▸ {name}"
            pct = _task_percent(r)
            task_rows.append({"Label": label, "Start": s, "Finish": e, "Status": r.get("status", "To Do"),
                              "Priority": r.get("priority", "Medium"), "Assignees": who, "Task": name, "Percent": pct})
            for m in (r.get("milestones", []) or []):
                mx = _clamp_to_project(pid_t, _pdate(m.get("start_date")))
                if pd.notna(mx):
                    overlay_ms_x.append(mx); overlay_ms_y.append(label); overlay_ms_txt.append(m.get("name", "Milestone"))

        if task_rows:
            df_t = pd.DataFrame(task_rows).sort_values("Start", ascending=True).reset_index(drop=True)
            fig_t = px.timeline(
                df_t, x_start="Start", x_end="Finish", y="Label", color="Status",
                hover_data={"Task": True, "Assignees": True, "Priority": True, "Start": "|%Y-%m-%d", "Finish": "|%Y-%m-%d", "Percent": True}
            )
            fig_t.add_trace(go.Scatter(x=df_t["Finish"], y=df_t["Label"], mode="text",
                                       text=[f"{int(p)}%" for p in df_t["Percent"]],
                                       textposition="middle right", showlegend=False, hoverinfo="skip"))
            if overlay_ms_x:
                fig_t.add_trace(go.Scatter(
                    x=overlay_ms_x, y=overlay_ms_y, mode="markers", marker_symbol="diamond", marker_size=10,
                    name="Milestone", hovertext=overlay_ms_txt,
                    hovertemplate="Milestone: %{hovertext}<br>Date: %{x|%Y-%m-%d}<extra></extra>"
                ))
            fig_t.add_vline(x=_today(), line_width=2, line_dash="dash")

            labels_sorted = df_t["Label"].tolist()
            fig_t.update_layout(
                height=420, title=f"Gantt — Tasks  ({pstart.date()} → {pend.date()})",
                xaxis=dict(range=[pstart, pend]),
                yaxis=dict(categoryorder="array", categoryarray=list(reversed(labels_sorted)))
            )

            id_to_label = {}
            for r in proj_tasks:
                assignees_nm = _names_for_emails(list(r.get("assignees", []) or []))
                who = ", ".join(assignees_nm) or "—"
                id_to_label[_safe_int(r.get("task_id"), -1)] = f"{who} ▸ {r.get('name','(no name)')}"
            label_to_span = {row["Label"]: (row["Start"], row["Finish"]) for _, row in df_t.iterrows()}

            for r in proj_tasks:
                succ_label = id_to_label.get(_safe_int(r.get("task_id"), -1))
                if not succ_label or succ_label not in label_to_span: continue
                succ_start, _succ_finish = label_to_span[succ_label]
                for pred in (r.get("dependencies", []) or []):
                    pred_label = id_to_label.get(_safe_int(pred, -1))
                    if not pred_label or pred_label not in label_to_span: continue
                    _pred_start, pred_finish = label_to_span[pred_label]
                    fig_t.add_annotation(
                        x=succ_start, y=succ_label, ax=pred_finish, ay=pred_label,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1
                    )

            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No tasks to plot yet.")

    else:
        ms_rows = []
        for r in proj_tasks:
            tname = r.get("name", "(no name)")
            assignees_nm = _names_for_emails(list(r.get("assignees", []) or []))
            who = ", ".join(assignees_nm) or "—"
            for m in (r.get("milestones", []) or []):
                s, e = _safe_span(m.get("start_date"), m.get("end_date"), pstart)
                label = f"{tname} — {m.get('name','(milestone)')}"
                ms_rows.append({
                    "Label": label, "Start": s, "Finish": e,
                    "Status": m.get("status", "To Do"),
                    "Owners": ", ".join(_names_for_emails(m.get("owners", []) or [])) or who,
                    "Task": tname,
                    "Percent": int(_safe_float(m.get("percent", 0)))
                })

        if ms_rows:
            df_m = pd.DataFrame(ms_rows).sort_values("Start", ascending=True).reset_index(drop=True)
            fig_m = px.timeline(
                df_m, x_start="Start", x_end="Finish", y="Label", color="Status",
                hover_data={"Task": True, "Owners": True, "Start": "|%Y-%m-%d", "Finish": "|%Y-%m-%d", "Percent": True}
            )
            fig_m.add_trace(go.Scatter(x=df_m["Finish"], y=df_m["Label"], mode="text",
                                       text=[f"{int(p)}%" for p in df_m["Percent"]],
                                       textposition="middle right", showlegend=False, hoverinfo="skip"))
            fig_m.add_vline(x=_today(), line_width=2, line_dash="dash")
            labels_sorted_m = df_m["Label"].tolist()
            fig_m.update_layout(
                height=420, title=f"Gantt — Milestones  ({pstart.date()} → {pend.date()})",
                xaxis=dict(range=[pstart, pend]),
                yaxis=dict(categoryorder="array", categoryarray=list(reversed(labels_sorted_m)))
            )
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("No milestones to plot.")

# -------- 📊 Analytics ----------
with a_tab:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime

    def _pid_int(x):
        try:
            return int(str(x).strip())
        except Exception:
            return -1

    def _build_proj_rows(df: pd.DataFrame) -> pd.DataFrame:
        rows = df.copy().reset_index(drop=True)
        rows["__ord"] = rows.index + 1
        rows["__label"] = rows.apply(
            lambda r: f"{int(r['__ord'])} — {r.get('project_number','PRJ-????')} — {r.get('name','(no name)')}[{r.get('status','')}]",
            axis=1
        )
        rows["project_id_int"] = rows["project_id"].astype("string").fillna("-1").apply(_pid_int)
        return rows

    def _tasks_df_all() -> pd.DataFrame:
        base = pd.DataFrame(tasks or [])
        if base.empty and 'task_df' in globals() and isinstance(task_df, pd.DataFrame) and not task_df.empty:
            base = task_df.copy()
        if base.empty:
            base = pd.DataFrame(columns=["task_id","project_id","status","estimate_h","actual_h","due_date","milestones","assignees"])
        if "task_id" not in base.columns:
            base["task_id"] = range(1, len(base) + 1)
        if "status" not in base.columns:
            base["status"] = ""
        if "estimate_h" not in base.columns:
            base["estimate_h"] = 0.0
        if "actual_h" not in base.columns:
            base["actual_h"] = 0.0
        if "assignees" not in base.columns:
            base["assignees"] = [[] for _ in range(len(base))]
        base["project_id_int"] = base.get("project_id", pd.Series([None]*len(base))).astype("string").fillna("-1").apply(_pid_int)
        base["estimate_h"] = pd.to_numeric(base["estimate_h"], errors="coerce").fillna(0.0)
        base["actual_h"]   = pd.to_numeric(base["actual_h"], errors="coerce").fillna(0.0)
        base["due_date"]   = pd.to_datetime(base.get("due_date", None), errors="coerce")
        return base

    def _task_percent_like_gantt(trow: dict) -> int:
        ms = trow.get("milestones", []) or []
        vals = []
        for m in ms:
            try: vals.append(float(m.get("percent", 0)))
            except Exception: pass
        if vals:
            return int(max(0, min(100, round(sum(vals)/len(vals)))))
        est = _safe_float(trow.get("estimate_h", 0.0))
        act = _safe_float(trow.get("actual_h", 0.0))
        if est > 0:
            return int(max(0, min(100, round((act/est)*100.0))))
        stat = str(trow.get("status", "To Do"))
        return {"Done": 100, "Review": 70, "In Progress": 50, "Blocked": 10}.get(stat, 0)

    def _project_progress_for(pid: int, tasks_list: list) -> int:
        pobj = next((p for p in projects if _pid_int(p.get("project_id")) == pid), None)
        mode = str((pobj or {}).get("progress_mode", "auto")).lower()
        weight = str((pobj or {}).get("progress_weight", "tasks")).lower()
        override = int(float((pobj or {}).get("progress_override", 0)))
        if mode == "manual":
            return max(0, min(100, int(override)))
        ptasks = [t for t in tasks_list if _pid_int(t.get("project_id")) == pid]
        if not ptasks:
            return 0
        if weight == "hours":
            wsum = 0.0; acc  = 0.0
            for t in ptasks:
                pct = _task_percent_like_gantt(t)
                w   = float(_safe_float(t.get("estimate_h", 0.0)))
                if w <= 0: w = 1.0
                acc += pct * w; wsum += w
            return int(round(acc / wsum)) if wsum > 0 else int(round(sum(_task_percent_like_gantt(t) for t in ptasks) / len(ptasks)))
        else:
            return int(round(sum(_task_percent_like_gantt(t) for t in ptasks) / len(ptasks)))

    def _portfolio_frame(proj_rows: pd.DataFrame, tdf: pd.DataFrame) -> pd.DataFrame:
        if proj_rows.empty:
            return pd.DataFrame(columns=["__ord","__label","project_id_int","Status","Start","End","Progress %","Done/Total","Overdue","On-time %"])
        today = _today().normalize()
        overdue_mask = (tdf["due_date"] < today) & (tdf["status"] != "Done")
        overdue = overdue_mask.groupby(tdf["project_id_int"]).sum().rename("Overdue").astype(int)
        totals = tdf.groupby("project_id_int")["task_id"].count().rename("Total")
        done   = tdf[tdf["status"] == "Done"].groupby("project_id_int")["task_id"].count().rename("Done")
        mt_rows = []
        for _, r in tdf.iterrows():
            for m in (r.get("milestones", []) or []):
                try:
                    pid = _pid_int(r.get("project_id"))
                    end = pd.to_datetime(m.get("end_date"), errors="coerce")
                    pct = float(m.get("percent", 0) or 0)
                    mt_rows.append({"project_id_int": pid, "end_date": end, "percent": pct})
                except Exception:
                    pass
        mdf = pd.DataFrame(mt_rows)
        if not mdf.empty:
            mdf["on_time"] = (mdf["percent"] >= 100) & (mdf["end_date"] <= today)
            ontime = (mdf.groupby("project_id_int")["on_time"].sum() /
                      mdf.groupby("project_id_int")["on_time"].count() * 100).fillna(0.0).round(0).rename("On-time %")
        else:
            ontime = pd.Series(dtype=float, name="On-time %")
        progress_map = {}
        for pid in proj_rows["project_id_int"]:
            progress_map[pid] = _project_progress_for(pid, tasks or [])
        progress = pd.Series(progress_map, name="Progress %")

        out = proj_rows.copy()
        out = out.rename(columns={"status":"Status"})
        out = out.merge(totals, left_on="project_id_int", right_index=True, how="left")
        out = out.merge(done,   left_on="project_id_int", right_index=True, how="left")
        out = out.merge(overdue, left_on="project_id_int", right_index=True, how="left")
        out = out.merge(ontime, left_on="project_id_int", right_index=True, how="left")
        out["Total"]   = out["Total"].fillna(0).astype(int)
        out["Done"]    = out["Done"].fillna(0).astype(int)
        out["Overdue"] = out["Overdue"].fillna(0).astype(int)
        out["Progress %"] = out["project_id_int"].map(progress).fillna(0).astype(int)
        out["On-time %"]  = out["On-time %"].fillna(0).astype(int)
        out["Done/Total"] = out["Done"].astype(str) + " / " + out["Total"].astype(str)

        return out[["__ord","__label","project_id_int","Status","start_date","end_date","Progress %","Done/Total","Overdue","On-time %"]] \
                  .rename(columns={"start_date":"Start","end_date":"End","__label":"Project"})

    if proj_df.empty:
        st.info("No projects available.")
        st.stop()

    proj_rows = _build_proj_rows(proj_df)
    tdf_all   = _tasks_df_all()
    portfolio = _portfolio_frame(proj_rows, tdf_all)

    st.subheader("Portfolio Overview")
    k1, k2, k3, k4 = st.columns(4)
    total_projects = int(len(portfolio))
    avg_progress   = int(round(portfolio["Progress %"].mean())) if total_projects > 0 else 0
    total_open     = int((tdf_all["status"] != "Done").sum())
    total_overdue  = int(((tdf_all["due_date"] < _today().normalize()) & (tdf_all["status"] != "Done")).sum())
    k1.metric("Projects", total_projects)
    k2.metric("Avg Progress", f"{avg_progress}%")
    k3.metric("Open Tasks", total_open)
    k4.metric("Overdue Tasks", total_overdue)

    fc1, fc2 = st.columns([1.2, 2])
    status_filter = fc1.multiselect("Filter by Status", sorted(portfolio["Status"].dropna().unique().tolist()))
    search_text   = fc2.text_input("Search (number / name)")

    pf = portfolio.copy()
    if status_filter:
        pf = pf[pf["Status"].isin(status_filter)]
    if search_text.strip():
        q = search_text.lower()
        pf = pf[pf["Project"].str.lower().str.contains(q)]

    st.dataframe(
        pf[["Project","Status","Start","End","Progress %","Done/Total","Overdue","On-time %"]],
        use_container_width=True, hide_index=True
    )

    label_to_pid = {r["__label"]: int(r["project_id_int"]) for _, r in proj_rows.iterrows()}
    labels = proj_rows["__label"].tolist()

    current_pid = st.session_state.get("t7_selected_pid")
    if (current_pid is None) or (current_pid not in proj_rows["project_id_int"].tolist()):
        current_pid = int(proj_rows.iloc[0]["project_id_int"])
    try:
        cur_label = proj_rows.loc[proj_rows["project_id_int"] == current_pid, "__label"].iloc[0]
    except Exception:
        cur_label = labels[0]

    pick_label = st.selectbox("Select Project", options=labels, index=max(0, labels.index(cur_label)), key="t7_analytics_pick")
    selected_pid = label_to_pid.get(pick_label, current_pid)
    if st.session_state.get("t7_selected_pid") != selected_pid:
        st.session_state["t7_selected_pid"] = selected_pid

    sel_row = proj_rows[proj_rows["project_id_int"] == selected_pid]
    if not sel_row.empty:
        r0 = sel_row.iloc[0]
        st.markdown("---")
        st.subheader("Project Deep-Dive")

        cL, cR = st.columns([1.3, 1.0])
        with cL:
            st.markdown(f"**{r0.get('project_number','PRJ-????')} — {r0.get('name','(no name)')}**")
            st.caption(f"Status: {r0.get('status','—')}  |  Start: {r0.get('start_date','—')}  |  End: {r0.get('end_date','—')}")
        with cR:
            if st.button("Open in Tasks & Milestones"):
                st.session_state["t7_selected_pid"] = selected_pid
                st.success("Selected for Tasks & Milestones. Switch to that subtab.")
            if st.button("Open in Project Details"):
                st.session_state["t7_selected_pid"] = selected_pid
                st.success("Selected for Project Details. Switch to that subtab.")

        proj_obj = next((p for p in projects if _pid_int(p.get("project_id")) == selected_pid), None)
        current_progress = 0
        if proj_obj is None:
            st.warning("Project not found in memory.")
        else:
            prog_mode = str(proj_obj.get("progress_mode", "auto")).lower()
            weight    = str(proj_obj.get("progress_weight", "tasks")).lower()
            override  = int(float(proj_obj.get("progress_override", 0)))

            pc1, pc2, pc3 = st.columns([1.3, 1.0, 1.2])
            new_mode = pc1.selectbox(
                "Progress mode",
                ["Auto (from tasks)", "Manual override"],
                index=(0 if prog_mode == "auto" else 1),
                key=f"t7_prog_mode_{selected_pid}"
            )
            if "Auto" in new_mode:
                w_choice = pc2.selectbox(
                    "Weighting",
                    ["Equal by task", "Weighted by estimate hours"],
                    index=(0 if weight == "tasks" else 1),
                    key=f"t7_prog_weight_{selected_pid}"
                )
                proj_obj["progress_mode"] = "auto"
                proj_obj["progress_weight"] = "hours" if w_choice.endswith("hours") else "tasks"
                current_progress = _project_progress_for(selected_pid, tasks or [])
                st.progress(current_progress / 100)
                st.metric("Project progress", f"{current_progress}%")
                proj_obj["updated_at"] = datetime.now().isoformat(timespec="seconds")
                _write_json(PROJECTS_JSON, projects)
            else:
                new_override = pc3.slider(
                    "Set progress %",
                    min_value=0, max_value=100, value=int(override),
                    key=f"t7_prog_override_{selected_pid}"
                )
                proj_obj["progress_mode"] = "manual"
                proj_obj["progress_override"] = int(new_override)
                current_progress = int(new_override)
                st.progress(current_progress / 100)
                st.metric("Project progress", f"{current_progress}%")
                proj_obj["updated_at"] = datetime.now().isoformat(timespec="seconds")
                _write_json(PROJECTS_JSON, projects)

        st.markdown("### Analytics")

        fig_d = go.Figure(go.Pie(values=[current_progress, 100 - current_progress],
                                 labels=["Done", "Remaining"], hole=0.7))
        fig_d.update_layout(title="Overall Progress")
        st.plotly_chart(fig_d, use_container_width=True)

        sel_tasks_df = _tasks_df_all()
        sel_tasks_df = sel_tasks_df[sel_tasks_df["project_id_int"] == selected_pid].copy()

        try:
            bd = _compute_burndown(sel_tasks_df)
        except Exception:
            bd = pd.DataFrame()
        if not bd.empty:
            fig_b = go.Figure(go.Scatter(x=bd["date"], y=bd["remaining"], mode="lines+markers", name="Remaining"))
            fig_b.update_layout(title="Burndown", height=260, xaxis_title="Date", yaxis_title="Remaining Tasks")
            st.plotly_chart(fig_b, use_container_width=True)
        else:
            st.info("Burndown appears after tasks are completed (with completion dates).")

        if not sel_tasks_df.empty:
            def _norm_list(v):
                if v is None or v == "" or (isinstance(v, float) and pd.isna(v)): return []
                return v if isinstance(v, list) else [str(v)]
            w = sel_tasks_df.copy()
            w["assignees"] = w["assignees"].apply(_norm_list)
            w = w.explode("assignees")
            w["assignees"] = w["assignees"].replace("", pd.NA)
            w = w.dropna(subset=["assignees"])
            agg = w.groupby("assignees").size().reset_index(name="tasks")
            if not agg.empty:
                fig_w = px.bar(agg, x="assignees", y="tasks", title="Workload by Member (by tasks)")
                st.plotly_chart(fig_w, use_container_width=True)

        mt = []
        for _, r in sel_tasks_df.iterrows():
            for m in (r.get("milestones", []) or []):
                mt.append(m)
        if mt:
            mdf = pd.DataFrame(mt)
            mdf["end_date"] = pd.to_datetime(mdf["end_date"], errors="coerce")
            mdf["percent"] = pd.to_numeric(mdf["percent"], errors="coerce").fillna(0)
            on_time = int(((mdf["percent"] >= 100) & (mdf["end_date"] <= _today())).sum())
            total_m = len(mdf)
            rate = (on_time / total_m * 100) if total_m > 0 else 0
            st.metric("On-time Milestones", f"{rate:.0f}%")

# -------- 🔗 Linked Items ----------
with l_tab:
    if proj_df.empty:
        st.info("No projects yet.")
    else:
        options = (
            proj_df.assign(
                label=lambda df: df.apply(
                    lambda r: f"{int(r['project_id'])} | {r.get('project_number','PRJ-????')} — {r.get('name','(no name)')}", axis=1
                )
            )[["project_id","label"]].sort_values("label")
        )
        pick_l = st.selectbox("Select Project", options=options["label"].tolist(), index=0, key="t7_link_pick")
        try:
            pid_l = int(str(pick_l).split("|")[0].strip())
        except Exception:
            pid_l = int(options.iloc[0]["project_id"])

        row = proj_df[proj_df["project_id"].astype("string").fillna("-1").astype(int)==pid_l]
        if row.empty:
            st.warning("Project not found.")
        else:
            r0 = row.iloc[0]
            st.markdown(f"#### {r0.get('project_number','')} — {r0.get('name','')}")

            st.markdown("**Linked Assets**")
            asset_opts = sorted(map(str, (assets_map or {}).keys()))
            cur_assets = list(_normalize_list(r0.get("linked_assets", [])))
            new_assets = st.multiselect("Assets", options=asset_opts, default=cur_assets, key=f"lnk_assets_{pid_l}")
            if st.button("Save Assets Links", key=f"lnk_save_assets_{pid_l}"):
                for p in projects:
                    if _safe_int(p.get("project_id"),-1)==pid_l:
                        p["linked_assets"]= list(_normalize_list(new_assets))
                        p["updated_at"]= datetime.now().isoformat(timespec="seconds")
                        break
                _write_json(PROJECTS_JSON, projects); st.success("Assets updated."); _rerun()

            st.markdown("**Linked FMECAs**")
            fm_opts=[]
            if not fmeca_df.empty:
                for _, rf in fmeca_df.iterrows():
                    fm_opts.append(f"{rf.get('fmeca_id')} - {rf.get('linked_id','Untitled')}")
            cur_f = _normalize_int_list(r0.get("linked_fmecas",[]))
            new_f = st.multiselect("FMECAs", options=fm_opts,
                                   default=[f for f in fm_opts if (f.split(' - ')[0].isdigit() and int(f.split(' - ')[0]) in cur_f)],
                                   key=f"lnk_fmecas_{pid_l}")
            if st.button("Save FMECA Links", key=f"lnk_save_fm_{pid_l}"):
                for p in projects:
                    if _safe_int(p.get("project_id"),-1)==pid_l:
                        p["linked_fmecas"]=[int(f.split(" - ")[0]) for f in _normalize_list(new_f)] if new_f else []
                        p["updated_at"]= datetime.now().isoformat(timespec="seconds")
                        break
                _write_json(PROJECTS_JSON, projects); st.success("FMECAs updated."); _rerun()

# =================== 🔍 FMECA HUB — Phase 1.2 (Multi-Asset, Project-Linked, BOM-Aware) =========================
with sub_fmeca:
    # -------- Imports (local safety) --------
    import json
    from datetime import datetime, timedelta
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("FMECA Hub (Risk Management Center)")
    st.caption("Multi-asset FMECAs linked to projects, BOM-aware risk matrices, and task generation.")

    # ================== SCHEMA ==================
    def _init_fmeca_schema():
        """Ensure base table & indexes (backwards compatible)."""
        db.conn.executescript("""
        CREATE TABLE IF NOT EXISTS fmecas (
            fmeca_id INTEGER PRIMARY KEY AUTOINCREMENT,
            linked_asset TEXT,               -- legacy single-asset link (kept for backward-compat)
            linked_project INTEGER,
            type TEXT DEFAULT 'System',
            description TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            closed_at TEXT,
            version INTEGER DEFAULT 1,
            risk_matrix_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_fmeca_project ON fmecas (linked_project);
        CREATE INDEX IF NOT EXISTS idx_fmeca_asset   ON fmecas (linked_asset);

        -- NEW: many-to-many assets per FMECA
        CREATE TABLE IF NOT EXISTS fmeca_assets (
            fmeca_id INTEGER NOT NULL,
            asset_tag TEXT NOT NULL,
            PRIMARY KEY (fmeca_id, asset_tag)
        );
        CREATE INDEX IF NOT EXISTS idx_fmeca_assets_fid ON fmeca_assets (fmeca_id);
        CREATE INDEX IF NOT EXISTS idx_fmeca_assets_tag ON fmeca_assets (asset_tag);
        """)
        db.conn.commit()

    _init_fmeca_schema()

    # ================== SMALL HELPERS ==================
    def _safe_int(v, default=0):
        try:
            if v is None:
                return default
            return int(str(v).strip())
        except Exception:
            return default

    def _auto_table_height(n):
        try:
            return auto_table_height(n)  # if you already have a helper
        except Exception:
            return min(560, 56 + 28 * max(1, int(n)))

    def _write_tasks_json_safely(tlist: list):
        try:
            _write_json(TASKS_JSON, tlist)
        except Exception:
            # Silent fallback; data is still in memory/session
            pass

    def _tasks_store_load() -> list:
        """Load tasks from globals/session; never crash."""
        t = globals().get("tasks", None)
        if isinstance(t, list):
            return t
        t2 = st.session_state.get("tasks")
        if isinstance(t2, list):
            return t2
        return []

    def _tasks_store_set(tlist: list):
        """Persist tasks to both global and session, and write to JSON if available."""
        try:
            globals()["tasks"] = tlist
        except Exception:
            pass
        st.session_state["tasks"] = tlist
        _write_tasks_json_safely(tlist)

    # -------- Projects & Assets plumbing --------
    def _project_options():
        """Return (labels, pids) from proj_df for UI pickers."""
        opts, pids = [], []
        if 'proj_df' in globals() and isinstance(proj_df, pd.DataFrame) and not proj_df.empty:
            rows = proj_df.copy().reset_index(drop=True)
            rows["__ord"] = rows.index + 1
            rows["pid"]   = rows["project_id"].astype("string").fillna("-1").apply(_safe_int)
            rows["__label"] = rows.apply(
                lambda r: f"{int(r['__ord'])} — {r.get('project_number','PRJ-????')} — {r.get('name','(no name)')} [{r.get('status','')}]",
                axis=1
            )
            for _, r in rows.iterrows():
                opts.append(r["__label"])
                pids.append(int(r["pid"]))
        return opts, pids

    def _label_to_pid(label: str, opts: list, pids: list) -> int:
        try:
            return int(pids[opts.index(label)])
        except Exception:
            return 0

    def _known_assets_list() -> list:
        """All known asset tags from session (preferred) or from fmeca_assets table."""
        ss_assets = (st.session_state.get("assets") or {})
        if isinstance(ss_assets, dict) and ss_assets:
            return sorted(list(ss_assets.keys()))
        try:
            df = pd.read_sql_query("SELECT DISTINCT asset_tag FROM fmeca_assets ORDER BY asset_tag", db.conn)
            return sorted(df["asset_tag"].dropna().astype(str).tolist())
        except Exception:
            return []

    def _get_project_linked_assets(pid: int) -> list:
        """Pull linked assets from your Projects store (projects list preferred, fallback to proj_df)."""
        assets = []
        # Try projects list (authoritative JSON)
        try:
            for p in (globals().get("projects") or []):
                if int(p.get("project_id",-1)) == int(pid):
                    assets = list(p.get("linked_assets") or [])
                    break
        except Exception:
            pass
        # Fallback to proj_df column if available
        if not assets and 'proj_df' in globals() and isinstance(proj_df, pd.DataFrame) and not proj_df.empty:
            row = proj_df[proj_df["project_id"].astype("string").fillna("-1").astype(int)==int(pid)]
            if not row.empty:
                cand = row.iloc[0].get("linked_assets", [])
                try:
                    if isinstance(cand, str):
                        cand = json.loads(cand)
                except Exception:
                    pass
                if isinstance(cand, (list, tuple, set)):
                    assets = list(cand)
        # Normalize
        out = sorted({str(a).strip() for a in (assets or []) if str(a).strip()})
        return out

    # -------- FMECA assets (many-to-many) --------
    def _get_assets_for_fmeca(fid: int) -> list:
        try:
            df = pd.read_sql_query("SELECT asset_tag FROM fmeca_assets WHERE fmeca_id=? ORDER BY asset_tag",
                                   db.conn, params=[int(fid)])
            return df["asset_tag"].dropna().astype(str).tolist()
        except Exception:
            return []

    def _set_assets_for_fmeca(fid: int, assets: list):
        """Replace asset set for a FMECA (idempotent)."""
        aset = sorted({str(a).strip() for a in (assets or []) if str(a).strip()})
        cur = db.conn.cursor()
        cur.execute("DELETE FROM fmeca_assets WHERE fmeca_id=?", (int(fid),))
        if aset:
            cur.executemany("INSERT OR IGNORE INTO fmeca_assets (fmeca_id, asset_tag) VALUES (?, ?)",
                            [(int(fid), a) for a in aset])
        db.conn.commit()

    # ================== LOAD/SAVE ==================
    @st.cache_data(ttl=15, show_spinner=False)
    def _load_fmecas(rev: int) -> pd.DataFrame:
        try:
            df = pd.read_sql_query("SELECT * FROM fmecas ORDER BY fmeca_id DESC", db.conn)
            if "created_at" in df.columns: df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
            if "closed_at"  in df.columns: df["closed_at"]  = pd.to_datetime(df["closed_at"],  errors="coerce")
            return df
        except Exception as e:
            st.warning(f"Failed to load FMECAs: {e}")
            return pd.DataFrame(columns=["fmeca_id","linked_asset","linked_project","type","description","created_at","closed_at","version","risk_matrix_json"])

    def _save_fmeca(fid: int, data: dict) -> int | None:
        """Insert or update FMECA. Returns fmeca_id or None."""
        try:
            cur = db.conn.cursor()
            if fid and int(fid) > 0:
                cur.execute("""
                    UPDATE fmecas SET
                        linked_asset=?, linked_project=?, type=?, description=?,
                        closed_at=?, version=version+1, risk_matrix_json=?
                    WHERE fmeca_id=?
                """, (
                    data.get("linked_asset"),
                    data.get("linked_project"),
                    data.get("type"),
                    data.get("description"),
                    data.get("closed_at"),
                    json.dumps(data.get("risk_matrix") or []),
                    int(fid)
                ))
                db.conn.commit()
                return int(fid)
            else:
                cur.execute("""
                    INSERT INTO fmecas (linked_asset, linked_project, type, description, closed_at, risk_matrix_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    data.get("linked_asset"),
                    data.get("linked_project"),
                    data.get("type"),
                    data.get("description"),
                    data.get("closed_at"),
                    json.dumps(data.get("risk_matrix") or [])
                ))
                db.conn.commit()
                return int(cur.lastrowid)
        except Exception as e:
            st.error(f"Failed to save FMECA: {e}")
            return None

    def _delete_fmeca(fid: int):
        try:
            db.conn.execute("DELETE FROM fmecas WHERE fmeca_id=?", (int(fid),))
            db.conn.execute("DELETE FROM fmeca_assets WHERE fmeca_id=?", (int(fid),))
            db.conn.commit()
        except Exception as e:
            st.error(f"Failed to delete FMECA: {e}")

    # ================== RISK MATRIX ==================
    RM_COLS = [
        "Asset", "Function/Item", "Failure Mode", "Effect", "Cause",
        "Severity", "Occurrence", "Detection", "RPN",
        "Recommended Actions", "Revised Severity", "Revised Occurrence", "Revised Detection", "Revised RPN"
    ]
    NUM_COLS = {"Severity","Occurrence","Detection","RPN","Revised Severity","Revised Occurrence","Revised Detection","Revised RPN"}

    def compute_rpn(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=RM_COLS)
        df = df.copy()
        for c in ["Severity","Occurrence","Detection","Revised Severity","Revised Occurrence","Revised Detection"]:
            if c not in df.columns: df[c] = 1
        for c in ["Severity","Occurrence","Detection","Revised Severity","Revised Occurrence","Revised Detection"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1).clip(1,10).astype(int)
        df["RPN"]         = df["Severity"] * df["Occurrence"] * df["Detection"]
        df["Revised RPN"] = df["Revised Severity"] * df["Revised Occurrence"] * df["Revised Detection"]
        return df

    def _ensure_rm_schema(df: pd.DataFrame, default_asset: str | None = None) -> pd.DataFrame:
        """Guarantee columns; fill defaults; recompute RPNs; ensure 'Asset' and 'Recommended Actions' exist."""
        if df is None or df.empty:
            base = pd.DataFrame(columns=RM_COLS)
            if default_asset:
                base["Asset"] = [default_asset]
            return base
        df = df.copy()
        for c in RM_COLS:
            if c not in df.columns:
                df[c] = 1 if c in NUM_COLS else ""
        # Ensure Asset
        df["Asset"] = df["Asset"].astype(str).fillna("").replace("nan","")
        if default_asset:
            df.loc[df["Asset"].str.strip()=="", "Asset"] = default_asset
        # numeric safety then RPN
        for c in ["Severity","Occurrence","Detection","Revised Severity","Revised Occurrence","Revised Detection"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1).clip(1, 10).astype(int)
        return compute_rpn(df)

    def _dedup_union_rows(df: pd.DataFrame, add_df: pd.DataFrame) -> pd.DataFrame:
        """Append rows but avoid duplicates by key (Asset, Function/Item, Failure Mode)."""
        if df is None or df.empty:
            return add_df.copy()
        df = df.copy(); add_df = add_df.copy()
        df["_key"] = df.apply(lambda r: (str(r.get("Asset","")).strip(),
                                         str(r.get("Function/Item","")).strip(),
                                         str(r.get("Failure Mode","")).strip()), axis=1)
        add_df["_key"] = add_df.apply(lambda r: (str(r.get("Asset","")).strip(),
                                                 str(r.get("Function/Item","")).strip(),
                                                 str(r.get("Failure Mode","")).strip()), axis=1)
        exist = set(df["_key"].tolist())
        add_df = add_df[~add_df["_key"].isin(exist)].drop(columns=["_key"])
        df = df.drop(columns=["_key"])
        return pd.concat([df, add_df], ignore_index=True)

    def _coalesce_cols(df: pd.DataFrame, candidates: list) -> pd.Series:
        """Row-wise first non-empty among candidates (string)."""
        if df is None or df.empty:
            return pd.Series([], dtype="object")
        s = pd.Series([""] * len(df), index=df.index, dtype="object")
        for c in candidates:
            if c in df.columns:
                vals = df[c].astype(str).fillna("")
                mask = s.astype(str).str.strip() == ""
                s = s.mask(mask, vals)
        return s.fillna("")

    # ================== SESSION INIT ==================
    if "fmeca_df" not in st.session_state:
        st.session_state.fmeca_df = _load_fmecas(st.session_state.get("data_rev", 0))
    fmeca_df = st.session_state.fmeca_df

    # ================== TOP ACTIONS ==================
    r1, r2 = st.columns([1,1])
    if r1.button("🔄 Refresh"):
        st.session_state.fmeca_df = _load_fmecas(st.session_state.get("data_rev", 0))
        st.rerun()
    if r2.button("🧱 Ensure DB Schema/Indexes"):
        _init_fmeca_schema()
        st.success("Schema & indexes ensured.")

    # ================== SUB-TABS ==================
    tab_list, tab_add, tab_edit = st.tabs(["🔍 List FMECAs", "➕ Add New", "✏️ Edit Selected"])

    # ---------- LIST ----------
    with tab_list:
        if fmeca_df.empty:
            st.info("No FMECAs yet. Use 'Add New' to create one.")
        else:
            c1, c2, c3 = st.columns([1.2, 1.2, 2])
            proj_opts_labels, proj_opts_ids = _project_options()
            proj_filter = c1.selectbox("Filter by Project", options=["(All)"] + proj_opts_labels, index=0)
            type_filter = c2.multiselect("Type", options=sorted(fmeca_df["type"].dropna().astype(str).unique().tolist()))
            search = c3.text_input("Search by ID, Asset, or Description")

            filtered = fmeca_df.copy()
            if proj_filter != "(All)" and proj_opts_labels:
                pid = _label_to_pid(proj_filter, proj_opts_labels, proj_opts_ids)
                filtered = filtered[filtered["linked_project"].fillna(0).astype(int) == int(pid)]
            if type_filter:
                filtered = filtered[filtered["type"].astype(str).isin(type_filter)]
            if search:
                mask = (
                    filtered["fmeca_id"].astype(str).str.contains(search, case=False) |
                    filtered["linked_asset"].astype(str).str.contains(search, case=False) |
                    filtered["description"].astype(str).str.contains(search, case=False)
                )
                filtered = filtered[mask]

            st.dataframe(
                filtered[["fmeca_id","linked_project","type","linked_asset","created_at","closed_at","version"]],
                use_container_width=True,
                height=_auto_table_height(len(filtered)),
                hide_index=True
            )
            st.download_button(
                "⬇️ Download Filtered FMECAs (CSV)",
                data=filtered.to_csv(index=False),
                file_name="fmecas.csv"
            )

    # ---------- ADD NEW ----------
    with tab_add:
        st.markdown("### Add New FMECA")

        labels, pids = _project_options()
        default_pid = st.session_state.get("t7_selected_pid")
        idx_default = 0
        if labels and default_pid in pids:
            idx_default = pids.index(default_pid) + 1  # +1 for "(None)"
        col1, col2 = st.columns([1.2, 2])

        with col1:
            proj_label = st.selectbox("Linked Project", options=["(None)"] + labels, index=idx_default, key="fm_add_proj_lbl")
            new_proj = _label_to_pid(proj_label, labels, pids) if proj_label != "(None)" else None

        with col2:
            known_assets = _known_assets_list()
            suggested = _get_project_linked_assets(new_proj) if new_proj else []
            add_assets = st.multiselect(
                "Linked Assets (multi-select)",
                options=known_assets,
                default=[a for a in suggested if a in known_assets],
                key="fm_add_assets"
            )

        c3, c4 = st.columns([1,1])
        with c3:
            new_type = st.selectbox("Type", ["Design","Process","System","Other"], key="fm_add_type")
        with c4:
            new_closed = st.date_input("Closed Date (optional)", value=None, key="fm_add_closed")

        new_desc = st.text_area("Description", key="fm_add_desc", placeholder="Purpose/scope of this FMECA...")

        # ----- Risk Matrix (Unified, Multi-Asset) -----
        st.markdown("### Risk Matrix (multi-asset)")
        state_key_add = "fm_add_rm_state"
        if state_key_add not in st.session_state:
            st.session_state[state_key_add] = pd.DataFrame(columns=RM_COLS)

        if add_assets:
            a1, a2 = st.columns([1.2, 2])
            with a1:
                add_asset_filter = st.selectbox("Filter rows by asset", options=["(All Assets)"] + add_assets, key="fm_add_asset_filter")
            with a2:
                seed_cols = st.columns(max(1, min(4, len(add_assets))))
                for i, atag in enumerate(add_assets):
                    if seed_cols[i % len(seed_cols)].button(f"📥 Seed BOM for {atag}", key=f"fm_add_seed_{atag}"):
                        seed_df = pd.DataFrame(columns=RM_COLS)
                        try:
                            assets_store = (st.session_state.get("assets") or {})
                            aobj = assets_store.get(atag, {})
                            if 'build_bom_table' in globals():
                                bom = build_bom_table(aobj)
                            else:
                                bom = pd.DataFrame()
                            if isinstance(bom, pd.DataFrame) and not bom.empty:
                                failure_modes = ["Wear","Leakage","Fracture"]
                                effects       = ["Reduced efficiency","Fluid loss","System failure"]
                                causes        = ["Aging","Overpressure","Material defect"]
                                descr_col = "DESCRIPTION" if "DESCRIPTION" in bom.columns else (bom.columns[0] if not bom.empty else "Item")
                                n = len(bom)
                                seed_df = pd.DataFrame({
                                    "Asset":        [atag]*n,
                                    "Function/Item": bom[descr_col].astype(str).fillna(""),
                                    "Failure Mode":  [failure_modes[i % len(failure_modes)] for i in range(n)],
                                    "Effect":        [effects[i % len(effects)]       for i in range(n)],
                                    "Cause":         [causes[i % len(causes)]         for i in range(n)],
                                    "Severity":      [5]*n, "Occurrence":[3]*n, "Detection":[4]*n,
                                    "Recommended Actions": ["" for _ in range(n)],
                                    "Revised Severity":[5]*n, "Revised Occurrence":[3]*n, "Revised Detection":[4]*n
                                })
                                seed_df = compute_rpn(seed_df)
                        except Exception:
                            pass
                        merged = _dedup_union_rows(_ensure_rm_schema(st.session_state[state_key_add]), _ensure_rm_schema(seed_df, default_asset=atag))
                        st.session_state[state_key_add] = _ensure_rm_schema(merged)
                        st.rerun()
        else:
            add_asset_filter = "(All Assets)"
            st.info("Select one or more assets to enable BOM seeding and per-asset filtering.")

        base_df = _ensure_rm_schema(st.session_state[state_key_add], default_asset=(add_assets[0] if add_assets else None))
        view_df = base_df[base_df["Asset"] == add_asset_filter].copy() if add_asset_filter != "(All Assets)" else base_df.copy()

        rm_cfg_add = {
            "Asset":                st.column_config.SelectboxColumn(options=add_assets or _known_assets_list(), help="Row's asset."),
            "Function/Item":        st.column_config.TextColumn(),
            "Failure Mode":         st.column_config.TextColumn(),
            "Effect":               st.column_config.TextColumn(),
            "Cause":                st.column_config.TextColumn(),
            "Severity":             st.column_config.SelectboxColumn(options=list(range(1,11)), default=5),
            "Occurrence":           st.column_config.SelectboxColumn(options=list(range(1,11)), default=3),
            "Detection":            st.column_config.SelectboxColumn(options=list(range(1,11)), default=4),
            "RPN":                  st.column_config.NumberColumn(disabled=True),
            "Recommended Actions":  st.column_config.TextColumn(),
            "Revised Severity":     st.column_config.SelectboxColumn(options=list(range(1,11)), default=5),
            "Revised Occurrence":   st.column_config.SelectboxColumn(options=list(range(1,11)), default=3),
            "Revised Detection":    st.column_config.SelectboxColumn(options=list(range(1,11)), default=4),
            "Revised RPN":          st.column_config.NumberColumn(disabled=True),
        }

        edited_add = st.data_editor(
            view_df,
            column_config=rm_cfg_add,
            num_rows="dynamic",
            use_container_width=True,
            height=280,
            key="fm_add_rm_editor"
        )

        merged_all = pd.concat([base_df[base_df["Asset"] != add_asset_filter], edited_add], ignore_index=True) if add_asset_filter != "(All Assets)" else edited_add.copy()
        st.session_state[state_key_add] = _ensure_rm_schema(merged_all)

        if st.button("💾 Save New FMECA"):
            la = (add_assets[0] if add_assets else None)  # legacy single-asset column
            data = {
                "linked_asset": la,
                "linked_project": int(new_proj) if new_proj else None,
                "type": new_type,
                "description": (new_desc or "").strip(),
                "closed_at": str(new_closed) if new_closed else None,
                "risk_matrix": _ensure_rm_schema(st.session_state[state_key_add]).to_dict("records")
            }
            new_id = _save_fmeca(0, data)
            if new_id:
                _set_assets_for_fmeca(new_id, add_assets)
                if new_proj:
                    st.session_state["t7_selected_pid"] = int(new_proj)
                st.success(f"Added FMECA ID {new_id}.")
                st.session_state.fmeca_df = _load_fmecas(st.session_state.get("data_rev", 0))
                st.session_state.pop(state_key_add, None)
                try:
                    _bump_rev()
                except Exception:
                    pass
                st.rerun()

    # ---------- EDIT ----------
    with tab_edit:
        if fmeca_df.empty:
            st.info("No FMECAs to edit.")
        else:
            sel_id = st.selectbox("Select FMECA ID", options=sorted(fmeca_df["fmeca_id"].tolist()), key="fm_edit_sel")
            row = fmeca_df[fmeca_df["fmeca_id"] == sel_id].iloc[0] if sel_id else None

            if row is not None:
                # MIGRATION: If legacy linked_asset exists but fmeca_assets empty, migrate it.
                try:
                    if (not _get_assets_for_fmeca(int(sel_id))) and (str(row.get("linked_asset") or "").strip()):
                        _set_assets_for_fmeca(int(sel_id), [str(row["linked_asset"]).strip()])
                except Exception:
                    pass

                st.markdown(f"### Editing FMECA {int(sel_id)} (v{int(row['version'])})")
                labels, pids = _project_options()
                cur_pid = _safe_int(row["linked_project"], 0)
                cur_idx = (pids.index(cur_pid) + 1) if (cur_pid and cur_pid in pids) else 0

                c1, c2 = st.columns([1.2, 2])
                with c1:
                    edit_proj_label = st.selectbox("Linked Project", options=["(None)"] + labels, index=cur_idx, key=f"fm_edit_proj_label_{sel_id}")
                    edit_proj = _label_to_pid(edit_proj_label, labels, pids) if edit_proj_label != "(None)" else None
                with c2:
                    known_assets = _known_assets_list()
                    proj_assets = _get_project_linked_assets(edit_proj) if edit_proj else []
                    current_assets = _get_assets_for_fmeca(int(sel_id))
                    default_assets = sorted({*(current_assets or []), *proj_assets})
                    edit_assets = st.multiselect("Linked Assets (multi-select)",
                                                options=known_assets,
                                                default=[a for a in default_assets if a in known_assets],
                                                key=f"fm_edit_assets_{sel_id}")

                c3, c4 = st.columns([1,1])
                with c3:
                    types = ["Design","Process","System","Other"]
                    t_idx = types.index(str(row["type"])) if str(row["type"]) in types else 2
                    edit_type = st.selectbox("Type", types, index=t_idx, key=f"fm_edit_type_{sel_id}")
                with c4:
                    edit_closed = st.date_input(
                        "Closed Date",
                        value=pd.to_datetime(row["closed_at"]).date() if not pd.isna(row["closed_at"]) else None,
                        key=f"fm_edit_closed_{sel_id}"
                    )

                edit_desc = st.text_area("Description", value=str(row["description"] or ""), key=f"fm_edit_desc_{sel_id}")

                # -------- Risk Matrix (Editor with Asset filter + BOM seeding) --------
                st.markdown("### Risk Matrix")
                try:
                    rm_data = json.loads(row.get("risk_matrix_json") or "[]")
                except Exception:
                    rm_data = []
                base_rm = _ensure_rm_schema(pd.DataFrame(rm_data), default_asset=(edit_assets[0] if edit_assets else None))

                state_key_edit = f"fm_edit_rm_state_{sel_id}"
                if state_key_edit not in st.session_state:
                    st.session_state[state_key_edit] = base_rm.copy()
                else:
                    st.session_state[state_key_edit] = _ensure_rm_schema(st.session_state[state_key_edit])

                if edit_assets:
                    a1, a2 = st.columns([1.2, 2])
                    with a1:
                        edit_asset_filter = st.selectbox("Filter rows by asset",
                                                         options=["(All Assets)"] + edit_assets,
                                                         key=f"fm_edit_asset_filter_{sel_id}")
                    with a2:
                        seed_cols = st.columns(max(1, min(4, len(edit_assets))))
                        for i, atag in enumerate(edit_assets):
                            if seed_cols[i % len(seed_cols)].button(f"📥 Seed BOM for {atag}", key=f"fm_edit_seed_{sel_id}_{atag}"):
                                seed_df = pd.DataFrame(columns=RM_COLS)
                                try:
                                    assets_store = (st.session_state.get("assets") or {})
                                    aobj = assets_store.get(atag, {})
                                    if 'build_bom_table' in globals():
                                        bom = build_bom_table(aobj)
                                    else:
                                        bom = pd.DataFrame()
                                    if isinstance(bom, pd.DataFrame) and not bom.empty:
                                        failure_modes = ["Wear","Leakage","Fracture"]
                                        effects       = ["Reduced efficiency","Fluid loss","System failure"]
                                        causes        = ["Aging","Overpressure","Material defect"]
                                        descr_col = "DESCRIPTION" if "DESCRIPTION" in bom.columns else (bom.columns[0] if not bom.empty else "Item")
                                        n = len(bom)
                                        seed_df = pd.DataFrame({
                                            "Asset":        [atag]*n,
                                            "Function/Item": bom[descr_col].astype(str).fillna(""),
                                            "Failure Mode":  [failure_modes[i % len(failure_modes)] for i in range(n)],
                                            "Effect":        [effects[i % len(effects)]       for i in range(n)],
                                            "Cause":         [causes[i % len(causes)]         for i in range(n)],
                                            "Severity":      [5]*n, "Occurrence":[3]*n, "Detection":[4]*n,
                                            "Recommended Actions": ["" for _ in range(n)],
                                            "Revised Severity":[5]*n, "Revised Occurrence":[3]*n, "Revised Detection":[4]*n
                                        })
                                        seed_df = compute_rpn(seed_df)
                                except Exception:
                                    pass
                                merged = _dedup_union_rows(_ensure_rm_schema(st.session_state[state_key_edit]), _ensure_rm_schema(seed_df, default_asset=atag))
                                st.session_state[state_key_edit] = _ensure_rm_schema(merged)
                                st.rerun()
                else:
                    edit_asset_filter = "(All Assets)"
                    st.info("Select one or more assets to enable BOM seeding and per-asset filtering.")

                base_df = _ensure_rm_schema(st.session_state[state_key_edit], default_asset=(edit_assets[0] if edit_assets else None))
                view_df = base_df[base_df["Asset"] == edit_asset_filter].copy() if edit_asset_filter != "(All Assets)" else base_df.copy()

                rm_cfg_edit = {
                    "Asset":                st.column_config.SelectboxColumn(options=edit_assets or _known_assets_list(), help="Row's asset."),
                    "Function/Item":        st.column_config.TextColumn(),
                    "Failure Mode":         st.column_config.TextColumn(),
                    "Effect":               st.column_config.TextColumn(),
                    "Cause":                st.column_config.TextColumn(),
                    "Severity":             st.column_config.SelectboxColumn(options=list(range(1,11)), default=5),
                    "Occurrence":           st.column_config.SelectboxColumn(options=list(range(1,11)), default=3),
                    "Detection":            st.column_config.SelectboxColumn(options=list(range(1,11)), default=4),
                    "RPN":                  st.column_config.NumberColumn(disabled=True),
                    "Recommended Actions":  st.column_config.TextColumn(),
                    "Revised Severity":     st.column_config.SelectboxColumn(options=list(range(1,11)), default=5),
                    "Revised Occurrence":   st.column_config.SelectboxColumn(options=list(range(1,11)), default=3),
                    "Revised Detection":    st.column_config.SelectboxColumn(options=list(range(1,11)), default=4),
                    "Revised RPN":          st.column_config.NumberColumn(disabled=True),
                }

                edited_view = st.data_editor(
                    view_df,
                    column_config=rm_cfg_edit,
                    num_rows="dynamic",
                    use_container_width=True,
                    height=280,
                    key=f"fm_edit_rm_editor_{sel_id}"
                )
                merged_all = pd.concat([base_df[base_df["Asset"] != edit_asset_filter], edited_view], ignore_index=True) if edit_asset_filter != "(All Assets)" else edited_view.copy()
                st.session_state[state_key_edit] = _ensure_rm_schema(merged_all)

                # -------- Visualizations --------
                with st.expander("📊 Risk Analytics"):
                    viz_df = _ensure_rm_schema(st.session_state[state_key_edit])
                    if edit_asset_filter != "(All Assets)":
                        viz_df = viz_df[viz_df["Asset"] == edit_asset_filter]
                    if not viz_df.empty:
                        try:
                            pivot = viz_df.pivot_table(values="RPN", index="Severity", columns="Occurrence", aggfunc="mean").fillna(0)
                            fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="Reds"))
                            fig_heat.update_layout(title="Risk Heatmap (Severity vs Occurrence)", height=300)
                            st.plotly_chart(fig_heat, use_container_width=True)
                        except Exception:
                            pass
                        try:
                            pareto = viz_df.sort_values("RPN", ascending=False).head(12)
                            fig_pareto = px.bar(pareto, x="Failure Mode", y="RPN", title="Top Risks by RPN")
                            st.plotly_chart(fig_pareto, use_container_width=True)
                        except Exception:
                            pass
                        try:
                            mitigated = (viz_df["Revised RPN"] < viz_df["RPN"]).sum() / len(viz_df) * 100 if len(viz_df) > 0 else 0
                            st.metric("Mitigated Risks (%)", f"{mitigated:.1f}%")
                            high_risks = int((viz_df["RPN"] > 100).sum())
                            st.metric("High Risks (RPN > 100)", high_risks)
                        except Exception:
                            pass

                # -------- Generate Tasks from Actions (safe IDs + de-dup + renumber) --------
                def _task_ids_for_project_list(tlist, pid):
                    ids = []
                    for t in (tlist or []):
                        if _safe_int(t.get("project_id"), -1) == int(pid):
                            tid = t.get("task_id")
                            if tid is not None and str(tid).strip() != "":
                                try:
                                    ids.append(int(tid))
                                except Exception:
                                    pass
                    return sorted(set(ids))

                def _next_task_id_for_project_local(tlist, pid):
                    used = _task_ids_for_project_list(tlist, pid)
                    n = 1
                    for u in used:
                        if u == n:
                            n += 1
                        elif u > n:
                            break
                    return n

                def _renumber_tasks_for_project_local(tlist, pid):
                    """Reassign IDs 1..N for this project (stable order: start_date asc, name asc, old_id)."""
                    pid = int(pid)
                    idxs = [i for i, t in enumerate(tlist) if _safe_int(t.get("project_id"), -1) == pid]
                    if not idxs:
                        return

                    def _pdate_coerce(x):
                        try: return pd.to_datetime(x, errors="coerce")
                        except Exception: return pd.NaT

                    sortable = []
                    for i in idxs:
                        t = tlist[i]
                        sd = _pdate_coerce(t.get("start_date"))
                        nm = str(t.get("name", "")).lower()
                        old = _safe_int(t.get("task_id"), 10**9)
                        key = (sd if pd.notna(sd) else pd.Timestamp.max, nm, old)
                        sortable.append((key, i, old))

                    sortable.sort(key=lambda z: z[0])

                    mapping, new_id = {}, 1
                    for _key, i, old in sortable:
                        mapping[old] = new_id
                        new_id += 1

                    for _key, i, old in sortable:
                        tlist[i]["task_id"] = mapping.get(old, _safe_int(tlist[i].get("task_id"), 10**9))

                    for i in idxs:
                        deps = tlist[i].get("dependencies", [])
                        if not isinstance(deps, list): deps = []
                        new_deps = []
                        for d in deps:
                            did = _safe_int(d, None)
                            if did is not None and did in mapping:
                                new_deps.append(mapping[did])
                        tlist[i]["dependencies"] = sorted(set(new_deps))

                def _action_fingerprint(fmeca_id, asset_tag, failure_mode, desc):
                    return "|".join([
                        f"fmeca:{int(fmeca_id)}",
                        f"asset:{str(asset_tag).strip().lower()}",
                        f"fm:{str(failure_mode).strip().lower()}",
                        f"desc:{str(desc).strip().lower()}"
                    ])

                if (edit_proj is not None) and not _ensure_rm_schema(st.session_state[state_key_edit]).empty:
                    if st.button("🛠️ Generate Tasks from Recommended Actions", key=f"fm_gen_tasks_{sel_id}"):

                        full_df = _ensure_rm_schema(st.session_state[state_key_edit])
                        if edit_asset_filter != "(All Assets)":
                            full_df = full_df[full_df["Asset"] == edit_asset_filter].copy()

                        action_series = _coalesce_cols(
                            full_df,
                            ["Recommended Actions", "Action", "Mitigation", "Recommendation", "Corrective Action"]
                        )
                        mask = action_series.astype(str).str.strip() != ""
                        actions = full_df[mask].reset_index(drop=True)
                        action_series = action_series[mask].reset_index(drop=True)

                        if actions.empty:
                            st.info("No recommended actions found in the matrix.")
                        else:
                            tlist = _tasks_store_load()

                            existing_fps = set()
                            for t in tlist:
                                if _safe_int(t.get("project_id"), -1) == _safe_int(edit_proj, -1):
                                    src = t.get("source", {}) or {}
                                    if isinstance(src, dict) and _safe_int(src.get("fmeca_id"), None) == _safe_int(sel_id, None):
                                        fp = src.get("fingerprint")
                                        if fp:
                                            existing_fps.add(str(fp))

                            next_id = _next_task_id_for_project_local(tlist, int(edit_proj))
                            new_tasks = []
                            today_str = datetime.now().strftime("%Y-%m-%d")
                            due_str   = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

                            for i, act in actions.iterrows():
                                asset_tag = str(act.get("Asset","")).strip() or "Unknown Asset"
                                fm = str(act.get("Failure Mode","Issue")).strip() or "Issue"
                                desc = str(action_series.iloc[i]).strip()

                                fp = _action_fingerprint(sel_id, asset_tag, fm, desc)
                                if fp in existing_fps:
                                    continue

                                rpn = _safe_int(act.get("RPN"), 0)
                                pri = "High" if rpn > 100 else "Medium"

                                new_tasks.append({
                                    "task_id": next_id,         # assign real int ID now
                                    "project_id": int(edit_proj),
                                    "parent_task_id": None,
                                    "name": f"FMECA {sel_id} — {asset_tag}: Mitigate {fm}",
                                    "description": desc,
                                    "status": "To Do",
                                    "priority": pri,
                                    "assignees": [],
                                    "start_date": today_str,
                                    "due_date":   due_str,
                                    "estimate_h": 0.0,
                                    "actual_h":   0.0,
                                    "dependencies": [],
                                    "milestones": [],
                                    "comments": [],
                                    "links": [],
                                    "attachments": [],
                                    "completed_at": "",
                                    "source": { "type": "FMECA", "fmeca_id": int(sel_id), "fingerprint": fp }
                                })
                                next_id += 1
                                existing_fps.add(fp)

                            if not new_tasks:
                                st.info("No new tasks to add (duplicates were skipped).")
                            else:
                                tlist.extend(new_tasks)
                                _renumber_tasks_for_project_local(tlist, int(edit_proj))
                                _tasks_store_set(tlist)

                                st.session_state["t7_selected_pid"] = int(edit_proj)
                                st.success(f"Generated {len(new_tasks)} task(s) in Project {int(edit_proj)} from FMECA {sel_id}.")
                                try:
                                    st.toast("Open Tasks & Milestones to review these tasks.", icon="✅")
                                except Exception:
                                    pass

                # -------- Save / Delete / Cross-links --------
                b1, b2, b3, b4 = st.columns([1,1,1,1])
                with b1:
                    if st.button("💾 Update FMECA", key=f"fm_update_{sel_id}"):
                        df_to_save = _ensure_rm_schema(st.session_state[state_key_edit])
                        legacy_la = (edit_assets[0] if edit_assets else (str(row.get("linked_asset") or "").strip() or None))
                        data = {
                            "linked_asset": legacy_la,
                            "linked_project": int(edit_proj) if edit_proj is not None else None,
                            "type": edit_type,
                            "description": (edit_desc or "").strip(),
                            "closed_at": str(edit_closed) if edit_closed else None,
                            "risk_matrix": df_to_save.to_dict("records")
                        }
                        saved_id = _save_fmeca(int(sel_id), data)
                        if saved_id:
                            _set_assets_for_fmeca(int(sel_id), edit_assets)
                            if edit_proj is not None:
                                st.session_state["t7_selected_pid"] = int(edit_proj)
                            st.success("FMECA updated.")
                            st.session_state.fmeca_df = _load_fmecas(st.session_state.get("data_rev", 0))
                            try:
                                _bump_rev()
                            except Exception:
                                pass
                            st.rerun()
                with b2:
                    if st.button("🗑️ Delete FMECA", key=f"fm_del_{sel_id}"):
                        st.session_state[f"fm_del_confirm_{sel_id}"] = True
                    if st.session_state.get(f"fm_del_confirm_{sel_id}", False):
                        if st.button("✔️ Confirm Delete", key=f"fm_del_do_{sel_id}"):
                            _delete_fmeca(int(sel_id))
                            st.success("Deleted.")
                            st.session_state.fmeca_df = _load_fmecas(st.session_state.get("data_rev", 0))
                            st.session_state.pop(state_key_edit, None)
                            try:
                                _bump_rev()
                            except Exception:
                                pass
                            st.rerun()
                with b3:
                    if edit_proj is not None and st.button("↪️ Open in Tasks & Milestones", key=f"fm_open_tasks_{sel_id}"):
                        st.session_state["t7_selected_pid"] = int(edit_proj)
                        st.success("Project selected for Tasks & Milestones. Switch to that subtab.")
                with b4:
                    if edit_proj and st.button("🔗 Add these assets to Project links", key=f"fm_sync_proj_assets_{sel_id}"):
                        try:
                            changed = False
                            for p in (globals().get("projects") or []):
                                if int(p.get("project_id",-1)) == int(edit_proj):
                                    cur = list(p.get("linked_assets") or [])
                                    aset = sorted({*(cur or []), *(edit_assets or [])})
                                    if aset != cur:
                                        p["linked_assets"] = aset
                                        p["updated_at"] = datetime.now().isoformat(timespec="seconds")
                                        changed = True
                                    break
                            if changed:
                                _write_json(PROJECTS_JSON, projects)
                                st.success("Project linked assets updated.")
                            else:
                                st.info("No changes to project linked assets.")
                        except Exception as e:
                            st.warning(f"Unable to update project links: {e}")

    # ---------- Getting Started ----------
    if st.session_state.fmeca_df.empty:
        with st.expander("Getting Started with FMECA"):
            st.markdown("""
            - **Link a project** → assets auto-suggest.
            - **Link multiple assets** to the same FMECA.
            - Click **“Seed BOM for <asset>”** to add rows from that asset’s BOM (no duplicates).
            - Use **S/O/D** (1–10) → **RPN = S×O×D**. Add **Recommended Actions**.
            - **Generate Tasks** to push actions into your Tasks & Milestones for the selected project.
            """)

