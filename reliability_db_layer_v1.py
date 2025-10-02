"""
reliability_db_layer_v1.py

Phase 1 deliverable for the Site Reliability App (v3.0.0) upgrade:
- SQLite schema (normalized enough for speed, flexible enough for ADD-never-remove)
- Safe connection & pragmas
- Idempotent schema init
- Migration from legacy JSON/CSV (assets.json, runtime.csv, history.csv, jobs_index.json + jobs/*.json)
- Central read/write service (row-level UPSERTs; no full-table hacks)
- Input validation helpers (non-negative, date ≤ today, unique keys)
- Attachment registry (metadata table) while keeping files on disk

This module is importable from Streamlit. Example (in your app):

    from reliability_db_layer_v1 import DB
    db = DB(db_path="data/reliability.db", data_dir="data")
    db.init_schema()
    # Optional one-time migration
    report = db.migrate_from_legacy()
    st.success(report.summary())

    # Read examples
    assets_df = db.assets_list(department=dept_filter, search=search_text)
    runtime_df = db.runtime_for_tags(tags=[...], date_from=None, date_to=None)

    # Write examples
    db.assets_upsert([{"tag": "242-PP-011", "department": "Pumps", ...}])
    db.runtime_upsert_rows(rows)  # rows = list[dict]
    db.history_upsert_rows(rows)

Notes
- Designed for Africa/Johannesburg site-local dates (naive date stored as TEXT ISO YYYY-MM-DD).
- Uses ONLY Python stdlib + pandas; no ORMs to keep deployment simple.
- All writes are transactional; validation runs before commit.
- Jobs kept in DB; attachments kept on disk with metadata rows.

"""
from __future__ import annotations
import os
import json
import csv
import hashlib
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional, Sequence

import sqlite3
import pandas as pd

# -------------------------------
# Small utilities
# -------------------------------

def _today_str() -> str:
    return date.today().isoformat()


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------
# Validation rules (centralized)
# -------------------------------
class ValidationError(Exception):
    pass


class Validator:
    @staticmethod
    def non_negative(v, field: str):
        if v is None:
            return
        try:
            if float(v) < 0:
                raise ValidationError(f"{field} must be ≥ 0")
        except (TypeError, ValueError):
            raise ValidationError(f"{field} must be numeric")

    @staticmethod
    def date_not_future(d: str, field: str):
        if not d:
            return
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").date()
        except ValueError:
            raise ValidationError(f"{field} must be YYYY-MM-DD")
        if dt > date.today():
            raise ValidationError(f"{field} cannot be in the future")


# -------------------------------
# Migration report dataclass
# -------------------------------
@dataclass
class MigrationReport:
    assets_in: int = 0
    assets_ok: int = 0
    runtime_in: int = 0
    runtime_ok: int = 0
    history_in: int = 0
    history_ok: int = 0
    jobs_in: int = 0
    jobs_ok: int = 0
    backup_dir: Optional[str] = None

    def summary(self) -> str:
        return (
            f"Migration complete.\n"
            f"Assets: {self.assets_ok}/{self.assets_in}\n"
            f"Runtime: {self.runtime_ok}/{self.runtime_in}\n"
            f"History: {self.history_ok}/{self.history_in}\n"
            f"Jobs: {self.jobs_ok}/{self.jobs_in}\n"
            f"Backup: {self.backup_dir or '-'}"
        )


# -------------------------------
# DB service
# -------------------------------
class DB:
    def __init__(self, db_path: str = "data/reliability.db", data_dir: str = "data"):
        self.db_path = db_path
        self.data_dir = data_dir
        _ensure_dir(os.path.dirname(self.db_path))
        _ensure_dir(self.data_dir)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode = WAL;")
        con.execute("PRAGMA synchronous = NORMAL;")
        con.execute("PRAGMA foreign_keys = ON;")
        return con

    # ---------------------------
    # Schema
    # ---------------------------
    def init_schema(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            # assets (JSON details for flexibility)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS assets (
                  tag TEXT PRIMARY KEY,
                  func_location TEXT,
                  section TEXT,
                  department TEXT,
                  type TEXT,
                  model TEXT,
                  criticality TEXT DEFAULT 'Medium',
                  details_json TEXT,
                  created_at TEXT DEFAULT (date('now')),
                  updated_at TEXT DEFAULT (date('now'))
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_assets_dept ON assets(department, criticality);")

            # components (optional granular MTBF targets per asset)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS components (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tag TEXT NOT NULL REFERENCES assets(tag) ON DELETE CASCADE,
                  name TEXT NOT NULL,
                  mtbf_target_h REAL,
                  notes TEXT,
                  UNIQUE(tag, name)
                );
                """
            )

            # bom_items
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bom_items (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tag TEXT NOT NULL REFERENCES assets(tag) ON DELETE CASCADE,
                  cmms_code TEXT,
                  oem_part TEXT,
                  description TEXT,
                  uom TEXT,
                  std_price REAL,
                  default_qty REAL,
                  UNIQUE(tag, cmms_code, oem_part, description)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_bom_tag ON bom_items(tag);")

            # runtime (row-level upserts)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tag TEXT NOT NULL REFERENCES assets(tag) ON DELETE CASCADE,
                  component_name TEXT,
                  date TEXT NOT NULL,
                  mtbf_h REAL,
                  hours_since REAL,
                  remaining_h REAL,
                  status TEXT,
                  last_overhaul TEXT,
                  UNIQUE(tag, component_name, date)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_runtime_tag_date ON runtime(tag, date);")

            # history (work orders & events)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tag TEXT NOT NULL REFERENCES assets(tag) ON DELETE CASCADE,
                  wo TEXT,
                  date TEXT NOT NULL,
                  pm_code TEXT,
                  downtime_h REAL,
                  labour_h REAL,
                  spares_cost REAL,
                  qty REAL,
                  notes TEXT,
                  attachments_json TEXT,
                  UNIQUE(tag, wo, date)
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_history_tag_date ON history(tag, date);")
            cur.execute("CREATE INDEX IF NOT EXISTS ix_history_wo ON history(wo);")

            # jobs + job_spares
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT UNIQUE,
                  tag TEXT REFERENCES assets(tag),
                  wo TEXT,
                  title TEXT,
                  status TEXT,
                  planned_start TEXT,
                  planned_end TEXT,
                  kitting_ready_pct REAL,
                  logistics_json TEXT,
                  created_at TEXT DEFAULT (datetime('now')),
                  updated_at TEXT DEFAULT (datetime('now'))
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS job_spares (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_id TEXT NOT NULL,
                  bom_item_id INTEGER,
                  pick_qty REAL,
                  price_at_pick REAL,
                  line_total REAL,
                  FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
                  FOREIGN KEY(bom_item_id) REFERENCES bom_items(id) ON DELETE SET NULL
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS ix_job_spares_job ON job_spares(job_id);")

            # attachments registry (files stay on disk)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS attachments (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  owner_table TEXT NOT NULL,
                  owner_id INTEGER NOT NULL,
                  name TEXT,
                  mime TEXT,
                  path TEXT,
                  size_b INTEGER,
                  checksum TEXT
                );
                """
            )

            # cm_observations / failure_log (Tab 6)
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS cm_observations (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tag TEXT REFERENCES assets(tag),
                  date TEXT,
                  severity TEXT,
                  type TEXT,
                  description TEXT,
                  src TEXT,
                  parsed_src_id TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_log (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  tag TEXT REFERENCES assets(tag),
                  date TEXT,
                  failure_mode TEXT,
                  cause TEXT,
                  detection TEXT,
                  downtime_h REAL,
                  pm_code TEXT,
                  costs_json TEXT
                );
                """
            )
            con.commit()

    # ---------------------------
    # MIGRATION from legacy files
    # ---------------------------
    def migrate_from_legacy(self) -> MigrationReport:
        rep = MigrationReport()
        backup_dir = os.path.join(self.data_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        _ensure_dir(backup_dir)
        rep.backup_dir = backup_dir

        assets_json = os.path.join(self.data_dir, "assets.json")
        runtime_csv = os.path.join(self.data_dir, "runtime.csv")
        history_csv = os.path.join(self.data_dir, "history.csv")
        jobs_dir = os.path.join(self.data_dir, "jobs")
        jobs_index_json = os.path.join(self.data_dir, "jobs_index.json")

        with self._connect() as con:
            cur = con.cursor()
            # Assets
            if os.path.exists(assets_json):
                shutil.copy2(assets_json, os.path.join(backup_dir, "assets.json"))
                with open(assets_json, "r", encoding="utf-8") as f:
                    assets = json.load(f) or {}
                rep.assets_in = len(assets)
                for tag, a in assets.items():
                    rec = {
                        "tag": tag,
                        "func_location": a.get("FL") or a.get("Functional_Location"),
                        "section": str(tag)[:3] if tag else None,
                        "department": a.get("Department") or a.get("Dept") or "Pumps",
                        "type": a.get("Type"),
                        "model": a.get("Model"),
                        "criticality": a.get("Criticality") or "Medium",
                        "details_json": json.dumps(a, ensure_ascii=False),
                    }
                    self._upsert(con, "assets", rec, keys=["tag"]) ; rep.assets_ok += 1

            # Runtime
            if os.path.exists(runtime_csv):
                shutil.copy2(runtime_csv, os.path.join(backup_dir, "runtime.csv"))
                df = pd.read_csv(runtime_csv)
                rep.runtime_in = len(df)
                for _, r in df.iterrows():
                    tag = str(r.get("Asset Tag") or r.get("Asset_Tag") or "").strip()
                    if not tag:
                        continue
                    comp = str(r.get("Component") or r.get("component_name") or "").strip() or None
                    dt = str(r.get("Date") or r.get("date") or _today_str())[:10]
                    try:
                        Validator.date_not_future(dt, "runtime.date")
                        Validator.non_negative(r.get("MTBF (Hours)"), "MTBF (Hours)")
                        Validator.non_negative(r.get("Running Hours Since Last Major Maintenance"), "Hours Since")
                    except ValidationError:
                        continue
                    mtbf = float(r.get("MTBF (Hours)") or 0)
                    hours_since = float(r.get("Running Hours Since Last Major Maintenance") or 0)
                    remaining = max(mtbf - hours_since, 0)
                    status = str(r.get("STATUS") or ("Overdue" if remaining <= 0 else "OK"))
                    rec = {
                        "tag": tag,
                        "component_name": comp,
                        "date": dt,
                        "mtbf_h": mtbf,
                        "hours_since": hours_since,
                        "remaining_h": remaining,
                        "status": status,
                        "last_overhaul": (str(r.get("Last Overhaul"))[:10] if r.get("Last Overhaul") else None),
                    }
                    self._upsert(con, "runtime", rec, keys=["tag", "component_name", "date"]) ; rep.runtime_ok += 1

            # History
            if os.path.exists(history_csv):
                shutil.copy2(history_csv, os.path.join(backup_dir, "history.csv"))
                dfh = pd.read_csv(history_csv)
                rep.history_in = len(dfh)
                for _, r in dfh.iterrows():
                    tag = str(r.get("Asset Tag") or r.get("Asset_Tag") or "").strip()
                    if not tag:
                        continue
                    wo = str(r.get("WO Number") or r.get("WO_Number") or r.get("WO") or "").strip() or None
                    dt = str(r.get("Date of Maintenance") or r.get("Date") or _today_str())[:10]
                    try:
                        Validator.date_not_future(dt, "history.date")
                        Validator.non_negative(r.get("Asset Downtime Hours"), "Downtime Hours")
                        Validator.non_negative(r.get("Labour Hours"), "Labour Hours")
                        Validator.non_negative(r.get("QTY"), "QTY")
                    except ValidationError:
                        continue
                    rec = {
                        "tag": tag,
                        "wo": wo,
                        "date": dt,
                        "pm_code": str(r.get("Maintenance Code") or r.get("PM Code") or "").strip() or None,
                        "downtime_h": float(r.get("Asset Downtime Hours") or 0),
                        "labour_h": float(r.get("Labour Hours") or 0),
                        "spares_cost": float(r.get("Spares Cost") or 0),
                        "qty": float(r.get("QTY") or 0),
                        "notes": str(r.get("Notes and Findings") or r.get("Notes") or "").strip() or None,
                        "attachments_json": None,
                    }
                    self._upsert(con, "history", rec, keys=["tag", "wo", "date"]) ; rep.history_ok += 1

            # Jobs (index + job files) ➜ DB
            if os.path.isdir(jobs_dir):
                # backup folder
                backup_jobs = os.path.join(backup_dir, "jobs")
                _ensure_dir(backup_jobs)
                rep.jobs_in = 0
                rep.jobs_ok = 0
                for root, _, files in os.walk(jobs_dir):
                    for fn in files:
                        if not fn.lower().endswith(".json"):
                            continue
                        src = os.path.join(root, fn)
                        dst = os.path.join(backup_jobs, os.path.relpath(src, jobs_dir))
                        _ensure_dir(os.path.dirname(dst))
                        shutil.copy2(src, dst)
                        try:
                            data = json.load(open(src, "r", encoding="utf-8"))
                        except Exception:
                            continue
                        rep.jobs_in += 1
                        job_id = data.get("job_id") or os.path.splitext(fn)[0]
                        rec = {
                            "job_id": job_id,
                            "tag": data.get("tag"),
                            "wo": data.get("wo"),
                            "title": data.get("title"),
                            "status": data.get("status"),
                            "planned_start": (data.get("planned_start") or "")[:19],
                            "planned_end": (data.get("planned_end") or "")[:19],
                            "kitting_ready_pct": float(data.get("kitting_ready_pct") or 0),
                            "logistics_json": json.dumps(data.get("logistics") or {}, ensure_ascii=False),
                        }
                        self._upsert(con, "jobs", rec, keys=["job_id"]) ; rep.jobs_ok += 1
                        # job spares if present
                        for s in (data.get("spares") or []):
                            sp = {
                                "job_id": job_id,
                                "bom_item_id": None,  # unknown mapping at migration time
                                "pick_qty": float(s.get("Pick QTY") or s.get("pick_qty") or 0),
                                "price_at_pick": float(s.get("Price") or s.get("price") or 0),
                                "line_total": float(s.get("Line Total") or s.get("line_total") or 0),
                            }
                            self._insert(con, "job_spares", sp)

            # Write migration log
            with open(os.path.join(backup_dir, "migration_log.json"), "w", encoding="utf-8") as f:
                json.dump(rep.__dict__, f, indent=2)

            con.commit()
        return rep

    # ---------------------------
    # READ API (DataFrames)
    # ---------------------------
    def assets_list(self, department: Optional[str] = None, criticality: Optional[str] = None, search: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT * FROM assets WHERE 1=1"
        params = []
        if department:
            q += " AND department = ?"; params.append(department)
        if criticality:
            q += " AND criticality = ?"; params.append(criticality)
        if search:
            q += " AND (tag LIKE ? OR func_location LIKE ? OR model LIKE ?)"
            s = f"%{search}%"; params += [s, s, s]
        with self._connect() as con:
            return pd.read_sql_query(q, con, params=params)

    def runtime_for_tags(self, tags: Sequence[str], date_from: Optional[str] = None, date_to: Optional[str] = None) -> pd.DataFrame:
        if not tags:
            return pd.DataFrame()
        q = "SELECT * FROM runtime WHERE tag IN (" + ",".join(["?"]*len(tags)) + ")"
        params = list(tags)
        if date_from:
            q += " AND date >= ?"; params.append(date_from)
        if date_to:
            q += " AND date <= ?"; params.append(date_to)
        q += " ORDER BY tag, date"
        with self._connect() as con:
            return pd.read_sql_query(q, con, params=params)

    def history_for_tags(self, tags: Sequence[str], date_from: Optional[str] = None, date_to: Optional[str] = None) -> pd.DataFrame:
        if not tags:
            return pd.DataFrame()
        q = "SELECT * FROM history WHERE tag IN (" + ",".join(["?"]*len(tags)) + ")"
        params = list(tags)
        if date_from:
            q += " AND date >= ?"; params.append(date_from)
        if date_to:
            q += " AND date <= ?"; params.append(date_to)
        q += " ORDER BY date DESC"
        with self._connect() as con:
            return pd.read_sql_query(q, con, params=params)

    # ---------------------------
    # WRITE API (row-level)
    # ---------------------------
    def assets_upsert(self, rows: Iterable[dict]) -> int:
        with self._connect() as con:
            n = 0
            for r in rows:
                r = dict(r)
                if not r.get("tag"):
                    continue
                if not r.get("criticality"):
                    r["criticality"] = "Medium"
                r.setdefault("updated_at", _today_str())
                self._upsert(con, "assets", r, keys=["tag"]) ; n += 1
            con.commit()
            return n

    def runtime_upsert_rows(self, rows: Iterable[dict]) -> int:
        with self._connect() as con:
            n = 0
            for r in rows:
                tag = (r.get("tag") or r.get("Asset Tag") or r.get("Asset_Tag") or "").strip()
                if not tag:
                    continue
                comp = (r.get("component_name") or r.get("Component") or None)
                dt = (r.get("date") or r.get("Date") or _today_str())[:10]
                mtbf = float(r.get("mtbf_h") or r.get("MTBF (Hours)") or 0)
                hours_since = float(r.get("hours_since") or r.get("Running Hours Since Last Major Maintenance") or 0)
                remaining = max(mtbf - hours_since, 0)
                status = r.get("status") or ("Overdue" if remaining <= 0 else "OK")
                try:
                    Validator.date_not_future(dt, "runtime.date")
                    Validator.non_negative(mtbf, "MTBF (Hours)")
                    Validator.non_negative(hours_since, "Hours Since")
                except ValidationError:
                    continue
                rec = {
                    "tag": tag,
                    "component_name": comp,
                    "date": dt,
                    "mtbf_h": mtbf,
                    "hours_since": hours_since,
                    "remaining_h": remaining,
                    "status": status,
                    "last_overhaul": (str(r.get("last_overhaul"))[:10] if r.get("last_overhaul") else None),
                }
                self._upsert(con, "runtime", rec, keys=["tag", "component_name", "date"]) ; n += 1
            con.commit()
            return n

    def history_upsert_rows(self, rows: Iterable[dict]) -> int:
        with self._connect() as con:
            n = 0
            for r in rows:
                tag = (r.get("tag") or r.get("Asset Tag") or r.get("Asset_Tag") or "").strip()
                if not tag:
                    continue
                wo = (r.get("wo") or r.get("WO Number") or r.get("WO_Number") or r.get("WO") or None)
                dt = (r.get("date") or r.get("Date") or r.get("Date of Maintenance") or _today_str())[:10]
                pm = (r.get("pm_code") or r.get("PM Code") or r.get("Maintenance Code") or None)
                downtime = float(r.get("downtime_h") or r.get("Asset Downtime Hours") or 0)
                labour = float(r.get("labour_h") or r.get("Labour Hours") or 0)
                qty = float(r.get("qty") or r.get("QTY") or 0)
                notes = (r.get("notes") or r.get("Notes and Findings") or r.get("Notes") or None)
                try:
                    Validator.date_not_future(dt, "history.date")
                    Validator.non_negative(downtime, "Downtime Hours")
                    Validator.non_negative(labour, "Labour Hours")
                    Validator.non_negative(qty, "QTY")
                except ValidationError:
                    continue
                rec = {
                    "tag": tag,
                    "wo": wo,
                    "date": dt,
                    "pm_code": pm,
                    "downtime_h": downtime,
                    "labour_h": labour,
                    "spares_cost": float(r.get("spares_cost") or 0),
                    "qty": qty,
                    "notes": notes,
                    "attachments_json": json.dumps(r.get("attachments") or [], ensure_ascii=False) if r.get("attachments") else None,
                }
                self._upsert(con, "history", rec, keys=["tag", "wo", "date"]) ; n += 1
            con.commit()
            return n

    # ---------------------------
    # Low-level helpers
    # ---------------------------
    @staticmethod
    def _insert(con: sqlite3.Connection, table: str, rec: dict):
        keys = list(rec.keys())
        q = f"INSERT INTO {table} (" + ",".join(keys) + ") VALUES (" + ",".join(["?"]*len(keys)) + ")"
        con.execute(q, [rec[k] for k in keys])

    @staticmethod
    def _upsert(con: sqlite3.Connection, table: str, rec: dict, keys: Sequence[str]):
        # Build an INSERT ... ON CONFLICT(...) DO UPDATE SET ... statement
        all_cols = list(rec.keys())
        placeholders = ",".join(["?"] * len(all_cols))
        insert_q = f"INSERT INTO {table} (" + ",".join(all_cols) + ") VALUES (" + placeholders + ")"
        conflict_cols = ",".join(keys)
        update_cols = [c for c in all_cols if c not in keys]
        if not update_cols:
            on_conflict = f" ON CONFLICT({conflict_cols}) DO NOTHING"
            con.execute(insert_q + on_conflict, [rec[c] for c in all_cols])
            return
        set_clause = ",".join([f"{c}=excluded.{c}" for c in update_cols])
        on_conflict = f" ON CONFLICT({conflict_cols}) DO UPDATE SET {set_clause}"
        con.execute(insert_q + on_conflict, [rec[c] for c in all_cols])


if __name__ == "__main__":
    # Simple smoke test (does not run in Streamlit context)
    db = DB()
    db.init_schema()
    print("Schema initialized at:", db.db_path)
