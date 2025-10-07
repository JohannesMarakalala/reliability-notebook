"""Flet implementation of the VIGIL reliability application.

The original code base was a Streamlit prototype with a large collection of
widgets, authentication helpers and data synchronisation logic.  The goal of
this rewrite is to provide the same data model (SQLite/JSON backed assets,
runtime trackers and maintenance history) but expose it through a modern
Flet UI so it can be packaged as a desktop or web experience.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import flet as ft
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ASSETS_JSON = os.path.join(DATA_DIR, "assets.json")
RUNTIME_CSV = os.path.join(DATA_DIR, "runtime.csv")
HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")
USERS_JSON = os.path.join(DATA_DIR, "users.json")
DB_PATH = os.path.join(DATA_DIR, "reliability.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "attachments"), exist_ok=True)

DEFAULT_CONFIG: Dict[str, Any] = {
    "default_asset_filter": [],
    "craft_presets": [
        "Mechanical Technician",
        "Electrician",
        "Instrument Technician",
        "Rigger",
        "Boilermaker",
        "Planner / Scheduler",
        "Reliability Engineer",
        "Other",
    ],
}


# ---------------------------------------------------------------------------
# Authentication helpers (ported from the original Streamlit prototype)
# ---------------------------------------------------------------------------
def _load_users() -> List[dict]:
    try:
        if os.path.exists(USERS_JSON):
            with open(USERS_JSON, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
        return []
    except Exception:
        return []


def _save_users(users: List[dict]) -> None:
    os.makedirs(os.path.dirname(USERS_JSON), exist_ok=True)
    with open(USERS_JSON, "w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2, ensure_ascii=False)


def _hash_pbkdf2_sha256(password: str, salt: bytes) -> str:
    import hashlib

    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000).hex()


def _create_password_hash(password: str, salt_hex: Optional[str] = None) -> tuple[str, str]:
    salt_hex = salt_hex or secrets.token_hex(16)
    try:
        salt_bytes = bytes.fromhex(salt_hex)
    except ValueError:
        salt_bytes = salt_hex.encode("utf-8")
    return salt_hex, _hash_pbkdf2_sha256(password, salt_bytes)


def _find_user(email: str) -> Optional[dict]:
    for record in _load_users():
        if record.get("email", "").strip().lower() == email.strip().lower():
            return record
    return None


def register_user(name: str, email: str, password: str) -> tuple[bool, str]:
    name = name.strip()
    email = email.strip().lower()
    password = password or ""

    if not name or not email or not password:
        return False, "Please provide name, email and password."
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email format."
    if _find_user(email):
        return False, "An account with this email already exists."

    salt, password_hash = _create_password_hash(password)
    users = _load_users()
    users.append(
        {
            "name": name,
            "email": email,
            "salt": salt,
            "hash": password_hash,
            "algo": "pbkdf2_sha256_v1",
            "created": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save_users(users)
    return True, "Account created successfully."


def authenticate(email: str, password: str) -> tuple[bool, Optional[dict], str]:
    account = _find_user(email)
    if not account:
        return False, None, "Account not found."

    algo = account.get("algo", "pbkdf2_sha256_v1")
    salt = account.get("salt", "")
    stored_hash = account.get("hash", "")

    try:
        salt_bytes = bytes.fromhex(salt)
    except ValueError:
        salt_bytes = salt.encode("utf-8")

    if algo == "sha256_legacy":
        import hashlib

        candidate = hashlib.sha256((str(salt) + password).encode("utf-8")).hexdigest()
    else:
        candidate = _hash_pbkdf2_sha256(password, salt_bytes)

    if candidate == stored_hash:
        return True, {"name": account.get("name", ""), "email": account.get("email", "")}, "Welcome back!"
    return False, None, "Invalid credentials."


# ---------------------------------------------------------------------------
# SQLite helper – keeps the assets table in sync with JSON edits
# ---------------------------------------------------------------------------
class AssetsDB:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS assets (
                tag TEXT PRIMARY KEY,
                details_json TEXT
            );
            """
        )
        self.conn.commit()

    def assets_as_dict(self) -> Dict[str, dict]:
        try:
            cursor = self.conn.execute("SELECT tag, details_json FROM assets")
        except sqlite3.Error:
            return {}
        assets: Dict[str, dict] = {}
        for tag, payload in cursor.fetchall():
            try:
                details = json.loads(payload or "{}")
            except json.JSONDecodeError:
                details = {}
            assets[tag] = details if isinstance(details, dict) else {}
        return assets

    def sync(self, assets: Dict[str, dict]) -> None:
        with self.conn:
            existing = {row[0] for row in self.conn.execute("SELECT tag FROM assets")}
            desired = set(map(str, assets.keys()))

            for tag in desired:
                details = assets.get(tag, {})
                try:
                    payload = json.dumps(details, ensure_ascii=False, default=str)
                except TypeError:
                    payload = json.dumps(dict(details), ensure_ascii=False, default=str)
                self.conn.execute(
                    "INSERT INTO assets(tag, details_json) VALUES(?, ?) "
                    "ON CONFLICT(tag) DO UPDATE SET details_json = excluded.details_json",
                    (tag, payload),
                )

            for obsolete in existing - desired:
                self.conn.execute("DELETE FROM assets WHERE tag = ?", (obsolete,))


# ---------------------------------------------------------------------------
# Data access layer for the Flet application
# ---------------------------------------------------------------------------
@dataclass
class AssetRecord:
    tag: str
    description: str = ""
    functional_location: str = ""
    criticality: str = ""
    manufacturer: str = ""
    model: str = ""

    @classmethod
    def from_mapping(cls, tag: str, payload: Dict[str, Any]) -> "AssetRecord":
        return cls(
            tag=tag,
            description=str(payload.get("description", "")),
            functional_location=str(payload.get("functional_location", "")),
            criticality=str(payload.get("criticality", "")),
            manufacturer=str(payload.get("manufacturer", "")),
            model=str(payload.get("model", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "functional_location": self.functional_location,
            "criticality": self.criticality,
            "manufacturer": self.manufacturer,
            "model": self.model,
        }


class AppData:
    def __init__(self) -> None:
        self.db = AssetsDB(DB_PATH)
        self.assets: Dict[str, AssetRecord] = {}
        self.runtime_df = pd.DataFrame()
        self.history_df = pd.DataFrame()
        self.load_assets()
        self.load_runtime()
        self.load_history()

    # ---------------------- Assets ----------------------
    def load_assets(self) -> None:
        assets: Dict[str, dict] = {}
        if os.path.exists(ASSETS_JSON):
            try:
                with open(ASSETS_JSON, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    assets.update({str(k): dict(v) for k, v in payload.items()})
            except Exception:
                pass

        db_assets = self.db.assets_as_dict()
        if db_assets:
            assets.update(db_assets)
        elif assets:
            self.db.sync(assets)

        self.assets = {tag: AssetRecord.from_mapping(tag, meta) for tag, meta in assets.items()}

    def save_assets(self) -> None:
        serialised = {tag: record.to_dict() for tag, record in self.assets.items()}
        with open(ASSETS_JSON, "w", encoding="utf-8") as handle:
            json.dump(serialised, handle, indent=2, ensure_ascii=False)
        self.db.sync(serialised)

    def upsert_asset(self, record: AssetRecord) -> None:
        self.assets[record.tag] = record
        self.save_assets()

    def query_assets(self, search: str = "") -> List[AssetRecord]:
        search = search.strip().lower()
        records = list(self.assets.values())
        if not search:
            return sorted(records, key=lambda r: r.tag)
        return sorted(
            [
                asset
                for asset in records
                if search in asset.tag.lower()
                or search in asset.description.lower()
                or search in asset.functional_location.lower()
            ],
            key=lambda r: r.tag,
        )

    def asset_summary(self) -> Dict[str, Any]:
        total = len(self.assets)
        critical = sum(1 for record in self.assets.values() if record.criticality.lower() in {"critical", "high"})
        manufacturers = {record.manufacturer for record in self.assets.values() if record.manufacturer}
        return {
            "total": total,
            "critical": critical,
            "manufacturers": len(manufacturers),
        }

    # ---------------------- Runtime ----------------------
    def load_runtime(self) -> None:
        if os.path.exists(RUNTIME_CSV):
            try:
                self.runtime_df = pd.read_csv(RUNTIME_CSV)
            except Exception:
                self.runtime_df = pd.DataFrame()
        else:
            self.runtime_df = pd.DataFrame(
                columns=["asset_tag", "run_hours", "criticality", "updated_at"]
            )

    def load_history(self) -> None:
        if os.path.exists(HISTORY_CSV):
            try:
                self.history_df = pd.read_csv(HISTORY_CSV)
            except Exception:
                self.history_df = pd.DataFrame()
        else:
            self.history_df = pd.DataFrame(
                columns=["work_order", "asset_tag", "performed_on", "summary", "technician"]
            )

    def runtime_summary(self) -> Dict[str, Any]:
        df = self.runtime_df
        if df.empty:
            return {"avg": 0, "max": 0, "min": 0}
        return {
            "avg": float(df.get("run_hours", pd.Series(dtype=float)).mean() or 0),
            "max": float(df.get("run_hours", pd.Series(dtype=float)).max() or 0),
            "min": float(df.get("run_hours", pd.Series(dtype=float)).min() or 0),
        }


# ---------------------------------------------------------------------------
# Flet application implementation
# ---------------------------------------------------------------------------
class ReliabilityApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.page.title = "VIGIL Reliability"
        self.page.horizontal_alignment = ft.CrossAxisAlignment.STRETCH
        self.page.vertical_alignment = ft.MainAxisAlignment.START
        self.page.scroll = ft.ScrollMode.AUTO
        self.page.theme_mode = ft.ThemeMode.LIGHT

        self.data = AppData()
        self.beta_ok = False
        self.user: Optional[dict] = None
        self.auth_mode = "login"

        self.root = ft.Column(expand=True, alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        self.page.add(self.root)

        self.show_beta_gate()

    # ---------------------- Helpers ----------------------
    def show_snack(self, message: str, *, error: bool = False) -> None:
        self.page.snack_bar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor="#dc2626" if error else "#2563eb",
            open=True,
        )
        self.page.update()

    # ---------------------- Beta gate ----------------------
    def show_beta_gate(self) -> None:
        description = ft.Text(
            "VIGIL • Public Beta v0.2 — enter your access code to continue.",
            text_align=ft.TextAlign.CENTER,
        )
        self.beta_code = ft.TextField(
            label="Beta access code",
            password=True,
            autofocus=True,
            on_submit=self.handle_beta_submit,
            width=320,
        )
        enter_button = ft.FilledButton("Enter", on_click=self.handle_beta_submit, width=320)

        self.root.controls = [
            ft.Container(
                content=ft.Column(
                    [
                        ft.Icon(name=ft.icons.CONSTRUCTION, size=42, color="#0ea5e9"),
                        description,
                        self.beta_code,
                        enter_button,
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=16,
                ),
                padding=40,
                border_radius=20,
                bgcolor=ft.colors.with_opacity(0.85, ft.colors.BLUE_GREY_50),
            )
        ]
        self.page.update()

    def handle_beta_submit(self, event: ft.ControlEvent) -> None:
        expected = os.environ.get("BETA_ACCESS_CODE", "").strip()
        provided = (self.beta_code.value or "").strip()
        if expected and provided != expected:
            self.show_snack("Incorrect access code", error=True)
            return
        self.beta_ok = True
        self.show_auth_view()

    # ---------------------- Authentication view ----------------------
    def show_auth_view(self) -> None:
        self.email_field = ft.TextField(label="Email", width=320, on_submit=self.handle_auth_submit)
        self.password_field = ft.TextField(
            label="Password",
            password=True,
            can_reveal_password=True,
            width=320,
            on_submit=self.handle_auth_submit,
        )
        self.name_field = ft.TextField(label="Full name", width=320)

        toggle_button = ft.TextButton("Need an account? Sign up", on_click=self.toggle_auth_mode)
        self.auth_submit = ft.FilledButton("Sign in", on_click=self.handle_auth_submit, width=320)

        self.auth_card = ft.Container(
            content=ft.Column(
                [
                    ft.Text("VIGIL Login", size=24, weight=ft.FontWeight.BOLD),
                    self.name_field,
                    self.email_field,
                    self.password_field,
                    self.auth_submit,
                    toggle_button,
                ],
                spacing=14,
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=40,
            border_radius=20,
            bgcolor=ft.colors.with_opacity(0.92, ft.colors.WHITE),
            width=420,
        )

        self.root.controls = [self.auth_card]
        self.toggle_auth_mode(None)  # initialise visibility
        self.page.update()

    def toggle_auth_mode(self, event: Optional[ft.ControlEvent]) -> None:
        if self.auth_mode == "login":
            self.auth_mode = "register"
            self.auth_card.content.controls[0].value = "Create account"
            self.name_field.visible = True
            self.auth_submit.text = "Sign up"
            if isinstance(self.auth_card.content.controls[-1], ft.TextButton):
                self.auth_card.content.controls[-1].text = "Already have an account? Sign in"
        else:
            self.auth_mode = "login"
            self.auth_card.content.controls[0].value = "VIGIL Login"
            self.name_field.visible = False
            self.auth_submit.text = "Sign in"
            if isinstance(self.auth_card.content.controls[-1], ft.TextButton):
                self.auth_card.content.controls[-1].text = "Need an account? Sign up"
        self.page.update()

    def handle_auth_submit(self, event: Optional[ft.ControlEvent]) -> None:
        email = self.email_field.value or ""
        password = self.password_field.value or ""
        if self.auth_mode == "register":
            name = self.name_field.value or ""
            ok, message = register_user(name, email, password)
            if ok:
                self.show_snack(message)
                self.auth_mode = "login"
                self.toggle_auth_mode(None)
            else:
                self.show_snack(message, error=True)
        else:
            ok, payload, message = authenticate(email, password)
            if ok:
                self.user = payload
                self.show_snack(message)
                self.show_main_view()
            else:
                self.show_snack(message, error=True)

    # ---------------------- Main application view ----------------------
    def show_main_view(self) -> None:
        header = ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text("VIGIL Reliability", size=26, weight=ft.FontWeight.BOLD),
                            ft.Text(
                                f"Logged in as {self.user.get('name', '')}"
                                if self.user
                                else "",
                                size=14,
                                color=ft.colors.BLUE_GREY,
                            ),
                        ]
                    ),
                    ft.FilledButton("Sign out", on_click=self.handle_sign_out),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.only(left=12, right=12, top=16, bottom=8),
        )

        self.tabs = ft.Tabs(
            tabs=[
                ft.Tab(text="Overview", icon=ft.icons.DASHBOARD, content=self.build_overview_tab()),
                ft.Tab(text="Assets", icon=ft.icons.STORAGE, content=self.build_assets_tab()),
                ft.Tab(text="Runtime", icon=ft.icons.TIMER, content=self.build_runtime_tab()),
                ft.Tab(text="History", icon=ft.icons.HISTORY, content=self.build_history_tab()),
            ],
            expand=True,
        )

        self.root.controls = [header, ft.Container(content=self.tabs, expand=True)]
        self.page.update()

    def handle_sign_out(self, event: ft.ControlEvent) -> None:
        self.user = None
        self.show_snack("Signed out successfully.")
        self.show_auth_view()

    # ---------------------- Overview ----------------------
    def build_metric_card(self, title: str, value: str, icon: str, color: str) -> ft.Container:
        return ft.Container(
            content=ft.Column(
                [
                    ft.Icon(icon, color=color, size=28),
                    ft.Text(title, size=14, color=ft.colors.BLUE_GREY_400),
                    ft.Text(value, size=24, weight=ft.FontWeight.BOLD),
                ],
                spacing=6,
            ),
            padding=20,
            border_radius=16,
            bgcolor=ft.colors.with_opacity(0.8, ft.colors.WHITE),
            width=220,
        )

    def build_overview_tab(self) -> ft.Control:
        summary = self.data.asset_summary()
        runtime = self.data.runtime_summary()
        body = ft.Column(
            [
                ft.ResponsiveRow(
                    [
                        ft.Container(
                            self.build_metric_card("Assets", str(summary.get("total", 0)), ft.icons.INVENTORY_2, "#0ea5e9"),
                            col={'xs': 12, 'md': 4},
                        ),
                        ft.Container(
                            self.build_metric_card("Critical assets", str(summary.get("critical", 0)), ft.icons.WARNING, "#f97316"),
                            col={'xs': 12, 'md': 4},
                        ),
                        ft.Container(
                            self.build_metric_card("Vendors", str(summary.get("manufacturers", 0)), ft.icons.FACTORY, "#14b8a6"),
                            col={'xs': 12, 'md': 4},
                        ),
                    ]
                ),
                ft.ResponsiveRow(
                    [
                        ft.Container(
                            self.build_metric_card(
                                "Average runtime hours",
                                f"{runtime.get('avg', 0):,.1f}",
                                ft.icons.SPEED,
                                "#6366f1",
                            ),
                            col={'xs': 12, 'md': 4},
                        ),
                        ft.Container(
                            self.build_metric_card(
                                "Max runtime hours",
                                f"{runtime.get('max', 0):,.1f}",
                                ft.icons.TRENDING_UP,
                                "#10b981",
                            ),
                            col={'xs': 12, 'md': 4},
                        ),
                        ft.Container(
                            self.build_metric_card(
                                "Min runtime hours",
                                f"{runtime.get('min', 0):,.1f}",
                                ft.icons.TRENDING_DOWN,
                                "#ef4444",
                            ),
                            col={'xs': 12, 'md': 4},
                        ),
                    ]
                ),
                ft.Divider(),
                ft.Text("Recently updated runtime records", size=18, weight=ft.FontWeight.BOLD),
                self._build_runtime_preview_table(limit=5),
            ],
            spacing=24,
            scroll=ft.ScrollMode.AUTO,
        )
        return ft.Container(content=body, padding=20, expand=True)

    def _build_runtime_preview_table(self, limit: int | None = None) -> ft.Control:
        df = self.data.runtime_df
        if df.empty:
            return ft.Text("No runtime entries available yet.")
        preview = df.sort_values(by="updated_at", ascending=False).head(limit) if "updated_at" in df.columns else df.head(limit)
        columns = [
            ft.DataColumn(ft.Text("Asset")),
            ft.DataColumn(ft.Text("Run hours")),
            ft.DataColumn(ft.Text("Criticality")),
            ft.DataColumn(ft.Text("Last update")),
        ]
        rows = [
            ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(str(row.get("asset_tag", "")))),
                    ft.DataCell(ft.Text(f"{row.get('run_hours', 0):,.1f}" if pd.notna(row.get("run_hours")) else "")),
                    ft.DataCell(ft.Text(str(row.get("criticality", "")))),
                    ft.DataCell(ft.Text(str(row.get("updated_at", "")))),
                ]
            )
            for _, row in preview.iterrows()
        ]
        return ft.DataTable(columns=columns, rows=rows, heading_row_color=ft.colors.BLUE_GREY_50)

    # ---------------------- Assets tab ----------------------
    def build_assets_tab(self) -> ft.Control:
        self.asset_search = ft.TextField(
            label="Search assets",
            prefix_icon=ft.icons.SEARCH,
            on_change=self.refresh_asset_table,
            width=320,
        )
        self.asset_table = ft.DataTable(columns=self._asset_table_columns(), rows=[])
        self.refresh_asset_table(None)

        add_button = ft.FilledButton("Add asset", icon=ft.icons.ADD, on_click=self.open_add_asset_dialog)

        body = ft.Column(
            [
                ft.Row([self.asset_search, add_button], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(self.asset_table, expand=True),
            ],
            spacing=16,
        )
        return ft.Container(content=body, padding=20, expand=True)

    def _asset_table_columns(self) -> List[ft.DataColumn]:
        return [
            ft.DataColumn(ft.Text("Tag")),
            ft.DataColumn(ft.Text("Description")),
            ft.DataColumn(ft.Text("Location")),
            ft.DataColumn(ft.Text("Criticality")),
            ft.DataColumn(ft.Text("Manufacturer")),
            ft.DataColumn(ft.Text("Model")),
        ]

    def refresh_asset_table(self, event: Optional[ft.ControlEvent]) -> None:
        search = self.asset_search.value if hasattr(self, "asset_search") else ""
        rows: List[ft.DataRow] = []
        for record in self.data.query_assets(search or ""):
            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(record.tag)),
                        ft.DataCell(ft.Text(record.description)),
                        ft.DataCell(ft.Text(record.functional_location)),
                        ft.DataCell(ft.Text(record.criticality)),
                        ft.DataCell(ft.Text(record.manufacturer)),
                        ft.DataCell(ft.Text(record.model)),
                    ]
                )
            )
        self.asset_table.rows = rows
        self.page.update()

    def open_add_asset_dialog(self, event: ft.ControlEvent) -> None:
        self.asset_tag_field = ft.TextField(label="Tag", autofocus=True)
        self.asset_description_field = ft.TextField(label="Description")
        self.asset_location_field = ft.TextField(label="Functional location")
        self.asset_criticality_field = ft.Dropdown(
            label="Criticality",
            options=[ft.dropdown.Option(v) for v in ["Low", "Medium", "High", "Critical"]],
            value="Medium",
        )
        self.asset_manufacturer_field = ft.TextField(label="Manufacturer")
        self.asset_model_field = ft.TextField(label="Model")

        dialog_content = ft.Column(
            [
                self.asset_tag_field,
                self.asset_description_field,
                self.asset_location_field,
                self.asset_criticality_field,
                self.asset_manufacturer_field,
                self.asset_model_field,
            ],
            tight=True,
        )

        self.asset_dialog = ft.AlertDialog(
            title=ft.Text("Add asset"),
            content=dialog_content,
            actions=[
                ft.TextButton("Cancel", on_click=self.close_asset_dialog),
                ft.FilledButton("Save", on_click=self.save_asset_from_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda _: self.close_asset_dialog(None),
        )

        self.page.dialog = self.asset_dialog
        self.asset_dialog.open = True
        self.page.update()

    def close_asset_dialog(self, event: Optional[ft.ControlEvent]) -> None:
        if hasattr(self, "asset_dialog"):
            self.asset_dialog.open = False
            self.page.update()

    def save_asset_from_dialog(self, event: ft.ControlEvent) -> None:
        tag = (self.asset_tag_field.value or "").strip()
        if not tag:
            self.show_snack("Asset tag is required", error=True)
            return
        record = AssetRecord(
            tag=tag,
            description=self.asset_description_field.value or "",
            functional_location=self.asset_location_field.value or "",
            criticality=self.asset_criticality_field.value or "",
            manufacturer=self.asset_manufacturer_field.value or "",
            model=self.asset_model_field.value or "",
        )
        self.data.upsert_asset(record)
        self.close_asset_dialog(None)
        self.refresh_asset_table(None)
        self.show_snack(f"Asset {tag} saved")

    # ---------------------- Runtime tab ----------------------
    def build_runtime_tab(self) -> ft.Control:
        refresh_button = ft.IconButton(ft.icons.REFRESH, tooltip="Reload data", on_click=self.reload_runtime_data)
        header = ft.Row(
            [ft.Text("Runtime dashboard", size=20, weight=ft.FontWeight.BOLD), refresh_button],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.runtime_chart = ft.Container(expand=True)
        self.runtime_table = ft.DataTable(columns=self._runtime_table_columns(), rows=[], expand=True)

        self.populate_runtime_chart()
        self.populate_runtime_table()

        body = ft.Column(
            [header, ft.Container(self.runtime_chart, height=340), ft.Divider(), self.runtime_table],
            spacing=16,
        )
        return ft.Container(content=body, padding=20, expand=True)

    def _runtime_table_columns(self) -> List[ft.DataColumn]:
        return [
            ft.DataColumn(ft.Text("Asset")),
            ft.DataColumn(ft.Text("Run hours")),
            ft.DataColumn(ft.Text("Criticality")),
            ft.DataColumn(ft.Text("Updated")),
        ]

    def populate_runtime_chart(self) -> None:
        df = self.data.runtime_df
        if df.empty:
            self.runtime_chart.content = ft.Container(
                content=ft.Column(
                    [ft.Icon(ft.icons.INSIGHTS, size=48, color=ft.colors.BLUE_GREY_200), ft.Text("No runtime data available")],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                alignment=ft.alignment.center,
            )
        else:
            chart_df = df.sort_values(by="run_hours", ascending=False).head(20)
            fig = px.bar(
                chart_df,
                x="asset_tag",
                y="run_hours",
                color="criticality" if "criticality" in chart_df.columns else None,
                labels={"asset_tag": "Asset", "run_hours": "Run hours"},
                title="Runtime hours by asset",
            )
            fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
            self.runtime_chart.content = ft.PlotlyChart(fig, expand=True)
        self.page.update()

    def populate_runtime_table(self) -> None:
        df = self.data.runtime_df
        rows: List[ft.DataRow] = []
        if not df.empty:
            for _, row in df.sort_values(by="asset_tag").iterrows():
                rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(row.get("asset_tag", "")))),
                            ft.DataCell(ft.Text(f"{row.get('run_hours', 0):,.1f}" if pd.notna(row.get("run_hours")) else "")),
                            ft.DataCell(ft.Text(str(row.get("criticality", "")))),
                            ft.DataCell(ft.Text(str(row.get("updated_at", "")))),
                        ]
                    )
                )
        self.runtime_table.rows = rows
        self.page.update()

    def reload_runtime_data(self, event: ft.ControlEvent) -> None:
        self.data.load_runtime()
        self.populate_runtime_chart()
        self.populate_runtime_table()
        self.show_snack("Runtime data refreshed")

    # ---------------------- History tab ----------------------
    def build_history_tab(self) -> ft.Control:
        refresh_button = ft.IconButton(ft.icons.REFRESH, tooltip="Reload", on_click=self.reload_history_data)
        header = ft.Row(
            [ft.Text("Maintenance history", size=20, weight=ft.FontWeight.BOLD), refresh_button],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        self.history_table = ft.DataTable(columns=self._history_table_columns(), rows=[], expand=True)
        self.populate_history_table()

        body = ft.Column([header, self.history_table], spacing=16)
        return ft.Container(content=body, padding=20, expand=True)

    def _history_table_columns(self) -> List[ft.DataColumn]:
        return [
            ft.DataColumn(ft.Text("Work order")),
            ft.DataColumn(ft.Text("Asset")),
            ft.DataColumn(ft.Text("Performed on")),
            ft.DataColumn(ft.Text("Summary")),
            ft.DataColumn(ft.Text("Technician")),
        ]

    def populate_history_table(self) -> None:
        df = self.data.history_df
        rows: List[ft.DataRow] = []
        if not df.empty:
            for _, row in df.sort_values(by="performed_on", ascending=False).iterrows():
                rows.append(
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(str(row.get("work_order", "")))),
                            ft.DataCell(ft.Text(str(row.get("asset_tag", "")))),
                            ft.DataCell(ft.Text(str(row.get("performed_on", "")))),
                            ft.DataCell(ft.Text(str(row.get("summary", "")))),
                            ft.DataCell(ft.Text(str(row.get("technician", "")))),
                        ]
                    )
                )
        self.history_table.rows = rows
        self.page.update()

    def reload_history_data(self, event: ft.ControlEvent) -> None:
        self.data.load_history()
        self.populate_history_table()
        self.show_snack("History reloaded")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main(page: ft.Page) -> None:
    ReliabilityApp(page)


if __name__ == "__main__":
    ft.app(target=main)
