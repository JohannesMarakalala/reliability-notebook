# jobcard_template.py

from datetime import datetime
from datetime import date, datetime as dt  # keep both if you need later imports

# We call to_num that lives in the main app. Provide a safe local fallback so
# importing this module never crashes if main doesn't inject one.
def _to_num_local(x, default=0.0):
    try:
        if x in (None, "", "NaN"):
            return float(default)
        return float(x)
    except Exception:
        return float(default)

# Main app will replace this reference at runtime:
to_num = _to_num_local

def render_jobcard_html(job: dict, assets: dict, currency: str, fx_rate: float) -> str:
    """Return HTML (inline CSS) for the PM MAINTENANCE ORDER (PDF-ready)."""

    asset = assets.get(job.get("asset_tag",""), {})
    wo = job.get("wo_number","")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------- D) Spares & Kitting rows (no prices on PDF) ----------
    sp_rows = job.get("spares", [])
    picked_rows_html = ""
    for i, s in enumerate(sp_rows, start=1):
        picked_rows_html += f"""
        <tr>
          <td>{i}</td>
          <td>{s.get('cmms','')}</td>
          <td>{s.get('oem','')}</td>
          <td>{s.get('desc','')}</td>
          <td style="text-align:right">{int(_to_num_local(s.get('qty',0)))}</td>
        </tr>
        """
    if not picked_rows_html:
        picked_rows_html = '<tr><td colspan="5" style="text-align:center;color:#777">No spares picked</td></tr>'

    # ---------- E) Labour Plan table (incl. names) ----------
    lb_rows_html = ""
    for i, r in enumerate(job.get("labour", []), start=1):
        lb_rows_html += f"""
        <tr>
          <td>{i}</td>
          <td>{r.get('craft','')}</td>
          <td>{r.get('dept','')}</td>
          <td>{r.get('names','')}</td>
          <td style="text-align:right">{int(_to_num_local(r.get('qty',0)))}</td>
          <td style="text-align:right">{float(_to_num_local(r.get('hours',0.0))):.1f}</td>
        </tr>
        """
    if not lb_rows_html:
        lb_rows_html = '<tr><td colspan="6" style="text-align:center;color:#777">No labour planned</td></tr>'

    # ---------- C) Safety (permits from planning) ----------
    permits = (job.get("logistics", {}) or {}).get("permits", []) or []
    if permits:
        safety_html = "".join(f'<span class="chip">{p}</span>' for p in permits)
    else:
        safety_html = '<span class="muted">No specific permits selected.</span>'

    # ---------- F) Execution Time Sheet (prefill names) ----------
    timesheet_rows_html = ""
    total_rows = 0
    for r in job.get("labour", []):
        planned_h = float(_to_num_local(r.get("hours", 0.0)))
        qty = int(_to_num_local(r.get("qty", 0)))
        names_str = str(r.get("names","")).strip()
        names_list = [n.strip() for n in names_str.split(",") if n.strip()] if names_str else []

        for nm in names_list:
            timesheet_rows_html += f"""
            <tr>
              <td>{nm}</td>
              <td style="text-align:right">{planned_h:.1f}</td>
              <td></td><td></td><td></td><td></td><td></td>
            </tr>
            """
            total_rows += 1

        remaining = max(qty - len(names_list), 0)
        craft_hint = f"({r.get('craft','')})" if r.get("craft") else ""
        for _ in range(remaining):
            timesheet_rows_html += f"""
            <tr>
              <td>{craft_hint}</td>
              <td style="text-align:right">{planned_h:.1f}</td>
              <td></td><td></td><td></td><td></td><td></td>
            </tr>
            """
            total_rows += 1

    for _ in range(2):
        timesheet_rows_html += """
        <tr>
          <td></td><td></td><td></td><td></td><td></td><td></td><td></td>
        </tr>
        """
        total_rows += 1

    if total_rows == 0:
        timesheet_rows_html = '<tr><td colspan="7" class="muted" style="text-align:center">To be completed by the crew</td></tr>'

    # ---------- G) Spares Used (tick box per line) ----------
    used_rows_html = ""
    for i, s in enumerate(sp_rows, start=1):
        used_rows_html += f"""
        <tr>
          <td>{i}</td>
          <td>{s.get('cmms','')}</td>
          <td>{s.get('oem','')}</td>
          <td>{s.get('desc','')}</td>
          <td style="text-align:right">{int(_to_num_local(s.get('qty',0)))}</td>
          <td style="text-align:center">☐</td>
        </tr>
        """
    if not used_rows_html:
        used_rows_html = '<tr><td colspan="6" class="muted" style="text-align:center">To be completed by the crew</td></tr>'

    # ---------- CSS & Layout ----------
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>PM MAINTENANCE ORDER {wo}</title>
<style>
  html, body {{ font-family:Segoe UI, Roboto, Arial, sans-serif; color:#111; }}
  .sheet {{ width: 960px; margin: 16px auto; }}
  .head {{ display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #444; padding-bottom:8px; }}
  .brand {{ font-size:13px; line-height:1.2 }}
  .title {{ text-align:center; }}
  .title h1 {{ margin:0; font-size:20px; letter-spacing:0.5px; }}
  .title small {{ color:#666 }}
  .meta {{ text-align:right; font-size:12px; }}
  .box {{ border:1px solid #ccc; border-radius:6px; padding:10px; margin:10px 0; }}
  .label {{ font-weight:700; font-size:11px; color:#555; margin:0 0 6px 2px; }}
  .grid-4 {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; }}
  .cell {{ border:1px solid #e7e7e7; border-radius:4px; padding:8px; min-height:28px; font-size:12px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th, td {{ border:1px solid #e4e4e4; padding:6px 8px; }}
  th {{ background:#f6f7fb; text-align:left; white-space:nowrap }}
  .muted {{ color:#777; }}
  .signrow {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap:12px; }}
  .signbox {{ border:1px dashed #bbb; border-radius:6px; padding:12px; }}
  .small {{ font-size:11px; }}
  .chip {{ display:inline-block; background:#eef5ff; border:1px solid #cfdcff; color:#2b5cab; border-radius:14px; padding:2px 8px; margin:2px 6px 2px 0; font-size:11px; }}
</style>
</head>
<body>
<div class="sheet">
  <div class="head">
    <div class="brand small">
      <div><strong>PM MAINTENANCE ORDER</strong></div>
      <div class="muted">Generated: {now}</div>
    </div>
    <div class="title">
      <h1>PM MAINTENANCE ORDER</h1>
      <small class="muted">Works Order Number</small>
      <div style="font-size:18px;font-weight:700;margin-top:2px">{wo or "—"}</div>
    </div>
    <div class="meta">
      <div><strong>Asset:</strong> {job.get('asset_tag','')}</div>
      <div><strong>Priority:</strong> {job.get('priority','')}</div>
    </div>
  </div>

  <!-- A) Job Summary -->
  <div class="label">A) JOB SUMMARY</div>
  <div class="box">
    <div class="grid-4">
      <div class="cell"><strong>Asset Tag</strong><br>{job.get('asset_tag','')}</div>
      <div class="cell"><strong>Functional Location</strong><br>{asset.get('Functional Location','')}</div>
      <div class="cell"><strong>Model</strong><br>{asset.get('Model','')}</div>
      <div class="cell"><strong>Priority</strong><br>{job.get('priority','')}</div>
    </div>
    <div class="grid-4" style="margin-top:8px">
      <div class="cell"><strong>Basic Start Date</strong><br>{job.get('planned_start','')}</div>
      <div class="cell"><strong>Basic End Date</strong><br>{job.get('required_end_date','')}</div>
      <div class="cell"><strong>Required End Date</strong><br>{job.get('required_end_date','')}</div>
      <div class="cell"><strong>Department</strong><br>{job.get('lead_department','')}</div>
    </div>
    <div class="grid-4" style="margin-top:8px">
      <div class="cell"><strong>Creator</strong><br>{job.get('creator','')}</div>
      <div class="cell"><strong>PM Code</strong><br>{job.get('pm_code','')}</div>
      <div class="cell"><strong>Lead Role</strong><br>{job.get('lead_role','')}</div>
      <div class="cell"><strong>WO Number</strong><br>{wo or "—"}</div>
    </div>
  </div>

  <!-- B) Main Description of the Works Order -->
  <div class="label">B) MAIN DESCRIPTION OF THE WORKS ORDER</div>
  <div class="box">
    <div class="cell" style="min-height:90px">{job.get('task_description','')}</div>
  </div>

  <!-- C) Safety (selected permits) -->
  <div class="label">C) SAFETY</div>
  <div class="box">
    {safety_html}
  </div>

  <!-- D) Spares & Kitting (no prices) -->
  <div class="label">D) SPARES & KITTING</div>
  <div class="box">
    <table>
      <thead>
        <tr><th>#</th><th>CMMS Code</th><th>OEM Part</th><th>Description</th><th>Qty Picked</th></tr>
      </thead>
      <tbody>
        {picked_rows_html}
      </tbody>
    </table>
  </div>

  <!-- E) Labour Plan -->
  <div class="label">E) LABOUR PLAN</div>
  <div class="box">
    <table>
      <thead>
        <tr><th>#</th><th>Craft</th><th>Dept</th><th>Names</th><th>Qty</th><th>Planned Hours</th></tr>
      </thead>
      <tbody>
        {lb_rows_html}
      </tbody>
    </table>
  </div>

  <!-- F) Execution Time Sheet (prefilled from names; +2 blank rows) -->
  <div class="label">F) EXECUTION TIME SHEET</div>
  <div class="box">
    <table>
      <thead>
        <tr>
          <th>Person Name</th>
          <th>Planned Work Dur. (h)</th>
          <th>Actual Work Dur. (h)</th>
          <th>Start Date</th>
          <th>Finish Date</th>
          <th>Start Time</th>
          <th>Finish Time</th>
        </tr>
      </thead>
      <tbody>
        {timesheet_rows_html}
      </tbody>
    </table>
  </div>

  <!-- G) Spares Used (tick used / not used) -->
  <div class="label">G) SPARES USED (TICK)</div>
  <div class="box">
    <table>
      <thead>
        <tr><th>#</th><th>CMMS</th><th>OEM</th><th>Description</th><th>Picked Qty</th><th>Used?</th></tr>
      </thead>
      <tbody>
        {used_rows_html}
      </tbody>
    </table>
  </div>

  <!-- H) Work Execution Comments -->
  <div class="label">H) WORK EXECUTION COMMENTS</div>
  <div class="box" style="min-height:120px"></div>

  <!-- I) Over-inspection Comments -->
  <div class="label">I) OVER-INSPECTION COMMENTS</div>
  <div class="box" style="min-height:80px"></div>

  <!-- J) Signatures -->
  <div class="label">J) SIGNATURES</div>
  <div class="signrow">
    <div class="signbox small"><strong>Craftsman</strong><br><br>Signature: ___________<br>Date: ___________</div>
    <div class="signbox small"><strong>Supervisor</strong><br><br>Signature: ___________<br>Date: ___________</div>
    <div class="signbox small"><strong>Planner/Approver</strong><br><br>Signature: ___________<br>Date: ___________</div>
  </div>
</div>
</body>
</html>
"""
    return html
