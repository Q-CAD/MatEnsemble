const app = document.querySelector("#app");
const ROOT = "__root__";
const colors = {
  pending: "#d9b763",
  running: "#69a7d8",
  completed: "#70b894",
  failed: "#d87575",
  cores: "#70b894",
  gpus: "#a58bc4",
};
const view = {
  catalog: null,
  campaign: null,
  workflow: null,
  detail: null,
  history: [],
  lastSequence: null,
  catalogOffline: false,
  detailOffline: false,
};

const esc = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const token = (value) => String(value || "unknown").toLowerCase()
  .replace(/[^a-z0-9_-]/g, "-");
const campaignOf = (workflow) => workflow.campaign || ROOT;
const campaignName = (campaign) =>
  campaign === ROOT ? "(dashboard root)" : campaign;

function time(value, short = false) {
  if (!value) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) return String(value);
  return new Intl.DateTimeFormat(undefined, short
    ? { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }
    : { dateStyle: "medium", timeStyle: "medium" }).format(parsed);
}

function duration(seconds) {
  if (seconds == null) return "—";
  let value = Math.max(0, Math.round(Number(seconds)));
  const days = Math.floor(value / 86400); value %= 86400;
  const hours = Math.floor(value / 3600); value %= 3600;
  const minutes = Math.floor(value / 60); value %= 60;
  return [days && `${days}d`, (hours || days) && `${hours}h`,
    (minutes || hours || days) && `${minutes}m`, `${value}s`]
    .filter(Boolean).join(" ");
}

function campaigns() {
  return [...new Set((view.catalog?.workflows || []).map(campaignOf))]
    .sort((a, b) => campaignName(a).localeCompare(campaignName(b)));
}

function campaignWorkflows(campaign) {
  return (view.catalog?.workflows || [])
    .filter((workflow) => campaignOf(workflow) === campaign);
}

function readUrl() {
  const params = new URLSearchParams(location.search);
  view.campaign = params.get("campaign");
  view.workflow = params.get("workflow");
}

function writeUrl(campaign, workflow, replace = false) {
  const params = new URLSearchParams();
  if (campaign) params.set("campaign", campaign);
  if (workflow) params.set("workflow", workflow);
  history[replace ? "replaceState" : "pushState"](
    {}, "", `${location.pathname}${params.size ? `?${params}` : ""}`,
  );
}

function pill(label, kind) {
  return `<span class="pill ${token(kind)}"><i></i>${esc(label)}</span>`;
}

function sidebar() {
  return `<aside class="sidebar">
    <div class="brand"><span class="logo">▥</span><div><strong>MatEnsemble</strong><small>workflow observatory</small></div></div>
    <div class="side-title"><span>Campaigns</span><span>${campaigns().length}</span></div>
    <nav>${campaigns().map((campaign) => {
      const workflows = campaignWorkflows(campaign);
      const active = workflows.filter((item) =>
        ["initializing", "running"].includes(item.state)).length;
      const attention = workflows.filter((item) =>
        item.health !== "healthy" || ["failed", "interrupted"].includes(item.state)).length;
      return `<button class="campaign-link ${view.campaign === campaign ? "selected" : ""}" data-campaign="${esc(campaign)}">
        <span class="folder">⌁</span><span><strong>${esc(campaignName(campaign))}</strong><small>${workflows.length} workflows</small></span>
        <em>${active ? `<i class="blue">${active}</i>` : ""}${attention ? `<i class="amber">${attention}</i>` : ""}</em>
      </button>`;
    }).join("") || '<p class="muted">No campaigns found.</p>'}</nav>
    <div class="connection"><i class="${view.catalogOffline ? "offline" : ""}"></i><span><strong>${view.catalogOffline ? "Reconnecting" : "Live catalog"}</strong><small>Scanned ${esc(time(view.catalog?.scanned_at, true))}</small></span></div>
  </aside>`;
}

function topbar(title, path = "") {
  return `<header class="topbar">
    <button class="menu" aria-label="Toggle campaigns">☰</button>
    <div><span>Dashboard / ${esc(title)}</span>${path ? `<code>${esc(path)}</code>` : ""}</div>
    <div class="refresh"><span>Refreshes every 5 seconds</span><button data-refresh aria-label="Refresh now">↻</button></div>
  </header>`;
}

function overview() {
  const workflows = view.catalog?.workflows || [];
  const active = workflows.filter((item) =>
    ["initializing", "running"].includes(item.state)).length;
  const healthy = workflows.filter((item) => item.health === "healthy").length;
  const issues = workflows.filter((item) =>
    item.health !== "healthy" || ["failed", "interrupted"].includes(item.state)).length;
  return `<main>${topbar("Overview", view.catalog?.root_name)}
    <section class="intro"><p class="eyebrow">Dashboard root</p><h1>${esc(view.catalog?.root_name)}</h1><p>Choose a campaign to browse its directory tree and workflow runs.</p></section>
    <section class="overview-stats">
      ${summary("Campaigns", campaigns().length, "Directory groups")}
      ${summary("Workflows", workflows.length, `${active} active`)}
      ${summary("Healthy", healthy, "Readable and current")}
      ${summary("Needs attention", issues, "Stale, failed, or unreadable")}
    </section>
    <section class="panel"><div class="panel-head"><div><p class="eyebrow">Directory index</p><h2>Campaigns</h2></div></div>
      <div class="campaign-cards">${campaigns().map((campaign) => {
        const items = campaignWorkflows(campaign);
        return `<button data-campaign="${esc(campaign)}"><span class="folder">⌁</span><strong>${esc(campaignName(campaign))}</strong><small>${items.length} workflow${items.length === 1 ? "" : "s"}</small></button>`;
      }).join("") || '<div class="empty">No stamped workflow directories discovered yet.</div>'}</div>
    </section>
  </main>`;
}

function summary(label, value, note) {
  return `<div><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`;
}

function groups(workflows) {
  const result = new Map();
  workflows.forEach((workflow) => {
    const key = workflow.parent_path || "(dashboard root)";
    if (!result.has(key)) result.set(key, []);
    result.get(key).push(workflow);
  });
  return [...result.entries()].sort(([a], [b]) => a.localeCompare(b));
}

function browserView() {
  const workflows = campaignWorkflows(view.campaign);
  return `<main>${topbar(campaignName(view.campaign), `${workflows.length} workflows`)}
    <section class="intro browser-intro"><div><p class="eyebrow">Campaign</p><h1>${esc(campaignName(view.campaign))}</h1><p>Placement follows the directory tree beneath the dashboard root.</p></div>
      <div>${pill(`${workflows.filter((x) => x.state === "running").length} running`, "running")}${pill(`${workflows.filter((x) => x.health !== "healthy").length} attention`, "stale")}</div></section>
    <section class="directories">${groups(workflows).map(([parent, items]) => `
      <article class="directory"><header><div><span class="folder">⌁</span><span><small>Directory</small><strong>${esc(parent)}</strong></span></div><em>${items.length} runs</em></header>
        ${items.map(workflowRow).join("")}</article>`).join("") || '<div class="empty">No workflows in this campaign.</div>'}</section>
  </main>`;
}

function workflowRow(workflow) {
  const current = workflow.current || {};
  const condition = workflow.health !== "healthy" ? workflow.health : workflow.state;
  return `<button class="workflow-row" data-workflow="${esc(workflow.id)}">
    <i class="state-line ${token(condition)}"></i>
    <span class="identity"><strong>${esc(workflow.display_name)}</strong>${workflow.display_name !== workflow.directory_name ? `<small>${esc(workflow.directory_name)}</small>` : ""}${workflow.error ? `<em>${esc(workflow.error)}</em>` : ""}</span>
    <span class="row-pills">${pill(workflow.state || "unknown", workflow.state)}${workflow.health !== "healthy" ? pill(workflow.health, workflow.health) : ""}</span>
    <span class="updated"><small>Updated</small><strong>${esc(time(workflow.updated_at || workflow.started_at, true))}</strong></span>
    <span class="tiny-metrics">${tiny("P", current.pending, "pending")}${tiny("R", current.running, "running")}${tiny("C", current.completed, "completed")}${tiny("F", current.failed, "failed")}</span>
    <b>→</b>
  </button>`;
}

function tiny(label, value, kind) {
  return `<span title="${kind}"><i class="${kind}">${label}</i>${Number(value || 0)}</span>`;
}

function detailView() {
  const catalogItem = view.catalog?.workflows.find((item) => item.id === view.workflow);
  const envelope = view.detail;
  const status = envelope?.status;
  if (!status) {
    return `<main>${topbar(catalogItem?.display_name || "Workflow")}
      <section class="waiting"><i class="${token(envelope?.health || catalogItem?.health || "starting")}"></i><h1>${esc(catalogItem?.display_name || "Workflow unavailable")}</h1><p>${esc(envelope?.error || catalogItem?.error || "Waiting for workflow status…")}</p><button data-back>Back to campaign</button></section></main>`;
  }
  const workflow = status.workflow || {};
  const current = status.current || {};
  const allocation = status.allocation || {};
  const failures = status.failures || [];
  const total = ["pending", "running", "completed", "failed"]
    .reduce((sum, key) => sum + Number(current[key] || 0), 0);
  const settled = Number(current.completed || 0) + Number(current.failed || 0);
  const progress = total ? settled / total * 100 : 0;
  const usedCores = Math.max(0, Number(allocation.total_cores || 0) - Number(current.free_cores || 0));
  const usedGpus = Math.max(0, Number(allocation.total_gpus || 0) - Number(current.free_gpus || 0));
  const health = envelope.health || "healthy";
  return `<main>${topbar(workflow.name, envelope.relative_path)}
    ${view.detailOffline || health !== "healthy" ? warning(health, envelope.error) : ""}
    <section class="detail-title"><div><span>${pill(workflow.state, workflow.state)}${pill(health, health)}</span><h1>${esc(workflow.name)}</h1><code>${esc(envelope.relative_path)}</code></div><em>schema <strong>v${status.schema_version}</strong></em></section>
    <section class="times">${timeBox("Started", time(workflow.started_at))}${timeBox("Updated", time(workflow.updated_at))}${timeBox("Finished", time(workflow.finished_at))}${timeBox("Elapsed", duration(workflow.elapsed_seconds))}</section>
    <section class="metrics">${metric("Pending", current.pending, "pending", `${current.ready || 0} ready`)}${metric("Ready", current.ready, "ready", "Eligible to launch")}${metric("Blocked", current.blocked, "blocked", "Waiting on dependencies")}${metric("Running", current.running, "running", "In flight")}${metric("Completed", current.completed, "completed", "Successful")}${metric("Failed", current.failed, "failed", "Needs attention")}</section>
    <section class="detail-grid">
      <article class="panel progress"><div class="panel-head"><div><p class="eyebrow">Known workload</p><h2>Workflow progress</h2></div><strong>${progress.toFixed(1)}%</strong></div><div class="progress-bar"><i style="width:${Math.min(100, progress)}%"></i></div><div class="progress-labels"><span>${settled} settled</span><span>${total} known chores</span></div><p class="note">Adaptive workflows can add chores, so progress may decrease during a run.</p></article>
      <article class="panel allocation"><div class="panel-head"><div><p class="eyebrow">HPC resources</p><h2>Allocation</h2></div></div><div>${allocationBox("Nodes", allocation.nodes)}${allocationBox("Cores / node", allocation.cores_per_node)}${allocationBox("GPUs / node", allocation.gpus_per_node)}${allocationBox("Total cores", allocation.total_cores)}${allocationBox("Total GPUs", allocation.total_gpus)}</div></article>
      <article class="panel queue-chart"><div class="panel-head"><div><p class="eyebrow">Snapshots</p><h2>Queue history</h2></div><span>${view.history.length} points</span></div>${lineChart(view.history, ["pending", "running", "completed", "failed"], "Queue history", "chores")}</article>
      <article class="panel resources"><div class="panel-head"><div><p class="eyebrow">Current pressure</p><h2>Utilization</h2></div></div>${gauge("CPU cores", usedCores, Number(allocation.total_cores || 0), "cores")}${gauge("GPUs", usedGpus, Number(allocation.total_gpus || 0), "gpus")}<p>${current.free_cores || 0} free cores · ${current.free_gpus || 0} free GPUs</p><h3>Usage history</h3>${resourceChart(view.history, allocation)}</article>
      <article class="panel failures"><div class="panel-head"><div><p class="eyebrow">Exceptions</p><h2>Failures</h2></div><span>${failures.length} recorded</span></div>${failureTable(failures)}</article>
    </section>
  </main>`;
}

function warning(health, error) {
  const message = view.detailOffline
    ? "A refresh failed. The last successful data remains visible."
    : error || { stale: "This active workflow has not published a recent update.",
      missing: "The workflow directory is no longer available.",
      unreadable: "The workflow status cannot currently be read.",
      starting: "Waiting for status.json to be created." }[health];
  return `<div class="warning ${token(health)}"><strong>${view.detailOffline ? "Disconnected" : esc(health)}</strong><span>${esc(message)}</span></div>`;
}
function timeBox(label, value) { return `<div><span>${label}</span><strong>${esc(value)}</strong></div>`; }
function metric(label, value, kind, note) { return `<div class="metric ${kind}"><span>${label}</span><strong>${Number(value || 0)}</strong><small>${note}</small></div>`; }
function allocationBox(label, value) { return `<span><small>${label}</small><strong>${Number(value || 0)}</strong></span>`; }

function lineChart(records, fields, ariaLabel, yLabel) {
  if (!records.length) return '<div class="empty">No history snapshots yet.</div>';
  const width = 760, height = 245, pad = { left: 44, right: 12, top: 14, bottom: 34 };
  const max = Math.max(1, ...records.flatMap((row) => fields.map((field) => Number(row[field] || 0))));
  const x = (index) => pad.left + index / Math.max(1, records.length - 1) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - Number(value || 0) / max * (height - pad.top - pad.bottom);
  const grid = [0, .25, .5, .75, 1].map((part) => `<line x1="${pad.left}" y1="${y(max * part)}" x2="${width - pad.right}" y2="${y(max * part)}"/><text x="${pad.left - 8}" y="${y(max * part) + 3}" text-anchor="end">${Math.round(max * part)}</text>`).join("");
  const paths = fields.map((field) => `<polyline points="${records.map((row, index) => `${x(index)},${y(row[field])}`).join(" ")}" stroke="${colors[field]}"/>`).join("");
  const axis = (row) => row?.elapsed_seconds != null ? duration(row.elapsed_seconds) : time(row?.timestamp, true);
  const latest = records.at(-1);
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}"><g>${grid}</g>${paths}<text class="axis-label" transform="translate(11 ${height / 2}) rotate(-90)" text-anchor="middle">${yLabel}</text><text x="${pad.left}" y="${height - 7}">${esc(axis(records[0]))}</text><text x="${width - pad.right}" y="${height - 7}" text-anchor="end">${esc(axis(latest))}</text></svg><div class="legend">${fields.map((field) => `<span><i style="background:${colors[field]}"></i>${field}</span>`).join("")}</div><p class="sr-only">Latest snapshot: ${fields.map((field) => `${field} ${latest[field] || 0}`).join(", ")}.</p>`;
}

function gauge(label, used, total, kind) {
  const percent = total ? Math.max(0, Math.min(100, used / total * 100)) : 0;
  return `<div class="gauge"><div><span>${label}</span><strong>${used} <small>/ ${total}</small></strong></div><div class="gauge-bar"><i class="${kind}" style="width:${percent}%"></i></div><em>${percent.toFixed(0)}%</em></div>`;
}
function resourceChart(records, allocation) {
  return lineChart(records.map((row) => ({ ...row,
    cores: Math.max(0, Number(allocation.total_cores || 0) - Number(row.free_cores || 0)),
    gpus: Math.max(0, Number(allocation.total_gpus || 0) - Number(row.free_gpus || 0)),
  })), ["cores", "gpus"], "Resource utilization history", "resources");
}
function failureTable(failures) {
  if (!failures.length) return '<div class="success">✓ <span><strong>No failures recorded</strong><small>The workflow has not reported failed chores.</small></span></div>';
  return `<div class="table-wrap"><table><thead><tr><th>Chore</th><th>Timestamp</th><th>Reason</th><th>Upstream</th><th>Message</th><th>Artifact</th></tr></thead><tbody>${failures.map((failure) => `<tr><td><code>${esc(failure.chore_id || "—")}</code></td><td>${esc(time(failure.timestamp))}</td><td class="failure">${esc(failure.reason || "unknown")}</td><td>${esc(failure.upstream || "—")}</td><td>${esc(failure.message || "—")}</td><td>${failure.chore_id ? `<a target="_blank" rel="noreferrer" href="/api/workflows/${encodeURIComponent(view.workflow)}/artifacts/${encodeURIComponent(failure.chore_id)}/stderr">stderr ↗</a>` : "—"}</td></tr>`).join("")}</tbody></table></div>`;
}

function render() {
  if (!view.catalog) return;
  app.innerHTML = `<div class="shell">${sidebar()}${view.workflow ? detailView() : view.campaign ? browserView() : overview()}</div>`;
  document.querySelectorAll("[data-campaign]").forEach((button) =>
    button.addEventListener("click", () => selectCampaign(button.dataset.campaign)));
  document.querySelectorAll("[data-workflow]").forEach((button) =>
    button.addEventListener("click", () => selectWorkflow(button.dataset.workflow)));
  document.querySelector("[data-back]")?.addEventListener("click", () => selectCampaign(view.campaign));
  document.querySelector("[data-refresh]")?.addEventListener("click", async () => {
    await refreshCatalog(); await refreshSelected(false);
  });
  document.querySelector(".menu")?.addEventListener("click", () =>
    document.querySelector(".sidebar")?.classList.toggle("open"));
}

function selectCampaign(campaign, push = true) {
  view.campaign = campaign; view.workflow = null; view.detail = null;
  view.history = []; view.lastSequence = null;
  if (push) writeUrl(campaign, null);
  render();
}
function selectWorkflow(id, push = true) {
  const workflow = view.catalog?.workflows.find((item) => item.id === id);
  if (workflow) view.campaign = campaignOf(workflow);
  view.workflow = id; view.detail = null; view.history = []; view.lastSequence = null;
  if (push) writeUrl(view.campaign, id);
  render(); refreshSelected(true);
}
async function json(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.error?.message || `Request failed (${response.status})`);
  }
  return response.json();
}
async function refreshCatalog() {
  try {
    view.catalog = await json("/api/catalog"); view.catalogOffline = false;
    render();
  } catch { view.catalogOffline = true; if (view.catalog) render(); }
}
async function refreshSelected(initial = false) {
  const id = view.workflow;
  if (!id) return;
  try {
    const detail = await json(`/api/workflows/${encodeURIComponent(id)}/status`);
    if (id !== view.workflow) return;
    view.detail = detail; view.detailOffline = false;
    if (detail.status) {
      const query = !initial && view.lastSequence != null
        ? `?after_sequence=${view.lastSequence}&max_points=1000` : "?max_points=1000";
      const payload = await json(`/api/workflows/${encodeURIComponent(id)}/history${query}`);
      if (id !== view.workflow) return;
      view.history = initial || view.lastSequence == null
        ? payload.records
        : [...view.history, ...payload.records].slice(-1000);
      view.lastSequence = payload.last_sequence ?? view.history.at(-1)?.sequence ?? view.lastSequence;
    }
    render();
  } catch (error) {
    view.detailOffline = true;
    if (view.detail && view.catalog?.workflows.find((item) => item.id === id)?.health === "missing") {
      view.detail = { ...view.detail, health: "missing", error: error.message };
    }
    render();
  }
}

window.addEventListener("popstate", () => {
  readUrl(); view.detail = null; view.history = []; view.lastSequence = null;
  render(); refreshSelected(true);
});
readUrl();
await refreshCatalog();
await refreshSelected(true);
setInterval(refreshCatalog, 5000);
setInterval(() => refreshSelected(false), 5000);
