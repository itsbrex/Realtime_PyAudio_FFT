// Custom tooltips with a short hover delay. Replaces the browser's native
// `title` tooltip (which has a ~500ms+ delay that can't be configured).
//
// Tooltip content is rendered as a small subset of Markdown:
//   **bold**, *italic*, `code`, # / ## headings, - bullet lists,
//   blank-line-separated paragraphs.

const SHOW_DELAY_MS = 150;
const HIDE_DELAY_MS = 60;

let tipEl = null;
let showTimer = null;
let hideTimer = null;
let activeTarget = null;

function injectStyles() {
  if (document.getElementById("tooltip-styles")) return;
  const s = document.createElement("style");
  s.id = "tooltip-styles";
  s.textContent = `
    .tooltip { font: 12px/1.45 system-ui, sans-serif; }
    .tooltip p { margin: 0 0 6px 0; }
    .tooltip p:last-child { margin-bottom: 0; }
    .tooltip h4, .tooltip h5, .tooltip h6 {
      margin: 8px 0 3px 0; font-size: 12px; font-weight: 600; color: #fff;
    }
    .tooltip h4:first-child, .tooltip h5:first-child, .tooltip h6:first-child { margin-top: 0; }
    .tooltip ul { margin: 2px 0 6px 0; padding-left: 16px; }
    .tooltip ul:last-child { margin-bottom: 0; }
    .tooltip li { margin: 1px 0; }
    .tooltip code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      background: rgba(255,255,255,0.08);
      padding: 0 4px;
      border-radius: 3px;
      color: #f2cf6a;
    }
    .tooltip strong { color: #fff; font-weight: 600; }
    .tooltip em { color: #b8b8b8; font-style: italic; }
  `;
  document.head.appendChild(s);
}

function ensureEl() {
  if (tipEl) return tipEl;
  injectStyles();
  tipEl = document.createElement("div");
  tipEl.className = "tooltip";
  tipEl.style.cssText = [
    "position:fixed",
    "z-index:9999",
    "max-width:420px",
    "padding:8px 11px",
    "background:rgba(20,20,24,0.96)",
    "color:#ddd",
    "border:1px solid #444",
    "border-radius:5px",
    "pointer-events:none",
    "opacity:0",
    "transition:opacity 80ms",
    "box-shadow:0 4px 12px rgba(0,0,0,0.4)",
  ].join(";");
  document.body.appendChild(tipEl);
  return tipEl;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function inlineMd(s) {
  // Order matters: code first (so its contents aren't re-parsed), then bold,
  // then italic. We're operating on already-escaped HTML.
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1<em>$2</em>");
}

function renderMarkdown(src) {
  const escaped = escapeHtml(src.trim());
  const lines = escaped.split("\n");
  let html = "";
  let inList = false;
  let para = [];
  const flushPara = () => {
    if (para.length) {
      html += "<p>" + inlineMd(para.join(" ")) + "</p>";
      para = [];
    }
  };
  const closeList = () => {
    if (inList) { html += "</ul>"; inList = false; }
  };
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) { flushPara(); closeList(); continue; }
    let m;
    if ((m = line.match(/^(#{1,3})\s+(.+)$/))) {
      flushPara(); closeList();
      const lvl = Math.min(6, 3 + m[1].length); // # → h4, ## → h5, ### → h6
      html += `<h${lvl}>${inlineMd(m[2])}</h${lvl}>`;
      continue;
    }
    if ((m = line.match(/^[-*]\s+(.+)$/))) {
      flushPara();
      if (!inList) { html += "<ul>"; inList = true; }
      html += "<li>" + inlineMd(m[1]) + "</li>";
      continue;
    }
    closeList();
    para.push(line);
  }
  flushPara(); closeList();
  return html;
}

function position(ev) {
  const el = ensureEl();
  const pad = 12;
  let x = ev.clientX + pad;
  let y = ev.clientY + pad;
  const r = el.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 4) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 4) y = ev.clientY - r.height - pad;
  el.style.left = x + "px";
  el.style.top = y + "px";
}

function show(target, ev) {
  const txt = target.getAttribute("data-tooltip");
  if (!txt) return;
  const el = ensureEl();
  el.innerHTML = renderMarkdown(txt);
  position(ev);
  el.style.opacity = "1";
}

function hide() {
  if (tipEl) tipEl.style.opacity = "0";
  activeTarget = null;
}

function findTipTarget(node) {
  // Suppress tooltips while pointing at the slider itself — they get in the
  // way of dragging. The wrapping label still shows on its text/readout.
  if (node && node.tagName === "INPUT" && node.type === "range") return null;
  while (node && node.nodeType === 1) {
    if (node.hasAttribute && node.hasAttribute("data-tooltip")) return node;
    node = node.parentNode;
  }
  return null;
}

function migrateTitles(root) {
  const els = root.querySelectorAll("[title]");
  els.forEach((el) => {
    const t = el.getAttribute("title");
    if (!t) return;
    el.setAttribute("data-tooltip", t);
    el.removeAttribute("title");
  });
}

export function setupTooltips() {
  migrateTitles(document);

  document.addEventListener("mouseover", (ev) => {
    const t = findTipTarget(ev.target);
    if (!t || t === activeTarget) return;
    activeTarget = t;
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    showTimer = setTimeout(() => show(t, ev), SHOW_DELAY_MS);
  });

  document.addEventListener("mousemove", (ev) => {
    if (tipEl && tipEl.style.opacity === "1") position(ev);
  });

  document.addEventListener("mouseout", (ev) => {
    const t = findTipTarget(ev.target);
    if (!t) return;
    const related = findTipTarget(ev.relatedTarget);
    if (related === t) return;
    clearTimeout(showTimer);
    hideTimer = setTimeout(hide, HIDE_DELAY_MS);
  });

  document.addEventListener("mousedown", () => {
    clearTimeout(showTimer);
    hide();
  }, true);
}
