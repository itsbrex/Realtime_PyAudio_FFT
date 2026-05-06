// Custom tooltips with a short hover delay. Replaces the browser's native
// `title` tooltip (which has a ~500ms+ delay that can't be configured).

const SHOW_DELAY_MS = 150;
const HIDE_DELAY_MS = 60;

let tipEl = null;
let showTimer = null;
let hideTimer = null;
let activeTarget = null;

function ensureEl() {
  if (tipEl) return tipEl;
  tipEl = document.createElement("div");
  tipEl.className = "tooltip";
  tipEl.style.cssText = [
    "position:fixed",
    "z-index:9999",
    "max-width:420px",
    "padding:6px 9px",
    "background:rgba(20,20,24,0.96)",
    "color:#eee",
    "font:12px/1.4 system-ui,sans-serif",
    "border:1px solid #444",
    "border-radius:4px",
    "pointer-events:none",
    "opacity:0",
    "transition:opacity 80ms",
    "box-shadow:0 4px 12px rgba(0,0,0,0.4)",
  ].join(";");
  document.body.appendChild(tipEl);
  return tipEl;
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
  el.textContent = txt;
  position(ev);
  el.style.opacity = "1";
}

function hide() {
  if (tipEl) tipEl.style.opacity = "0";
  activeTarget = null;
}

function findTipTarget(node) {
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
