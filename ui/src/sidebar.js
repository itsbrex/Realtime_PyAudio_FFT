// Side-panel resize + collapse. The controls panel shares the main grid with
// the viz area, separated by a draggable splitter. Below 10% of page width
// the drag snaps to fully collapsed; the floating expand button restores the
// previous width. Width and collapsed state persist in localStorage.

const STORAGE_KEY_W = "controls.width.px";
const STORAGE_KEY_C = "controls.collapsed";
const SNAP_FRAC = 0.10;
const DEFAULT_W = 304;
const MIN_W = 0;
const MAX_FRAC = 0.8;

export function setupSidebar() {
  const grid = document.getElementById("main-grid");
  const resizer = document.getElementById("grid-resizer");
  const controls = grid?.querySelector(".controls");
  const collapseBtn = document.getElementById("controls-collapse");
  const expandBtn = document.getElementById("controls-expand");
  if (!grid || !resizer || !controls || !collapseBtn || !expandBtn) return;

  const stored = parseInt(localStorage.getItem(STORAGE_KEY_W), 10);
  let savedW = Number.isFinite(stored) ? clamp(stored, 120, 1600) : DEFAULT_W;

  function applyWidth(px) {
    grid.style.setProperty("--controls-w", `${Math.round(px)}px`);
  }
  function applyCollapsed(c) {
    grid.classList.toggle("controls-collapsed", c);
    localStorage.setItem(STORAGE_KEY_C, c ? "1" : "0");
  }

  applyWidth(savedW);
  applyCollapsed(localStorage.getItem(STORAGE_KEY_C) === "1");

  let drag = null;
  resizer.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    drag = {
      startX: e.clientX,
      startW: controls.getBoundingClientRect().width,
    };
    resizer.classList.add("dragging");
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp, { once: true });
  });

  function onMove(e) {
    if (!drag) return;
    const dx = e.clientX - drag.startX;
    const pageW = window.innerWidth;
    const next = clamp(drag.startW - dx, MIN_W, pageW * MAX_FRAC);
    applyWidth(next);
  }

  function onUp() {
    if (!drag) return;
    drag = null;
    resizer.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    window.removeEventListener("mousemove", onMove);

    const w = controls.getBoundingClientRect().width;
    const pageW = window.innerWidth;
    if (w / pageW < SNAP_FRAC) {
      // Snap closed; preserve the previously committed width as the "restore"
      // target so the expand button returns the panel to a sensible size.
      applyWidth(savedW);
      applyCollapsed(true);
    } else {
      savedW = Math.round(w);
      localStorage.setItem(STORAGE_KEY_W, String(savedW));
      applyCollapsed(false);
    }
  }

  collapseBtn.addEventListener("click", () => {
    const w = controls.getBoundingClientRect().width;
    if (w >= 60) {
      savedW = Math.round(w);
      localStorage.setItem(STORAGE_KEY_W, String(savedW));
    }
    applyWidth(savedW);
    applyCollapsed(true);
  });

  expandBtn.addEventListener("click", () => {
    applyCollapsed(false);
    applyWidth(savedW);
  });
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
