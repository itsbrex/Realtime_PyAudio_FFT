// WebSocket connection with exponential backoff (cap 2s) and message routing.

import { store } from "./store.js";

const WS_URL = `ws://${location.hostname || "127.0.0.1"}:8765`;

let socket = null;
let backoffMs = 250;
const handlers = {};
const errSinks = [];

export function onMessage(type, fn) {
  handlers[type] = fn;
}

export function onError(fn) {
  errSinks.push(fn);
}

export function send(obj) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify(obj));
  return true;
}

export function isConnected() {
  return socket && socket.readyState === WebSocket.OPEN;
}

function setStatus(text, ok) {
  const el = document.getElementById("ws-status");
  if (el) {
    el.textContent = text;
    el.className = "badge " + (ok ? "connected" : "disconnected");
  }
}

function onOpen() {
  setStatus("connected", true);
  backoffMs = 250;
}

function onClose() {
  setStatus("disconnected", false);
  setTimeout(connect, backoffMs);
  backoffMs = Math.min(backoffMs * 2, 2000);
}

function onWsError(_e) {
  // close handler will fire
}

function decodeFftBinary(buf) {
  // [type=1:u8][reserved:u8][n_bins:u16][float32 * n_bins] LE
  const dv = new DataView(buf);
  const type = dv.getUint8(0);
  if (type !== 1) return null;
  const n = dv.getUint16(2, true);
  const f32 = new Float32Array(buf, 4, n);
  return f32;
}

function onMsg(ev) {
  if (typeof ev.data !== "string") {
    // Binary -> FFT frame
    const buf = ev.data instanceof ArrayBuffer ? ev.data : null;
    if (buf) {
      const f32 = decodeFftBinary(buf);
      if (f32) store.fft_bins = f32;
    }
    return;
  }

  let msg;
  try { msg = JSON.parse(ev.data); }
  catch { return; }
  // Track snapshot rate independently of FFT binary frames so the badge
  // reflects "how often does the server push L/M/H state" — invariant under
  // the FFT enable toggle.
  if (msg.type === "snapshot") {
    const now = performance.now();
    store.snapshotTimestamps.push(now);
    if (store.snapshotTimestamps.length > 60) store.snapshotTimestamps.shift();
  }
  const h = handlers[msg.type];
  if (h) h(msg);
  if (msg.type === "error") {
    for (const f of errSinks) f(msg.reason || "(no reason)");
  }
}

export function connect() {
  try {
    socket = new WebSocket(WS_URL);
    socket.binaryType = "arraybuffer";
    socket.addEventListener("open", onOpen);
    socket.addEventListener("close", onClose);
    socket.addEventListener("error", onWsError);
    socket.addEventListener("message", onMsg);
  } catch (e) {
    setTimeout(connect, backoffMs);
    backoffMs = Math.min(backoffMs * 2, 2000);
  }
}
