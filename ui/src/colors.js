// Single source of truth for L/M/H band colors across the UI.
// Keep these aligned wherever a band is rendered (bars, lines, FFT overlay,
// frequency axis picker, scene viz, raw meters).

export const LMH = {
  low:  { hex: "#5a8dee", rgb: "90,141,238",  hue: 220 },
  mid:  { hex: "#79d17a", rgb: "121,209,122", hue: 121 },
  high: { hex: "#e8a857", rgb: "232,168,87",  hue: 30  },
};

export const LMH_ORDER = ["low", "mid", "high"];
export const LMH_HEX = LMH_ORDER.map(k => LMH[k].hex);

export function lmhRgba(name, alpha) {
  return `rgba(${LMH[name].rgb},${alpha})`;
}
