import { useEffect, useRef, useState } from "react";
import { usePrefersReducedMotion } from "./lib";

/* Deterministic PRNG so every visit renders the same membrane */
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Pt = { x: number; y: number; h: number };

const STOPS: [number, number, number][] = [
  [16, 46, 68],
  [21, 94, 99],
  [30, 143, 122],
  [55, 230, 196],
  [143, 240, 208],
  [255, 208, 138],
  [255, 180, 84],
  [255, 138, 112],
  [255, 110, 156],
];

export function rampRGB(t: number): [number, number, number] {
  const c = Math.min(STOPS.length - 1.001, Math.max(0, t * (STOPS.length - 1)));
  const i = Math.floor(c);
  const f = c - i;
  const a = STOPS[i];
  const b = STOPS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function makeSprite(rgb: [number, number, number], alpha: number, size = 36) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  const grad = g.createRadialGradient(
    size / 2,
    size / 2,
    0,
    size / 2,
    size / 2,
    size / 2
  );
  grad.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`);
  grad.addColorStop(1, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, size, size);
  return c;
}

function buildScene(W: number, H: number): Pt[] {
  const rnd = mulberry32(1337);
  const pts: Pt[] = [];

  /* hexagonal trimer lattice — a stylized membrane patch */
  const cx = W * 0.33;
  const cy = H * 0.52;
  const rx = W * 0.27;
  const ry = H * 0.4;
  const s = Math.max(22, Math.min(W, H) * 0.088);
  let row = 0;
  for (let y = H * 0.05; y < H * 0.95; y += s * 0.866, row++) {
    for (let x = W * 0.03 + (row % 2) * s * 0.5; x < W * 0.63; x += s) {
      const nx = (x - cx) / rx;
      const ny = (y - cy) / ry;
      const d = nx * nx + ny * ny;
      if (d > 0.9 + 0.22 * rnd()) continue;
      const rr = s * 0.27;
      const base = 0.5 + 0.36 * rnd();
      for (let k = 0; k < 6; k++) {
        const a = (k / 6) * Math.PI * 2 + 0.35;
        pts.push({
          x: x + Math.cos(a) * rr,
          y: y + Math.sin(a) * rr,
          h: base,
        });
      }
      pts.push({ x, y, h: base * 0.55 });
    }
  }

  /* filaments crossing the right third */
  for (let f = 0; f < 3; f++) {
    const base = H * (0.2 + 0.28 * f);
    const amp = H * 0.07;
    const fr = ((2 + f) * Math.PI) / (W * 0.5);
    const ph = f * 2.1;
    for (let x = W * 0.6; x < W * 0.975; x += 4) {
      pts.push({
        x: x + (rnd() - 0.5) * 2,
        y: base + Math.sin(x * fr + ph) * amp,
        h: 0.28 + 0.14 * Math.sin(x * 0.02 + f) + 0.12 * rnd(),
      });
    }
  }

  pts.sort((a, b) => a.y - b.y);
  return pts;
}

function gaussPair(rnd: () => number): [number, number] {
  let u = 0;
  let v = 0;
  while (u === 0) u = rnd();
  while (v === 0) v = rnd();
  const m = Math.sqrt(-2 * Math.log(u));
  return [m * Math.cos(2 * Math.PI * v), m * Math.sin(2 * Math.PI * v)];
}

type Hud = {
  frame: number;
  total: number;
  locs: number;
  res: string;
  util: number;
  phase: "scan" | "hold";
};

export default function SimCanvas() {
  const reduced = usePrefersReducedMotion();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hud, setHud] = useState<Hud>({
    frame: 0,
    total: 0,
    locs: 0,
    res: "—",
    util: 0,
    phase: "scan",
  });

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(1.75, window.devicePixelRatio || 1);
    const acc = document.createElement("canvas");
    const actx = acc.getContext("2d")!;
    const rawC = document.createElement("canvas");
    const rctx = rawC.getContext("2d")!;
    const rnd = mulberry32(4242);
    const sprites = Array.from({ length: 40 }, (_, i) =>
      makeSprite(rampRGB(i / 39), 0.85)
    );
    const rawSprite = makeSprite([96, 148, 178], 0.16, 24);

    let W = 0;
    let H = 0;
    let pts: Pt[] = [];
    let ptr = 0;
    let rowY = 0;
    let locs = 0;
    let raf = 0;
    let mode: "scan" | "hold" = "scan";
    let holdUntil = 0;
    let frameCount = 0;
    let t0 = performance.now();
    const events: { x: number; y: number; ttl: number }[] = [];

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      W = Math.max(300, Math.floor(rect.width));
      H = Math.max(260, Math.floor(rect.height));
      for (const c of [canvas, acc, rawC]) {
        c.width = Math.floor(W * dpr);
        c.height = Math.floor(H * dpr);
      }
      for (const g of [ctx, actx, rctx]) g.setTransform(dpr, 0, 0, dpr, 0, 0);
      pts = buildScene(W, H);
      ptr = 0;
      rowY = 0;
      locs = 0;
      mode = "scan";
      events.length = 0;
      t0 = performance.now();
      actx.clearRect(0, 0, W, H);
      rctx.clearRect(0, 0, W, H);
      for (const p of pts) rctx.drawImage(rawSprite, p.x - 4, p.y - 4, 8, 8);
    };

    const localize = (p: Pt) => {
      const [gx, gy] = gaussPair(rnd);
      const x = p.x + gx * 1.7;
      const y = p.y + gy * 1.7;
      const si = Math.min(
        39,
        Math.max(0, Math.round(p.h * 39))
      );
      actx.globalCompositeOperation = "lighter";
      actx.drawImage(sprites[si], x - 7.5, y - 7.5, 15, 15);
      actx.globalCompositeOperation = "source-over";
      locs++;
      events.push({ x, y, ttl: 1 });
      if (events.length > 26) events.shift();
    };

    const drawGrid = () => {
      ctx.strokeStyle = "rgba(95,178,255,0.05)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < W; x += 42) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
      }
      for (let y = 0; y < H; y += 42) {
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
      }
      ctx.stroke();
    };

    const draw = (t: number) => {
      ctx.fillStyle = "#071019";
      ctx.fillRect(0, 0, W, H);
      drawGrid();

      /* scanned territory: raw topo + accumulating reconstruction */
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, W, Math.max(0, rowY));
      ctx.clip();
      ctx.globalAlpha = 0.5;
      ctx.drawImage(rawC, 0, 0, W, H);
      ctx.globalAlpha = 1;
      ctx.drawImage(acc, 0, 0, W, H);
      ctx.restore();

      /* unscanned territory */
      if (rowY < H) {
        ctx.fillStyle = "rgba(7,16,25,0.93)";
        ctx.fillRect(0, rowY, W, H - rowY);
      }

      /* scan line + tip */
      if (rowY < H) {
        ctx.strokeStyle = "rgba(55,230,196,0.45)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, rowY);
        ctx.lineTo(W, rowY);
        ctx.stroke();
        const cyc = (t * 0.45) % (2 * W);
        const tipX = cyc < W ? cyc : 2 * W - cyc;
        const glow = ctx.createRadialGradient(tipX, rowY, 0, tipX, rowY, 16);
        glow.addColorStop(0, "rgba(143,240,208,0.9)");
        glow.addColorStop(1, "rgba(143,240,208,0)");
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(tipX, rowY, 16, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#eafff8";
        ctx.beginPath();
        ctx.moveTo(tipX, rowY + 5);
        ctx.lineTo(tipX - 4, rowY - 4);
        ctx.lineTo(tipX + 4, rowY - 4);
        ctx.closePath();
        ctx.fill();
      }

      /* live localization crosshairs */
      for (let i = events.length - 1; i >= 0; i--) {
        const e = events[i];
        const a = Math.max(0, e.ttl);
        ctx.strokeStyle = `rgba(255,180,84,${0.85 * a})`;
        ctx.lineWidth = 1;
        ctx.strokeRect(e.x - 8, e.y - 8, 16, 16);
        ctx.beginPath();
        ctx.moveTo(e.x - 12, e.y);
        ctx.lineTo(e.x - 5, e.y);
        ctx.moveTo(e.x + 5, e.y);
        ctx.lineTo(e.x + 12, e.y);
        ctx.moveTo(e.x, e.y - 12);
        ctx.lineTo(e.x, e.y - 5);
        ctx.moveTo(e.x, e.y + 5);
        ctx.lineTo(e.x, e.y + 12);
        ctx.stroke();
        e.ttl -= 0.045;
        if (e.ttl <= 0) events.splice(i, 1);
      }
    };

    const publishHud = (t: number) => {
      const progress = Math.min(1, rowY / H);
      const res = mode === "hold" ? 4.1 : Math.max(4.1, 34 - 31 * Math.sqrt(progress));
      const elapsed = Math.max(0.001, (t - t0) / 1000);
      setHud({
        frame: Math.min(512, Math.floor(progress * 512)),
        total: 512,
        locs,
        res: res.toFixed(1),
        util: Math.max(
          4,
          Math.min(99, Math.round(66 + 26 * Math.sin(t / 640) + (rnd() * 8 - 4)))
        ),
        phase: mode,
      });
    };

    const tick = (t: number) => {
      if (mode === "scan") {
        const prev = rowY;
        rowY += Math.max(1.15, H / 330);
        while (ptr < pts.length && pts[ptr].y <= rowY) {
          localize(pts[ptr]);
          ptr++;
        }
        if (rowY >= H) {
          rowY = H;
          while (ptr < pts.length) {
            localize(pts[ptr]);
            ptr++;
          }
          mode = "hold";
          holdUntil = t + 1900;
        }
        void prev;
      } else if (t > holdUntil) {
        mode = "scan";
        ptr = 0;
        rowY = 0;
        locs = 0;
        events.length = 0;
        t0 = t;
        actx.clearRect(0, 0, W, H);
      }
      draw(t);
      frameCount++;
      if (frameCount % 9 === 0) publishHud(t);
      raf = requestAnimationFrame(tick);
    };

    resize();

    if (reduced) {
      /* static completed reconstruction — no animation loop */
      const renderStatic = () => {
        resize();
        for (const p of pts) localize(p);
        rowY = H;
        mode = "hold";
        draw(performance.now());
        events.length = 0;
        draw(performance.now());
        setHud({
          frame: 512,
          total: 512,
          locs,
          res: "4.1",
          util: 0,
          phase: "hold",
        });
      };
      renderStatic();
      window.addEventListener("resize", renderStatic);
      return () => window.removeEventListener("resize", renderStatic);
    }

    const ro = new ResizeObserver(() => resize());
    ro.observe(wrap);
    const onVis = () => {
      if (document.hidden) cancelAnimationFrame(raf);
      else raf = requestAnimationFrame(tick);
    };
    document.addEventListener("visibilitychange", onVis);
    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [reduced]);

  const stats: { k: string; v: string }[] = [
    { k: "FRAME", v: `${hud.frame}/${hud.total}` },
    { k: "LOCALIZATIONS", v: hud.locs.toLocaleString("en-US") },
    { k: "EST. RES", v: `${hud.res} nm` },
    { k: "GPU UTIL", v: reduced ? "—" : `${hud.util}%` },
  ];

  return (
    <div
      ref={wrapRef}
      className="corner-frame relative h-[340px] w-full overflow-hidden border border-line bg-ink2 sm:h-[420px] lg:h-[540px]"
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      <div className="scanlines pointer-events-none absolute inset-0" />

      {/* top strip */}
      <div className="absolute left-4 top-3 z-10 flex items-center gap-2 font-mono text-[10px] tracking-[0.18em] text-dim">
        <span
          className={`inline-block h-2 w-2 rounded-full ${
            reduced ? "bg-amber" : "bg-teal pulse-dot"
          }`}
        />
        {reduced ? "STATIC PREVIEW · REDUCED MOTION" : "LIVE · LAFM RENDER LOOP"}
      </div>
      <div className="absolute right-4 top-3 z-10 font-mono text-[10px] tracking-[0.18em] text-faint">
        PX 0.49 nm · σ 2.1 · CH HEIGHT
      </div>

      {/* height colorbar */}
      <div className="absolute right-4 top-1/2 z-10 hidden -translate-y-1/2 flex-col items-center gap-1 sm:flex">
        <span className="font-mono text-[9px] text-faint">12nm</span>
        <div
          className="h-36 w-1.5 rounded-full"
          style={{
            background:
              "linear-gradient(to bottom, #ff6e9c, #ffb454, #8ff0d0, #37e6c4, #1e8f7a, #155e63, #102e44)",
          }}
        />
        <span className="font-mono text-[9px] text-faint">0nm</span>
      </div>

      {/* complete chip */}
      {!reduced && hud.phase === "hold" && (
        <div className="absolute left-1/2 top-1/2 z-10 -translate-x-1/2 -translate-y-1/2 border border-teal/60 bg-ink/85 px-4 py-2 font-mono text-[11px] tracking-[0.22em] text-teal backdrop-blur-sm">
          RENDER COMPLETE · FRC 4.1 nm
        </div>
      )}

      {/* bottom HUD */}
      <div className="absolute inset-x-0 bottom-0 z-10 grid grid-cols-4 divide-x divide-line/70 border-t border-line bg-ink/80 backdrop-blur-sm">
        {stats.map((s) => (
          <div key={s.k} className="px-3 py-2 sm:px-4">
            <div className="font-mono text-[8.5px] tracking-[0.2em] text-faint">
              {s.k}
            </div>
            <div className="font-mono text-[12px] font-medium text-teal2 tabular-nums sm:text-[13px]">
              {s.v}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
