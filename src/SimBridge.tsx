import { useEffect, useMemo, useRef, useState } from "react";
import { SectionHead } from "./Sections";
import {
  FIRST_SESSIONS,
  KICKOFF_MD,
  SIM_LEDGER,
  Reveal,
  copyText,
} from "./lib";
import { rampRGB } from "./SimCanvas";

/* ------------------------------------------------------------------ */
/* shared bits                                                         */
/* ------------------------------------------------------------------ */

function CopyBtn({ text, label = "COPY" }: { text: string; label?: string }) {
  const [ok, setOk] = useState(false);
  return (
    <button
      onClick={async () => {
        const done = await copyText(text);
        if (done) {
          setOk(true);
          window.setTimeout(() => setOk(false), 1400);
        }
      }}
      className={`inline-flex items-center gap-1.5 border px-2.5 py-1.5 font-mono text-[10px] tracking-[0.18em] transition-all duration-200 ${
        ok
          ? "border-teal/70 bg-teal/10 text-teal"
          : "border-line2 text-dim hover:border-teal/60 hover:text-teal"
      }`}
    >
      {ok ? "COPIED ✓" : label}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* the virtual tip bench — live pseudo-AFM + rough fit                 */
/* ------------------------------------------------------------------ */

const X0 = -70;
const Y0 = -55;
const STEP = 1.25; // Å per cell
const NX = 112;
const NY = 88;
const R0 = 12; // tip the "experiment" was recorded with (Å)
const TRUTH = { dx: 8.4, dy: -5.2 }; // hidden offset (Å)

type Bead = { x: number; y: number; r: number; z: number };

const BEADS: Bead[] = (() => {
  const out: Bead[] = [];
  for (let k = 0; k < 7; k++) {
    const a = (k / 7) * Math.PI * 2;
    out.push({ x: Math.cos(a) * 30, y: Math.sin(a) * 30, r: 9, z: 20 });
  }
  out.push({ x: 0, y: 0, r: 7, z: 13 });
  out.push({ x: Math.cos(0.9) * 34, y: Math.sin(0.9) * 34, r: 6.5, z: 27 });
  out.push({ x: Math.cos(3.8) * 34, y: Math.sin(3.8) * 34, r: 6.5, z: 27 });
  out.push({ x: -46, y: 22, r: 5, z: 15 }); // asymmetric tail — makes the fit unique
  return out;
})();

const TAN_A = Math.tan((10 * Math.PI) / 180); // cone half-angle α = 10°

function synth(R: number, ox: number, oy: number, out: Float64Array) {
  for (let j = 0; j < NY; j++) {
    const py = Y0 + (j + 0.5) * STEP;
    for (let i = 0; i < NX; i++) {
      const px = X0 + (i + 0.5) * STEP;
      let h = 0;
      for (const b of BEADS) {
        const ddx = px - (b.x + ox);
        const ddy = py - (b.y + oy);
        const d = Math.sqrt(ddx * ddx + ddy * ddy);
        const rr = R + b.r;
        let hs = 0;
        if (d < rr) hs = b.z + Math.sqrt(rr * rr - d * d) - R;
        const hc = b.z + Math.max(0, d - b.r) * TAN_A;
        const hh = hs > hc ? hs : hc;
        if (hh > h) h = hh;
      }
      out[j * NX + i] = h;
    }
  }
}

function ncc(a: Float64Array, b: Float64Array): number {
  const n = a.length;
  let ma = 0;
  let mb = 0;
  for (let i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma;
    const y = b[i] - mb;
    num += x * y;
    da += x * x;
    db += y * y;
  }
  return num / Math.sqrt(da * db + 1e-12);
}

function paint(cv: HTMLCanvasElement | null, grid: Float64Array) {
  if (!cv) return;
  const ctx = cv.getContext("2d");
  if (!ctx) return;
  const img = ctx.createImageData(NX, NY);
  for (let i = 0; i < grid.length; i++) {
    const t = Math.min(1, grid[i] / 28);
    const [r, g, b] = rampRGB(t);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="font-mono text-[9.5px] tracking-[0.22em] text-faint">
          {label}
        </span>
        <span className="font-mono text-[12px] text-teal2 tabular-nums">
          {value > 0 && label !== "TIP RADIUS R" ? "+" : ""}
          {value.toFixed(1)} {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
        style={{ accentColor: "#37e6c4" }}
        aria-label={label}
      />
    </div>
  );
}

function VirtualTipBench() {
  const [tipR, setTipR] = useState(12);
  const [dx, setDx] = useState(0);
  const [dy, setDy] = useState(0);
  const targetRef = useRef<HTMLCanvasElement | null>(null);
  const userRef = useRef<HTMLCanvasElement | null>(null);

  const targetGrid = useMemo(() => {
    const g = new Float64Array(NX * NY);
    synth(R0, TRUTH.dx, TRUTH.dy, g);
    return g;
  }, []);

  const userGrid = useMemo(() => {
    const g = new Float64Array(NX * NY);
    synth(tipR, dx, dy, g);
    return g;
  }, [tipR, dx, dy]);

  const c = useMemo(() => ncc(targetGrid, userGrid), [targetGrid, userGrid]);

  useEffect(() => paint(targetRef.current, targetGrid), [targetGrid]);
  useEffect(() => paint(userRef.current, userGrid), [userGrid]);

  const dist = Math.hypot(dx - TRUTH.dx, dy - TRUTH.dy);
  const locked = dist < 1.2 && Math.abs(tipR - R0) < 0.3;
  const rFactor = Math.max(0, 1 - c);
  const barColor = c > 0.95 ? "bg-teal" : c > 0.8 ? "bg-amber" : "bg-mag";

  return (
    <div className="corner-frame overflow-hidden border border-line bg-ink2">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-ink px-4 py-2.5">
        <span className="font-mono text-[10.5px] tracking-[0.2em] text-dim">
          THE VIRTUAL TIP BENCH — LIVE PSEUDO-AFM + ROUGH FIT
        </span>
        <span className="font-mono text-[9.5px] tracking-[0.14em] text-faint">
          α 10° FIXED · STEP a = 1.25 Å · TOY BEAD MODEL
        </span>
      </div>

      <div className="grid gap-6 p-4 sm:p-6 lg:grid-cols-[1.15fr_1fr]">
        {/* two renders */}
        <div>
          <div className="grid grid-cols-2 gap-3">
            <div className="border border-line bg-ink p-2">
              <div className="mb-1.5 flex items-center justify-between px-1">
                <span className="font-mono text-[9px] tracking-[0.18em] text-amber">
                  TARGET
                </span>
                <span className="font-mono text-[8.5px] text-faint">
                  OFFSET HIDDEN
                </span>
              </div>
              <canvas
                ref={targetRef}
                width={NX}
                height={NY}
                className="pixelated aspect-[112/88] w-full border border-line/60"
              />
            </div>
            <div className="border border-line bg-ink p-2">
              <div className="mb-1.5 flex items-center justify-between px-1">
                <span className="font-mono text-[9px] tracking-[0.18em] text-teal">
                  YOUR FIT
                </span>
                <span className="font-mono text-[8.5px] text-faint">
                  R = {tipR.toFixed(1)} Å
                </span>
              </div>
              <canvas
                ref={userRef}
                width={NX}
                height={NY}
                className={`pixelated aspect-[112/88] w-full border transition-colors duration-300 ${
                  locked ? "border-teal/70" : "border-line/60"
                }`}
              />
            </div>
          </div>
          <p className="mt-3 text-[12px] leading-relaxed text-dim">
            The target was rendered with a <span className="text-amber">hidden
            offset</span> and a <span className="text-amber">12 Å probe</span>.
            Match the tip radius, then steer Δx / Δy until the correlation locks —
            exactly the search the GPU kernel will do over thousands of candidates
            at once.
          </p>
          <div
            className={`mt-3 border px-3 py-2 font-mono text-[10.5px] tracking-[0.18em] transition-all duration-300 ${
              locked
                ? "border-teal/70 bg-teal/10 text-teal"
                : "border-line2 text-faint"
            }`}
          >
            {locked ? "✔ FIT LOCKED · C " + c.toFixed(3) + " · TRANSFORM RECOVERED" : "SEARCHING… MATCH THE TIP, THEN THE OFFSET"}
          </div>
        </div>

        {/* controls + score */}
        <div className="space-y-4">
          <Slider label="TIP RADIUS R" value={tipR} min={6} max={24} step={0.5} unit="Å" onChange={setTipR} />
          <Slider label="SHIFT Δx" value={dx} min={-25} max={25} step={0.2} unit="Å" onChange={setDx} />
          <Slider label="SHIFT Δy" value={dy} min={-25} max={25} step={0.2} unit="Å" onChange={setDy} />

          <div className="border border-line bg-ink p-3.5">
            <div className="flex items-baseline justify-between">
              <span className="font-mono text-[9.5px] tracking-[0.22em] text-faint">
                NORMALIZED CROSS-CORRELATION
              </span>
              <span
                className={`font-mono text-[16px] font-semibold tabular-nums ${
                  c > 0.95 ? "text-teal" : c > 0.8 ? "text-amber" : "text-mag"
                }`}
              >
                C = {c.toFixed(3)}
              </span>
            </div>
            <div className="mt-2 h-1.5 w-full overflow-hidden bg-ink3">
              <div
                className={`h-full ${barColor} transition-all duration-150`}
                style={{ width: `${Math.max(0, Math.min(100, c * 100))}%` }}
              />
            </div>
            <div className="mt-2.5 grid grid-cols-2 gap-3">
              <div>
                <div className="font-mono text-[8.5px] tracking-[0.2em] text-faint">
                  R-FACTOR (1 − C)
                </div>
                <div className="font-mono text-[12.5px] text-dim tabular-nums">
                  {rFactor.toFixed(3)}
                </div>
              </div>
              <div>
                <div className="font-mono text-[8.5px] tracking-[0.2em] text-faint">
                  OFFSET ERROR
                </div>
                <div className={`font-mono text-[12.5px] tabular-nums ${dist < 1.2 ? "text-teal2" : "text-dim"}`}>
                  {dist.toFixed(1)} Å
                </div>
              </div>
            </div>
          </div>

          <p className="font-mono text-[9.5px] leading-relaxed tracking-[0.06em] text-faint">
            PAPER'S BAR: C &gt; 0.9 VS EXPERIMENT (ClpB · Cas9 · F1-ATPase).
            WRONG TIP RADIUS CAPS C EVEN AT THE TRUE OFFSET — TIP MISMATCH IS THE
            CLASSIC FIT-FAILURE MODE.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* section 05 — the simulation bridge                                  */
/* ------------------------------------------------------------------ */

export function SimBridge() {
  return (
    <section id="simbridge" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="06"
          kicker="THE SIMULATION BRIDGE"
          title="Put any PDB under a virtual tip"
          aside="BioAFMviewer (Amyot & Flechsig, PLOS Comput Biol 2020) pioneered PDB → pseudo-AFM. We rebuild its core on GPU — and ship the automated fitting the authors called future work."
        />

        {/* facts strip */}
        <Reveal>
          <div className="mb-12 grid gap-px overflow-hidden border border-line bg-line sm:grid-cols-3">
            <div className="bg-ink2 p-5">
              <div className="font-display text-3xl font-black text-teal">CC-BY</div>
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-dim">
                The method paper is open access and specifies the full algorithm —
                VdW hard-sphere collision, cone α + probe R, grid step a. Reimplement
                freely; cite Amyot & Flechsig alongside Heath et al.
              </p>
            </div>
            <div className="bg-ink2 p-5">
              <div className="font-display text-3xl font-black text-amber">C &gt; 0.9</div>
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-dim">
                The paper's quantitative bar: image correlation between simulated and
                experimental hs-AFM for ClpB, Cas9 and F1-ATPase. Our parity target
                against BioAFMviewer renders: C &gt; 0.95, same PDB + parameters.
              </p>
            </div>
            <div className="bg-ink2 p-5">
              <div className="font-display text-3xl font-black text-mag">2020 → NOW</div>
              <p className="mt-1.5 text-[12.5px] leading-relaxed text-dim">
                “An automatized procedure to detect optimal fitting” is the authors'
                own stated future work. A batched GPU grid search closes exactly that
                gap — the bridge is the differentiator, not the renderer.
              </p>
            </div>
          </div>
        </Reveal>

        {/* feature ledger */}
        <div className="border-t border-line">
          {SIM_LEDGER.map((m, i) => (
            <Reveal key={m.from} delay={Math.min(i * 45, 220)}>
              <div className="group grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1 border-b border-line/70 py-3.5 transition-all duration-300 hover:bg-ink3/70 hover:pl-3 sm:grid-cols-[1fr_36px_1.15fr]">
                <div>
                  <span className="font-display text-[14.5px] font-bold text-fog">
                    {m.from}
                  </span>
                  <p className="mt-0.5 text-[11.5px] text-faint sm:hidden">{m.note}</p>
                </div>
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="hidden h-4 w-4 text-teal opacity-0 transition-opacity duration-300 group-hover:opacity-100 sm:block"
                >
                  <path d="M4 12h15M13 6l6 6-6 6" />
                </svg>
                <div>
                  <div className="font-mono text-[12px] text-teal2">{m.to}</div>
                  <div className="mt-0.5 hidden text-[11px] text-faint sm:block">
                    {m.note}
                  </div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>

        {/* the math */}
        <Reveal className="mt-14">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="font-mono text-[11px] tracking-[0.3em] text-faint">
                THE SPEC IN THREE EQUATIONS
              </div>
              <h3 className="font-display mt-1.5 text-xl font-black text-fog sm:text-2xl">
                Small enough to fit on one index card
              </h3>
            </div>
            <p className="max-w-md font-mono text-[11px] leading-relaxed text-faint">
              This is the entire algorithm an agent has to implement — which is why
              it's a two-week workstream, not a quarter.
            </p>
          </div>
        </Reveal>
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <Reveal>
            <div className="h-full border border-line bg-ink p-5 transition-colors duration-300 hover:border-teal/50">
              <div className="font-mono text-[10px] tracking-[0.24em] text-teal">
                1 · HEIGHT FIELD
              </div>
              <pre className="mt-3 overflow-x-auto font-mono text-[12px] leading-[1.8] text-dim">
{`h_s = max_i[ z_i + √((R+r_i)² − d_i²) − R ]
h_c = max_i[ z_i + (d_i − r_i)·tan α ]
h   = max(h_s, h_c, 0)   grid step a`}
              </pre>
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                Probe sphere of radius R plus cone half-angle α hard-colliding with
                VdW spheres; virtual surface = min-z plane of the oriented structure.
              </p>
            </div>
          </Reveal>
          <Reveal delay={90}>
            <div className="h-full border border-line bg-ink p-5 transition-colors duration-300 hover:border-amber/50">
              <div className="font-mono text-[10px] tracking-[0.24em] text-amber">
                2 · SCORE
              </div>
              <pre className="mt-3 overflow-x-auto font-mono text-[12px] leading-[1.8] text-dim">
{`C = Σ(a−ā)(b−b̄)
    ─────────────────────────
    √( Σ(a−ā)² · Σ(b−b̄)² )

masked ROI · heights normalized`}
              </pre>
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                Normalized cross-correlation between simulated and experimental
                image. Simulation overshoots real heights, so z is always
                normalized before scoring.
              </p>
            </div>
          </Reveal>
          <Reveal delay={180}>
            <div className="h-full border border-line bg-ink p-5 transition-colors duration-300 hover:border-mag/50">
              <div className="font-mono text-[10px] tracking-[0.24em] text-mag">
                3 · ROUGH-FIT SEARCH
              </div>
              <pre className="mt-3 overflow-x-auto font-mono text-[12px] leading-[1.8] text-dim">
{`tx, ty ∈ ±50 Å   @ 2 Å
θ      ∈ 0–358°  @ 2°
→ one batched CUDA launch
→ Nelder–Mead refine
   (tx, ty, θ, z-scale)`}
              </pre>
              <p className="mt-3 text-[11.5px] leading-relaxed text-faint">
                Rotate the bead cloud per candidate (cheap), synthesize ~4,600
                candidates in a single kernel batch, keep the ΔC map as the
                uncertainty surface.
              </p>
            </div>
          </Reveal>
        </div>

        {/* live bench */}
        <Reveal className="mt-12">
          <VirtualTipBench />
        </Reveal>

        {/* kickoff for the agent */}
        <div className="mt-16 grid gap-8 lg:grid-cols-[1.35fr_1fr]">
          <Reveal>
            <div className="flex h-full flex-col overflow-hidden border border-line bg-ink shadow-[0_20px_60px_-30px_rgba(95,178,255,0.16)]">
              <div className="flex items-center justify-between border-b border-line bg-ink2 px-4 py-2.5">
                <span className="font-mono text-[11px] tracking-[0.22em] text-sky2">
                  KICKOFF.md — HAND THIS TO YOUR AGENT
                </span>
                <CopyBtn text={KICKOFF_MD} label="COPY FULL BRIEF" />
              </div>
              <pre className="max-h-[520px] flex-1 overflow-auto whitespace-pre-wrap p-5 font-mono text-[11.5px] leading-[1.7] text-dim">
                {KICKOFF_MD}
              </pre>
              <div className="border-t border-line bg-ink2 px-4 py-2.5 font-mono text-[10px] tracking-[0.16em] text-faint">
                SELF-CONTAINED ON PURPOSE · ALGORITHM + CONVENTIONS + ACCEPTANCE + ORACLE
              </div>
            </div>
          </Reveal>

          <div>
            <Reveal>
              <div className="font-mono text-[11px] tracking-[0.3em] text-faint">
                THE FIRST WEEK
              </div>
              <h3 className="font-display mt-1.5 text-xl font-black text-fog sm:text-2xl">
                Five sessions to a working bridge
              </h3>
              <p className="mt-2 text-[13px] leading-relaxed text-dim">
                Each session maps to one work-breakdown card with its own acceptance
                test — a fresh agent never needs to know what happened before.
              </p>
            </Reveal>
            <div className="mt-5 space-y-2.5">
              {FIRST_SESSIONS.map((s, i) => (
                <Reveal key={s.n} delay={Math.min(i * 70, 280)}>
                  <div className="group flex items-center gap-4 border border-line bg-ink2/60 px-4 py-3 transition-all duration-300 hover:border-sky2/60 hover:bg-ink3">
                    <span className="font-display text-xl font-black text-line2 transition-colors duration-300 group-hover:text-sky2">
                      {s.n}
                    </span>
                    <span className="flex-1 text-[13px] leading-snug text-dim group-hover:text-fog">
                      {s.t}
                    </span>
                    <span className="shrink-0 border border-line2 px-1.5 py-0.5 font-mono text-[9.5px] tracking-[0.12em] text-faint">
                      {s.card}
                    </span>
                  </div>
                </Reveal>
              ))}
            </div>
            <Reveal delay={340}>
              <div className="mt-5 border-l-2 border-sky2/70 pl-4 text-[12.5px] leading-relaxed text-dim">
                Oracle loop: keep BioAFMviewer installed locally. After NL-52, the
                parity suite diffs your renders against its exports at C &gt; 0.95 —
                the original tool becomes your golden-file generator.
              </div>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}
