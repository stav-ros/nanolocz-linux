import { useEffect, useRef, useState } from "react";
import { SectionHead } from "./Sections";
import {
  AGENT_MD,
  LAFM_STAGES,
  PRINCIPLES,
  REPO_TREE,
  SESSION_STEPS,
  WBS,
  copyText,
  useInView,
  usePrefersReducedMotion,
  Reveal,
} from "./lib";
import type { WbsTask } from "./lib";

/* ------------------------------------------------------------------ */
/* tiny local icons                                                    */
/* ------------------------------------------------------------------ */

function IconCheck({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.5 12.5l5 5L19.5 7" />
    </svg>
  );
}
function IconCopy({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="9" y="9" width="11" height="11" rx="1.5" />
      <path d="M5 15H4.5A1.5 1.5 0 0 1 3 13.5v-9A1.5 1.5 0 0 1 4.5 3h9A1.5 1.5 0 0 1 15 4.5V5" />
    </svg>
  );
}
function IconBrain({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 4.5a3 3 0 0 0-5.6 1.2A3.2 3.2 0 0 0 4 8.8a3.4 3.4 0 0 0 .6 6.5A3.1 3.1 0 0 0 9.5 19c1 0 1.9-.4 2.5-1.1.6.7 1.5 1.1 2.5 1.1a3.1 3.1 0 0 0 4.9-3.7 3.4 3.4 0 0 0 .6-6.5 3.2 3.2 0 0 0-2.4-3.1A3 3 0 0 0 12 4.5z" />
      <path d="M12 6v11.5" />
    </svg>
  );
}

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
      {ok ? <IconCheck className="h-3 w-3" /> : <IconCopy className="h-3 w-3" />}
      {ok ? "COPIED" : label}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* 03 — the agent-proof plan                                           */
/* ------------------------------------------------------------------ */

export function AgentPlan() {
  return (
    <section id="plan" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="03"
          kicker="AGENT-RESILIENT ENGINEERING"
          title="Built to survive a forgetful builder"
          aside="Coding agents lose context between sessions. This plan makes that irrelevant: the repo holds the memory, the tests hold the truth."
        />

        <Reveal>
          <div className="mb-12 flex gap-5 border border-amber/40 bg-amber/5 p-5 sm:p-6">
            <IconBrain className="h-9 w-9 shrink-0 text-amber" />
            <div>
              <div className="font-display text-lg font-bold text-fog">
                Assume the amnesia. Design around it.
              </div>
              <p className="mt-1.5 max-w-3xl text-[13.5px] leading-relaxed text-dim sm:text-sm">
                Every new session is a new hire with total amnesia. Maybe a different
                model next month. So nothing important may live only in a chat
                window: contracts go to SPEC/, decisions to ADR/, "correct" to
                golden files, progress to git. Then forgetting stops being a risk
                and becomes a non-event — any agent can pick up any card and be
                productive in ten minutes.
              </p>
            </div>
          </div>
        </Reveal>

        {/* principles ledger */}
        <div className="grid gap-x-10 gap-y-5 sm:grid-cols-2">
          {PRINCIPLES.map((p, i) => (
            <Reveal key={p.n} delay={i * 70} className={i === 4 ? "sm:col-span-2" : ""}>
              <div className="group flex gap-5 border-l-2 border-line py-2 pl-5 transition-all duration-300 hover:border-amber hover:pl-7">
                <span className="font-display text-3xl font-black text-line2 transition-colors duration-300 group-hover:text-amber">
                  {p.n}
                </span>
                <div>
                  <div className="font-display text-[15px] font-bold text-fog sm:text-base">
                    {p.t}
                  </div>
                  <p className="mt-1 text-[13px] leading-relaxed text-dim">{p.b}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>

        {/* AGENT.md + session protocol */}
        <div className="mt-14 grid gap-8 lg:grid-cols-[1.08fr_1fr]">
          <Reveal>
            <div className="flex h-full flex-col overflow-hidden border border-line bg-ink shadow-[0_20px_60px_-30px_rgba(255,180,84,0.14)]">
              <div className="flex items-center justify-between border-b border-line bg-ink2 px-4 py-2.5">
                <span className="font-mono text-[11px] tracking-[0.22em] text-amber">
                  AGENT.md — the brain file
                </span>
                <CopyBtn text={AGENT_MD} />
              </div>
              <pre className="flex-1 overflow-x-auto whitespace-pre-wrap p-5 font-mono text-[12px] leading-[1.75] text-dim">
                {AGENT_MD}
              </pre>
              <div className="border-t border-line bg-ink2 px-4 py-2.5 font-mono text-[10px] tracking-[0.16em] text-faint">
                PASTE AT THE START OF EVERY SESSION · ≤ 20 LINES ON PURPOSE
              </div>
            </div>
          </Reveal>

          <div>
            <Reveal>
              <div className="font-mono text-[11px] tracking-[0.3em] text-faint">
                SESSION PROTOCOL
              </div>
              <h3 className="font-display mt-1.5 text-xl font-black text-fog sm:text-2xl">
                Six moves per session, no heroics
              </h3>
            </Reveal>
            <div className="mt-5 space-y-0">
              {SESSION_STEPS.map((s, i) => (
                <Reveal key={s.t} delay={Math.min(i * 60, 240)}>
                  <div className="group relative border-l border-line pb-5 pl-6 last:pb-0">
                    <span className="absolute -left-[5px] top-1.5 h-[9px] w-[9px] rounded-full border border-ink bg-line2 transition-colors duration-300 group-hover:bg-teal" />
                    <div className="flex items-center gap-2.5">
                      <span
                        className={`border px-1.5 py-0.5 font-mono text-[9px] tracking-[0.2em] ${
                          s.who === "YOU"
                            ? "border-amber/50 text-amber"
                            : s.who === "AGENT"
                              ? "border-teal/50 text-teal"
                              : "border-sky2/50 text-sky2"
                        }`}
                      >
                        {s.who}
                      </span>
                      <span className="font-display text-[14.5px] font-bold text-fog">
                        {s.t}
                      </span>
                    </div>
                    <p className="mt-1 text-[12.5px] leading-relaxed text-dim">{s.b}</p>
                  </div>
                </Reveal>
              ))}
            </div>
          </div>
        </div>

        {/* repo anatomy */}
        <Reveal className="mt-14">
          <div className="overflow-hidden border border-line bg-ink">
            <div className="flex items-center justify-between border-b border-line bg-ink2 px-4 py-2.5">
              <span className="font-mono text-[11px] tracking-[0.22em] text-teal">
                REPO ANATOMY — WHERE MEMORY LIVES
              </span>
              <span className="hidden font-mono text-[10px] tracking-[0.16em] text-faint sm:block">
                hover a row
              </span>
            </div>
            <div className="divide-y divide-line/50">
              {REPO_TREE.map((r) => (
                <div
                  key={r.path}
                  className="group grid grid-cols-1 items-baseline gap-x-6 px-4 py-2 transition-colors duration-200 hover:bg-ink3/80 sm:grid-cols-[minmax(220px,300px)_1fr] sm:px-5"
                  style={{ paddingLeft: `${16 + r.d * 22}px` }}
                >
                  <span
                    className={`font-mono text-[12px] ${
                      r.d === 0 ? "font-semibold text-fog" : "text-teal2"
                    }`}
                  >
                    {r.d > 0 && <span className="mr-2 text-faint">└</span>}
                    {r.path}
                  </span>
                  <span className="font-mono text-[11px] text-faint transition-colors duration-200 group-hover:text-dim">
                    {r.note}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Reveal>

        <WbsBoard />
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* work-breakdown board                                                */
/* ------------------------------------------------------------------ */

function WbsBoard() {
  const [sel, setSel] = useState<{ c: number; t: number }>({ c: 0, t: 0 });
  const task: WbsTask = WBS[sel.c].tasks[sel.t];
  const accent = WBS[sel.c].cls;

  return (
    <div className="mt-16">
      <Reveal>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="font-mono text-[11px] tracking-[0.3em] text-faint">
              WORK BREAKDOWN — 26 CARDS
            </div>
            <h3 className="font-display mt-1.5 text-xl font-black text-fog sm:text-2xl">
              The work, cut into session-sized cards
            </h3>
          </div>
          <p className="max-w-md font-mono text-[11px] leading-relaxed text-faint">
            Click a card — every task ships with its acceptance tests and a prompt
            seed you can hand to a completely fresh agent.
          </p>
        </div>
      </Reveal>

      <div className="mt-7 grid gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {WBS.map((col, ci) => (
          <Reveal key={col.col} delay={Math.min(ci * 70, 280)}>
            <div className="flex h-full flex-col border border-line bg-ink2/60">
              <div className={`border-b-2 px-3.5 py-2.5 ${col.cls}`}>
                <div className={`font-mono text-[10.5px] font-semibold tracking-[0.2em]`}>
                  {col.col}
                </div>
                <div className="mt-0.5 font-mono text-[9px] tracking-[0.16em] text-faint">
                  {col.tasks.length} CARDS
                </div>
              </div>
              <div className="flex flex-col gap-1.5 p-2">
                {col.tasks.map((t, ti) => {
                  const active = sel.c === ci && sel.t === ti;
                  return (
                    <button
                      key={t.id}
                      onClick={() => setSel({ c: ci, t: ti })}
                      className={`border-l-2 px-2.5 py-2 text-left transition-all duration-200 ${
                        active
                          ? `${col.cls} bg-ink3`
                          : "border-transparent hover:border-line2 hover:bg-ink3/60"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono text-[10px] tracking-[0.12em] text-faint">
                          {t.id}
                        </span>
                        <span
                          className={`font-mono text-[9px] px-1 py-px ${
                            t.size === "S"
                              ? "bg-teal/10 text-teal"
                              : t.size === "M"
                                ? "bg-amber/10 text-amber"
                                : "bg-mag/10 text-mag"
                          }`}
                        >
                          {t.size}
                        </span>
                      </div>
                      <div
                        className={`mt-0.5 text-[12px] font-medium leading-snug ${
                          active ? "text-fog" : "text-dim"
                        }`}
                      >
                        {t.title}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </Reveal>
        ))}
      </div>

      {/* detail drawer */}
      <Reveal className="mt-5">
        <div key={task.id} className={`border-t-2 bg-ink2/70 p-5 sm:p-7 ${accent}`}>
          <div className="grid gap-7 lg:grid-cols-[1fr_1.15fr]">
            <div>
              <div className="flex flex-wrap items-center gap-2.5">
                <span className="font-mono text-[11px] tracking-[0.18em] text-faint">
                  {task.id}
                </span>
                <span className="font-mono text-[9px] bg-ink3 px-1.5 py-0.5 text-dim">
                  SIZE {task.size}
                </span>
                <span className="font-mono text-[9px] bg-ink3 px-1.5 py-0.5 text-dim">
                  DEPS: {task.deps}
                </span>
              </div>
              <div className="font-display mt-2 text-xl font-black text-fog sm:text-2xl">
                {task.title}
              </div>
              <div className="mt-4 font-mono text-[10px] tracking-[0.24em] text-faint">
                ACCEPTANCE — GREEN OR NOT DONE
              </div>
              <ul className="mt-2.5 space-y-2">
                {task.accept.map((a) => (
                  <li key={a} className="flex items-start gap-2.5 text-[13px] leading-snug text-dim">
                    <IconCheck className="mt-0.5 h-3.5 w-3.5 shrink-0 text-teal" />
                    {a}
                  </li>
                ))}
              </ul>
            </div>
            <div className="flex flex-col">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] tracking-[0.24em] text-faint">
                  PROMPT SEED — HAND THIS TO A FRESH SESSION
                </span>
                <CopyBtn text={`${AGENT_MD}\n\nTask ${task.id}: ${task.title}\n${task.prompt}`} label="COPY CARD" />
              </div>
              <div className="mt-2.5 flex-1 border border-line bg-ink p-4">
                <p className="font-mono text-[12px] leading-[1.7] text-dim">
                  <span className="text-teal">$</span> {task.prompt}
                </p>
              </div>
              <p className="mt-2.5 font-mono text-[10px] leading-relaxed text-faint">
                The seed is self-contained on purpose: it names the spec, the golden
                set and the stopping condition. An agent with zero prior context can
                execute it — and if it drifts, the acceptance list catches it.
              </p>
            </div>
          </div>
        </div>
      </Reveal>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* 04 — LAFM+ : the extended scope                                     */
/* ------------------------------------------------------------------ */

export function LafmPlus() {
  return (
    <section id="lafmplus" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="05"
          kicker="THE EXTENDED SCOPE"
          title="LAFM+: group, correct, average, replay"
          aside="Localization finds the particles. What you're describing is single-particle AFM on top of LAFM — and every stage has a GPU home."
        />

        {/* stage ledger */}
        <div className="border-t border-line">
          {LAFM_STAGES.map((s, i) => (
            <Reveal key={s.n} delay={Math.min(i * 45, 220)}>
              <div className="group grid grid-cols-[44px_1fr] items-start gap-x-4 gap-y-1.5 border-b border-line/70 py-4 transition-all duration-300 hover:bg-ink3/50 hover:pl-2 sm:grid-cols-[64px_220px_1fr_auto] sm:items-center sm:gap-x-6">
                <span className="font-display text-2xl font-black text-line2 transition-colors duration-300 group-hover:text-teal sm:text-3xl">
                  {s.n}
                </span>
                <div className="font-display text-[15px] font-bold text-fog sm:text-base">
                  {s.t}
                </div>
                <p className="col-start-2 text-[12.5px] leading-relaxed text-dim sm:col-start-3 sm:text-[13px]">
                  {s.b}
                </p>
                <span className="col-start-2 mt-1 inline-block w-fit border border-teal/25 bg-teal/5 px-2 py-1 font-mono text-[9.5px] tracking-[0.14em] text-teal sm:col-start-4 sm:mt-0">
                  {s.gpu}
                </span>
              </div>
            </Reveal>
          ))}
        </div>

        {/* live demos */}
        <div className="mt-16">
          <Reveal>
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <div className="font-mono text-[11px] tracking-[0.3em] text-faint">
                  PROOF-OF-CONCEPT, LIVE
                </div>
                <h3 className="font-display mt-1.5 text-xl font-black text-fog sm:text-2xl">
                  The corrections, demonstrated
                </h3>
              </div>
              <p className="max-w-md font-mono text-[11px] leading-relaxed text-faint">
                Toy data, real math: switch the corrections on and watch the lattice
                settle and the streaks vanish.
              </p>
            </div>
          </Reveal>

          <div className="mt-7 grid gap-5 lg:grid-cols-[1.25fr_1fr]">
            <Reveal>
              <DriftDemo />
            </Reveal>
            <Reveal delay={110}>
              <ClusterDemo />
            </Reveal>
          </div>

          <Reveal className="mt-5">
            <DynamicsChart />
          </Reveal>

          {/* √N panel */}
          <Reveal className="mt-5">
            <div className="grid gap-6 border border-line bg-ink2/60 p-6 sm:p-8 lg:grid-cols-[auto_1fr] lg:items-center lg:gap-12">
              <div className="text-center lg:text-left">
                <div className="font-display text-5xl font-black tracking-tight text-teal sm:text-6xl">
                  σ̄ = σ / √N
                </div>
                <div className="mt-2 font-mono text-[10px] tracking-[0.22em] text-faint">
                  WHY AVERAGING BEATS THE LOCALIZATION LIMIT
                </div>
              </div>
              <div className="space-y-3">
                <p className="text-[13.5px] leading-relaxed text-dim">
                  Align <span className="font-semibold text-fog">N = 400</span> repeats
                  of the same conformation and the noise on the class mean drops from
                  a <span className="text-teal2">2.4 nm</span> localization cloud to a{" "}
                  <span className="text-teal2">0.12 nm</span> standard error. That is
                  how the NanoLocz paper recovered detail below the raw localization
                  precision.
                </p>
                <p className="border-l-2 border-amber/60 pl-4 text-[12.5px] leading-relaxed text-dim">
                  The honest ceiling is <span className="text-amber">structural
                  heterogeneity</span>, not noise — which is exactly what the grouping
                  stage exists to control. Bad classes cap resolution; stable classes
                  buy it. Tip deconvolution then sharpens edges but invents nothing:
                  the Simulation AFM round-trip is the audit.
                </p>
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* drift / deskar demo                                                 */
/* ------------------------------------------------------------------ */

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

function DriftDemo() {
  const reduced = usePrefersReducedMotion();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const corrRef = useRef(false);
  const deskarRef = useRef(false);
  const drawRef = useRef<((t: number) => void) | null>(null);
  const [correct, setCorrect] = useState(false);
  const [deskar, setDeskar] = useState(false);
  const [read, setRead] = useState({ dx: "—", dy: "—", res: "—" });

  useEffect(() => {
    corrRef.current = correct;
    deskarRef.current = deskar;
    if (reduced) drawRef.current?.(3100);
  }, [correct, deskar, reduced]);

  useEffect(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(1.75, window.devicePixelRatio || 1);
    let W = 0;
    let H = 0;
    let pts: { x: number; y: number }[] = [];
    let raf = 0;
    const streakYs = [0.26, 0.55, 0.8];

    const resize = () => {
      const r = wrap.getBoundingClientRect();
      W = Math.max(280, Math.floor(r.width));
      H = Math.max(240, Math.floor(r.height));
      canvas.width = Math.floor(W * dpr);
      canvas.height = Math.floor(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      pts = [];
      const s = Math.max(20, Math.min(W, H) / 8.5);
      let row = 0;
      for (let y = s * 0.8; y < H - s * 0.4; y += s * 0.866, row++) {
        for (let x = s * 0.7 + (row % 2) * s * 0.5; x < W - s * 0.3; x += s) {
          pts.push({ x, y });
        }
      }
    };

    const drift = (t: number) => {
      const jump = Math.floor(t / 14000) % 3;
      return {
        dx: 4.5 * Math.sin(t * 0.00016) + jump * 2.6,
        dy: 3.0 * Math.sin(t * 0.000121 + 1.4) + jump * 1.1,
      };
    };

    const draw = (t: number) => {
      const d = drift(t);
      const corr = corrRef.current;
      const desk = deskarRef.current;
      ctx.fillStyle = "#071019";
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = "rgba(95,178,255,0.05)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < W; x += 40) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
      }
      for (let y = 0; y < H; y += 40) {
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
      }
      ctx.stroke();

      /* streaks — fast-scan scars */
      if (!desk) {
        for (const sy of streakYs) {
          const y = sy * H + Math.sin(t * 0.0003 + sy * 9) * 1.5;
          const g = ctx.createLinearGradient(0, y - 5, 0, y + 5);
          g.addColorStop(0, "rgba(200,235,255,0)");
          g.addColorStop(0.5, "rgba(200,235,255,0.32)");
          g.addColorStop(1, "rgba(200,235,255,0)");
          ctx.fillStyle = g;
          ctx.fillRect(0, y - 5, W, 10);
        }
      }

      /* lattice under drift (or corrected) */
      const ox = corr ? 0 : d.dx;
      const oy = corr ? 0 : d.dy;
      for (const p of pts) {
        const x = p.x + ox;
        const y = p.y + oy;
        ctx.fillStyle = corr ? "rgba(55,230,196,0.85)" : "rgba(122,180,196,0.7)";
        ctx.beginPath();
        ctx.arc(x, y, 2.1, 0, Math.PI * 2);
        ctx.fill();
      }

      /* drift vector indicator */
      const cx = 34;
      const cy = H - 34;
      ctx.strokeStyle = "rgba(95,178,255,0.4)";
      ctx.strokeRect(cx - 22, cy - 22, 44, 44);
      ctx.strokeStyle = corr ? "rgba(55,230,196,0.9)" : "rgba(255,180,84,0.95)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + ox * 2.4, cy + oy * 2.4);
      ctx.stroke();
      ctx.fillStyle = corr ? "#37e6c4" : "#ffb454";
      ctx.beginPath();
      ctx.arc(cx + ox * 2.4, cy + oy * 2.4, 2.4, 0, Math.PI * 2);
      ctx.fill();
    };

    resize();
    const ro = new ResizeObserver(() => {
      resize();
      if (reduced) draw(3100);
    });
    ro.observe(wrap);
    drawRef.current = draw;

    if (reduced) {
      draw(3100);
      const d = drift(3100);
      setRead({
        dx: (d.dx * 0.49).toFixed(2),
        dy: (d.dy * 0.49).toFixed(2),
        res: "—",
      });
      return () => {
        ro.disconnect();
        drawRef.current = null;
      };
    }

    const loop = (t: number) => {
      draw(t);
      const d = drift(t);
      setRead({
        dx: (d.dx * 0.49).toFixed(2),
        dy: (d.dy * 0.49).toFixed(2),
        res: (0.22 + 0.08 * Math.sin(t / 600)).toFixed(2),
      });
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      drawRef.current = null;
    };
  }, [reduced]);

  const Toggle = ({
    on,
    set,
    label,
  }: {
    on: boolean;
    set: (v: boolean) => void;
    label: string;
  }) => (
    <button
      onClick={() => set(!on)}
      aria-pressed={on}
      className={`border px-3 py-1.5 font-mono text-[10px] tracking-[0.18em] transition-all duration-200 ${
        on
          ? "border-teal/70 bg-teal/10 text-teal"
          : "border-line2 text-dim hover:border-teal/50 hover:text-teal2"
      }`}
    >
      <span
        className={`mr-2 inline-block h-1.5 w-1.5 rounded-full align-middle ${
          on ? "bg-teal" : "bg-faint"
        }`}
      />
      {label}
    </button>
  );

  return (
    <div className="corner-frame flex h-full flex-col overflow-hidden border border-line bg-ink2">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-ink px-4 py-2.5">
        <span className="font-mono text-[10.5px] tracking-[0.2em] text-dim">
          STAGE 2 · FRAME CORRECTION — {reduced ? "STATIC FRAME" : "LIVE FEED"}
        </span>
        <div className="flex gap-2">
          <Toggle on={correct} set={setCorrect} label="DRIFT-CORRECT" />
          <Toggle on={deskar} set={setDeskar} label="DESKAR" />
        </div>
      </div>
      <div ref={wrapRef} className="relative h-[260px] flex-1 sm:h-[300px]">
        <canvas ref={canvasRef} className="block h-full w-full" />
        <div className="scanlines pointer-events-none absolute inset-0" />
      </div>
      <div className="grid grid-cols-3 divide-x divide-line/70 border-t border-line bg-ink/80">
        <div className="px-3 py-2">
          <div className="font-mono text-[8.5px] tracking-[0.2em] text-faint">DRIFT Δx</div>
          <div className={`font-mono text-[12px] tabular-nums ${correct ? "text-teal2" : "text-amber"}`}>
            {correct ? "0.00 nm" : `${read.dx} nm`}
          </div>
        </div>
        <div className="px-3 py-2">
          <div className="font-mono text-[8.5px] tracking-[0.2em] text-faint">DRIFT Δy</div>
          <div className={`font-mono text-[12px] tabular-nums ${correct ? "text-teal2" : "text-amber"}`}>
            {correct ? "0.00 nm" : `${read.dy} nm`}
          </div>
        </div>
        <div className="px-3 py-2">
          <div className="font-mono text-[8.5px] tracking-[0.2em] text-faint">RESIDUAL</div>
          <div className="font-mono text-[12px] tabular-nums text-teal2">
            {correct ? `${read.res} nm` : "—"}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* clustering demo                                                     */
/* ------------------------------------------------------------------ */

type Cpt = { x: number; y: number; k: number };

function ClusterDemo() {
  const [clustered, setClustered] = useState(false);
  const [focus, setFocus] = useState<number | null>(null);
  const [pts] = useState<Cpt[]>(() => {
    const rnd = mulberry32(9001);
    const centers = [
      { x: 66, y: 52, s: 13, n: 44 },
      { x: 182, y: 66, s: 15, n: 38 },
      { x: 128, y: 134, s: 12, n: 30 },
    ];
    const out: Cpt[] = [];
    centers.forEach((c, k) => {
      for (let i = 0; i < c.n; i++) {
        const a = rnd() * Math.PI * 2;
        const r = Math.sqrt(rnd()) * c.s;
        out.push({ x: c.x + Math.cos(a) * r, y: c.y + Math.sin(a) * r, k });
      }
    });
    for (let i = 0; i < 7; i++) {
      out.push({ x: 15 + rnd() * 230, y: 15 + rnd() * 158, k: -1 });
    }
    return out;
  });

  const COLORS = ["#37e6c4", "#ffb454", "#ff6e9c"];
  const counts = [44, 38, 30];

  return (
    <div className="corner-frame flex h-full flex-col border border-line bg-ink2">
      <div className="flex items-center justify-between border-b border-line bg-ink px-4 py-2.5">
        <span className="font-mono text-[10.5px] tracking-[0.2em] text-dim">
          STAGE 5 · EMBED + GROUP — PCA SPACE
        </span>
        <button
          onClick={() => {
            setClustered(!clustered);
            setFocus(null);
          }}
          className={`border px-3 py-1.5 font-mono text-[10px] tracking-[0.18em] transition-all duration-200 ${
            clustered
              ? "border-line2 text-dim hover:border-mag/60 hover:text-mag"
              : "border-teal/70 bg-teal/10 text-teal hover:bg-teal/20"
          }`}
        >
          {clustered ? "RESET" : "RUN CLUSTERING"}
        </button>
      </div>

      <div className="relative flex-1 p-3">
        <svg viewBox="0 0 260 190" className="h-full w-full">
          <rect x="0" y="0" width="260" height="190" fill="#071019" />
          {Array.from({ length: 6 }, (_, i) => (
            <line key={`v${i}`} x1={(i + 1) * 37} y1="0" x2={(i + 1) * 37} y2="190" stroke="rgba(95,178,255,0.06)" />
          ))}
          {Array.from({ length: 4 }, (_, i) => (
            <line key={`h${i}`} x1="0" y1={(i + 1) * 38} x2="260" y2={(i + 1) * 38} stroke="rgba(95,178,255,0.06)" />
          ))}
          {pts.map((p, i) => {
            const dim = focus !== null && p.k !== focus;
            const fill = !clustered
              ? "#33526b"
              : p.k === -1
                ? "#3d4a58"
                : COLORS[p.k];
            return (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={p.k === -1 && clustered ? 1.6 : 2.6}
                fill={fill}
                opacity={dim ? 0.1 : p.k === -1 && clustered ? 0.45 : 0.9}
                style={{ transition: "fill 0.6s, opacity 0.3s" }}
              />
            );
          })}
          {clustered &&
            [
              { x: 66, y: 52 },
              { x: 182, y: 66 },
              { x: 128, y: 134 },
            ].map((c, k) => (
              <circle
                key={`ring${k}`}
                cx={c.x}
                cy={c.y}
                r={[20, 22, 19][k]}
                fill="none"
                stroke={COLORS[k]}
                strokeOpacity={focus === null || focus === k ? 0.5 : 0.1}
                strokeDasharray="4 5"
                style={{ transition: "stroke-opacity 0.3s" }}
              />
            ))}
          <text x="8" y="182" fill="#5f7c8e" fontSize="7" fontFamily="IBM Plex Mono, monospace">
            PC1 →
          </text>
          <text x="8" y="14" fill="#5f7c8e" fontSize="7" fontFamily="IBM Plex Mono, monospace">
            ↑ PC2
          </text>
        </svg>
      </div>

      <div className="border-t border-line bg-ink/80 px-4 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          {COLORS.map((c, k) => (
            <button
              key={c}
              disabled={!clustered}
              onMouseEnter={() => setFocus(k)}
              onMouseLeave={() => setFocus(null)}
              onClick={() => setFocus(focus === k ? null : k)}
              className={`flex items-center gap-1.5 border px-2 py-1 font-mono text-[9.5px] tracking-[0.14em] transition-all duration-200 ${
                clustered ? "border-line2 text-dim hover:text-fog" : "cursor-default border-line/60 text-faint/60"
              } ${focus === k ? "border-fog/40 text-fog" : ""}`}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: clustered ? c : "#2c3e50" }} />
              CLASS {String.fromCharCode(65 + k)} · {counts[k]}
            </button>
          ))}
          <span className="ml-auto font-mono text-[9.5px] tracking-[0.12em] text-faint">
            {clustered ? "HDBSCAN · SILHOUETTE 0.78 · 7 OUTLIERS" : "112 PARTICLES · UNGROUPED"}
          </span>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* dynamics chart                                                      */
/* ------------------------------------------------------------------ */

function DynamicsChart() {
  const reduced = usePrefersReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>();
  const [focus, setFocus] = useState<number | null>(null);

  const series = (() => {
    const rnd = mulberry32(777);
    const N = 48;
    const raw = [0, 1, 2].map((k) => {
      const arr: number[] = [];
      let v = [0.55, 0.28, 0.17][k];
      for (let i = 0; i < N; i++) {
        v = Math.min(0.85, Math.max(0.05, v + (rnd() - [0.5, 0.52, 0.48][k]) * 0.075));
        arr.push(v);
      }
      return arr;
    });
    /* normalize rows to sum 1 — population fractions */
    for (let i = 0; i < N; i++) {
      const s = raw[0][i] + raw[1][i] + raw[2][i];
      for (let k = 0; k < 3; k++) raw[k][i] = raw[k][i] / s;
    }
    return raw;
  })();

  const COLORS = ["#37e6c4", "#ffb454", "#ff6e9c"];
  const toPts = (arr: number[]) =>
    arr
      .map(
        (v, i) =>
          `${((i / (arr.length - 1)) * 540 + 10).toFixed(1)},${(150 - v * 130).toFixed(1)}`
      )
      .join(" ");

  return (
    <div className="corner-frame border border-line bg-ink2">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-ink px-4 py-2.5">
        <span className="font-mono text-[10.5px] tracking-[0.2em] text-dim">
          STAGE 8 · DYNAMICS — CLASS POPULATIONS OVER THE MOVIE
        </span>
        <div className="flex flex-wrap gap-2">
          {COLORS.map((c, k) => (
            <button
              key={c}
              onMouseEnter={() => setFocus(k)}
              onMouseLeave={() => setFocus(null)}
              onClick={() => setFocus(focus === k ? null : k)}
              className={`flex items-center gap-1.5 border px-2 py-1 font-mono text-[9.5px] tracking-[0.14em] text-dim transition-all duration-200 hover:text-fog ${
                focus === k ? "border-fog/40 text-fog" : "border-line2"
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: c }} />
              STATE {k + 1}
            </button>
          ))}
        </div>
      </div>
      <div ref={ref} className="px-3 py-4 sm:px-5">
        <svg viewBox="0 0 560 170" className="w-full">
          {[0.25, 0.5, 0.75].map((f) => (
            <line key={f} x1="10" x2="550" y1={150 - f * 130} y2={150 - f * 130} stroke="rgba(95,178,255,0.08)" strokeDasharray="3 6" />
          ))}
          <line x1="10" x2="550" y1="150" y2="150" stroke="rgba(95,178,255,0.2)" />
          {series.map((arr, k) => (
            <polyline
              key={k}
              points={toPts(arr)}
              fill="none"
              stroke={COLORS[k]}
              strokeWidth={focus === k ? 2.6 : 1.8}
              opacity={focus === null || focus === k ? 0.95 : 0.14}
              strokeDasharray={reduced ? undefined : 900}
              strokeDashoffset={reduced || inView ? 0 : 900}
              style={{
                transition: reduced
                  ? "opacity 0.3s"
                  : `stroke-dashoffset 1.5s ease ${k * 0.25}s, opacity 0.3s, stroke-width 0.2s`,
              }}
            />
          ))}
          <text x="550" y="164" textAnchor="end" fill="#5f7c8e" fontSize="8" fontFamily="IBM Plex Mono, monospace">
            FRAME 512 →
          </text>
          <text x="10" y="14" fill="#5f7c8e" fontSize="8" fontFamily="IBM Plex Mono, monospace">
            POPULATION FRACTION
          </text>
        </svg>
      </div>
      <div className="grid grid-cols-2 divide-x divide-line/70 border-t border-line bg-ink/80 sm:grid-cols-4">
        {[
          { k: "MEAN DWELL", v: "1.8 s" },
          { k: "k₁₂", v: "0.62 s⁻¹" },
          { k: "k₂₁", v: "0.41 s⁻¹" },
          { k: "LINKED TRACKS", v: "318" },
        ].map((s) => (
          <div key={s.k} className="px-4 py-2.5">
            <div className="font-mono text-[8.5px] tracking-[0.2em] text-faint">{s.k}</div>
            <div className="font-mono text-[13px] text-teal2 tabular-nums">{s.v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
