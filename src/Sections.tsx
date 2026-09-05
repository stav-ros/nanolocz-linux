import { useEffect, useState, type ReactNode } from "react";
import SimCanvas from "./SimCanvas";
import {
  BENCH,
  ENV_NOTES,
  FAQS,
  FEATURES,
  MAPPING,
  PHASES,
  PIPELINE,
  RISKS,
  TERM_LINES,
  TICKER,
  copyText,
  useCountUp,
  useInView,
  usePrefersReducedMotion,
  useScramble,
  Reveal,
  type PortSize,
} from "./lib";

/* ------------------------------------------------------------------ */
/* icons — hand-drawn inline SVG                                       */
/* ------------------------------------------------------------------ */

function IconFork({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="5" r="2.4" />
      <circle cx="18" cy="5" r="2.4" />
      <circle cx="12" cy="19" r="2.4" />
      <path d="M6 7.5v2.2c0 2 1.6 3.3 3.6 3.3h4.8c2 0 3.6-1.3 3.6-3.3V7.5" />
      <path d="M12 13v3.6" />
    </svg>
  );
}
function IconChip({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="6" y="6" width="12" height="12" rx="1.5" />
      <rect x="9.5" y="9.5" width="5" height="5" />
      <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
    </svg>
  );
}
function IconTip({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 16l8-12 8 12" />
      <path d="M2 20c3-2.4 6-2.4 10 0s7 2.4 10 0" />
      <circle cx="12" cy="12.5" r="1.6" fill="currentColor" stroke="none" />
    </svg>
  );
}
function IconScale({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3v18M8 21h8M12 5l-6 3m6-3l6 3" />
      <path d="M3 13l3-5 3 5c0 1.6-1.3 3-3 3s-3-1.4-3-3zM15 13l3-5 3 5c0 1.6-1.3 3-3 3s-3-1.4-3-3z" />
    </svg>
  );
}
function IconTerm({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M7 9l4 3-4 3M13 15h4" />
    </svg>
  );
}
function IconArrow({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 12h15M13 6l6 6-6 6" />
    </svg>
  );
}
function IconCheck({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.5 12.5l5 5L19.5 7" />
    </svg>
  );
}
function IconPlus({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}
function IconWave({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
      <path d="M2 12c2.5-5 5-5 7.5 0s5 5 7.5 0 3.5-4 5-2" />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* header + ticker                                                     */
/* ------------------------------------------------------------------ */

const NAV: { href: string; label: string }[] = [
  { href: "#verdict", label: "VERDICT" },
  { href: "#stack", label: "STACK MAP" },
  { href: "#plan", label: "THE PLAN" },
  { href: "#operator", label: "OPERATE" },
  { href: "#lafmplus", label: "LAFM+" },
  { href: "#simbridge", label: "SIM BRIDGE" },
  { href: "#phase4", label: "PHASE 4" },
  { href: "#pipeline", label: "PIPELINE" },
  { href: "#benchmarks", label: "BENCHMARKS" },
  { href: "#roadmap", label: "ROADMAP" },
  { href: "#faq", label: "FAQ" },
];

function Header() {
  const [prog, setProg] = useState(0);
  useEffect(() => {
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const h = document.documentElement;
        const max = h.scrollHeight - h.clientHeight;
        setProg(max > 0 ? (h.scrollTop / max) * 100 : 0);
      });
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-line bg-ink/85 backdrop-blur-md">
      <div className="mx-auto flex h-[54px] max-w-7xl items-center justify-between px-4 sm:px-6">
        <a href="#top" className="group flex items-center gap-2.5">
          <IconTip className="h-6 w-6 text-teal transition-transform duration-300 group-hover:-translate-y-0.5" />
          <span className="font-display text-lg font-black tracking-tight text-fog">
            NANOLOCZ
          </span>
          <IconArrow className="h-3.5 w-3.5 text-faint" />
          <span className="font-mono text-[11px] font-medium tracking-[0.2em] text-teal">
            PY/GPU
          </span>
        </a>
        <nav className="hidden items-center gap-5 lg:flex">
          {NAV.map((n) => (
            <a
              key={n.href}
              href={n.href}
              className="font-mono text-[10.5px] tracking-[0.18em] text-dim transition-colors duration-200 hover:text-teal"
            >
              {n.label}
            </a>
          ))}
        </nav>
        <div className="hidden items-center gap-2 border border-teal/40 bg-teal/5 px-3 py-1.5 sm:flex">
          <span className="pulse-dot h-1.5 w-1.5 rounded-full bg-teal" />
          <span className="font-mono text-[10px] tracking-[0.22em] text-teal">
            VERDICT: YES
          </span>
        </div>
      </div>
      <div className="overflow-hidden border-t border-line/70 bg-ink2/90">
        <div className="marquee-track flex w-max items-center">
          {[0, 1].map((dup) => (
            <div key={dup} className="flex items-center" aria-hidden={dup === 1}>
              {TICKER.map((t, i) => (
                <span
                  key={`${dup}-${i}`}
                  className="flex items-center whitespace-nowrap py-1.5 font-mono text-[10px] tracking-[0.22em] text-faint"
                >
                  <span className="px-6">{t}</span>
                  <span className="text-teal/70">▸</span>
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>
      <div
        className="absolute bottom-[-2px] left-0 h-[2px] bg-teal transition-[width] duration-150 ease-out"
        style={{ width: `${prog}%` }}
      />
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* opening                                                             */
/* ------------------------------------------------------------------ */

const CHIPS: { k: string; v: string }[] = [
  { k: "LICENSE", v: "GPL-3.0 — forking explicitly permitted" },
  { k: "TOOLBOXES", v: "6 MATLAB toolboxes → the SciPy stack" },
  { k: "GPU TODAY", v: "0 calls — CUDA would be a pure upgrade" },
  { k: "ARCHITECTURE", v: "v1.42 already split core lib from GUI" },
];

function Opening() {
  const reduced = usePrefersReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>();
  const [go2, setGo2] = useState(false);
  useEffect(() => {
    if (!inView) return;
    const id = window.setTimeout(() => setGo2(true), reduced ? 0 : 420);
    return () => window.clearTimeout(id);
  }, [inView, reduced]);
  const l1 = useScramble("NO MATLAB.", inView, reduced);
  const l2 = useScramble("FULL GPU.", go2, reduced);
  const cites = useCountUp(45, inView, reduced, 1600);

  return (
    <section id="top" className="relative overflow-hidden pt-[118px]">
      <div className="mx-auto grid max-w-7xl items-center gap-10 px-4 pb-16 pt-10 sm:px-6 lg:grid-cols-[1.04fr_0.96fr] lg:gap-12 lg:pb-24 lg:pt-16">
        <Reveal>
          <div className="corner-frame">
            <SimCanvas />
          </div>
          <p className="mt-4 max-w-xl font-mono text-[11px] leading-relaxed tracking-wide text-faint">
            SIMULATED LAFM ACQUISITION — a hexagonal membrane lattice plus
            filaments, rebuilt point-by-point from noisy tip localizations.
            This is the exact loop NanoLocz runs. The fork just moves it off
            the license server and onto the GPU.
          </p>
        </Reveal>

        <div ref={ref}>
          <Reveal>
            <div className="flex items-center gap-3">
              <span className="h-px w-10 bg-amber" />
              <span className="font-mono text-[11px] tracking-[0.3em] text-amber">
                FEASIBILITY REPORT · NANOLOCZ → LINUX / GPU
              </span>
            </div>
          </Reveal>
          <p className="mt-6 font-mono text-[12.5px] leading-relaxed text-dim">
            <span className="text-faint">// your question —</span> can{" "}
            <a
              href="https://github.com/George-R-Heath/NanoLocz"
              target="_blank"
              rel="noreferrer"
              className="text-sky2 underline decoration-sky2/40 underline-offset-4 transition-colors hover:text-teal"
            >
              NanoLocz
            </a>{" "}
            be forked to run without MATLAB, with GPU support, on Linux?
          </p>
          <h1 className="font-display mt-4 text-[13vw] font-black leading-[0.95] text-fog sm:text-7xl xl:text-[86px]">
            <span className="block whitespace-nowrap">{l1}</span>
            <span className="block whitespace-nowrap text-teal">{l2}</span>
          </h1>
          <Reveal delay={120}>
            <p className="mt-6 max-w-xl text-[16px] leading-relaxed text-dim">
              Short answer: <strong className="font-semibold text-fog">yes</strong>.
              NanoLocz ships under GPL-3.0, so the fork is legal — and its
              January 2026 release (v1.42) already split the processing
              library out of the App Designer GUI, which is precisely what
              makes a MATLAB-free, CUDA-accelerated Linux port realistic. The
              math ports cleanly. The interface is the real project.
            </p>
          </Reveal>
          <Reveal delay={220}>
            <div className="mt-8 grid grid-cols-1 gap-3 sm:grid-cols-2">
              {CHIPS.map((c) => (
                <div
                  key={c.k}
                  className="group border border-line bg-ink2/70 p-4 transition-all duration-300 hover:-translate-y-0.5 hover:border-teal/50 hover:bg-ink3"
                >
                  <div className="font-mono text-[9.5px] tracking-[0.25em] text-amber">
                    {c.k}
                  </div>
                  <div className="mt-1.5 text-[13.5px] font-medium leading-snug text-fog">
                    {c.v}
                  </div>
                </div>
              ))}
            </div>
          </Reveal>
          <Reveal delay={320}>
            <div className="mt-8 flex flex-wrap items-center gap-4">
              <a
                href="#verdict"
                className="group inline-flex items-center gap-2.5 border border-teal bg-teal/10 px-6 py-3 font-mono text-[12px] tracking-[0.18em] text-teal transition-all duration-300 hover:bg-teal hover:text-ink"
              >
                READ THE BLUEPRINT
                <span className="transition-transform duration-300 group-hover:translate-y-0.5">↓</span>
              </a>
              <a
                href="https://github.com/George-R-Heath/NanoLocz"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-2.5 border border-line px-6 py-3 font-mono text-[12px] tracking-[0.18em] text-dim transition-all duration-300 hover:border-fog/50 hover:text-fog"
              >
                <IconFork className="h-4 w-4" />
                ORIGINAL REPO
              </a>
              <span className="font-mono text-[11px] text-faint">
                cited {cites}+ times · <em className="not-italic text-dim">Small Methods</em> 2024
              </span>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* shared section head                                                 */
/* ------------------------------------------------------------------ */

export function SectionHead({
  no,
  kicker,
  title,
  aside,
}: {
  no: string;
  kicker: string;
  title: string;
  aside?: string;
}) {
  return (
    <Reveal>
      <div className="mb-10 flex items-end justify-between gap-6 border-b border-line pb-5">
        <div>
          <div className="font-mono text-[11px] tracking-[0.3em] text-amber">
            {no} / {kicker}
          </div>
          <h2 className="font-display mt-2.5 text-3xl font-black leading-tight text-fog sm:text-4xl xl:text-5xl">
            {title}
          </h2>
        </div>
        {aside && (
          <div className="hidden max-w-[280px] text-right font-mono text-[11px] leading-relaxed text-faint md:block">
            {aside}
          </div>
        )}
      </div>
    </Reveal>
  );
}

/* ------------------------------------------------------------------ */
/* 01 — verdict                                                        */
/* ------------------------------------------------------------------ */

const PILLARS: { n: string; t: string; b: string }[] = [
  {
    n: "01",
    t: "The license says yes",
    b: "GPL-3.0 grants fork, modify and redistribute rights up front. Your derivative stays GPL-3.0 — that's the whole price.",
  },
  {
    n: "02",
    t: "The core is already a library",
    b: "v1.42 (Jan 2026) moved the processing functions into external .m files 'to simplify future maintenance'. The port surface is bounded.",
  },
  {
    n: "03",
    t: "No GPU code to untangle",
    b: "NanoLocz uses zero gpuArray or Parallel Computing Toolbox calls today. CUDA support is additive, not a risky rewrite.",
  },
  {
    n: "04",
    t: "Escape hatches already exist",
    b: "HDF5 export landed in v1.20 — explicitly noted as 'openable with Python'. The authors have been paving this road.",
  },
];

const SIZE_META: Record<PortSize, { label: string; w: string; cls: string }> = {
  S: { label: "STRAIGHT PORT", w: "30%", cls: "bg-teal" },
  M: { label: "REAL WORK", w: "62%", cls: "bg-amber" },
  L: { label: "LONG TAIL", w: "96%", cls: "bg-mag" },
};

function Verdict() {
  return (
    <section id="verdict" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="01"
          kicker="WHY THIS IS FEASIBLE"
          title="Four facts make the fork a bet, not a gamble"
          aside="Verified against the public repository, README changelog and the Small Methods paper."
        />
        <div className="grid gap-x-10 gap-y-6 sm:grid-cols-2">
          {PILLARS.map((p, i) => (
            <Reveal key={p.n} delay={i * 90}>
              <div className="group flex gap-5 border-l-2 border-line py-2 pl-5 transition-all duration-300 hover:border-teal hover:pl-7">
                <span className="font-mono text-[13px] font-medium text-amber">
                  {p.n}
                </span>
                <div>
                  <h3 className="font-display text-xl font-bold text-fog">
                    {p.t}
                  </h3>
                  <p className="mt-2 text-[14px] leading-relaxed text-dim">
                    {p.b}
                  </p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>

        <div className="mt-16">
          <Reveal>
            <div className="mb-5 flex items-center justify-between">
              <h3 className="font-display text-xl font-bold text-fog sm:text-2xl">
                Every capability, rated for portability
              </h3>
              <div className="hidden items-center gap-4 font-mono text-[10px] tracking-[0.15em] text-faint sm:flex">
                <span className="flex items-center gap-1.5"><span className="h-1.5 w-4 bg-teal" />S</span>
                <span className="flex items-center gap-1.5"><span className="h-1.5 w-4 bg-amber" />M</span>
                <span className="flex items-center gap-1.5"><span className="h-1.5 w-4 bg-mag" />L</span>
              </div>
            </div>
          </Reveal>
          <div className="border-t border-line">
            {FEATURES.map((f, i) => (
              <Reveal key={f.name} delay={Math.min(i * 60, 300)}>
                <div className="group grid grid-cols-1 items-center gap-x-8 gap-y-2 border-b border-line/70 py-4 transition-all duration-300 hover:bg-ink2/70 hover:pl-3 md:grid-cols-[minmax(200px,1.1fr)_1.4fr_1fr]">
                  <div>
                    <div className="text-[15px] font-semibold text-fog">
                      {f.name}
                    </div>
                    <div className="mt-0.5 font-mono text-[10.5px] text-faint">
                      {f.how}
                    </div>
                  </div>
                  <div className="text-[13px] leading-relaxed text-dim">
                    {f.note}
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="h-[5px] flex-1 overflow-hidden bg-ink3">
                      <div
                        className={`h-full ${SIZE_META[f.size].cls} transition-all duration-500 group-hover:brightness-125`}
                        style={{ width: SIZE_META[f.size].w }}
                      />
                    </div>
                    <span
                      className={`w-28 text-right font-mono text-[9.5px] tracking-[0.18em] ${
                        f.size === "S"
                          ? "text-teal"
                          : f.size === "M"
                            ? "text-amber"
                            : "text-mag"
                      }`}
                    >
                      {SIZE_META[f.size].label}
                    </span>
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 02 — stack map                                                      */
/* ------------------------------------------------------------------ */

function StackMap() {
  return (
    <section id="stack" className="scroll-mt-28 border-y border-line/60 bg-ink2/40">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="02"
          kicker="MATLAB → LINUX / GPU"
          title="The eleven-line substitution map"
          aside="Everything NanoLocz leans on has a maintained, GPU-aware Python counterpart. Nothing here is exotic."
        />
        <div className="border-t border-line">
          {MAPPING.map((m, i) => (
            <Reveal key={m.from} delay={Math.min(i * 50, 250)}>
              <div className="group grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1 border-b border-line/70 py-3.5 transition-all duration-300 hover:bg-ink3/70 hover:pl-3 sm:grid-cols-[1fr_36px_1.15fr]">
                <div className="font-mono text-[12.5px] text-amber/90">
                  {m.from}
                </div>
                <IconArrow className="hidden h-4 w-4 text-teal transition-transform duration-300 group-hover:translate-x-1 sm:block" />
                <div className="col-span-2 sm:col-span-1">
                  <div className="text-[14px] font-semibold text-fog">
                    {m.to}
                  </div>
                  <div className="font-mono text-[10.5px] text-faint">
                    {m.note}
                  </div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
        <Reveal delay={150}>
          <p className="mt-6 flex items-start gap-3 font-mono text-[11.5px] leading-relaxed text-faint">
            <IconChip className="mt-0.5 h-4 w-4 shrink-0 text-teal" />
            The interesting row is the seventh: NanoLocz never touches
            gpuArray, so there is no MATLAB-GPU code to translate — the CUDA
            layer is designed from scratch for the fork, unburdened by
            matlab.gpu semantics.
          </p>
        </Reveal>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 03 — pipeline (sticky two-column)                                   */
/* ------------------------------------------------------------------ */

const NODES: { t: string; s: string; c: "amber" | "teal" | "sky" | "hot" }[] = [
  { t: "OPENERS", s: ".spm .asd .jpk .ibw .gwy", c: "amber" },
  { t: "ZARR STORE", s: "(data, meta) contract", c: "teal" },
  { t: "LEVEL", s: "CuPy batched lstsq", c: "teal" },
  { t: "DETECT", s: "filters · label · stats", c: "teal" },
  { t: "LAFM SPLAT", s: "raw CUDA kernel", c: "hot" },
  { t: "FRC", s: "cuFFT resolution", c: "teal" },
  { t: "NAPARI / CLI", s: "render · export", c: "sky" },
];

const NODE_FILL: Record<string, string> = {
  amber: "#ffb454",
  teal: "#37e6c4",
  sky: "#5fb2ff",
  hot: "#37e6c4",
};

function PipelineDiagram() {
  return (
    <svg viewBox="0 0 240 640" className="w-full max-w-[300px]">
      {NODES.map((n, i) => {
        const y = 14 + i * 92;
        const col = NODE_FILL[n.c];
        const hot = n.c === "hot";
        return (
          <g key={n.t}>
            {i < NODES.length - 1 && (
              <line
                x1="120"
                y1={y + 52}
                x2="120"
                y2={y + 92}
                stroke="#1f425e"
                strokeWidth="1.5"
                className="dash-anim"
              />
            )}
            <rect
              x="18"
              y={y}
              width="204"
              height="52"
              fill={hot ? "rgba(55,230,196,0.10)" : "rgba(14,33,48,0.9)"}
              stroke={hot ? col : "#1f425e"}
              strokeWidth={hot ? 2 : 1.2}
            />
            <rect x="18" y={y} width="4" height="52" fill={col} />
            <text
              x="34"
              y={y + 22}
              fill="#e8f2f4"
              fontSize="13"
              fontWeight="700"
              fontFamily="Archivo, sans-serif"
              letterSpacing="1"
            >
              {n.t}
            </text>
            <text
              x="34"
              y={y + 40}
              fill="#5f7c8e"
              fontSize="9.5"
              fontFamily="IBM Plex Mono, monospace"
            >
              {n.s}
            </text>
            {hot && (
              <text
                x="214"
                y={y + 22}
                fill={col}
                fontSize="11"
                textAnchor="end"
                fontFamily="IBM Plex Mono, monospace"
              >
                ★
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function CodePanel({
  title,
  tone,
  children,
}: {
  title: string;
  tone: "amber" | "teal";
  children: ReactNode;
}) {
  return (
    <div className="overflow-hidden border border-line bg-ink">
      <div
        className={`flex items-center justify-between border-b border-line px-4 py-2.5 ${
          tone === "teal" ? "bg-teal/5" : "bg-amber/5"
        }`}
      >
        <span
          className={`font-mono text-[10px] tracking-[0.2em] ${
            tone === "teal" ? "text-teal" : "text-amber"
          }`}
        >
          {title}
        </span>
        <span className="flex gap-1.5">
          <span className="h-2 w-2 rounded-full bg-line2" />
          <span className="h-2 w-2 rounded-full bg-line2" />
          <span className="h-2 w-2 rounded-full bg-line2" />
        </span>
      </div>
      <pre className="overflow-x-auto p-4 font-mono text-[12px] leading-[1.75] text-dim">
        {children}
      </pre>
    </div>
  );
}

function Pipeline() {
  return (
    <section id="pipeline" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="07"
          kicker="THE GPU PIPELINE"
          title="Six stages, one device"
          aside="Sticky schematic on the left; the engineering notes walk on the right. The LAFM stage is where the 50× lives."
        />
        <div className="grid gap-10 lg:grid-cols-[300px_1fr] lg:gap-14">
          <div className="hidden lg:block">
            <div className="sticky top-32">
              <PipelineDiagram />
              <p className="mt-4 font-mono text-[10.5px] leading-relaxed text-faint">
                ★ = stage implemented as a raw CUDA kernel via
                cupy.RawKernel — no wrapper overhead.
              </p>
            </div>
          </div>
          <div className="space-y-5">
            {PIPELINE.map((p, i) => (
              <Reveal key={p.n} delay={Math.min(i * 70, 280)}>
                <div className="group border border-line bg-ink2/60 p-5 transition-all duration-300 hover:-translate-y-0.5 hover:border-teal/50 sm:p-6">
                  <div className="flex flex-wrap items-center gap-3">
                    <span className="font-mono text-[12px] font-medium text-teal">
                      {p.n}
                    </span>
                    <h3 className="font-display text-xl font-bold text-fog sm:text-2xl">
                      {p.title}
                    </h3>
                    <span className="ml-auto border border-teal/30 bg-teal/5 px-2.5 py-1 font-mono text-[9.5px] tracking-[0.12em] text-teal">
                      {p.gpu}
                    </span>
                  </div>
                  <p className="mt-3 text-[14px] leading-relaxed text-dim">
                    {p.body}
                  </p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>

        {/* side-by-side code */}
        <div className="mt-14">
          <Reveal>
            <h3 className="font-display mb-5 text-xl font-bold text-fog sm:text-2xl">
              The port, side by side
            </h3>
          </Reveal>
          <div className="grid gap-5 lg:grid-cols-2">
            <Reveal>
              <CodePanel title="MATLAB · SIMPLIFIED FROM v1.42 STYLE" tone="amber">
                <code>
                  <span className="tok-c">% render localizations — CPU loops today</span>
                  {"\n"}acc = <span className="tok-f">zeros</span>(H, W);{"\n"}
                  <span className="tok-k">for</span> i = 1:<span className="tok-f">numel</span>(x){"\n"}
                  {"  "}xi = <span className="tok-f">round</span>(x(i)); yi = <span className="tok-f">round</span>(y(i));{"\n"}
                  {"  "}<span className="tok-k">for</span> dy = -r:r, <span className="tok-k">for</span> dx = -r:r{"\n"}
                  {"    "}g = <span className="tok-f">exp</span>(-((dx^2+dy^2))/(2*s^2));{"\n"}
                  {"    "}acc(yi+dy, xi+dx) = acc(yi+dy, xi+dx) + w(i)*g;{"\n"}
                  {"  "}<span className="tok-k">end</span>, <span className="tok-k">end</span>{"\n"}
                  <span className="tok-k">end</span>{"  "}<span className="tok-c">% 2.1M locs → ~3.5 min</span>
                </code>
              </CodePanel>
            </Reveal>
            <Reveal delay={120}>
              <CodePanel title="PYTHON · CuPy DROP-IN" tone="teal">
                <code>
                  <span className="tok-c"># same intent — one launch, one device</span>
                  {"\n"}<span className="tok-k">import</span> cupy <span className="tok-k">as</span> cp{"\n"}
                  x = cp.<span className="tok-f">asarray</span>(x)  <span className="tok-c"># host → device</span>{"\n"}
                  acc = cp.<span className="tok-f">zeros</span>((H, W), cp.float32){"\n"}
                  splat((n // <span className="tok-n">256</span> + <span className="tok-n">1</span>,), (<span className="tok-n">256</span>,),{"\n"}
                  {"      "}(x, y, w, acc, n, W, H, sigma)){"\n"}
                  cp.cuda.Stream.null.<span className="tok-f">synchronize</span>(){"\n  "}<span className="tok-c"># 2.1M locs → ~4 s</span>
                </code>
              </CodePanel>
            </Reveal>
          </div>
          <Reveal delay={160}>
            <div className="mt-5">
              <CodePanel title="THE KERNEL ITSELF · cupy.RawKernel (C++)" tone="teal">
                <code>
                  splat = cp.<span className="tok-f">RawKernel</span>(<span className="tok-s">r'''</span>{"\n"}
                  {"  "}<span className="tok-k">extern "C" __global__</span>{"\n"}
                  {"  "}<span className="tok-k">void</span> <span className="tok-f">lafm_splat</span>(<span className="tok-k">const float</span>* loc, <span className="tok-k">float</span>* img,{"\n"}
                  {"                  "}<span className="tok-k">int</span> n, <span className="tok-k">int</span> W, <span className="tok-k">int</span> H, <span className="tok-k">float</span> s) {"{"}{"\n"}
                  {"    "}<span className="tok-k">int</span> i = blockIdx.x * blockDim.x + threadIdx.x;{"\n"}
                  {"    "}<span className="tok-k">if</span> (i &gt;= n) <span className="tok-k">return</span>;{"\n"}
                  {"    "}<span className="tok-k">float</span> x = loc[<span className="tok-n">4</span>*i], y = loc[<span className="tok-n">4</span>*i+<span className="tok-n">1</span>];{"\n"}
                  {"    "}<span className="tok-k">float</span> h = loc[<span className="tok-n">4</span>*i+<span className="tok-n">2</span>], w = loc[<span className="tok-n">4</span>*i+<span className="tok-n">3</span>];{"\n"}
                  {"    "}<span className="tok-k">int</span> r = (<span className="tok-k">int</span>)<span className="tok-f">ceilf</span>(<span className="tok-n">3.0f</span> * s);{"\n"}
                  {"    "}<span className="tok-k">for</span> (<span className="tok-k">int</span> dy = -r; dy &lt;= r; dy++){"\n"}
                  {"      "}<span className="tok-k">for</span> (<span className="tok-k">int</span> dx = -r; dx &lt;= r; dx++) {"{"}{"\n"}
                  {"        "}<span className="tok-k">int</span> X = (<span className="tok-k">int</span>)<span className="tok-f">rintf</span>(x)+dx, Y = (<span className="tok-k">int</span>)<span className="tok-f">rintf</span>(y)+dy;{"\n"}
                  {"        "}<span className="tok-k">if</span> (X&lt;<span className="tok-n">0</span>||Y&lt;<span className="tok-n">0</span>||X&gt;=W||Y&gt;=H) <span className="tok-k">continue</span>;{"\n"}
                  {"        "}<span className="tok-k">float</span> g = <span className="tok-f">expf</span>(-((X-x)*(X-x)+(Y-y)*(Y-y))/(<span className="tok-n">2</span>*s*s));{"\n"}
                  {"        "}<span className="tok-f">atomicAdd</span>(&amp;img[Y*W+X], w*h*g);{"  "}<span className="tok-c">// height-weighted splat</span>{"\n"}
                  {"      "}{"}"}{"\n"}
                  {"  "}{"}"},{" "}<span className="tok-s">'''</span>, <span className="tok-s">"lafm_splat"</span>)
                </code>
              </CodePanel>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 04 — benchmarks                                                     */
/* ------------------------------------------------------------------ */

function Bench() {
  const [ref, inView] = useInView<HTMLDivElement>();
  const allTimes = BENCH.flatMap((b) => [b.matlab.s, b.gpu.s]);
  const lo = Math.log10(Math.min(...allTimes));
  const hi = Math.log10(Math.max(...allTimes));
  const w = (s: number) => 9 + 91 * ((Math.log10(s) - lo) / (hi - lo));

  return (
    <section id="benchmarks" className="scroll-mt-28 border-y border-line/60 bg-ink2/40">
      <div ref={ref} className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="08"
          kicker="EXPECTED HEADROOM"
          title="Where the speedup hides"
          aside="Log-scaled bars. Figures are engineering estimates from comparable published GPU ports — verify on your own hardware."
        />
        <div className="space-y-10">
          {BENCH.map((b, i) => {
            const ratio = Math.round(b.matlab.s / b.gpu.s);
            return (
              <Reveal key={b.task} delay={i * 100}>
                <div>
                  <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                    <div>
                      <span className="font-display text-xl font-bold text-fog sm:text-2xl">
                        {b.task}
                      </span>
                      <span className="ml-3 font-mono text-[11px] text-faint">
                        {b.spec}
                      </span>
                    </div>
                    <span className="border border-teal/40 bg-teal/5 px-2.5 py-1 font-mono text-[11px] tracking-[0.15em] text-teal">
                      ≈ {ratio}× FASTER
                    </span>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center gap-3">
                      <div className="h-7 flex-1 bg-ink3/80">
                        <div
                          className="flex h-full items-center bg-amber/80 pl-2.5 transition-[width] duration-[1200ms] ease-out"
                          style={{
                            width: inView ? `${w(b.matlab.s)}%` : "0%",
                            transitionDelay: `${i * 120}ms`,
                          }}
                        >
                          <span className="whitespace-nowrap font-mono text-[10.5px] font-medium text-ink">
                            {b.matlab.label}
                          </span>
                        </div>
                      </div>
                      <span className="w-16 text-right font-mono text-[12px] text-amber tabular-nums">
                        {b.matlab.pretty}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="h-7 flex-1 bg-ink3/80">
                        <div
                          className="flex h-full items-center bg-teal pl-2.5 transition-[width] duration-[1200ms] ease-out"
                          style={{
                            width: inView ? `${w(b.gpu.s)}%` : "0%",
                            transitionDelay: `${i * 120 + 150}ms`,
                          }}
                        >
                          <span className="whitespace-nowrap font-mono text-[10.5px] font-medium text-ink">
                            {b.gpu.label}
                          </span>
                        </div>
                      </div>
                      <span className="w-16 text-right font-mono text-[12px] text-teal tabular-nums">
                        {b.gpu.pretty}
                      </span>
                    </div>
                  </div>
                </div>
              </Reveal>
            );
          })}
        </div>
        <Reveal delay={200}>
          <p className="mt-10 border-l-2 border-amber/60 pl-4 font-mono text-[11.5px] leading-relaxed text-faint">
            Estimate basis: element-wise filtering and reduction kernels
            routinely hit 30–60× over scalar MATLAB loops; scatter/splat
            workloads (the LAFM core) match published CUDA localisation
            renderers such as Picasso and DECODE. Treat these as targets for
            the Phase 2 benchmark report, not guarantees.
          </p>
        </Reveal>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 05 — install terminal                                               */
/* ------------------------------------------------------------------ */

function Terminal() {
  const reduced = usePrefersReducedMotion();
  const [ref, inView] = useInView<HTMLDivElement>();
  const [count, setCount] = useState(0);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      setCount(TERM_LINES.length);
      return;
    }
    setCount(0);
    const id = window.setInterval(() => {
      setCount((c) => {
        if (c >= TERM_LINES.length) {
          window.clearInterval(id);
          return c;
        }
        return c + 1;
      });
    }, 460);
    return () => window.clearInterval(id);
  }, [inView, reduced]);

  const script = TERM_LINES.filter((l) => l.t === "cmd")
    .map((l) => l.s)
    .join("\n");

  return (
    <section id="install" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="09"
          kicker="LINUX, HEADLESS, GPU"
          title="What day one looks like"
          aside="No license server, no MCR installer, no X11. The whole pipeline survives a plain SSH session."
        />
        <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr]">
          <Reveal>
            <div ref={ref} className="overflow-hidden border border-line bg-ink shadow-[0_20px_60px_-30px_rgba(55,230,196,0.15)]">
              <div className="flex items-center justify-between border-b border-line bg-ink2 px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-mag/80" />
                  <span className="h-2.5 w-2.5 rounded-full bg-amber/80" />
                  <span className="h-2.5 w-2.5 rounded-full bg-teal/80" />
                  <span className="ml-3 font-mono text-[10.5px] tracking-[0.15em] text-faint">
                    bash — nanolocz@linux
                  </span>
                </div>
                <button
                  onClick={async () => {
                    if (await copyText(script)) {
                      setCopied(true);
                      window.setTimeout(() => setCopied(false), 1800);
                    }
                  }}
                  className="flex items-center gap-1.5 border border-line px-2.5 py-1 font-mono text-[10px] tracking-[0.15em] text-dim transition-colors hover:border-teal/60 hover:text-teal"
                >
                  {copied ? <IconCheck className="h-3 w-3 text-teal" /> : <IconTerm className="h-3 w-3" />}
                  {copied ? "COPIED" : "COPY SCRIPT"}
                </button>
              </div>
              <div className="min-h-[320px] p-5">
                {TERM_LINES.slice(0, count).map((l, i) => (
                  <div
                    key={i}
                    className={`whitespace-pre-wrap font-mono text-[12.5px] leading-[1.9] ${
                      l.t === "cmd"
                        ? "text-fog"
                        : l.t === "ok"
                          ? "text-teal2"
                          : "text-faint"
                    }`}
                  >
                    {l.t === "cmd" && <span className="text-teal">$ </span>}
                    {l.s}
                  </div>
                ))}
                {!reduced && count < TERM_LINES.length && (
                  <span className="caret-blink inline-block h-[15px] w-[8px] translate-y-[2px] bg-teal" />
                )}
              </div>
            </div>
          </Reveal>
          <div className="space-y-3">
            {ENV_NOTES.map((n, i) => (
              <Reveal key={n.k} delay={i * 80}>
                <div className="group border border-line bg-ink2/60 p-4 transition-all duration-300 hover:border-teal/40 hover:pl-5">
                  <div className="font-mono text-[9.5px] tracking-[0.25em] text-amber">
                    {n.k}
                  </div>
                  <div className="mt-1 text-[13.5px] leading-relaxed text-dim">
                    {n.v}
                  </div>
                </div>
              </Reveal>
            ))}
            <Reveal delay={ENV_NOTES.length * 80}>
              <p className="border border-amber/30 bg-amber/5 p-4 font-mono text-[10.5px] leading-relaxed text-amber/90">
                ⚠ Package and command names above are placeholders for this
                study — a blueprint, not a release. The real first artifact is
                an issue on the upstream repo.
              </p>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 06 — roadmap                                                        */
/* ------------------------------------------------------------------ */

const PHASE_COLORS = ["#5fb2ff", "#37e6c4", "#37e6c4", "#ffb454", "#ff6e9c"];

function Roadmap() {
  return (
    <section id="roadmap" className="scroll-mt-28 border-y border-line/60 bg-ink2/40">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="10"
          kicker="TWENTY-SIX WEEKS, SEVEN PHASES"
          title="A route that ships early and often"
          aside="A usable GPU CLI exists by week 6. Everything after is interface, packaging and proof."
        />
        <div className="relative ml-2 border-l border-line pl-8 sm:ml-6 sm:pl-12">
          {PHASES.map((p, i) => (
            <Reveal key={p.tag} delay={i * 100}>
              <div className="group relative pb-12 last:pb-0">
                <span
                  className="absolute -left-[37px] top-1.5 h-3 w-3 rotate-45 border-2 bg-ink transition-transform duration-300 group-hover:scale-125 sm:-left-[53px]"
                  style={{ borderColor: PHASE_COLORS[i] }}
                />
                <div className="flex flex-wrap items-center gap-3">
                  <span
                    className="font-mono text-[11px] font-medium tracking-[0.25em]"
                    style={{ color: PHASE_COLORS[i] }}
                  >
                    {p.tag}
                  </span>
                  <h3 className="font-display text-2xl font-bold text-fog">
                    {p.title}
                  </h3>
                  <span className="ml-auto border border-line px-2.5 py-1 font-mono text-[10px] tracking-[0.15em] text-faint">
                    {p.when}
                  </span>
                </div>
                <ul className="mt-4 space-y-2">
                  {p.items.map((it) => (
                    <li key={it} className="flex items-start gap-3 text-[14px] leading-relaxed text-dim">
                      <span className="mt-[3px] font-mono text-[11px] text-teal">▸</span>
                      {it}
                    </li>
                  ))}
                </ul>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 07 — risks                                                          */
/* ------------------------------------------------------------------ */

function Risks() {
  return (
    <section id="risks" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="11"
          kicker="EYES OPEN"
          title="Ten things that could bite"
          aside="None are fatal. All are cheaper to handle in Phase 0 than in Phase 4."
        />
        <Reveal>
          <div className="mb-10 flex gap-5 border border-amber/40 bg-amber/5 p-5 sm:p-6">
            <IconScale className="h-9 w-9 shrink-0 text-amber" />
            <div>
              <h3 className="font-display text-lg font-bold text-amber sm:text-xl">
                THE LICENSE, IN ONE BREATH
              </h3>
              <p className="mt-2 text-[14px] leading-relaxed text-dim">
                NanoLocz is <strong className="text-fog">GNU GPL-3.0</strong>.
                Fork it, rewrite it in Python, ship it — as long as the
                derivative stays GPL-3.0, keeps its license headers, states
                what changed, and the science cites the paper. That's the
                entire deal.
              </p>
            </div>
          </div>
        </Reveal>
        <div className="grid gap-x-10 gap-y-8 md:grid-cols-2">
          {RISKS.map((r, i) => (
            <Reveal key={r.n} delay={Math.min(i * 70, 280)}>
              <div className="group flex gap-5 border-t border-line pt-5 transition-all duration-300 hover:border-amber/60">
                <span className="font-mono text-[13px] text-amber/70">{r.n}</span>
                <div>
                  <h3 className="text-[16px] font-semibold text-fog">{r.title}</h3>
                  <p className="mt-2 text-[13.5px] leading-relaxed text-dim">{r.body}</p>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 08 — faq                                                            */
/* ------------------------------------------------------------------ */

function Faq() {
  const [open, setOpen] = useState(0);
  return (
    <section id="faq" className="scroll-mt-28 border-t border-line/60 bg-ink2/40">
      <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="12"
          kicker="STRAIGHT ANSWERS"
          title="Questions you'd ask next"
        />
        <div className="space-y-3">
          {FAQS.map((f, i) => {
            const isOpen = open === i;
            return (
              <Reveal key={f.q} delay={Math.min(i * 60, 240)}>
                <div
                  className={`border transition-colors duration-300 ${
                    isOpen
                      ? "border-teal/50 bg-ink3/70"
                      : "border-line bg-ink2/60 hover:border-line2"
                  }`}
                >
                  <button
                    onClick={() => setOpen(isOpen ? -1 : i)}
                    className="flex w-full items-center justify-between gap-4 p-5 text-left"
                    aria-expanded={isOpen}
                  >
                    <span className="font-display text-[16px] font-bold text-fog sm:text-lg">
                      {f.q}
                    </span>
                    <IconPlus
                      className={`h-4 w-4 shrink-0 transition-transform duration-300 ${
                        isOpen ? "rotate-45 text-teal" : "text-faint"
                      }`}
                    />
                  </button>
                  <div className={`acc-body ${isOpen ? "open" : ""}`}>
                    <div className="acc-inner">
                      <p className="px-5 pb-5 text-[14px] leading-relaxed text-dim">
                        {f.a}
                      </p>
                    </div>
                  </div>
                </div>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* footer                                                              */
/* ------------------------------------------------------------------ */

function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto max-w-7xl px-4 py-14 sm:px-6">
        <Reveal>
          <div className="flex flex-wrap items-center gap-4">
            <IconWave className="h-8 w-8 text-teal" />
            <h2 className="font-display text-3xl font-black tracking-tight text-fog sm:text-5xl">
              PORT THE LIB<span className="text-teal">.</span> KEEP THE
              SCIENCE<span className="text-amber">.</span>
            </h2>
          </div>
        </Reveal>
        <div className="mt-10 grid gap-10 md:grid-cols-[1.3fr_1fr_1fr]">
          <Reveal>
            <div>
              <div className="font-mono text-[10px] tracking-[0.25em] text-amber">
                CITE THE ORIGINALS
              </div>
              <div className="mt-3 space-y-3 font-mono text-[11.5px] leading-relaxed text-dim">
                <p>
                  Heath, G.R., Micklethwaite, E. &amp; Storer, T.M. —{" "}
                  <em className="not-italic text-fog">
                    NanoLocz: Image analysis platform for AFM, high-speed AFM
                    and localization AFM.
                  </em>{" "}
                  <a
                    href="https://doi.org/10.1002/smtd.202301766"
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky2 underline decoration-sky2/40 underline-offset-4 hover:text-teal"
                  >
                    Small Methods (2024)
                  </a>
                </p>
                <p>
                  Heath, G.R., Kots, E., Robertson, J.L. et al. —{" "}
                  <em className="not-italic text-fog">
                    Localization atomic force microscopy.
                  </em>{" "}
                  <a
                    href="https://doi.org/10.1038/s41586-021-03551-x"
                    target="_blank"
                    rel="noreferrer"
                    className="text-sky2 underline decoration-sky2/40 underline-offset-4 hover:text-teal"
                  >
                    Nature 594, 385–390 (2021)
                  </a>
                </p>
              </div>
            </div>
          </Reveal>
          <Reveal delay={100}>
            <div>
              <div className="font-mono text-[10px] tracking-[0.25em] text-amber">
                UPSTREAM LINKS
              </div>
              <ul className="mt-3 space-y-2 text-[13.5px]">
                {[
                  ["GitHub repository", "https://github.com/George-R-Heath/NanoLocz"],
                  ["User guide & docs", "https://george-r-heath.github.io/NanoLocz/docs/"],
                  ["MATLAB File Exchange", "https://www.mathworks.com/matlabcentral/fileexchange/154880-nanolocz"],
                  ["Open AFM data resources", "https://george-r-heath.github.io/NanoLocz/docs/AFMDataRepositories"],
                ].map(([label, href]) => (
                  <li key={href}>
                    <a
                      href={href}
                      target="_blank"
                      rel="noreferrer"
                      className="group inline-flex items-center gap-2 text-dim transition-colors hover:text-teal"
                    >
                      <span className="h-px w-4 bg-line2 transition-all duration-300 group-hover:w-6 group-hover:bg-teal" />
                      {label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          </Reveal>
          <Reveal delay={200}>
            <div>
              <div className="font-mono text-[10px] tracking-[0.25em] text-amber">
                COLOPHON
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-dim">
                An unofficial feasibility study, unaffiliated with the
                NanoLocz authors. Benchmarks are estimates; package names are
                placeholders; the license facts are straight from the repo.
              </p>
              <a
                href="#top"
                className="mt-4 inline-flex items-center gap-2 font-mono text-[11px] tracking-[0.2em] text-faint transition-colors hover:text-teal"
              >
                BACK TO THE TIP ↑
              </a>
            </div>
          </Reveal>
        </div>
        <div className="mt-12 flex flex-wrap items-center justify-between gap-3 border-t border-line/60 pt-6 font-mono text-[10.5px] text-faint">
          <span>NANOLOCZ → PY/GPU · fork blueprint · 2026</span>
          <span className="flex items-center gap-2">
            <IconTip className="h-4 w-4 text-teal/70" />
            no MATLAB was harmed in this study
          </span>
        </div>
      </div>
    </footer>
  );
}

/* ------------------------------------------------------------------ */
export { Header, Opening, Verdict, StackMap, Pipeline, Bench, Terminal, Roadmap, Risks, Faq, Footer };
