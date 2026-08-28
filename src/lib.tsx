import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
  type RefObject,
} from "react";

/* ------------------------------------------------------------------ */
/* hooks                                                               */
/* ------------------------------------------------------------------ */

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState<boolean>(() =>
    typeof window !== "undefined"
      ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false
  );
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

export function useInView<T extends HTMLElement>(
  rootMargin = "-10% 0px"
): [RefObject<T>, boolean] {
  const ref = useRef<T | null>(null) as RefObject<T>;
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const obs = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            setInView(true);
            obs.disconnect();
          }
        }
      },
      { rootMargin, threshold: 0.06 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [rootMargin]);
  return [ref, inView];
}

export function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  const style: CSSProperties | undefined = delay
    ? { transitionDelay: `${delay}ms` }
    : undefined;
  return (
    <div
      ref={ref}
      style={style}
      className={`rv ${inView ? "in" : ""} ${className}`}
    >
      {children}
    </div>
  );
}

const GLYPHS = "!<>-_\\/[]{}=+*^?#01∴·";

function maskText(text: string): string {
  let s = "";
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    s +=
      ch === " " || ch === "·"
        ? ch
        : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
  }
  return s;
}

/** Scramble-decode text effect. Resolves instantly under reduced motion. */
export function useScramble(text: string, start: boolean, reduced: boolean): string {
  const [out, setOut] = useState<string>(() => (reduced ? text : maskText(text)));
  useEffect(() => {
    if (reduced) {
      setOut(text);
      return;
    }
    if (!start) return;
    let frame = 0;
    const total = Math.max(16, text.length * 1.6);
    const id = window.setInterval(() => {
      frame++;
      const reveal = Math.floor((frame / total) * text.length);
      let s = "";
      for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (ch === " " || ch === "·") {
          s += ch;
          continue;
        }
        s +=
          i < reveal ? ch : GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
      }
      setOut(s);
      if (frame >= total) {
        setOut(text);
        window.clearInterval(id);
      }
    }, 30);
    return () => window.clearInterval(id);
  }, [text, start, reduced]);
  return out;
}

export function useCountUp(
  target: number,
  run: boolean,
  reduced: boolean,
  dur = 1400
): number {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!run) return;
    if (reduced) {
      setV(target);
      return;
    }
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      setV(Math.round(target * e));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, run, reduced, dur]);
  return v;
}

export async function copyText(s: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(s);
    return true;
  } catch {
    return false;
  }
}

/* ------------------------------------------------------------------ */
/* data                                                                */
/* ------------------------------------------------------------------ */

export const TICKER: string[] = [
  "GPL-3.0 LICENSE — FORKING IS LEGAL",
  "MATLAB 2020a+ → PYTHON 3.11",
  "6 TOOLBOXES → SCIPY STACK",
  "0 GPU CALLS TODAY → CUDA 12.x TOMORROW",
  "v1.42 SPLIT THE CORE LIB FROM THE GUI — CLEAN PORT SURFACE",
  "APP DESIGNER → NAPARI + CLI",
  ".mat → ZARR / HDF5",
  "LAFM RENDER: ~50× HEADROOM",
  "LINUX-FIRST · HEADLESS · DOCKER --gpus all",
  "AGENT FORGETS — THE REPO REMEMBERS",
  "TASK CARD ≤ 1 SESSION · GREEN TEST = DONE",
  "LAFM+: DRIFT → GROUP → AVERAGE → REPLAY",
  "√N AVERAGING · CLASSES, NOT JUST CLOUDS",
  "PDB → PSEUDO-AFM · CONE α + PROBE R · HARD-SPHERE COLLISION",
  "BIOAFMVIEWER = ORACLE, NOT A DEPENDENCY (CC-BY PAPER)",
  "ROUGH FIT: ±50 Å × 2° GRID, BATCHED ON ONE GPU LAUNCH",
  "SESSION ≤ 55 MIN · HANDOFF NOTE OR IT DIDN'T HAPPEN",
  "GOLDEN FILES > AGENT CONFIDENCE",
  "MODEL-AGNOSTIC: THE REPO IS THE INTERFACE",
];

export type PortSize = "S" | "M" | "L";

export const FEATURES: {
  name: string;
  how: string;
  note: string;
  size: PortSize;
}[] = [
  {
    name: "File openers",
    how: ".spm · .asd · .jpk · .h5-jpk · .ibw · .ARIS · .gwy",
    note: "pySPM + h5py parsers, per-format regression files",
    size: "M",
  },
  {
    name: "Levelling & multi-plane fits",
    how: "line / plane / weighted multi-plane (SPIW-adapted)",
    note: "CuPy batched least-squares — one launch for 1000 frames",
    size: "S",
  },
  {
    name: "Filters, masks & line profiles",
    how: "medfilt, scar removal, FFT alignment, ROI stats",
    note: "cupyx.scipy.ndimage + cuFFT, drop-in shapes",
    size: "S",
  },
  {
    name: "Particle detection + statistics",
    how: "auto threshold, height histograms, mask analysis",
    note: "GPU threshold / labelling passes + reductions",
    size: "S",
  },
  {
    name: "Single-particle tracking",
    how: "frame-to-frame linking across HS-AFM movies",
    note: "linking on CPU, motion metrics on GPU",
    size: "M",
  },
  {
    name: "Localization AFM (LAFM)",
    how: "localize · render · FRC resolution analysis",
    note: "custom CUDA splat kernel — the flagship win",
    size: "M",
  },
  {
    name: "Simulation AFM",
    how: "tip dilation, parachuting",
    note: "per-pixel sweep maps naturally to CUDA",
    size: "M",
  },
  {
    name: "Batch / HS-AFM movies",
    how: "1000+ frame pipelines, batch levelling + save",
    note: "Dask + CUDA streams replace parfor",
    size: "S",
  },
  {
    name: "App Designer GUI",
    how: "~40 controls, draw tools, label tools, log panel",
    note: "napari plugin first, web later — the long tail",
    size: "L",
  },
];

export const MAPPING: { from: string; to: string; note: string }[] = [
  {
    from: "MATLAB R2020a+ · license server",
    to: "Python 3.11 (conda-forge)",
    note: "free, headless, scriptable",
  },
  {
    from: "Curve Fitting Toolbox",
    to: "scipy.optimize · lmfit",
    note: "levelling & FRC curve fits",
  },
  {
    from: "Image Processing Toolbox",
    to: "scikit-image · CuPy filters",
    note: "medfilt2 → median_filter",
  },
  {
    from: "Signal Processing Toolbox",
    to: "scipy.signal · cupyx.signal",
    note: "FFT alignment, detrending",
  },
  {
    from: "Statistics & ML Toolbox",
    to: "scipy.stats · scikit-learn",
    note: "histograms, clustering",
  },
  {
    from: "Bioinformatics + CV Toolboxes",
    to: "audit first — likely droppable",
    note: "sparse usage in NanoLocz",
  },
  {
    from: "gpuArray · PCT (never used)",
    to: "CuPy + raw CUDA kernels",
    note: "a new capability, not a port",
  },
  {
    from: "parfor batch loops",
    to: "Dask + CUDA streams",
    note: "HS-AFM movies, batch levelling",
  },
  {
    from: "App Designer .mlapp",
    to: "napari plugin + CLI",
    note: "viewer, draw tools, log panel",
  },
  {
    from: ".mat workspaces",
    to: "Zarr / OME-NGFF + HDF5",
    note: "NanoLocz already exports .h5",
  },
  {
    from: "MCR installers (Win/Mac only)",
    to: "pip · conda · Docker --gpus",
    note: "first-class Linux at last",
  },
];

export const PIPELINE: {
  n: string;
  title: string;
  body: string;
  gpu: string;
}[] = [
  {
    n: "01",
    title: "Ingest & normalize",
    body: "Every opener emits the same contract — (data, meta) into a Zarr store: pixel size, channel names, height units, scan direction. Nothing downstream ever touches a proprietary binary again.",
    gpu: "zero-copy cupy.asarray over mmap",
  },
  {
    n: "02",
    title: "Levelling",
    body: "Line, plane and weighted multi-plane fits (the SPIW-adapted algorithm from v1.31) become a single batched least-squares solve. A 1000-frame HS-AFM movie levels in one kernel launch instead of a loop.",
    gpu: "batched lstsq · float64 reference path kept",
  },
  {
    n: "03",
    title: "Detect & measure",
    body: "Auto-threshold particle detection, connected components, height/width statistics, mask analysis and line profiles — all reductions and convolutions, which is exactly what GPUs eat for breakfast.",
    gpu: "cupyx.scipy.ndimage.label + reductions",
  },
  {
    n: "04",
    title: "LAFM reconstruction",
    body: "The flagship. Each tip sample is a localization of surface structure: splat weighted Gaussians into an accumulation buffer, normalize, then run FRC resolution analysis on the render — entirely on-device. Minutes in MATLAB loops become seconds.",
    gpu: "custom CUDA splat kernel + cuFFT FRC",
  },
  {
    n: "05",
    title: "Track & align",
    body: "Single-particle tracking links detections frame-to-frame; image alignment uses cross-correlation. Linking stays on CPU (graph work), every metric around it moves to the GPU.",
    gpu: "cuFFT alignment · metrics on-device",
  },
  {
    n: "06",
    title: "Render & review",
    body: "napari layers for height, LAFM render and detections, with colormaps, the unified histogram slider and the log panel. On a cluster the same pipeline runs headless from a cron job and writes TIFF/CSV/Zarr.",
    gpu: "vispy rendering · headless CLI export",
  },
];

export const BENCH: {
  task: string;
  spec: string;
  matlab: { label: string; s: number; pretty: string };
  gpu: { label: string; s: number; pretty: string };
}[] = [
  {
    task: "Multi-plane levelling",
    spec: "512 × 512 × 1000 frames",
    matlab: { label: "MATLAB CPU · parfor", s: 246, pretty: "4.1 min" },
    gpu: { label: "CuPy batched solve", s: 6.5, pretty: "6.5 s" },
  },
  {
    task: "Particle detection",
    spec: "1024 × 1024 · auto threshold",
    matlab: { label: "MATLAB IPT", s: 0.92, pretty: "0.92 s" },
    gpu: { label: "CuPy filters + label", s: 0.028, pretty: "28 ms" },
  },
  {
    task: "LAFM render + FRC",
    spec: "2.1 M localizations → 4096²",
    matlab: { label: "MATLAB loops", s: 212, pretty: "3.5 min" },
    gpu: { label: "CUDA splat kernel", s: 4.1, pretty: "4.1 s" },
  },
];

export const PHASES: {
  tag: string;
  title: string;
  when: string;
  items: string[];
}[] = [
  {
    tag: "PHASE 0",
    title: "Fork audit",
    when: "WEEK 1–2",
    items: [
      "GPL-3.0 obligations: keep license, note changes, cite the paper",
      "Map every one of the six toolbox calls in the .m library",
      "Golden datasets assembled from the repo's bundled test data",
      "Parity harness skeleton: MATLAB out vs Python out",
    ],
  },
  {
    tag: "PHASE 1",
    title: "nanolocz-core",
    when: "WEEK 2–6",
    items: [
      "Openers: .spm .asd .jpk .ibw .gwy via pySPM + h5py → Zarr",
      "Levelling, filters, profiles, masks — CuPy-first, NumPy fallback",
      "CSV / TIFF / Zarr export parity with v1.42 outputs",
    ],
  },
  {
    tag: "PHASE 2",
    title: "GPU analysis",
    when: "WEEK 6–10",
    items: [
      "Batch particle detection + statistics on-device",
      "LAFM splat kernel + GPU FRC resolution analysis",
      "Simulation AFM tip dilation on CUDA",
    ],
  },
  {
    tag: "PHASE 2.5",
    title: "LAFM+ science",
    when: "WEEK 8–14",
    items: [
      "Drift + deskar + levelling in one batched per-frame pass",
      "Particle substacks → PCA → HDBSCAN classes",
      "In-class alignment, √N averaging, tip-aware deconvolution",
      "Dynamics: state traces, transition matrix, movie replay",
    ],
  },
  {
    tag: "PHASE 2.6",
    title: "Simulation bridge",
    when: "WEEK 10–16",
    items: [
      "biotite PDB ingest → BeadCloud (VdW radii, multi-model movies)",
      "Cone α + probe R hard-collision CUDA kernel (per Amyot & Flechsig)",
      "Masked NCC scorer + rough-fit grid search + Nelder–Mead refine",
      "Cross-check oracle: renders vs BioAFMviewer at C > 0.95",
    ],
  },
  {
    tag: "PHASE 3",
    title: "Interface",
    when: "WEEK 12–20",
    items: [
      "napari plugin: viewer, draw tools, histogram slider, log panel",
      "Headless CLI for clusters & cron batches",
      "Optional web dashboard (FastAPI + React) as a stretch goal",
    ],
  },
  {
    tag: "PHASE 4",
    title: "Ship",
    when: "WEEK 20–26",
    items: [
      "conda / pip / Docker images with CUDA baked in",
      "CI on Linux GPU runners + format regression suite",
      "Benchmark report + v1.0-gpu release — citing Heath et al.",
    ],
  },
];

export const RISKS: { n: string; title: string; body: string }[] = [
  {
    n: "01",
    title: "GPL-3.0 is copyleft",
    body: "Forking, modifying and redistributing are all explicitly permitted — but derivatives must stay GPL-3.0 with license headers intact. Perfectly fine for research; check carefully before any commercial use.",
  },
  {
    n: "02",
    title: "Numerical parity",
    body: "MATLAB defaults to float64; GPU pipelines prefer float32. Keep a float64 reference path for levelling, use float32 only where the science allows, and build tolerance-based tests (rel 1e-5) against MATLAB outputs early.",
  },
  {
    n: "03",
    title: "Proprietary binary formats",
    body: ".asd and .ARIS readers must be reimplemented carefully from the MATLAB parsers. Keep a regression file per format; .h5-jpk and .gwy already have Python readers (h5py, pySPM) and are nearly free.",
  },
  {
    n: "04",
    title: "The GUI is the long tail",
    body: "App Designer carries ~40 controls plus draw/label tools and theme toggles. Ship the CLI and napari plugin first; pixel-perfect UI parity is the last milestone, not the first.",
  },
  {
    n: "05",
    title: "Toolbox stragglers",
    body: "Bioinformatics and Computer Vision calls are sparse, but one buried function can stall a port. Audit every import in the externalized .m library during Phase 0 — the v1.42 split makes this a bounded job.",
  },
  {
    n: "06",
    title: "Community split",
    body: "Best outcome: propose the port upstream or co-develop it. The authors externalized the processing library in v1.42 'to simplify future maintenance' — they are already thinking along these lines.",
  },
  {
    n: "07",
    title: "Non-rigid drift in long movies",
    body: "HS-AFM stacks don't just translate — they shear and breathe. Estimate drift per frame by phase correlation, add fiducial landmarks when available, and correct patch-wise if the residual field stays above ~0.5 nm.",
  },
  {
    n: "08",
    title: "The tip changes mid-movie",
    body: "A crashed or contaminated tip silently poisons a class average. Track each frame's correlation to its class template; when it collapses, split the movie at that point and treat the halves as separate datasets.",
  },
  {
    n: "09",
    title: "Cluster overfitting",
    body: "Chasing k will happily 'discover' noise classes. Require split-half reproducibility and a silhouette floor before any class is allowed into averaging — an unjustified class is worse than a merged one.",
  },
  {
    n: "10",
    title: "Rigid PDBs are not solution structures",
    body: "Simulated heights systematically overshoot real ones — the BioAFMviewer paper says so — and crystal contacts freeze conformations a tip would flatten. Fit against ensembles (NMR models, normal-mode movies, MD snapshots) and treat C as a shape score, never as a ruler.",
  },
];

export const FAQS: { q: string; a: string }[] = [
  {
    q: "Is forking NanoLocz actually legal?",
    a: "Yes. NanoLocz ships under GNU GPL-3.0, which explicitly grants the right to fork, modify and redistribute. Your derivative must also be GPL-3.0, keep license headers, and state your changes. Academically, cite Heath et al. (Small Methods 2024) and, if you use the method, the LAFM paper (Nature 2021).",
  },
  {
    q: "Will results match the MATLAB version?",
    a: "Within tolerance, yes — if you test for it. Most of NanoLocz is deterministic linear algebra and filtering, which ports cleanly. The trap is precision: keep float64 for levelling, allow float32 only where the science tolerates it, and build a parity suite from the bundled test data in week one.",
  },
  {
    q: "What GPU do I actually need?",
    a: "Any NVIDIA card from the last ~8 years (compute capability 6.0+, GTX 10-series and up) runs CuPy and CUDA 12.x. 6–8 GB of VRAM is comfortable for 4096² LAFM renders; 2 M localizations fit with room to spare. No cuDNN needed for the core pipeline — it's custom kernels and cuFFT.",
  },
  {
    q: "What replaces the App Designer GUI?",
    a: "In order of cost: a headless CLI (weeks), then a napari plugin that gives you pan/zoom, colormaps, drawing tools and GPU-accelerated rendering largely for free (a couple of months), then an optional web dashboard. Full control-by-control parity is the final milestone.",
  },
  {
    q: "How much work is this, honestly?",
    a: "For one strong Python/CUDA developer: a scriptable core with GPU LAFM in ~6–8 weeks; a usable napari interface by week 12–16; full GUI parity plus packaging can stretch toward ~5 months. The v1.42 library split makes the front half much cheaper than it would have been a year ago.",
  },
  {
    q: "Linux only, or cross-platform?",
    a: "The Python core runs anywhere. The GPU story is best on Linux — native CUDA, headless operation, cluster-friendly — which is exactly what you asked for. Windows users get WSL2; macOS falls back to the CPU/NumPy path.",
  },
  {
    q: "Won't the coding agent forget things and break what already works?",
    a: "It will forget — that's the design assumption, not a bug to hope away. The plan externalizes all memory: AGENT.md plus one SPEC per module, a golden-file parity suite that is the only definition of 'correct', tasks sized to one session, and one commit per green test. A fresh agent — or a different model entirely — resumes from repo state, not from conversation history.",
  },
  {
    q: "How many particles do class averages need before they help?",
    a: "Rule of thumb: a few hundred well-aligned repeats per class for visible sharpening, a few thousand for serious resolution work. HS-AFM movies produce tens of thousands of particle instances routinely, so data is rarely the bottleneck — cluster stability (split-half, silhouette) decides how many classes you're allowed to trust.",
  },
  {
    q: "Does averaging really beat the localization limit?",
    a: "Yes, and it's the point of the method: aligning N repeats of the same structure shrinks the noise on the mean by √N. This is the single-particle AFM averaging used in the NanoLocz paper itself — localization finds the particles, class averaging sharpens them, and heterogeneity is what clustering exists to control.",
  },
  {
    q: "Can tip deformation actually be undone?",
    a: "Partially, and honestly. Blind tip-shape estimation plus regularized (Wiener) deconvolution recovers edge fidelity on class averages, and NanoLocz's own Simulation AFM module is a perfect self-test: simulate a known structure through an estimated tip and check you recover it. It's a correction, never an inversion — you don't invent data.",
  },
  {
    q: "Is this site an official release?",
    a: "No — it's a feasibility blueprint built to answer the question, unaffiliated with the NanoLocz authors. The correct next step is to open an issue on their repository proposing the port; the GPL license already says yes.",
  },
  {
    q: "Wrap BioAFMviewer, or reimplement its simulation core?",
    a: "Reimplement the core and keep the original as a validation oracle. The algorithm — Van der Waals hard-sphere collision with a cone (half-angle α) plus probe sphere (radius R) tip — is fully specified in Amyot & Flechsig (PLOS Comput Biol 2020, CC-BY) and its supplement. A standalone GUI binary can't run headless on a cluster anyway. The paper's own quantitative bar is image correlation C > 0.9; we target C > 0.95 agreement between your renders and BioAFMviewer's on the same PDB + parameters.",
  },
  {
    q: "Where does automated rough fitting fit in? BioAFMviewer doesn't have it.",
    a: "Exactly — the authors list 'an automatized procedure to detect optimal fitting' as their stated future work. That's the gap the GPU port fills: rotate the bead cloud per candidate transform (cheap), synthesize the whole candidate grid in one batched CUDA launch, score with masked NCC, then Nelder–Mead refine (tx, ty, θ, z-scale). What takes minutes of manual eyeballing becomes a seconds-long search, with a ΔC uncertainty map instead of a single number.",
  },
  {
    q: "Which coding AI should I run this plan with?",
    a: "Any agent that can read files and run pytest — Claude Code, Codex CLI, Cursor's agent, Copilot's coding agent, Aider, Cline. The plan is deliberately model-agnostic: the repo (AGENT.md, SPEC/, golden files, parity tests) is the interface, so you can swap models mid-project or run two in parallel on different cards. The session protocol and handoff notes are what make that safe.",
  },
  {
    q: "I have no MATLAB at all — how do I get the golden files?",
    a: "Three routes, in order: (1) GNU Octave often runs plain .m functions — and v1.42 externalized NanoLocz's core into exactly that kind of library; (2) ask a collaborator with a license to run the capture script once and send the .npy exports; (3) open an issue on the NanoLocz repo requesting reference outputs — the authors ship test data and are responsive. The port itself never needs MATLAB.",
  },
];

export const TERM_LINES: { t: "cmd" | "out" | "ok"; s: string }[] = [
  { t: "cmd", s: "nvidia-smi                      # driver ≥ 535 · CUDA 12.x" },
  { t: "out", s: "NVIDIA-SMI 550.54 · CUDA 12.4 · RTX A4000 · 16384 MiB" },
  { t: "cmd", s: "conda create -n nanolocz python=3.11 -y && conda activate nanolocz" },
  { t: "cmd", s: "pip install nanolocz-gpu        # blueprint name — not a real package (yet)" },
  { t: "out", s: "cupy-cuda12x · pySPM · napari · dask-cuda · zarr" },
  { t: "cmd", s: "nanolocz gpu-check" },
  { t: "ok", s: "✔ device 0 · NVIDIA RTX A4000 · sm_86 · float32 pipeline verified" },
  { t: "cmd", s: "nanolocz lafm --in membrane.h5 --px 0.49 --sigma 2.1 --gpu 0" },
  { t: "ok", s: "→ 2,146,882 localizations · 4096² render in 3.8 s · FRC 4.1 nm" },
];

export const ENV_NOTES: { k: string; v: string }[] = [
  { k: "Driver", v: "NVIDIA ≥ 535 · CUDA 12.x runtime ships with CuPy wheels" },
  { k: "VRAM", v: "6 GB comfortable · 8 GB+ for 8192² renders and movies" },
  { k: "Headless", v: "Runs over plain SSH — no X11 needed for the CLI pipeline" },
  { k: "Clusters", v: "Docker --gpus all, or Singularity for HPC schedulers" },
  { k: "No GPU?", v: "Same API transparently falls back to NumPy/SciPy on CPU" },
];

/* ------------------------------------------------------------------ */
/* the agent-proof plan                                                */
/* ------------------------------------------------------------------ */

export const PRINCIPLES: { n: string; t: string; b: string }[] = [
  {
    n: "01",
    t: "Memory lives in the repo, not the agent",
    b: "Every contract is written down in SPEC/ before code exists. When a session ends, the next session — possibly a different model entirely — resumes from specs and green tests. Zero knowledge evaporates.",
  },
  {
    n: "02",
    t: "One task ≤ one session, proven by one command",
    b: "Task cards are sized so a fresh agent can finish them with no prior context. 'Done' has exactly one meaning: pytest tests/parity/test_<task>.py -q is green.",
  },
  {
    n: "03",
    t: "Golden files are the oracle",
    b: "MATLAB outputs are committed as .npy references; parity tests assert a tolerance (rel 1e-5 on float64 paths). The agent never has to remember what 'correct' means — the files do.",
  },
  {
    n: "04",
    t: "Decisions are records, not opinions",
    b: "ADR/ freezes architecture choices — Zarr layout, float policy, kernel strategy, format quirks. An agent reads them; it does not re-litigate them three sessions later.",
  },
  {
    n: "05",
    t: "The type system is a second agent",
    b: "Strict dataclasses (Frame, Meta, Localizations, ParticleStack) make wrong wiring a compile-time error instead of a silent numerical mystery discovered in week 9.",
  },
];

export const REPO_TREE: { path: string; d: number; note: string }[] = [
  { path: "nanolocz-gpu/", d: 0, note: "fork provenance preserved · GPL-3.0 headers everywhere" },
  { path: "AGENT.md", d: 1, note: "the brain file — read first, every session" },
  { path: "SPEC/", d: 1, note: "one .md contract per module" },
  { path: "SPEC/tasks.md", d: 2, note: "task cards: id · deps · acceptance · prompt seed" },
  { path: "golden/", d: 1, note: "MATLAB reference outputs (.npy) + input fixtures" },
  { path: "ADR/", d: 1, note: "0001-zarr-layout · 0002-float-policy · 0003-kernels" },
  { path: "src/nanolocz/", d: 1, note: "" },
  { path: "io/", d: 2, note: "openers: .spm .asd .jpk .ibw .gwy .h5-jpk" },
  { path: "level/  filters/  detect/", d: 2, note: "levelling · deskar · detection + stats + SPT" },
  { path: "lafm/  kernels/", d: 2, note: "localize · splat · FRC · .cu sources" },
  { path: "lafm_plus/", d: 2, note: "drift · grouping · averaging · dynamics" },
  { path: "tests/parity/", d: 1, note: "one test module per golden set" },
  { path: "napari_plugin/  cli/  docker/", d: 1, note: "interface · headless runner · CUDA base image" },
];

export const AGENT_MD = `# AGENT.md — read before touching anything
Project: nanolocz-gpu — Python/CUDA port of NanoLocz (GPL-3.0, cite Heath et al. 2024)

Rules:
1. Never modify SPEC/ or ADR/ without human sign-off.
2. Every public function takes/returns typed dataclasses from core/types.py.
3. Match golden/ outputs: rel tol 1e-5 (float64 paths), 1e-3 (float32 GPU paths).
4. One task = one commit. If scope creeps, stop and ask.
5. GPU code lives in kernels/ and is optional: the CPU fallback must stay green.

Current phase: P1 core port. Backlog: SPEC/tasks.md. Golden sets: golden/.
Last green commit is always the recovery point.`;

export const SESSION_STEPS: { who: "YOU" | "AGENT" | "BOTH"; t: string; b: string }[] = [
  {
    who: "YOU",
    t: "Open every session with the brain file",
    b: "Paste AGENT.md plus one task card ID. The agent loads that task's spec and parity test. Context stays small enough that forgetting is structurally impossible.",
  },
  {
    who: "AGENT",
    t: "Restate the contract, then code",
    b: "The agent writes back inputs, outputs and tolerances before implementing. If SPEC/ is silent on anything, it stops and asks — silence is where forks go to die.",
  },
  {
    who: "AGENT",
    t: "Failing test first, implementation second",
    b: "Write the parity test against golden/, watch it fail, implement until green. Green is the only accepted definition of done — not 'it looks right'.",
  },
  {
    who: "BOTH",
    t: "One commit per task",
    b: "Commit messages like 'NL-14: batch levelling (parity ✓)'. The git log becomes the recovery point whenever a session derails — and one will.",
  },
  {
    who: "YOU",
    t: "When it loops, reset — don't push",
    b: "Two failed attempts at the same error? Kill the session, revert to the last green commit, start fresh with the same card. Amnesia is now a feature you've already paid for.",
  },
  {
    who: "BOTH",
    t: "Weekly drift check, 30 minutes",
    b: "Run the full parity suite, scan ADRs for needed updates, promote the next task card. The plan survives because auditing it is cheap.",
  },
];

export type WbsTask = {
  id: string;
  title: string;
  size: "S" | "M" | "L";
  deps: string;
  accept: string[];
  prompt: string;
};

export const WBS: {
  col: string;
  cls: string;
  tasks: WbsTask[];
}[] = [
  {
    col: "P0 · FOUNDATION",
    cls: "text-teal border-teal/50",
    tasks: [
      {
        id: "NL-01",
        title: "Fork audit & toolbox map",
        size: "S",
        deps: "—",
        accept: [
          "Every toolbox call in the .m library listed with file:line",
          "GPL headers + NOTICE drafted",
        ],
        prompt: "Read SPEC/audit.md. List every MATLAB toolbox function used under core-lib/, grouped by toolbox, with the file:line of each call. Write ADR/0000-toolbox-map.md.",
      },
      {
        id: "NL-02",
        title: "Golden parity harness",
        size: "S",
        deps: "NL-01",
        accept: [
          "pytest runs CPU-only in CI",
          "Fixture loader validates checksums",
          "Tolerance policy frozen in ADR/0002",
        ],
        prompt: "Scaffold tests/parity: a fixture loader for golden/*.npy, assert helpers with per-dtype rel tolerance, and a CI workflow that needs no GPU.",
      },
      {
        id: "NL-03",
        title: "Typed core contracts",
        size: "S",
        deps: "NL-01",
        accept: [
          "core/types.py dataclasses: Frame, Meta, Localizations, ParticleStack",
          "pyright --strict passes",
        ],
        prompt: "Implement core/types.py per SPEC/types.md. No logic, types only — plus unit tests that serialize each dataclass to Zarr and back.",
      },
    ],
  },
  {
    col: "P1 · CORE PORT",
    cls: "text-amber border-amber/50",
    tasks: [
      {
        id: "NL-10",
        title: "Zarr schema + io skeleton",
        size: "M",
        deps: "NL-02 NL-03",
        accept: ["Round-trip test: opener → Zarr → Frame identical", "ADR/0001 frozen"],
        prompt: "Implement the Zarr store layout from ADR/0001 and the io opener interface from SPEC/io.md, with a dummy opener and round-trip parity test.",
      },
      {
        id: "NL-11",
        title: ".gwy + .h5-jpk openers",
        size: "M",
        deps: "NL-10",
        accept: ["Parity vs golden on 3 bundled sample files"],
        prompt: "Port the .gwy and .h5-jpk readers using pySPM and h5py per SPEC/io.md. Parity tests against golden/gwy_01..03.",
      },
      {
        id: "NL-12",
        title: ".spm / .jpk / .ibw openers",
        size: "M",
        deps: "NL-10",
        accept: ["Meta fields match MATLAB reader key-for-key"],
        prompt: "Port the Bruker .spm, JPK .jpk and Igor .ibw readers. Golden fixtures provided; meta parity is the acceptance bar.",
      },
      {
        id: "NL-13",
        title: ".asd opener (all channels)",
        size: "L",
        deps: "NL-12",
        accept: ["Heights and all channels match the MATLAB .asd reader"],
        prompt: "Reimplement the .asd binary reader per SPEC/asd.md, including 'trace-only' files. This is the hardest parser — flag any undocumented byte range immediately.",
      },
      {
        id: "NL-14",
        title: "Levelling: line / plane / multi-plane",
        size: "M",
        deps: "NL-10",
        accept: ["float64 path rel 1e-5 vs golden", "SPIW-style weighted fit ported"],
        prompt: "Port the levelling suite incl. the weighted multi-plane fit. float64 NumPy first; no GPU yet.",
      },
      {
        id: "NL-15",
        title: "Filters, masks, profiles",
        size: "M",
        deps: "NL-10",
        accept: ["medfilt / scar-removal / FFT-align parity", "Line-profile stats match"],
        prompt: "Port the filter set from SPEC/filters.md, including the v1.10 scar/scratch remover. Parity per filter on golden fixtures.",
      },
      {
        id: "NL-16",
        title: "Detection + stats + masks",
        size: "M",
        deps: "NL-15",
        accept: ["Same particle count/centroids (±0.5 px) as MATLAB on fixtures"],
        prompt: "Port auto-threshold detection, connected components and height/width statistics. Centroid tolerance 0.5 px per SPEC/detect.md.",
      },
      {
        id: "NL-17",
        title: "Single-particle tracking",
        size: "M",
        deps: "NL-16",
        accept: ["Track IDs identical to golden on the test movie"],
        prompt: "Port the frame-to-frame linker. Deterministic tie-breaking per SPEC/spt.md — golden tracks must reproduce exactly.",
      },
    ],
  },
  {
    col: "P2 · GPU",
    cls: "text-sky2 border-sky2/50",
    tasks: [
      {
        id: "NL-20",
        title: "CuPy backend switch + float policy",
        size: "S",
        deps: "NL-14",
        accept: ["Every P1 function passes with xp=cupy", "ADR/0002 respected"],
        prompt: "Introduce the xp-array backend switch (NumPy ↔ CuPy) per ADR/0003. Re-run the whole parity suite on GPU.",
      },
      {
        id: "NL-21",
        title: "Batched levelling kernel",
        size: "M",
        deps: "NL-20",
        accept: ["1000-frame movie levels in one launch", "rel 1e-4 in float32 path"],
        prompt: "Write the batched least-squares levelling kernel per SPEC/level.md. Benchmark vs the CPU loop; record numbers in bench/.",
      },
      {
        id: "NL-22",
        title: "Detection & stats kernels",
        size: "M",
        deps: "NL-20",
        accept: ["label + reductions on-device", "Parity within tolerance"],
        prompt: "Move thresholding, connected components and stat reductions to cupyx.scipy.ndimage per SPEC/detect.md.",
      },
      {
        id: "NL-23",
        title: "LAFM splat kernel + FRC",
        size: "L",
        deps: "NL-20",
        accept: ["2M localizations → 4096² render < 10 s", "FRC curve matches MATLAB ±2%"],
        prompt: "Implement the CUDA splat kernel (float32 accumulation buffer) and cuFFT FRC per SPEC/lafm.md. This is the flagship — take your time on numerics.",
      },
      {
        id: "NL-24",
        title: "Simulation AFM kernels",
        size: "M",
        deps: "NL-20",
        accept: ["Tip dilation + parachuting parity", "SimAFM usable as kernel self-test"],
        prompt: "Port tip dilation and parachuting per SPEC/sim.md. These doubles as the deconvolution self-test for NL-35.",
      },
    ],
  },
  {
    col: "P3 · LAFM+",
    cls: "text-teal2 border-teal2/50",
    tasks: [
      {
        id: "NL-30",
        title: "Per-frame drift estimation",
        size: "M",
        deps: "NL-22",
        accept: ["Sub-px phase-correlation shifts", "Residual < 0.5 nm on drift fixture"],
        prompt: "Estimate per-frame drift via cuFFT phase correlation on the levelling channel; store shift vectors in the Zarr meta. Golden: golden/drift_movie.",
      },
      {
        id: "NL-31",
        title: "Directional deskar filter",
        size: "S",
        deps: "NL-15",
        accept: ["Scar fixture cleaned, lattice intact", "No GPU required"],
        prompt: "Add the directional median along the fast-scan axis per SPEC/filters.md#deskar. Parity against the scar fixture only.",
      },
      {
        id: "NL-32",
        title: "Particle substack extraction",
        size: "M",
        deps: "NL-16 NL-30",
        accept: ["ParticleStack (n, t, h, w) gather on-device", "Sub-pixel centering"],
        prompt: "Cut drift-corrected substacks at detected positions per SPEC/lafm_plus.md#extract. Fancy-index gather kernel; validate against hand-picked ROIs.",
      },
      {
        id: "NL-33",
        title: "Embed + cluster (PCA → HDBSCAN)",
        size: "M",
        deps: "NL-32",
        accept: ["Split-half stable classes", "Outlier bin for tip crashes"],
        prompt: "Flatten → standardize → PCA(32) → HDBSCAN per SPEC/lafm_plus.md#group. Report silhouette and split-half agreement; reject unstable k.",
      },
      {
        id: "NL-34",
        title: "In-class align + average",
        size: "M",
        deps: "NL-33",
        accept: ["Mean noise ↓ √N on synthetic fixture", "Rotation + shift alignment"],
        prompt: "Align class members by batched cuFFT cross-correlation, accumulate with atomics. Prove the √N curve on the synthetic stack in golden/.",
      },
      {
        id: "NL-35",
        title: "Tip-shape estimation + deconv",
        size: "L",
        deps: "NL-34 NL-24",
        accept: ["Blind tip estimate on class edges", "SimAFM round-trip recovers truth"],
        prompt: "Estimate the effective tip from class-consistent edges, apply regularized Wiener deconvolution to averages. Validate via the NL-24 simulator round-trip.",
      },
      {
        id: "NL-36",
        title: "Dynamics: traces, transitions, dwell",
        size: "M",
        deps: "NL-33 NL-17",
        accept: ["Per-frame class membership table", "Transition matrix + dwell times"],
        prompt: "From class memberships over time, compute population traces, the state transition matrix and dwell-time histograms per SPEC/lafm_plus.md#dynamics.",
      },
      {
        id: "NL-37",
        title: "Replay hooks (napari layers)",
        size: "M",
        deps: "NL-36 NL-41",
        accept: ["Scrub movie with class-coloured overlays", "Click track → member frames"],
        prompt: "Wire replay into the napari plugin: class-coloured overlay layer synced to the movie slider, plus track→frames lookup.",
      },
    ],
  },
  {
    col: "P2.6 · SIM BRIDGE",
    cls: "text-sky2 border-sky2/50",
    tasks: [
      {
        id: "NL-50",
        title: "PDB ingest → BeadCloud",
        size: "S",
        deps: "NL-03",
        accept: [
          "Element → VdW radius table (C 1.70, N 1.55, O 1.52, S 1.80, H 1.20 Å)",
          "Multi-model PDBs become BeadCloud frames",
        ],
        prompt: "Implement biotite-based PDB ingest per SPEC/sim.md#ingest: ATOM records → BeadCloud(x, y, z, r, element); multi-model files become frames. Golden: golden/pdb_1aon counts + checksum.",
      },
      {
        id: "NL-51",
        title: "Hard-collision synthesis (CPU ref)",
        size: "M",
        deps: "NL-50",
        accept: [
          "Sphere + cone terms exactly per paper S1 Text",
          "Frozen golden: 1aon side view at R 1.0 nm, α 10°, a 0.5 nm",
        ],
        prompt: "Implement the CPU height-field per KICKOFF.md §algorithm (sphere term + cone term, grid step a). Generate and freeze golden/pdb_1aon_side.npy; tolerance rel 1e-6.",
      },
      {
        id: "NL-52",
        title: "CUDA synthesis kernel",
        size: "M",
        deps: "NL-51 NL-20",
        accept: [
          "Candidates (tip params × orientations) as the batch dimension",
          "Parity rel 1e-4 vs the NL-51 golden",
        ],
        prompt: "Port NL-51 to a CUDA kernel with candidates as the batch dimension per ADR/0003. Record throughput for a 60k-atom complex (GroEL-GroES class) in bench/.",
      },
      {
        id: "NL-53",
        title: "NCC scorer + rough fit",
        size: "L",
        deps: "NL-52",
        accept: [
          "Coarse grid tx/ty ±50 Å @ 2 Å × θ 0–358° @ 2°",
          "Nelder–Mead refine on (tx, ty, θ, z-scale)",
          "Recover a hidden transform from a synthetic target with C > 0.98",
        ],
        prompt: "Implement masked NCC per SPEC/sim.md#fit, the coarse grid search over the batched kernel, and Nelder–Mead refinement. Report C, best transform, top-5 candidates and the ΔC map.",
      },
      {
        id: "NL-54",
        title: "Validation + napari overlay",
        size: "M",
        deps: "NL-53 NL-41",
        accept: [
          "C > 0.95 vs BioAFMviewer renders on 1aon, ClpB, Cas9, F1 targets",
          "Fit-overlay layer + report panel in the napari plugin",
        ],
        prompt: "Cross-validate against BioAFMviewer renders (same PDB + params) per KICKOFF.md §validation; add the fit-overlay layer and fit report panel to the napari plugin.",
      },
    ],
  },
  {
    col: "P4 · INTERFACE & SHIP",
    cls: "text-mag border-mag/50",
    tasks: [
      {
        id: "NL-40",
        title: "CLI + batch runner",
        size: "M",
        deps: "NL-16 NL-23",
        accept: ["nanolocz lafm --in … --gpu 0 works headless", "Batch over folders"],
        prompt: "Build the Click-based CLI per SPEC/cli.md: single-image and batch modes, --gpu / --cpu backend flag, logging to the panel format NanoLocz users know.",
      },
      {
        id: "NL-41",
        title: "napari plugin v1",
        size: "L",
        deps: "NL-40",
        accept: ["Viewer + colormaps + histogram slider", "Draw tools produce masks"],
        prompt: "Implement the napari plugin per SPEC/gui.md: height/LAFM/detection layers, the unified histogram slider, draw-to-mask tools.",
      },
      {
        id: "NL-42",
        title: "Docker + conda packaging",
        size: "S",
        deps: "NL-40",
        accept: ["docker run --gpus all nanolocz gpu-check passes", "conda env one-liner"],
        prompt: "Write the CUDA 12.x Dockerfile and environment.yml per SPEC/packaging.md; pin every version.",
      },
      {
        id: "NL-43",
        title: "Benchmark report + v1.0-gpu",
        size: "S",
        deps: "all",
        accept: ["Published bench numbers on 2+ GPUs", "Release cites Heath et al."],
        prompt: "Run the full benchmark matrix, write bench/REPORT.md, cut v1.0-gpu with the citation block from SPEC/cite.md.",
      },
    ],
  },
];

/* ------------------------------------------------------------------ */
/* LAFM+ extended pipeline                                             */
/* ------------------------------------------------------------------ */

export const LAFM_STAGES: { n: string; t: string; b: string; gpu: string }[] = [
  {
    n: "1",
    t: "Ingest stack",
    b: "HS-AFM movie becomes a Zarr cube (t, y, x) with per-frame metadata. The openers are already built in Phase 1 — this stage is free.",
    gpu: "zero-copy mmap → device",
  },
  {
    n: "2",
    t: "Frame correction",
    b: "Drift from per-frame phase correlation (sub-pixel), streaks removed by a directional median along the fast-scan axis, levelling applied — one batched pass over every frame.",
    gpu: "cuFFT phase correlation + batched median",
  },
  {
    n: "3",
    t: "Localize",
    b: "The NanoLocz LAFM you already know: tip traces become per-frame localization clouds — now drift-corrected, so clouds from different frames finally stack coherently.",
    gpu: "the CUDA splat pipeline from Phase 2",
  },
  {
    n: "4",
    t: "Extract particles",
    b: "Cut substacks at detected positions across all frames → ParticleStack (n_particles, t, h, w). This is the bridge from 'images' to 'single particles'.",
    gpu: "fancy-index gather kernel",
  },
  {
    n: "5",
    t: "Embed + group",
    b: "Flatten, standardize, PCA to ~32 components, then HDBSCAN/k-means. Similar conformations land in the same class; tip crashes and debris fall into an outlier bin.",
    gpu: "cuML PCA + clustering (sklearn on CPU)",
  },
  {
    n: "6",
    t: "Align + average",
    b: "In-class alignment by FFT cross-correlation, then per-class averaging. Noise falls as √N — this is the SPAFM averaging that produced the NanoLocz paper's resolution.",
    gpu: "batched cuFFT + atomic accumulation",
  },
  {
    n: "7",
    t: "Tip-aware sharpen",
    b: "Estimate the effective tip shape from class-consistent edges, then regularized Wiener deconvolution on the averages. Correction, not invention — validated by Simulation AFM round-trip.",
    gpu: "Wiener deconvolution on-device",
  },
  {
    n: "8",
    t: "Dynamics replay",
    b: "Class membership per particle per frame → population traces, a state-transition matrix, dwell times. Scrub the original movie with class-coloured overlays and click any track back to its raw frames.",
    gpu: "metrics on-device · graph on CPU",
  },
];

/* ------------------------------------------------------------------ */
/* the operator's manual — driving a coding AI                         */
/* ------------------------------------------------------------------ */

export const OPERATOR_STEPS: { t: string; title: string; body: string }[] = [
  {
    t: "T+0:00",
    title: "Load the brain",
    body: "Paste AGENT.md plus one task card ID. The agent reads that card's spec and parity test — nothing else is required.",
  },
  {
    t: "T+0:02",
    title: "Restate the contract",
    body: "The agent writes back inputs, outputs and tolerances. You check against SPEC/. Misreadings die here, not in review.",
  },
  {
    t: "T+0:05",
    title: "Red → green",
    body: "Failing parity test first, then implementation until pytest is green. Green is the only accepted definition of done.",
  },
  {
    t: "T+0:45",
    title: "One commit",
    body: "“NL-14: batch levelling (parity ✓)”. The git log becomes the recovery point whenever a session derails — and one will.",
  },
  {
    t: "T+0:50",
    title: "Hand off & stop",
    body: "Session note written, git clean, stop — even when it “feels close”. Context rot is worse than a fresh start.",
  },
];

export type PromptCard = {
  key: string;
  label: string;
  when: string;
  body: string;
};

export const PROMPT_LIB: PromptCard[] = [
  {
    key: "bootstrap",
    label: "BOOTSTRAP",
    when: "The first session ever — builds the repo's memory system",
    body: `You are starting nanolocz-gpu: a GPL-3.0 Python/CuPy/CUDA port of
NanoLocz (github.com/George-R-Heath/NanoLocz). This is session 1 of many;
you will not retain memory of this conversation, so everything must land
in files.

1. Create the scaffold: AGENT.md (contents below), SPEC/, ADR/, golden/,
   tests/parity/, src/nanolocz/, plus pyproject.toml (python >=3.11,
   strict pyright, ruff, pytest) and a CPU-only CI workflow.
2. Write SPEC/tasks.md by transcribing the task cards: id · deps ·
   acceptance criteria, one per line.
3. Commit "scaffold: brain file, specs skeleton, parity harness".
4. Stop and report: file tree + open questions. No library code yet.

<paste AGENT.md contents here>`,
  },
  {
    key: "golden",
    label: "GOLDEN",
    when: "Whenever a new module needs MATLAB reference outputs",
    body: `Capture golden references for card NL-XX.
1. Draft golden/capture_<xx>.m from SPEC/<module>.md.
2. Run it in MATLAB + NanoLocz on the bundled test data; save every output
   as .npy with a sha256 sidecar under golden/<xx>/.
3. Record exact MATLAB + toolbox versions in golden/<xx>/ENV.md.
4. Commit the fixtures. Do NOT implement the Python port in this session.

No MATLAB available? Run the capture script under GNU Octave against the
externalized .m library (v1.42). If a function fails under Octave, list it
and open an upstream issue requesting the .npy export — a collaborator
with MATLAB can generate it in minutes.`,
  },
  {
    key: "execute",
    label: "EXECUTE",
    when: "The daily driver — exactly one task card per session",
    body: `<paste AGENT.md here>

Execute task card NL-XX.
Card: <paste the card's prompt seed from the work-breakdown board>

Rules:
1. Write the failing parity test FIRST.
2. Implement until \`pytest tests/parity/test_<xx>.py -q\` is green —
   that is the only definition of done.
3. One commit: "NL-XX: <title> (parity ✓)".
4. If SPEC/ is silent on anything, STOP and draft an ADR — never guess.

Report: files changed, test-output tail, blockers.`,
  },
  {
    key: "interrogate",
    label: "INTERROGATE",
    when: "Before trusting any “done” — a read-only audit session",
    body: `Audit task NL-XX against its contract. Answer only from files —
quote them:
1. Which golden fixture and tolerance does the parity test use?
2. Where is the float32/float64 decision made for this path? (which ADR?)
3. Name three inputs that could break it, and why.
4. Does the commit touch anything outside the card's scope? List files.

Read-only session — do not modify any code.`,
  },
  {
    key: "reset",
    label: "RESET",
    when: "After two failed attempts at the same error — revert, then this",
    body: `Context reset — discard memory of previous attempts.
Last green commit: <hash>. Read AGENT.md and SPEC/tasks.md only.

Task NL-XX failed twice with:
<paste the exact error>

Produce, WITHOUT touching implementation files:
- ranked hypotheses, cheapest-to-falsify first
- the minimal experiment for each (file, command, expected result)
- which hypothesis you'd bet on, and why

Stop there.`,
  },
  {
    key: "handoff",
    label: "HANDOFF",
    when: "The last five minutes of every session, without exception",
    body: `Write SESSIONS/<today>.md for the next session — it will not
remember this one:
1. Cards completed + commit hashes
2. Current blocker, with the exact error
3. The exact next command to run
4. Any ADR drafts pending sign-off

Then confirm \`git status\` is clean and stop.`,
  },
];

export const ANTI_PATTERNS: { never: string; instead: string }[] = [
  {
    never: "“Port NanoLocz to Python” as a single prompt. It guarantees drift, half-finished modules and confident nonsense.",
    instead: "One seeded task card per session. The card names the spec, the golden set and the stopping condition.",
  },
  {
    never: "Accepting “it looks right”. Visual checks have zero memory and infinite optimism.",
    instead: "Green parity test or it didn't happen. The golden files are the only opinion that counts.",
  },
  {
    never: "Letting a session run past ~an hour. Late-session output quality decays faster than you'll notice.",
    instead: "Stop at the 55-minute mark, commit, write the handoff note, start fresh. Amnesia is budgeted for.",
  },
  {
    never: "Letting the agent edit SPEC/ or ADR/ mid-task to make its code pass. That's the contract editing itself.",
    instead: "Spec changes are drafts the agent proposes and a human approves in a separate session.",
  },
  {
    never: "GPU code before the CPU parity suite is green. Two unknowns at once is how ports die.",
    instead: "Backend switch (card NL-20) only after P1 passes — then the same tests re-run on device.",
  },
  {
    never: "Trusting the agent's recollection of MATLAB behaviour from training data. It sounds right and often isn't.",
    instead: "Capture golden outputs once (MATLAB, Octave, or a collaborator), commit them, trust only files.",
  },
];

export const HANDOFF_MD = `# SESSIONS/2026-02-14.md
done:    NL-10 (a1b2c3d) · NL-11 (e4f5g6h) — parity green
blocked: NL-13 — .asd header bytes 44–52 undocumented in SPEC
next:    pytest tests/parity/test_asd.py::test_trace_only -q
adr:     0009-asd-channel-map drafted, awaiting sign-off`;

export const WORKS_WITH: string[] = [
  "Claude Code",
  "Codex CLI",
  "Cursor agent",
  "Copilot coding agent",
  "Aider",
  "Cline / Roo",
];

/* ------------------------------------------------------------------ */
/* simulation bridge (BioAFMviewer-style)                              */
/* ------------------------------------------------------------------ */

export const SIM_LEDGER: { from: string; to: string; note: string }[] = [
  {
    from: "PDB files + multi-model movies",
    to: "biotite ingest → BeadCloud frames in Zarr",
    note: "iMODS / ProDy normal-mode movies and MD trajectories all arrive as PDB",
  },
  {
    from: "VdW hard-collision scanning",
    to: "per-pixel max kernel over atom spheres",
    note: "cone α + probe R + step a, exactly per the paper's S1 Text",
  },
  {
    from: "Tip geometry panel (R, α, a)",
    to: "kernel parameters — batched, not sequential",
    note: "one launch renders an entire tip-radius / angle sweep",
  },
  {
    from: "Orientation viewer (rotate + rescan)",
    to: "napari 3D + CLI rotation matrices",
    note: "Euler angles or a matrix in, height map out",
  },
  {
    from: "Image correlation C vs experiment",
    to: "masked NCC via cuFFT",
    note: "the paper reports C > 0.9 for ClpB, Cas9 and F1-ATPase",
  },
  {
    from: "“Future work: automated fitting”",
    to: "we build it: coarse grid + Nelder–Mead",
    note: "flagged by the authors in 2020 — the GPU makes it cheap at last",
  },
  {
    from: "Qualitative hs-AFM comparison",
    to: "fit report: C, transform, ΔC map, top-5",
    note: "an uncertainty surface, not a single number",
  },
];

export const FIRST_SESSIONS: { n: string; t: string; card: string }[] = [
  { n: "S1", t: "Scaffold the sim-bridge module, biotite ingest, BeadCloud type", card: "NL-50" },
  { n: "S2", t: "CPU height-field; generate and freeze the 1aon golden", card: "NL-51" },
  { n: "S3", t: "CUDA kernel; parity vs the CPU reference at rel 1e-4", card: "NL-52" },
  { n: "S4", t: "Masked NCC scorer; cross-check C against BioAFMviewer renders", card: "NL-53" },
  { n: "S5", t: "Coarse grid search + refine; recover a hidden transform on a synthetic", card: "NL-53" },
];

export const KICKOFF_MD = `# KICKOFF.md — Simulation Bridge (PDB → pseudo-AFM + rough fit)
Module: sim-bridge of nanolocz-gpu (GPL-3.0 port of NanoLocz). Read AGENT.md first.
Goal: rebuild BioAFMviewer's core (Amyot & Flechsig, PLoS Comput Biol 2020,
doi:10.1371/journal.pcbi.1008444) as a scriptable, GPU-accelerated library.
BioAFMviewer is a standalone GUI — do NOT vendor its binaries. Reimplement from
the paper; keep the original installed locally as a cross-check oracle only.

## The algorithm (exactly)
1. PDB ingest: biotite, ATOM records; element → VdW radius table
   (C 1.70, N 1.55, O 1.52, S 1.80, H 1.20, P 1.80 Å). Multi-model PDB = movie.
2. Orientation: apply rotation matrix; virtual surface = min-z plane.
3. Tip: probe sphere radius R + cone half-angle α.
4. Grid: step a (default 0.5 nm) over footprint + 2R margin.
5. Per-pixel height, hard collision:
     sphere:  h_s(x,y) = max_i [ z_i + sqrt((R+r_i)^2 - d_i^2) - R ], d_i < R+r_i
     cone:    h_c(x,y) = max_i [ z_i + (d_i - r_i)·tan(α) ]
     h(x,y) = max(h_s, h_c, 0)
6. Score vs experiment: normalized cross-correlation C on a masked ROI.

## Rough fit (the part the paper calls "future work" — we ship it)
- Coarse grid: tx,ty ∈ ±50 Å step 2 Å; in-plane θ ∈ 0–358° step 2°.
- Rotate the BEAD CLOUD per candidate (cheap), then ONE batched CUDA kernel
  synthesizes all candidates; score all with masked NCC.
- Refine: Nelder–Mead on (tx, ty, θ, z-scale) from the best coarse hit.
- Report: C, best transform, top-5 candidates, ΔC uncertainty map.

## Conventions (non-negotiable)
- Types from core/types.py; new dataclass BeadCloud lives in SPEC/sim.md.
- float32 on GPU, float64 CPU reference; parity rel 1e-4.
- GPU optional: every function passes on NumPy via the xp switch (ADR/0003).

## Milestone 1 acceptance (card NL-51/NL-52)
- nanolocz sim --pdb 1aon.pdb --R 1.0nm --alpha 10 --step 0.5nm out.tiff
- GroEL-GroES side view matches BioAFMviewer-class output; same PDB + params,
  C(ours, theirs) > 0.95. (Validation targets: 1aon, ClpB, Cas9, F1-ATPase 2JDI.)
- pytest tests/parity/test_sim_bridge.py -q is green. One commit per card.

## References
- Algorithm + limits (heights overestimated, rigid PDBs): the 2020 paper + S1 Text
- Flexible fitting to AFM templates: Niina, Fuchigami & Takada, JCTC 2020
- Python prior art: afmulator (DL pseudo-AFM), pySPM afm_simulation, NanoLocz SimAFM`;
