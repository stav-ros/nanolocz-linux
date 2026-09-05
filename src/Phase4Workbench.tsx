import { useMemo, useState } from "react";
import { Reveal, copyText } from "./lib";
import { SectionHead } from "./Sections";

type Track = "available" | "planned";

type PhaseCard = {
  id: string;
  title: string;
  track: Track;
  summary: string;
  evidence: string;
  command: string;
};

const CARDS: PhaseCard[] = [
  {
    id: "NL-40",
    title: "CLI + batch runner",
    track: "available",
    summary: "Headless entry point for preprocessing, detection, tracking, LAFM, and batch jobs.",
    evidence: "nanolocz/cli/ · tests/test_cli_nl40.py",
    command: "python -m nanolocz.cli --help",
  },
  {
    id: "NL-41",
    title: "Napari plugin v1",
    track: "planned",
    summary: "Interactive image, localization, track, and reconstruction inspection.",
    evidence: "Roadmap card; plugin wiring still pending",
    command: "python -m nanolocz.cli napari --help",
  },
  {
    id: "NL-42",
    title: "Docker + conda",
    track: "planned",
    summary: "Reproducible CPU and CUDA environments for researchers and CI runners.",
    evidence: "Roadmap card; packaging work still pending",
    command: "python -m pip install -e '.[test]'",
  },
  {
    id: "NL-43",
    title: "Benchmark + release",
    track: "planned",
    summary: "Publish parity, performance, and hardware-validation evidence for v1.0-gpu.",
    evidence: "Roadmap card; release gate still pending",
    command: "python tools/project_check.py",
  },
];

const WORKFLOWS = [
  {
    label: "AFM MOVIE",
    detail: "Open → level → detect → track",
    output: "tracks.zarr",
    accent: "text-teal",
  },
  {
    label: "LAFM",
    detail: "Localizations → splat → FRC",
    output: "reconstruction.npy",
    accent: "text-sky2",
  },
  {
    label: "STRUCTURE",
    detail: "PDB → simulate → fit",
    output: "fit-result.json",
    accent: "text-amber",
  },
];

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      onClick={async () => {
        const ok = await copyText(command);
        if (!ok) return;
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1400);
      }}
      className="border border-line2 px-2.5 py-1.5 font-mono text-[9px] tracking-[0.14em] text-dim transition-colors hover:border-teal/60 hover:text-teal"
      aria-label={`Copy command: ${command}`}
    >
      {copied ? "COPIED" : "COPY COMMAND"}
    </button>
  );
}

function StatusPill({ track }: { track: Track }) {
  const available = track === "available";
  return (
    <span
      className={`border px-2 py-1 font-mono text-[9px] tracking-[0.18em] ${
        available
          ? "border-teal/50 bg-teal/5 text-teal"
          : "border-line2 bg-ink3/40 text-faint"
      }`}
    >
      {available ? "AVAILABLE" : "PLANNED"}
    </span>
  );
}

export function Phase4Workbench() {
  const [filter, setFilter] = useState<"all" | Track>("all");
  const visibleCards = useMemo(
    () => (filter === "all" ? CARDS : CARDS.filter((card) => card.track === filter)),
    [filter]
  );

  return (
    <section id="phase4" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="08"
          kicker="PHASE 4 — INTERFACE + SHIPPING"
          title="A control room for the first usable release"
          aside="The current site is a project dashboard, not yet a scientific viewer. This workbench makes that boundary explicit: launch the available headless workflow today, then track the interface and release gates without implying that the browser runs Python."
        />

        <Reveal>
          <div className="corner-frame overflow-hidden border border-teal/30 bg-ink2 shadow-[0_24px_80px_-42px_rgba(55,230,196,0.35)]">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-ink px-4 py-3">
              <div>
                <div className="font-mono text-[10px] tracking-[0.24em] text-teal">PHASE 4 LAUNCHPAD</div>
                <div className="mt-1 font-display text-xl font-bold text-fog">Choose a workflow, then leave the browser honestly</div>
              </div>
              <div className="flex gap-1.5" role="group" aria-label="Filter phase four cards">
                {(["all", "available", "planned"] as const).map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setFilter(value)}
                    aria-pressed={filter === value}
                    className={`border px-2.5 py-1.5 font-mono text-[9px] tracking-[0.16em] transition-colors ${
                      filter === value
                        ? "border-teal/70 bg-teal/10 text-teal"
                        : "border-line2 text-faint hover:text-teal2"
                    }`}
                  >
                    {value.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid gap-3 p-4 sm:p-5 lg:grid-cols-3">
              {WORKFLOWS.map((workflow) => (
                <div key={workflow.label} className="border border-line bg-ink p-4">
                  <div className={`font-mono text-[10px] tracking-[0.2em] ${workflow.accent}`}>{workflow.label}</div>
                  <div className="mt-3 font-display text-lg font-bold text-fog">{workflow.detail}</div>
                  <div className="mt-5 flex items-center justify-between border-t border-line pt-3 font-mono text-[10px] text-faint">
                    <span>OUTPUT</span>
                    <span className="text-dim">{workflow.output}</span>
                  </div>
                </div>
              ))}
            </div>

            <div className="border-t border-line bg-teal/[0.035] px-4 py-3 sm:px-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 font-mono text-[10px] tracking-[0.12em] text-dim">
                  <span className="h-2 w-2 rounded-full bg-teal" />
                  PYTHON BACKEND BOUNDARY
                </div>
                <span className="font-mono text-[10px] text-faint">Browser preview only · commands run in nanolocz-gpu</span>
              </div>
            </div>
          </div>
        </Reveal>

        <div className="mt-8 grid gap-8 lg:grid-cols-[1.25fr_0.75fr]">
          <Reveal>
            <div className="overflow-hidden border border-line bg-ink2">
              <div className="flex items-center justify-between border-b border-line bg-ink px-4 py-3">
                <span className="font-mono text-[10px] tracking-[0.22em] text-dim">RELEASE TRACKER</span>
                <span className="font-mono text-[9px] tracking-[0.14em] text-faint">{visibleCards.length} VISIBLE CARDS</span>
              </div>
              <div className="divide-y divide-line/60">
                {visibleCards.map((card) => (
                  <div key={card.id} className="grid gap-4 px-4 py-4 transition-colors hover:bg-ink3/60 sm:grid-cols-[92px_1fr_auto] sm:items-center sm:px-5">
                    <div className="font-mono text-[11px] font-semibold tracking-[0.16em] text-teal2">{card.id}</div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-display text-[15px] font-bold text-fog">{card.title}</span>
                        <StatusPill track={card.track} />
                      </div>
                      <p className="mt-1 text-[12px] leading-relaxed text-dim">{card.summary}</p>
                      <div className="mt-2 font-mono text-[9px] tracking-[0.08em] text-faint">{card.evidence}</div>
                    </div>
                    <CopyCommand command={card.command} />
                  </div>
                ))}
              </div>
            </div>
          </Reveal>

          <Reveal delay={100}>
            <div className="h-full border border-amber/30 bg-amber/[0.045] p-5">
              <div className="font-mono text-[10px] tracking-[0.24em] text-amber">WHAT THIS GUI DOES NEXT</div>
              <h3 className="mt-2 font-display text-xl font-black text-fog">Turn the dashboard into a handoff, not a fake viewer.</h3>
              <ul className="mt-5 space-y-3 text-[12.5px] leading-relaxed text-dim">
                <li className="flex gap-3"><span className="font-mono text-teal">01</span><span>Keep workflow previews and evidence visible in the browser.</span></li>
                <li className="flex gap-3"><span className="font-mono text-teal">02</span><span>Use NL-40 as the executable bridge for real data processing.</span></li>
                <li className="flex gap-3"><span className="font-mono text-teal">03</span><span>Make NL-41 the first true scientific interface milestone.</span></li>
              </ul>
              <div className="mt-6 border-t border-amber/20 pt-4 font-mono text-[10px] leading-relaxed tracking-[0.08em] text-faint">
                NO CLAIMED LIVE ANALYSIS UNTIL THE PYTHON BACKEND IS CONNECTED.
              </div>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
