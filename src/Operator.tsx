import { useState } from "react";
import { SectionHead } from "./Sections";
import {
  ANTI_PATTERNS,
  HANDOFF_MD,
  OPERATOR_STEPS,
  PROMPT_LIB,
  WORKS_WITH,
  Reveal,
  copyText,
} from "./lib";

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
/* prompt library                                                      */
/* ------------------------------------------------------------------ */

function PromptLibrary() {
  const [key, setKey] = useState(PROMPT_LIB[0].key);
  const active = PROMPT_LIB.find((p) => p.key === key) ?? PROMPT_LIB[0];

  return (
    <div className="corner-frame overflow-hidden border border-line bg-ink2">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-ink px-4 py-2.5">
        <span className="font-mono text-[10.5px] tracking-[0.2em] text-dim">
          PROMPT LIBRARY — COPY, PASTE, FILL THE &lt;XX&gt; SLOTS
        </span>
        <span className="font-mono text-[9.5px] tracking-[0.14em] text-faint">
          6 PROMPTS COVER THE WHOLE PROJECT
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5 border-b border-line bg-ink px-3 py-2.5">
        {PROMPT_LIB.map((p) => (
          <button
            key={p.key}
            onClick={() => setKey(p.key)}
            aria-pressed={p.key === key}
            className={`border px-3 py-1.5 font-mono text-[10px] tracking-[0.18em] transition-all duration-200 ${
              p.key === key
                ? "border-teal/70 bg-teal/10 text-teal"
                : "border-line2 text-dim hover:border-teal/50 hover:text-teal2"
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div key={active.key} className="p-4 sm:p-5">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <span className="font-mono text-[10.5px] tracking-[0.14em] text-amber">
            USE WHEN: {active.when.toUpperCase()}
          </span>
          <CopyBtn text={active.body} label="COPY PROMPT" />
        </div>
        <pre className="max-h-[380px] overflow-auto whitespace-pre-wrap border border-line bg-ink p-4 font-mono text-[11.5px] leading-[1.75] text-dim">
          {active.body}
        </pre>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* section — the operator's manual                                     */
/* ------------------------------------------------------------------ */

export function Operator() {
  return (
    <section id="operator" className="scroll-mt-28">
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:py-24">
        <SectionHead
          no="04"
          kicker="THE OPERATOR'S MANUAL"
          title="Driving a coding AI through this plan"
          aside="Model-agnostic on purpose: the repo is the interface, so any agent that can read files and run pytest will drive it — and you can swap models mid-project."
        />

        {/* works-with strip */}
        <Reveal>
          <div className="mb-12 flex flex-wrap items-center gap-2.5">
            <span className="mr-2 font-mono text-[10px] tracking-[0.24em] text-faint">
              TESTED PATTERN WITH
            </span>
            {WORKS_WITH.map((w) => (
              <span
                key={w}
                className="border border-line2 bg-ink2 px-3 py-1.5 font-mono text-[10.5px] tracking-[0.14em] text-dim transition-all duration-200 hover:-translate-y-0.5 hover:border-teal/60 hover:text-teal2"
              >
                {w}
              </span>
            ))}
            <span className="basis-full pt-1 text-[12.5px] text-faint sm:basis-auto sm:pl-2 sm:text-[12px]">
              — or any future agent. The plan doesn't care; the files do the remembering.
            </span>
          </div>
        </Reveal>

        {/* session script */}
        <Reveal>
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="font-mono text-[11px] tracking-[0.3em] text-faint">
                THE SESSION SCRIPT
              </div>
              <h3 className="font-display mt-1.5 text-xl font-black text-fog sm:text-2xl">
                Fifty-five minutes, five moves, then stop
              </h3>
            </div>
            <span className="border border-amber/50 bg-amber/5 px-3 py-1.5 font-mono text-[10px] tracking-[0.2em] text-amber">
              TOTAL ≤ 55 MIN · THEN A FRESH SESSION
            </span>
          </div>
        </Reveal>
        <Reveal delay={90} className="mt-6">
          <div className="grid border border-line bg-ink2 lg:grid-cols-5">
            {OPERATOR_STEPS.map((s, i) => (
              <div
                key={s.t}
                className={`group relative p-5 transition-colors duration-300 hover:bg-ink3/70 ${
                  i < OPERATOR_STEPS.length - 1
                    ? "border-b border-line lg:border-b-0 lg:border-r"
                    : ""
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] tracking-[0.22em] text-amber">
                    {s.t}
                  </span>
                  <span className="font-display text-lg font-black text-line2 transition-colors duration-300 group-hover:text-teal">
                    0{i + 1}
                  </span>
                </div>
                <div className="font-display mt-2 text-[15px] font-bold text-fog">
                  {s.title}
                </div>
                <p className="mt-1.5 text-[12px] leading-relaxed text-dim">{s.body}</p>
              </div>
            ))}
          </div>
        </Reveal>

        {/* prompt library */}
        <div className="mt-14 grid gap-8 xl:grid-cols-[1.4fr_1fr]">
          <Reveal>
            <PromptLibrary />
          </Reveal>

          <div className="space-y-6">
            {/* golden capture without matlab */}
            <Reveal delay={100}>
              <div className="border border-line bg-ink2/60 p-5">
                <div className="font-mono text-[10px] tracking-[0.24em] text-faint">
                  THE ONE THING THAT NEEDS MATLAB (ONCE)
                </div>
                <h4 className="font-display mt-1.5 text-lg font-bold text-fog">
                  Golden capture — and what to do if you have no MATLAB
                </h4>
                <p className="mt-2 text-[12.5px] leading-relaxed text-dim">
                  Reference outputs are captured <em>once per module</em>, then
                  trusted forever. If you have zero MATLAB access, in order:
                </p>
                <ul className="mt-3 space-y-2">
                  {[
                    ["GNU Octave", "often runs plain .m functions — v1.42 externalized NanoLocz's core into exactly that kind of library."],
                    ["A collaborator", "runs the capture script once with a license and sends the .npy exports."],
                    ["An upstream issue", "request reference outputs on the NanoLocz repo — test data ships with it."],
                  ].map(([t, b], i) => (
                    <li key={t} className="flex gap-3">
                      <span className="font-mono text-[11px] font-semibold text-teal">{i + 1}.</span>
                      <span className="text-[12.5px] leading-relaxed text-dim">
                        <span className="font-semibold text-fog">{t}</span> — {b}
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 border-t border-line pt-3 font-mono text-[10px] leading-relaxed tracking-[0.08em] text-faint">
                  THE PORT ITSELF NEVER NEEDS MATLAB. ONLY THE ORACLE FILES DO.
                </p>
              </div>
            </Reveal>

            {/* anti-patterns */}
            <Reveal delay={180}>
              <div className="border border-line bg-ink2/60">
                <div className="border-b border-line bg-ink px-5 py-3">
                  <span className="font-mono text-[10px] tracking-[0.24em] text-mag">
                    HOW AGENTS ACTUALLY FAIL — AND THE FIX
                  </span>
                </div>
                <div>
                  {ANTI_PATTERNS.map((a, i) => (
                    <div
                      key={i}
                      className="group border-b border-line/60 last:border-0"
                    >
                      <div className="flex gap-3 px-5 pb-2 pt-3.5">
                        <span className="mt-0.5 shrink-0 font-mono text-[11px] font-bold text-mag">✕</span>
                        <p className="text-[12px] leading-relaxed text-dim/90">{a.never}</p>
                      </div>
                      <div className="flex gap-3 border-t border-dashed border-line/50 bg-teal/[0.03] px-5 py-2.5 transition-colors duration-200 group-hover:bg-teal/[0.06]">
                        <span className="mt-0.5 shrink-0 font-mono text-[11px] font-bold text-teal">✓</span>
                        <p className="text-[12px] leading-relaxed text-dim">{a.instead}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Reveal>
          </div>
        </div>

        {/* handoff artifact */}
        <div className="mt-14 grid gap-8 lg:grid-cols-[1fr_1.2fr]">
          <Reveal>
            <div>
              <div className="font-mono text-[11px] tracking-[0.3em] text-faint">
                THE HANDOFF ARTIFACT
              </div>
              <h3 className="font-display mt-1.5 text-xl font-black text-fog sm:text-2xl">
                What a session leaves behind
              </h3>
              <p className="mt-3 max-w-md text-[13px] leading-relaxed text-dim">
                Four lines are enough. The next session — a stranger with amnesia —
                reads <span className="font-mono text-[12px] text-teal2">SESSIONS/</span>{" "}
                and knows exactly where the frontier is: what shipped, what's stuck,
                and the very next command to type.
              </p>
              <p className="mt-3 max-w-md text-[12.5px] leading-relaxed text-faint">
                Over weeks, this folder becomes a searchable project diary — and the
                strongest defence against an agent confidently re-solving a problem
                you already paid to solve.
              </p>
            </div>
          </Reveal>
          <Reveal delay={110}>
            <div className="overflow-hidden border border-line bg-ink">
              <div className="flex items-center justify-between border-b border-line bg-ink2 px-4 py-2.5">
                <span className="font-mono text-[11px] tracking-[0.2em] text-teal">
                  SESSIONS/&lt;today&gt;.md
                </span>
                <CopyBtn text={HANDOFF_MD} label="COPY TEMPLATE" />
              </div>
              <pre className="overflow-x-auto whitespace-pre-wrap p-5 font-mono text-[12px] leading-[1.85] text-dim">
                {HANDOFF_MD}
              </pre>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
