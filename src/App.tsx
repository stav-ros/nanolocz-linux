import {
  Header,
  Opening,
  Verdict,
  StackMap,
  Pipeline,
  Bench,
  Terminal,
  Roadmap,
  Risks,
  Faq,
  Footer,
} from "./Sections";
import { AgentPlan, LafmPlus } from "./PlanSections";
import { SimBridge } from "./SimBridge";
import { Phase4Workbench } from "./Phase4Workbench";
import { Operator } from "./Operator";

export default function App() {
  return (
    <div className="relative min-h-screen bg-ink text-fog">
      {/* ambient layers */}
      <div className="pointer-events-none fixed inset-0 z-0">
        <div className="grid-ambient absolute inset-0" />
        <div className="absolute -left-40 -top-40 h-[560px] w-[560px] rounded-full bg-teal/[0.05] blur-[130px]" />
        <div className="absolute -right-40 top-1/3 h-[560px] w-[560px] rounded-full bg-amber/[0.045] blur-[130px]" />
        <div className="absolute bottom-0 left-1/4 h-[460px] w-[460px] rounded-full bg-mag/[0.04] blur-[130px]" />
      </div>
      <div className="noise-layer" />

      <Header />

      <main className="relative z-10">
        <Opening />
        <Verdict />
        <StackMap />
        <AgentPlan />
        <Operator />
        <LafmPlus />
        <Phase4Workbench />
        <SimBridge />
        <Pipeline />
        <Bench />
        <Terminal />
        <Roadmap />
        <Risks />
        <Faq />
      </main>

      <div className="relative z-10">
        <Footer />
      </div>
    </div>
  );
}
