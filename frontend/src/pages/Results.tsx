import AppLayout from "@/components/AppLayout";
import ExperimentCard from "@/components/ExperimentCard";
import QuantumOutputPanel from "@/components/QuantumOutputPanel";
import ADMETPanel from "@/components/ADMETPanel";
import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import { useMemo } from "react";
import { computeADMET, combinedScore, DEMO_MOLECULES } from "@/lib/admetEngine";

const experiments = [
  { id: "1", name: "SARS-CoV-2 Mpro Inhibitor", protein: "6LU7", status: "completed" as const, score: 0.94, date: "Feb 18, 2026" },
  { id: "2", name: "EGFR Kinase Blocker", protein: "1M17", status: "running" as const, score: undefined, date: "Feb 19, 2026" },
  { id: "3", name: "HIV-1 Protease Drug", protein: "1HHP", status: "completed" as const, score: 0.82, date: "Feb 17, 2026" },
  { id: "4", name: "Beta-Lactamase Inhibitor", protein: "1ZG4", status: "completed" as const, score: 0.78, date: "Feb 17, 2026" },
  { id: "5", name: "Estrogen Receptor Modulator", protein: "3ERT", status: "queued" as const, score: undefined, date: "Feb 20, 2026" },
  { id: "6", name: "CDK4/6 Inhibitor Screening", protein: "2EUF", status: "completed" as const, score: 0.89, date: "Feb 16, 2026" },
];

export default function Results() {
  const admetData = useMemo(() => computeADMET(DEMO_MOLECULES["Aspirin"]), []);
  const cs = useMemo(() => combinedScore(-8.2, -75.3, admetData.scores.overall), [admetData]);

  return (
    <AppLayout>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-8 space-y-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Activity className="h-6 w-6 text-quantum" />
            Experiment Results
          </h1>
          <p className="text-muted-foreground mt-1">{experiments.length} experiments · {experiments.filter((e) => e.status === "completed").length} completed</p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Experiment list */}
          <div className="lg:col-span-2 grid grid-cols-1 gap-3 sm:grid-cols-2">
            {experiments.map((exp, i) => (
              <ExperimentCard key={exp.id} {...exp} index={i} />
            ))}
          </div>

          {/* Right sidebar: Quantum output + ADMET */}
          <div className="space-y-6">
            <QuantumOutputPanel />
            <ADMETPanel data={admetData} bindingAffinity={-8.2} quantumEnergy={-75.3} combinedScore={cs} />
          </div>
        </div>
      </motion.div>
    </AppLayout>
  );
}
