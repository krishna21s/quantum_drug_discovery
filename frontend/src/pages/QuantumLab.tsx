import AppLayout from "@/components/AppLayout";
import QuantumCircuitDiagram from "@/components/QuantumCircuitDiagram";
import QuantumOutputPanel from "@/components/QuantumOutputPanel";
import BindingSimulation from "@/components/BindingSimulation";
import MoleculeViewer from "@/components/MoleculeViewer";
import ProteinTargetMap from "@/components/ProteinTargetMap";
import DrugInfoPanel from "@/components/DrugInfoPanel";
import DiseasePanel from "@/components/DiseasePanel";
import MoleculeStatsPanel from "@/components/MoleculeStatsPanel";
import PhysicoChemicalRadar from "@/components/PhysicoChemicalRadar";
import ADMETPanel from "@/components/ADMETPanel";
import QuantumChemPanel from "@/components/QuantumChemPanel";
import { computeADMET, combinedScore, DEMO_MOLECULES } from "@/lib/admetEngine";
import { motion } from "framer-motion";
import { useMemo } from "react";

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 },
  },
};

const item = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function QuantumLab() {
  const admetData = useMemo(() => computeADMET(DEMO_MOLECULES["Cetuximab"]), []);
  const cs = useMemo(() => combinedScore(-9.4, -82.1, admetData.scores.overall), [admetData]);

  return (
    <AppLayout>
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="p-8 space-y-6"
      >
        <motion.div variants={item}>
          <h1 className="text-2xl font-bold">Quantum Lab</h1>
          <p className="text-muted-foreground mt-1">VQE, VQC, and binding simulation workspace</p>
        </motion.div>

        {/* Drug Info Banner */}
        <motion.div variants={item}>
          <DrugInfoPanel />
        </motion.div>

        {/* Main grid: Protein Map + Right panels */}
        <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Center: Protein Target Map */}
          <div className="lg:col-span-2">
            <ProteinTargetMap />
          </div>

          {/* Right: Disease + Stats */}
          <div className="space-y-6">
            <DiseasePanel />
          </div>
        </motion.div>

        {/* Second row: Molecule Stats + Radar + Molecule Viewer */}
        <motion.div variants={item} className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <MoleculeStatsPanel />
          <PhysicoChemicalRadar />
          <MoleculeViewer />
        </motion.div>

        {/* Third row: Quantum section */}
        <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-6">
            <QuantumCircuitDiagram />
            <BindingSimulation />
          </div>
          <div className="space-y-6">
            <QuantumOutputPanel />
            <ADMETPanel data={admetData} bindingAffinity={-9.4} quantumEnergy={-82.1} combinedScore={cs} />
            <QuantumChemPanel />
          </div>
        </motion.div>
      </motion.div>
    </AppLayout>
  );
}
