import AppLayout from "@/components/AppLayout";
import InteractionAnalysisPanel from "@/components/InteractionAnalysisPanel";
import ResidueContactMap from "@/components/ResidueContactMap";
import QuantumChemPanel from "@/components/QuantumChemPanel";
import MultiObjectiveScorePanel from "@/components/MultiObjectiveScorePanel";
import { motion } from "framer-motion";
import { PieChart } from "lucide-react";

const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.12 } },
};
const item = {
    hidden: { opacity: 0, y: 16 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function Analysis() {
    return (
        <AppLayout>
            <motion.div variants={container} initial="hidden" animate="show" className="p-8 space-y-6">
                <motion.div variants={item}>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <PieChart className="h-6 w-6 text-quantum" />
                        Detailed Analysis
                    </h1>
                    <p className="text-muted-foreground mt-1">Interaction profiling, quantum chemistry, and multi-objective evaluation</p>
                </motion.div>

                {/* Top row: Interactions + Contact Map */}
                <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                    <div className="lg:col-span-2">
                        <InteractionAnalysisPanel />
                    </div>
                    <div>
                        <ResidueContactMap />
                    </div>
                </motion.div>

                {/* Bottom row: Quantum Chem + Multi-Objective */}
                <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                    <QuantumChemPanel />
                    <MultiObjectiveScorePanel />
                </motion.div>
            </motion.div>
        </AppLayout>
    );
}
