import AppLayout from "@/components/AppLayout";
import ProteinPrepPanel from "@/components/ProteinPrepPanel";
import MDSimulationPanel from "@/components/MDSimulationPanel";
import TrajectoryPlayer from "@/components/TrajectoryPlayer";
import FreeEnergyPanel from "@/components/FreeEnergyPanel";
import { motion } from "framer-motion";
import { Beaker } from "lucide-react";

const container = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.12 } },
};
const item = {
    hidden: { opacity: 0, y: 16 },
    show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function Simulation() {
    return (
        <AppLayout>
            <motion.div variants={container} initial="hidden" animate="show" className="p-8 space-y-6">
                <motion.div variants={item}>
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Beaker className="h-6 w-6 text-quantum" />
                        Simulation Studio
                    </h1>
                    <p className="text-muted-foreground mt-1">Protein preparation, molecular dynamics, and free energy estimation</p>
                </motion.div>

                {/* Top row: Protein Prep + MD Simulation */}
                <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                    <div>
                        <ProteinPrepPanel />
                    </div>
                    <div className="lg:col-span-2">
                        <MDSimulationPanel />
                    </div>
                </motion.div>

                {/* Bottom row: Trajectory + Free Energy */}
                <motion.div variants={item} className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                    <div className="lg:col-span-2">
                        <TrajectoryPlayer />
                    </div>
                    <div>
                        <FreeEnergyPanel />
                    </div>
                </motion.div>
            </motion.div>
        </AppLayout>
    );
}
