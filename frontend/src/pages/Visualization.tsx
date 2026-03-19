import AppLayout from "@/components/AppLayout";
import Protein3DViewer from "@/components/Protein3DViewer";
import Ligand3DViewer from "@/components/Ligand3DViewer";
import BindingCinematic from "@/components/BindingCinematic";
import BodyPartViewer from "@/components/BodyPartViewer";
import { motion } from "framer-motion";
import { Microscope, ChevronRight } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const pdbOptions = [
    { id: "1M17", name: "EGFR Kinase Domain" },
    { id: "6LU7", name: "SARS-CoV-2 Main Protease" },
    { id: "1HHP", name: "HIV-1 Protease" },
    { id: "3ERT", name: "Estrogen Receptor" },
];

const container = {
    hidden: { opacity: 0 },
    show:   { opacity: 1, transition: { staggerChildren: 0.1 } },
};
const item = {
    hidden: { opacity: 0, y: 20 },
    show:   { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] } },
};

export default function Visualization() {
    const [pdbId, setPdbId] = useState("1M17");
    const selected = pdbOptions.find((p) => p.id === pdbId);

    return (
        <AppLayout>
            <motion.div variants={container} initial="hidden" animate="show" className="p-6 space-y-6">

                {/* Header */}
                <motion.div variants={item} className="flex items-center justify-between">
                    <div>
                        <div className="flex items-center gap-2 mb-1">
                            <span className="stat-pill bg-quantum/15 text-quantum text-[11px] font-semibold">
                                <Microscope className="h-3 w-3" />
                                Molecular Visualization
                            </span>
                        </div>
                        <h1 className="text-3xl font-bold">3D Structure Viewer</h1>
                        <p className="text-muted-foreground text-sm mt-1">
                            Interactive proteins, ligands &amp; organ-level drug action
                        </p>
                    </div>

                    {/* PDB selector */}
                    <div className="flex rounded-2xl glass-surface p-1 gap-1">
                        {pdbOptions.map((opt) => (
                            <button
                                key={opt.id}
                                onClick={() => setPdbId(opt.id)}
                                className={cn(
                                    "relative px-3.5 py-2 text-xs font-semibold rounded-xl transition-all duration-300",
                                    pdbId === opt.id
                                        ? "text-white"
                                        : "text-muted-foreground hover:text-foreground"
                                )}
                            >
                                {pdbId === opt.id && (
                                    <motion.div
                                        layoutId="pdb-select"
                                        className="absolute inset-0 rounded-xl"
                                        style={{
                                            background: "linear-gradient(135deg, hsl(187 85% 40%), hsl(207 100% 50%))",
                                            boxShadow: "0 4px 16px hsl(207 100% 50% / 0.35)",
                                        }}
                                        transition={{ type: "spring", stiffness: 400, damping: 30 }}
                                    />
                                )}
                                <span className="relative z-10 font-mono">{opt.id}</span>
                            </button>
                        ))}
                    </div>
                </motion.div>

                {/* Two-column: 3D organ (left) + protein (right) */}
                <motion.div variants={item} className="grid grid-cols-1 lg:grid-cols-5 gap-5">
                    {/* 3D Body/Organ Viewer — larger, Dribbble-style */}
                    <div className="lg:col-span-2">
                        <BodyPartViewer className="h-full" />
                    </div>

                    {/* Protein + Ligand column */}
                    <div className="lg:col-span-3 space-y-5">
                        <Protein3DViewer pdbId={pdbId} />

                        {/* Structure info card */}
                        <div className="glass-card rounded-3xl p-5 relative overflow-hidden">
                            <div className="absolute top-0 left-5 right-5 h-[2px] rounded-full"
                                style={{ background: "linear-gradient(90deg, transparent, hsl(207 100% 50%), transparent)" }}
                            />
                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">
                                Structure Info · {selected?.name}
                            </h3>
                            <div className="grid grid-cols-2 gap-2">
                                {[
                                    ["PDB ID",     pdbId],
                                    ["Resolution", "2.6 Å"],
                                    ["Chains",     "A, B"],
                                    ["Method",     "X-ray"],
                                    ["Source",     "RCSB PDB"],
                                    ["Organism",   "H. sapiens"],
                                ].map(([k, v]) => (
                                    <div key={k} className="glass-surface rounded-2xl px-3 py-2 flex items-center justify-between">
                                        <span className="text-xs text-muted-foreground">{k}</span>
                                        <span className="text-xs font-mono font-semibold text-primary">{v}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </motion.div>

                {/* Ligand viewer */}
                <motion.div variants={item}>
                    <Ligand3DViewer name="Cetuximab" />
                </motion.div>

                {/* Cinematic binding */}
                <motion.div variants={item}>
                    <BindingCinematic />
                </motion.div>

            </motion.div>
        </AppLayout>
    );
}

