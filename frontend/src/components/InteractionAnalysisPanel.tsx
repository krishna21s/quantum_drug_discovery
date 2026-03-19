import { motion } from "framer-motion";
import { Link2, Droplets, Hexagon, Zap } from "lucide-react";
import { useMemo } from "react";
import { analyzeInteractions, type InteractionProfile } from "@/lib/interactionEngine";
import { cn } from "@/lib/utils";

function strengthColor(s: string) {
    if (s === "Strong") return "text-success bg-success/10 ring-success/30";
    if (s === "Moderate") return "text-warning bg-warning/10 ring-warning/30";
    return "text-destructive bg-destructive/10 ring-destructive/30";
}

const typeIcons: Record<string, typeof Link2> = {
    "H-bond": Link2,
    "Hydrophobic": Droplets,
    "π-π": Hexagon,
    "Salt bridge": Zap,
};

export default function InteractionAnalysisPanel() {
    const data = useMemo(analyzeInteractions, []);

    return (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="liquid-glass rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Link2 className="h-4 w-4 text-quantum" />
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Interaction Analysis</h3>
                </div>
                <div className="flex items-center gap-2">
                    <InteractionBadge label="H-bonds" count={data.hydrogenBonds.length} color="text-quantum" />
                    <InteractionBadge label="Hydrophobic" count={data.hydrophobicContacts.length} color="text-primary" />
                    <InteractionBadge label="π-stack" count={data.piInteractions.length} color="text-purple-400" />
                    <InteractionBadge label="Salt" count={data.saltBridges.length} color="text-warning" />
                </div>
            </div>

            {/* Hydrogen bonds table */}
            <div className="mb-4">
                <p className="text-xs font-semibold text-muted-foreground mb-2">Hydrogen Bonds</p>
                <div className="rounded-xl overflow-hidden ring-1 ring-white/5">
                    <table className="w-full text-xs">
                        <thead>
                            <tr className="glass-surface">
                                <th className="px-3 py-2 text-left text-muted-foreground font-medium">Donor</th>
                                <th className="px-3 py-2 text-left text-muted-foreground font-medium">Acceptor</th>
                                <th className="px-3 py-2 text-right text-muted-foreground font-medium">Dist (Å)</th>
                                <th className="px-3 py-2 text-right text-muted-foreground font-medium">Angle (°)</th>
                                <th className="px-3 py-2 text-center text-muted-foreground font-medium">Strength</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.hydrogenBonds.map((hb, i) => (
                                <motion.tr key={i} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.04 }} className="border-t border-white/3 hover:bg-quantum/5 transition-colors">
                                    <td className="px-3 py-2 font-mono">{hb.donor}</td>
                                    <td className="px-3 py-2 font-mono">{hb.acceptor}</td>
                                    <td className="px-3 py-2 text-right font-mono">{hb.distance.toFixed(2)}</td>
                                    <td className="px-3 py-2 text-right font-mono">{hb.angle}°</td>
                                    <td className="px-3 py-2 text-center">
                                        <span className={cn("px-1.5 py-0.5 rounded-md text-xs font-semibold ring-1", strengthColor(hb.strength))}>{hb.strength}</span>
                                    </td>
                                </motion.tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Hydrophobic contacts */}
            <div className="mb-4">
                <p className="text-xs font-semibold text-muted-foreground mb-2">Hydrophobic Contacts</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
                    {data.hydrophobicContacts.map((hc, i) => (
                        <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 + i * 0.03 }} className="flex items-center justify-between glass-surface rounded-lg px-3 py-1.5">
                            <div className="flex items-center gap-1.5">
                                <span className="font-mono text-xs text-primary">{hc.ligandAtom}</span>
                                <span className="text-muted-foreground">→</span>
                                <span className="font-mono text-xs">{hc.residue}</span>
                            </div>
                            <span className="text-xs text-muted-foreground">{hc.distance.toFixed(1)} Å · {hc.type}</span>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Pi interactions */}
            <div className="mb-4">
                <p className="text-xs font-semibold text-muted-foreground mb-2">π-Interactions</p>
                <div className="space-y-1.5">
                    {data.piInteractions.map((pi, i) => (
                        <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.04 }} className="flex items-center justify-between glass-surface rounded-lg px-3 py-1.5">
                            <div className="flex items-center gap-1.5">
                                <Hexagon className="h-3 w-3 text-purple-400" />
                                <span className="font-mono text-xs">{pi.ligandRing}</span>
                                <span className="text-muted-foreground">↔</span>
                                <span className="font-mono text-xs">{pi.residue}</span>
                            </div>
                            <span className="text-xs text-muted-foreground">{pi.type} · {pi.distance.toFixed(1)} Å</span>
                        </motion.div>
                    ))}
                </div>
            </div>

            {/* Interaction score */}
            <div className="glass-surface rounded-xl p-3 text-center">
                <p className="text-xs text-muted-foreground">Interaction Quality Score</p>
                <p className="text-xl font-bold font-mono text-quantum mt-0.5">{(data.interactionScore * 100).toFixed(0)}%</p>
                <p className="text-xs text-muted-foreground">{data.totalInteractions} total interactions identified</p>
            </div>
        </motion.div>
    );
}

function InteractionBadge({ label, count, color }: { label: string; count: number; color: string }) {
    return (
        <span className={cn("px-1.5 py-0.5 rounded-md text-xs font-mono", color)}>
            {count} {label}
        </span>
    );
}
