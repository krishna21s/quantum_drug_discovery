import { motion } from "framer-motion";
import { Grid3X3 } from "lucide-react";
import { useMemo } from "react";
import { analyzeInteractions, type ResidueContact } from "@/lib/interactionEngine";

function freqColor(f: number): string {
    if (f > 0.85) return "hsl(187, 79%, 54%)";    // quantum cyan
    if (f > 0.7) return "hsl(217, 91%, 60%)";     // primary blue
    if (f > 0.55) return "hsl(38, 92%, 50%)";     // warning amber
    return "hsl(215, 20%, 40%)";                    // low gray
}

export default function ResidueContactMap() {
    const data = useMemo(analyzeInteractions, []);
    const contacts = data.residueContacts;

    const cellSize = 40;
    const labelW = 60;
    const headerH = 20;
    const W = labelW + cellSize;
    const H = headerH + contacts.length * cellSize;

    return (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="liquid-glass rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

            <div className="flex items-center gap-2 mb-4">
                <Grid3X3 className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Residue Contact Map</h3>
            </div>

            <div className="overflow-x-auto">
                <svg viewBox={`0 0 ${W + 80} ${H + 30}`} className="w-full max-w-xs mx-auto">
                    {/* Header */}
                    <text x={labelW + cellSize / 2} y={14} textAnchor="middle" fontSize="8" fill="hsl(215,20%,55%)">Frequency</text>

                    {contacts.map((c, i) => {
                        const y = headerH + i * cellSize;
                        return (
                            <g key={c.residue}>
                                {/* Residue label */}
                                <text x={labelW - 4} y={y + cellSize / 2 + 3} textAnchor="end" fontSize="9" fontFamily="monospace" fill="hsl(215,20%,65%)">
                                    {c.residue}
                                </text>

                                {/* Heat cell */}
                                <motion.rect
                                    x={labelW}
                                    y={y + 2}
                                    width={cellSize - 4}
                                    height={cellSize - 4}
                                    rx={4}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: 0.1 + i * 0.03 }}
                                    fill={freqColor(c.contactFrequency)}
                                    fillOpacity={0.2 + c.contactFrequency * 0.6}
                                    stroke={freqColor(c.contactFrequency)}
                                    strokeWidth={0.5}
                                    strokeOpacity={0.4}
                                />

                                {/* Frequency text */}
                                <text x={labelW + cellSize / 2 - 2} y={y + cellSize / 2 + 3} textAnchor="middle" fontSize="9" fontFamily="monospace" fill={freqColor(c.contactFrequency)}>
                                    {(c.contactFrequency * 100).toFixed(0)}%
                                </text>

                                {/* Distance badge */}
                                <text x={labelW + cellSize + 6} y={y + cellSize / 2 + 3} fontSize="8" fontFamily="monospace" fill="hsl(215,20%,55%)">
                                    {c.avgDistance.toFixed(1)} Å
                                </text>

                                {/* Interaction type tags */}
                                {c.interactionTypes.map((t, j) => (
                                    <text key={j} x={labelW + cellSize + 42 + j * 50} y={y + cellSize / 2 + 3} fontSize="7" fill={t.includes("H-bond") ? "hsl(187,79%,54%)" : t.includes("π") ? "hsl(270,60%,65%)" : "hsl(217,91%,60%)"}>
                                        {t}
                                    </text>
                                ))}
                            </g>
                        );
                    })}

                    {/* Legend */}
                    <g transform={`translate(${labelW}, ${H + 10})`}>
                        <rect x={0} y={0} width={10} height={8} rx={2} fill="hsl(215,20%,40%)" fillOpacity={0.5} />
                        <text x={14} y={7} fontSize="7" fill="hsl(215,20%,55%)">Low</text>
                        <rect x={40} y={0} width={10} height={8} rx={2} fill="hsl(38,92%,50%)" fillOpacity={0.6} />
                        <text x={54} y={7} fontSize="7" fill="hsl(215,20%,55%)">Med</text>
                        <rect x={80} y={0} width={10} height={8} rx={2} fill="hsl(187,79%,54%)" fillOpacity={0.8} />
                        <text x={94} y={7} fontSize="7" fill="hsl(215,20%,55%)">High</text>
                    </g>
                </svg>
            </div>
        </motion.div>
    );
}
