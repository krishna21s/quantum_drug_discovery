import { motion } from "framer-motion";
import { Target } from "lucide-react";
import { useMemo } from "react";
import { multiObjectiveScore, type MultiObjectiveResult } from "@/lib/admetEngine";
import { cn } from "@/lib/utils";

interface MultiObjectiveScorePanelProps {
    result?: MultiObjectiveResult;
}

function verdictColor(v: string) {
    if (v === "Excellent") return "text-success bg-success/10 ring-success/30";
    if (v === "Good") return "text-quantum bg-quantum/10 ring-quantum/30";
    if (v === "Moderate") return "text-warning bg-warning/10 ring-warning/30";
    return "text-destructive bg-destructive/10 ring-destructive/30";
}

const axes: { key: keyof Pick<MultiObjectiveResult, "binding" | "admet" | "quantum" | "stability" | "freeEnergy">; label: string; weight: string }[] = [
    { key: "binding", label: "Binding Affinity", weight: "25%" },
    { key: "admet", label: "ADMET Score", weight: "20%" },
    { key: "quantum", label: "Quantum Energy", weight: "15%" },
    { key: "stability", label: "MD Stability", weight: "20%" },
    { key: "freeEnergy", label: "Free Energy", weight: "20%" },
];

export default function MultiObjectiveScorePanel({ result }: MultiObjectiveScorePanelProps) {
    const data = useMemo(() => result ?? multiObjectiveScore(-8.2, 0.82, -75.3, 0.65, -9.5), [result]);

    // Radar chart
    const size = 200;
    const cx = size / 2;
    const cy = size / 2;
    const R = 70;
    const N = axes.length;

    const angleOf = (i: number) => (Math.PI * 2 * i) / N - Math.PI / 2;

    const gridPolygon = (r: number) =>
        axes.map((_, i) => {
            const a = angleOf(i);
            return `${cx + Math.cos(a) * r},${cy + Math.sin(a) * r}`;
        }).join(" ");

    const dataPolygon = axes.map((ax, i) => {
        const a = angleOf(i);
        const v = data[ax.key] * R;
        return `${cx + Math.cos(a) * v},${cy + Math.sin(a) * v}`;
    }).join(" ");

    return (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="liquid-glass rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-quantum" />
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Multi-Objective Score</h3>
                </div>
                <span className={cn("px-2 py-0.5 rounded-lg text-xs font-semibold ring-1", verdictColor(data.verdict))}>
                    {data.verdict}
                </span>
            </div>

            {/* Composite score */}
            <div className="text-center mb-4">
                <motion.p initial={{ scale: 0.8 }} animate={{ scale: 1 }} className="text-4xl font-bold font-mono text-quantum">
                    {(data.composite * 100).toFixed(0)}<span className="text-lg text-muted-foreground">%</span>
                </motion.p>
                <p className="text-xs text-muted-foreground">Composite Drug Suitability</p>
            </div>

            {/* Radar chart */}
            <div className="flex justify-center mb-4">
                <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                    <defs>
                        <filter id="mo-glow"><feGaussianBlur stdDeviation="3" result="b" /><feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
                        <linearGradient id="mo-fill" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="hsl(187,79%,54%)" stopOpacity="0.3" />
                            <stop offset="100%" stopColor="hsl(217,91%,60%)" stopOpacity="0.1" />
                        </linearGradient>
                    </defs>

                    {/* Grid rings */}
                    {[0.25, 0.5, 0.75, 1].map((s) => (
                        <polygon key={s} points={gridPolygon(R * s)} fill="none" stroke="hsl(215,20%,30%)" strokeWidth="0.5" strokeDasharray={s < 1 ? "2,2" : "0"} />
                    ))}

                    {/* Axis lines + labels */}
                    {axes.map((ax, i) => {
                        const a = angleOf(i);
                        const lx = cx + Math.cos(a) * (R + 18);
                        const ly = cy + Math.sin(a) * (R + 18);
                        return (
                            <g key={ax.key}>
                                <line x1={cx} y1={cy} x2={cx + Math.cos(a) * R} y2={cy + Math.sin(a) * R} stroke="hsl(215,20%,30%)" strokeWidth="0.5" />
                                <text x={lx} y={ly + 3} textAnchor="middle" fontSize="7" fill="hsl(215,20%,55%)">{ax.label.split(" ")[0]}</text>
                            </g>
                        );
                    })}

                    {/* Data polygon */}
                    <motion.polygon
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.8 }}
                        points={dataPolygon}
                        fill="url(#mo-fill)"
                        stroke="hsl(187,79%,54%)"
                        strokeWidth="1.5"
                        filter="url(#mo-glow)"
                    />

                    {/* Data dots */}
                    {axes.map((ax, i) => {
                        const a = angleOf(i);
                        const v = data[ax.key] * R;
                        return (
                            <motion.circle
                                key={ax.key}
                                initial={{ r: 0 }}
                                animate={{ r: 3 }}
                                transition={{ delay: 0.3 + i * 0.05 }}
                                cx={cx + Math.cos(a) * v}
                                cy={cy + Math.sin(a) * v}
                                fill="hsl(187,79%,54%)"
                            />
                        );
                    })}
                </svg>
            </div>

            {/* Axis breakdown */}
            <div className="space-y-1.5">
                {axes.map((ax, i) => (
                    <motion.div key={ax.key} initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 + i * 0.04 }} className="flex items-center justify-between">
                        <div className="flex items-center gap-2 text-xs">
                            <span className="text-muted-foreground">{ax.label}</span>
                            <span className="text-muted-foreground/50">({ax.weight})</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className="w-20 h-1.5 rounded-full bg-muted/20 overflow-hidden ring-1 ring-white/5">
                                <motion.div initial={{ width: 0 }} animate={{ width: `${data[ax.key] * 100}%` }} transition={{ duration: 0.8, delay: i * 0.04 }} className="h-full rounded-full bg-gradient-to-r from-quantum to-primary" />
                            </div>
                            <span className="font-mono text-xs font-semibold w-8 text-right">{(data[ax.key] * 100).toFixed(0)}%</span>
                        </div>
                    </motion.div>
                ))}
            </div>
        </motion.div>
    );
}
