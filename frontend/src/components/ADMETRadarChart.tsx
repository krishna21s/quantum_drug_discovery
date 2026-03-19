import { motion } from "framer-motion";
import type { ADMETScores } from "@/lib/admetEngine";

interface ADMETRadarChartProps {
    scores: ADMETScores;
}

const axes = [
    { key: "absorption" as const, label: "Absorption", angle: -90 },
    { key: "distribution" as const, label: "Distribution", angle: -18 },
    { key: "metabolism" as const, label: "Metabolism", angle: 54 },
    { key: "excretion" as const, label: "Excretion", angle: 126 },
    { key: "toxicity" as const, label: "Toxicity", angle: 198 },
];

function scoreColor(score: number): string {
    if (score > 0.7) return "hsl(142, 71%, 45%)";
    if (score > 0.45) return "hsl(38, 92%, 50%)";
    return "hsl(0, 84%, 60%)";
}

export default function ADMETRadarChart({ scores }: ADMETRadarChartProps) {
    const size = 240;
    const cx = size / 2;
    const cy = size / 2;
    const r = 80;

    const dataPoints = axes.map((axis) => {
        const a = (axis.angle * Math.PI) / 180;
        const v = scores[axis.key];
        return {
            x: cx + r * v * Math.cos(a),
            y: cy + r * v * Math.sin(a),
            lx: cx + (r + 28) * Math.cos(a),
            ly: cy + (r + 28) * Math.sin(a),
            score: v,
            label: axis.label,
        };
    });

    const polygonPoints = dataPoints.map((p) => `${p.x},${p.y}`).join(" ");

    return (
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="max-w-full">
            <defs>
                <filter id="admet-glow">
                    <feGaussianBlur stdDeviation="5" result="blur" />
                    <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
                <radialGradient id="admet-fill" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="hsl(187, 79%, 54%)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="hsl(217, 91%, 60%)" stopOpacity={0.08} />
                </radialGradient>
            </defs>

            {/* Grid rings */}
            {[0.33, 0.66, 1].map((scale) => (
                <polygon
                    key={scale}
                    points={axes
                        .map((axis) => {
                            const a = (axis.angle * Math.PI) / 180;
                            return `${cx + r * scale * Math.cos(a)},${cy + r * scale * Math.sin(a)}`;
                        })
                        .join(" ")}
                    fill="none"
                    stroke="hsl(217, 33%, 18%)"
                    strokeWidth={0.5}
                    opacity={0.5}
                />
            ))}

            {/* Axes */}
            {axes.map((axis) => {
                const a = (axis.angle * Math.PI) / 180;
                return (
                    <line
                        key={axis.key}
                        x1={cx}
                        y1={cy}
                        x2={cx + r * Math.cos(a)}
                        y2={cy + r * Math.sin(a)}
                        stroke="hsl(217, 33%, 18%)"
                        strokeWidth={0.5}
                        opacity={0.5}
                    />
                );
            })}

            {/* Data polygon */}
            <motion.polygon
                initial={{ opacity: 0, scale: 0.3 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2, duration: 0.8, ease: "easeOut" }}
                style={{ transformOrigin: `${cx}px ${cy}px` }}
                points={polygonPoints}
                fill="url(#admet-fill)"
                stroke="hsl(187, 79%, 54%)"
                strokeWidth={1.5}
                filter="url(#admet-glow)"
            />

            {/* Data dots + labels */}
            {dataPoints.map((p, i) => (
                <g key={i}>
                    {/* Dot glow */}
                    <circle cx={p.x} cy={p.y} r={5} fill={scoreColor(p.score)} opacity={0.25} />
                    <circle cx={p.x} cy={p.y} r={3} fill={scoreColor(p.score)} />

                    {/* Label */}
                    <text
                        x={p.lx}
                        y={p.ly - 5}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill="hsl(215, 20%, 55%)"
                        fontSize="8"
                        fontFamily="Inter"
                        fontWeight="500"
                    >
                        {p.label}
                    </text>
                    <text
                        x={p.lx}
                        y={p.ly + 7}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill={scoreColor(p.score)}
                        fontSize="9"
                        fontFamily="JetBrains Mono"
                        fontWeight="600"
                    >
                        {(p.score * 100).toFixed(0)}%
                    </text>
                </g>
            ))}

            {/* Center score */}
            <text x={cx} y={cy - 6} textAnchor="middle" dominantBaseline="middle" fill="hsl(187, 79%, 54%)" fontSize="18" fontFamily="JetBrains Mono" fontWeight="700">
                {(scores.overall * 100).toFixed(0)}
            </text>
            <text x={cx} y={cy + 10} textAnchor="middle" dominantBaseline="middle" fill="hsl(215, 20%, 55%)" fontSize="7" fontFamily="Inter">
                ADMET Score
            </text>
        </svg>
    );
}
