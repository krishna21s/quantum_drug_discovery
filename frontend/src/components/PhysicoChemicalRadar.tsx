import { motion } from "framer-motion";
import { useState } from "react";

const properties = [
  { label: "NSATU", value: 0.48, angle: -90 },
  { label: "Lipo", value: 2.21, angle: -30 },
  { label: "Size", value: 206, angle: 30, unit: "g/mol" },
  { label: "Polar", value: 2.7, angle: 90 },
  { label: "INSOLU", value: -4.12, angle: 150 },
  { label: "FLEX", value: 0.4, angle: 210 },
];

export default function PhysicoChemicalRadar() {
  const [viewMode, setViewMode] = useState<"3D" | "2D">("2D");

  const size = 200;
  const cx = size / 2;
  const cy = size / 2;
  const r = 65;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-5 relative overflow-hidden"
    >
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Physicochemical Properties</h3>
        <div className="flex rounded-xl glass-surface p-0.5">
          {(["3D", "2D"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setViewMode(m)}
              className={`px-2.5 py-1 text-xs rounded-lg transition-all duration-300 ${viewMode === m ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                }`}
            >
              {m}
            </button>
          ))}
          <button className="px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground rounded-lg transition-colors">Edit mode</button>
        </div>
      </div>

      <div className="flex justify-center">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {/* Glow filter for data polygon */}
          <defs>
            <filter id="radar-glow">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Grid rings */}
          {[0.33, 0.66, 1].map((scale) => (
            <polygon
              key={scale}
              points={properties
                .map((p) => {
                  const a = (p.angle * Math.PI) / 180;
                  return `${cx + r * scale * Math.cos(a)},${cy + r * scale * Math.sin(a)}`;
                })
                .join(" ")}
              fill="none"
              stroke="hsl(217, 33%, 18%)"
              strokeWidth={0.5}
              opacity={0.6}
            />
          ))}

          {/* Axes */}
          {properties.map((p) => {
            const a = (p.angle * Math.PI) / 180;
            return (
              <line
                key={p.label}
                x1={cx}
                y1={cy}
                x2={cx + r * Math.cos(a)}
                y2={cy + r * Math.sin(a)}
                stroke="hsl(217, 33%, 18%)"
                strokeWidth={0.5}
                opacity={0.6}
              />
            );
          })}

          {/* Data polygon with glow */}
          <motion.polygon
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            style={{ transformOrigin: `${cx}px ${cy}px` }}
            points={properties
              .map((p) => {
                const a = (p.angle * Math.PI) / 180;
                const v = Math.min(Math.abs(p.value) / 5, 1);
                return `${cx + r * v * Math.cos(a)},${cy + r * v * Math.sin(a)}`;
              })
              .join(" ")}
            fill="hsl(187, 79%, 54%, 0.12)"
            stroke="hsl(187, 79%, 54%)"
            strokeWidth={1.5}
            filter="url(#radar-glow)"
          />

          {/* Data point dots */}
          {properties.map((p) => {
            const a = (p.angle * Math.PI) / 180;
            const v = Math.min(Math.abs(p.value) / 5, 1);
            const px = cx + r * v * Math.cos(a);
            const py = cy + r * v * Math.sin(a);
            return (
              <g key={p.label + "-dot"}>
                <circle cx={px} cy={py} r={4} fill="hsl(187, 79%, 54%)" opacity={0.3} />
                <circle cx={px} cy={py} r={2.5} fill="hsl(187, 79%, 54%)" />
              </g>
            );
          })}

          {/* Labels */}
          {properties.map((p) => {
            const a = (p.angle * Math.PI) / 180;
            const lx = cx + (r + 22) * Math.cos(a);
            const ly = cy + (r + 22) * Math.sin(a);
            return (
              <text
                key={p.label}
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="hsl(215, 20%, 55%)"
                fontSize="7"
                fontFamily="JetBrains Mono"
              >
                {p.label}: {p.value}
              </text>
            );
          })}
        </svg>
      </div>
    </motion.div>
  );
}
