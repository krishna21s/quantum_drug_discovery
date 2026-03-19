import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const circuitGates = [
  { type: "H", qubit: 0, col: 0 },
  { type: "H", qubit: 1, col: 0 },
  { type: "Ry", qubit: 0, col: 1 },
  { type: "Ry", qubit: 1, col: 1 },
  { type: "CNOT", qubit: 0, col: 2, target: 1 },
  { type: "Rz", qubit: 0, col: 3 },
  { type: "Ry", qubit: 1, col: 3 },
  { type: "CNOT", qubit: 1, col: 4, target: 0 },
  { type: "M", qubit: 0, col: 5 },
  { type: "M", qubit: 1, col: 5 },
];

export default function QuantumCircuitDiagram() {
  const [pulsePos, setPulsePos] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setPulsePos((p) => (p + 1) % 7);
    }, 800);
    return () => clearInterval(interval);
  }, []);

  const qubits = 2;
  const cols = 6;
  const gateW = 44;
  const gateH = 32;
  const rowH = 64;
  const startX = 65;
  const startY = 35;

  return (
    <div className="glass-card rounded-2xl p-6 relative overflow-hidden">
      {/* Top glow line */}
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

      <h3 className="mb-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">VQC Circuit Diagram</h3>
      <svg width="100%" viewBox={`0 0 ${startX + cols * (gateW + 22) + 40} ${startY + qubits * rowH + 20}`} className="max-w-lg">
        {/* Glow filter */}
        <defs>
          <filter id="glow-gate">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-line">
            <feGaussianBlur stdDeviation="2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <linearGradient id="qubit-line-grad" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stopColor="hsl(var(--border))" stopOpacity="0.3" />
            <stop offset="50%" stopColor="hsl(var(--border))" stopOpacity="1" />
            <stop offset="100%" stopColor="hsl(var(--border))" stopOpacity="0.3" />
          </linearGradient>
        </defs>

        {/* Qubit lines */}
        {Array.from({ length: qubits }).map((_, i) => (
          <g key={i}>
            <text x={12} y={startY + i * rowH + 5} fill="hsl(var(--muted-foreground))" fontSize="12" fontFamily="JetBrains Mono" fontWeight="500">
              q{i}
            </text>
            <line
              x1={startX - 10}
              y1={startY + i * rowH}
              x2={startX + cols * (gateW + 22)}
              y2={startY + i * rowH}
              stroke="url(#qubit-line-grad)"
              strokeWidth={1.5}
            />
            {/* Traveling pulse dot */}
            <motion.circle
              cx={startX + pulsePos * (gateW + 22)}
              cy={startY + i * rowH}
              r={3}
              fill="hsl(var(--quantum))"
              filter="url(#glow-line)"
              initial={{ opacity: 0 }}
              animate={{ opacity: [0, 1, 0] }}
              transition={{ duration: 0.8, repeat: Infinity, repeatDelay: 0.5 }}
            />
          </g>
        ))}

        {/* Gates */}
        {circuitGates.map((gate, idx) => {
          const x = startX + gate.col * (gateW + 22);
          const y = startY + gate.qubit * rowH;

          if (gate.type === "CNOT") {
            const targetY = startY + (gate.target ?? 0) * rowH;
            return (
              <motion.g key={idx} initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: idx * 0.1, type: "spring", stiffness: 200 }}>
                <line x1={x + gateW / 2} y1={y} x2={x + gateW / 2} y2={targetY} stroke="hsl(var(--quantum))" strokeWidth={2} filter="url(#glow-line)" />
                <circle cx={x + gateW / 2} cy={y} r={5} fill="hsl(var(--quantum))" filter="url(#glow-gate)" />
                <circle cx={x + gateW / 2} cy={targetY} r={10} fill="none" stroke="hsl(var(--quantum))" strokeWidth={2} filter="url(#glow-gate)" />
                <line x1={x + gateW / 2 - 7} y1={targetY} x2={x + gateW / 2 + 7} y2={targetY} stroke="hsl(var(--quantum))" strokeWidth={2} />
                <line x1={x + gateW / 2} y1={targetY - 7} x2={x + gateW / 2} y2={targetY + 7} stroke="hsl(var(--quantum))" strokeWidth={2} />
              </motion.g>
            );
          }

          const isM = gate.type === "M";
          return (
            <motion.g key={idx} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: idx * 0.1, type: "spring", stiffness: 200 }}>
              {/* Gate glow */}
              {!isM && (
                <rect
                  x={x - 2}
                  y={y - gateH / 2 - 2}
                  width={gateW + 4}
                  height={gateH + 4}
                  rx={8}
                  fill="none"
                  stroke="hsl(var(--primary) / 0.2)"
                  strokeWidth={1}
                  filter="url(#glow-gate)"
                />
              )}
              <rect
                x={x}
                y={y - gateH / 2}
                width={gateW}
                height={gateH}
                rx={6}
                fill={isM ? "hsl(var(--muted) / 0.8)" : "hsl(var(--primary) / 0.12)"}
                stroke={isM ? "hsl(var(--muted-foreground) / 0.5)" : "hsl(var(--primary) / 0.6)"}
                strokeWidth={1}
              />
              <text
                x={x + gateW / 2}
                y={y + 4}
                textAnchor="middle"
                fill={isM ? "hsl(var(--muted-foreground))" : "hsl(var(--primary))"}
                fontSize="11"
                fontFamily="JetBrains Mono"
                fontWeight="600"
              >
                {gate.type}
              </text>
            </motion.g>
          );
        })}
      </svg>
    </div>
  );
}
