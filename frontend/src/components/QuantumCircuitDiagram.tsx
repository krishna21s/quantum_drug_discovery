const circuitGates = [
  { type: "H", qubit: 0, col: 0 },
  { type: "H", qubit: 1, col: 0 },
  { type: "Ry", qubit: 0, col: 1 },
  { type: "Ry", qubit: 1, col: 1 },
  { type: "CNX", qubit: 0, col: 2, target: 1 },
  { type: "Rz", qubit: 0, col: 3 },
  { type: "Ry", qubit: 1, col: 3 },
  { type: "CNX", qubit: 1, col: 4, target: 0 },
  { type: "M", qubit: 0, col: 5 },
  { type: "M", qubit: 1, col: 5 },
];

export default function QuantumCircuitDiagram() {
  const qubits = 2;
  const cols = 6;
  const gateW = 44;
  const gateH = 32;
  const rowH = 64;
  const startX = 60;
  const startY = 35;

  return (
    <div className="h-full flex flex-col items-center justify-center py-4">
      <div className="w-full flex items-center justify-start mb-4 px-2">
          <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">VQC Circuit Diagram</h3>
      </div>
      
      <svg width="100%" viewBox={`0 0 ${startX + cols * (gateW + 22) + 20} ${startY + qubits * rowH}`} className="max-w-lg overflow-visible">
          {/* Gradients */}
          <defs>
          <linearGradient id="circuit-line-grad" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="hsl(var(--border))" stopOpacity="0.2" />
              <stop offset="50%" stopColor="hsl(var(--border))" stopOpacity="0.8" />
              <stop offset="100%" stopColor="hsl(var(--border))" stopOpacity="0.2" />
          </linearGradient>
          </defs>

          {/* Qubit lines */}
          {Array.from({ length: qubits }).map((_, i) => (
          <g key={i}>
              <text x={12} y={startY + i * rowH + 4} fill="hsl(var(--muted-foreground))" fontSize="11" fontFamily="JetBrains Mono" fontWeight="600">
              q_{i}
              </text>
              <line
              x1={startX - 10}
              y1={startY + i * rowH}
              x2={startX + cols * (gateW + 22)}
              y2={startY + i * rowH}
              stroke="url(#circuit-line-grad)"
              strokeWidth={1.5}
              />
          </g>
          ))}

          {/* Gates */}
          {circuitGates.map((gate, idx) => {
          const x = startX + gate.col * (gateW + 22);
          const y = startY + gate.qubit * rowH;

          if (gate.type === "CNX") {
              const targetY = startY + (gate.target ?? 0) * rowH;
              return (
              <g key={idx}>
                  {/* Control line */}
                  <line x1={x + gateW / 2} y1={y} x2={x + gateW / 2} y2={targetY} stroke="hsl(var(--foreground))" strokeWidth={1.5} />
                  {/* Control dot */}
                  <circle cx={x + gateW / 2} cy={y} r={4} fill="hsl(var(--foreground))" />
                  {/* Target cross */}
                  <circle cx={x + gateW / 2} cy={targetY} r={8} fill="hsl(var(--background))" stroke="hsl(var(--foreground))" strokeWidth={1.5} />
                  <line x1={x + gateW / 2 - 8} y1={targetY} x2={x + gateW / 2 + 8} y2={targetY} stroke="hsl(var(--foreground))" strokeWidth={1.5} />
                  <line x1={x + gateW / 2} y1={targetY - 8} x2={x + gateW / 2} y2={targetY + 8} stroke="hsl(var(--foreground))" strokeWidth={1.5} />
              </g>
              );
          }

          const isM = gate.type === "M";
          return (
              <g key={idx}>
              <rect
                  x={x}
                  y={y - gateH / 2}
                  width={gateW}
                  height={gateH}
                  rx={6}
                  fill={isM ? "hsl(var(--muted)/0.5)" : "hsl(var(--primary)/0.15)"}
                  stroke={isM ? "hsl(var(--border))" : "hsl(var(--primary))"}
                  strokeWidth={1}
              />
              <text
                  x={x + gateW / 2}
                  y={y + 4}
                  textAnchor="middle"
                  fill={isM ? "hsl(var(--muted-foreground))" : "hsl(var(--foreground))"}
                  fontSize="11"
                  fontFamily="JetBrains Mono"
                  fontWeight="600"
              >
                  {gate.type}
              </text>
              </g>
          );
          })}
      </svg>
    </div>
  );
}
