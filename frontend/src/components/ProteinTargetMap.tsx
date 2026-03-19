import { motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";

interface DrugCategory {
  label: string;
  count: number;
  color: string;
  drugs: string[];
  angle: number;
}

const categories: DrugCategory[] = [
  { label: "Approved", count: 3, color: "hsl(187, 79%, 54%)", drugs: ["Cetuximab", "Afatinib", "Osimertinib"], angle: -60 },
  { label: "Withdrawn", count: 0, color: "hsl(215, 20%, 40%)", drugs: [], angle: -10 },
  { label: "Investigational", count: 5, color: "hsl(245, 58%, 60%)", drugs: ["IGN311", "Rindopepimut", "Matuzumab", "Canertinib", "Varlitinib"], angle: 30 },
  { label: "Experimental", count: 1, color: "hsl(142, 50%, 40%)", drugs: ["PD-168393"], angle: 120 },
  { label: "Other", count: 3, color: "hsl(330, 65%, 55%)", drugs: ["Gefitinib", "Osimertinib", "Lapatinib"], angle: 200 },
];

export default function ProteinTargetMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCategory, setHoveredCategory] = useState<string | null>(null);
  const [time, setTime] = useState(0);

  useEffect(() => {
    let animFrame: number;
    const tick = () => {
      setTime((t) => t + 0.003);
      animFrame = requestAnimationFrame(tick);
    };
    animFrame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animFrame);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Background particles
    for (let i = 0; i < 80; i++) {
      const px = ((Math.sin(i * 7.3 + time * 0.5) + 1) / 2) * w;
      const py = ((Math.cos(i * 4.7 + time * 0.3) + 1) / 2) * h;
      const alpha = 0.12 + Math.sin(time + i) * 0.06;
      const size = 1 + Math.sin(i * 3.1) * 0.8;
      ctx.beginPath();
      ctx.arc(px, py, size, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(187, 79%, 54%, ${alpha})`;
      ctx.fill();
    }

    // Orbital rings (reference image style)
    for (let ring = 0; ring < 3; ring++) {
      const ringR = 100 + ring * 50;
      ctx.beginPath();
      ctx.ellipse(cx, cy, ringR, ringR * 0.7, 0.1, 0, Math.PI * 2);
      ctx.strokeStyle = `hsla(187, 79%, 54%, ${0.06 - ring * 0.015})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Central protein structure (ribbon-like)
    drawProteinStructure(ctx, cx, cy, time);

    // Category nodes
    categories.forEach((cat) => {
      const rad = (cat.angle * Math.PI) / 180;
      const dist = 160;
      const nodeX = cx + Math.cos(rad) * dist;
      const nodeY = cy + Math.sin(rad) * dist;
      const isHovered = hoveredCategory === cat.label;

      // Connection line to center
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nodeX, nodeY);
      const lineColor = cat.color.replace(")", ", 0.3)").replace("hsl(", "hsla(");
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = isHovered ? 2 : 1;
      if (!isHovered) {
        ctx.setLineDash([6, 4]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Hexagon node
      drawHexagon(ctx, nodeX, nodeY, isHovered ? 34 : 28, cat.color, isHovered);

      // Count inside hexagon
      ctx.fillStyle = "#fff";
      ctx.font = `bold ${isHovered ? 18 : 15}px Inter`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(cat.count), nodeX, nodeY);

      // Label
      ctx.font = "11px Inter";
      ctx.fillStyle = "hsl(215, 20%, 65%)";
      ctx.fillText(cat.label, nodeX, nodeY + (isHovered ? 48 : 42));

      // Drug sub-nodes
      cat.drugs.forEach((drug, di) => {
        const subAngle = rad + ((di - (cat.drugs.length - 1) / 2) * 0.35);
        const subDist = isHovered ? 80 : 60;
        const sx = nodeX + Math.cos(subAngle) * subDist;
        const sy = nodeY + Math.sin(subAngle) * subDist;

        // Sub connection
        ctx.beginPath();
        ctx.moveTo(nodeX, nodeY);
        ctx.lineTo(sx, sy);
        ctx.strokeStyle = cat.color.replace(")", ", 0.2)").replace("hsl(", "hsla(");
        ctx.lineWidth = 1;
        ctx.stroke();

        // Sub dot with glow
        if (isHovered) {
          ctx.beginPath();
          ctx.arc(sx, sy, 10, 0, Math.PI * 2);
          ctx.fillStyle = cat.color.replace(")", ", 0.15)").replace("hsl(", "hsla(");
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(sx, sy, isHovered ? 5 : 4, 0, Math.PI * 2);
        ctx.fillStyle = cat.color;
        ctx.fill();

        // Drug name (only if hovered or always for small sets)
        if (isHovered || cat.drugs.length <= 2) {
          ctx.font = "9px Inter";
          ctx.fillStyle = "hsl(215, 20%, 60%)";
          const textOffset = Math.cos(subAngle) > 0 ? 8 : -8;
          ctx.textAlign = Math.cos(subAngle) > 0 ? "left" : "right";
          ctx.fillText(drug, sx + textOffset, sy + 3);
        }
      });
    });

    // Center label
    ctx.font = "bold 10px JetBrains Mono";
    ctx.fillStyle = "hsl(187, 79%, 54%)";
    ctx.textAlign = "center";
    ctx.shadowColor = "hsl(187, 79%, 54%)";
    ctx.shadowBlur = 8;
    ctx.fillText("EGFR", cx, cy + 55);
    ctx.shadowBlur = 0;
    ctx.font = "9px Inter";
    ctx.fillStyle = "hsl(215, 20%, 50%)";
    ctx.fillText("Protein Target", cx, cy + 67);
  }, [time, hoveredCategory]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    let found: string | null = null;
    categories.forEach((cat) => {
      const rad = (cat.angle * Math.PI) / 180;
      const nodeX = cx + Math.cos(rad) * 160;
      const nodeY = cy + Math.sin(rad) * 160;
      if (Math.hypot(mx - nodeX, my - nodeY) < 35) {
        found = cat.label;
      }
    });
    setHoveredCategory(found);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6 }}
      className="glass-card rounded-2xl p-6 relative overflow-hidden"
    >
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Protein Target Map</h3>
        <span className="text-xs font-mono text-quantum">EGFR Kinase Domain · PDB: 1M17</span>
      </div>
      <div className="flex items-center justify-center rounded-xl bg-background/30 p-2 ring-1 ring-white/5">
        <canvas
          ref={canvasRef}
          width={600}
          height={450}
          className="max-w-full cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredCategory(null)}
        />
      </div>
    </motion.div>
  );
}

function drawHexagon(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string, glow: boolean) {
  ctx.beginPath();
  for (let i = 0; i < 6; i++) {
    const angle = (Math.PI / 3) * i - Math.PI / 6;
    const hx = x + r * Math.cos(angle);
    const hy = y + r * Math.sin(angle);
    if (i === 0) ctx.moveTo(hx, hy);
    else ctx.lineTo(hx, hy);
  }
  ctx.closePath();

  if (glow) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 25;
  }

  ctx.fillStyle = color.replace(")", ", 0.2)").replace("hsl(", "hsla(");
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

function drawProteinStructure(ctx: CanvasRenderingContext2D, cx: number, cy: number, t: number) {
  const ribbonColors = [
    "hsl(217, 70%, 55%)",
    "hsl(187, 60%, 50%)",
    "hsl(330, 50%, 55%)",
    "hsl(142, 50%, 45%)",
  ];

  ribbonColors.forEach((color, ri) => {
    ctx.beginPath();
    const offset = (ri * Math.PI) / 2;
    const baseR = 30 + ri * 8;
    for (let i = 0; i < 40; i++) {
      const angle = (i / 40) * Math.PI * 2 + offset + t * 0.5;
      const wobble = Math.sin(i * 0.5 + t * 2) * 8;
      const r = baseR + wobble;
      const px = cx + Math.cos(angle) * r;
      const py = cy + Math.sin(angle) * r * 0.7;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.6;
    ctx.stroke();
    ctx.globalAlpha = 1;
  });

  // Central glow
  const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 60);
  gradient.addColorStop(0, "hsla(187, 79%, 54%, 0.2)");
  gradient.addColorStop(1, "transparent");
  ctx.fillStyle = gradient;
  ctx.fillRect(cx - 60, cy - 60, 120, 120);
}
