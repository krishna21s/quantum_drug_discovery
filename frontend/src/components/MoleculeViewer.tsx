import { motion } from "framer-motion";
import { useEffect, useRef } from "react";

export default function MoleculeViewer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animFrame: number;
    let angle = 0;

    const atoms = [
      { x: 0, y: 0, z: 0, r: 14, color: "hsl(187, 79%, 54%)" },
      { x: 40, y: 30, z: 10, r: 10, color: "hsl(217, 91%, 60%)" },
      { x: -35, y: 35, z: -10, r: 10, color: "hsl(217, 91%, 60%)" },
      { x: -40, y: -25, z: 15, r: 10, color: "hsl(217, 91%, 60%)" },
      { x: 30, y: -35, z: -15, r: 10, color: "hsl(217, 91%, 60%)" },
      { x: 0, y: 50, z: -20, r: 8, color: "hsl(142, 71%, 45%)" },
      { x: 60, y: -10, z: 5, r: 8, color: "hsl(0, 84%, 60%)" },
      { x: -60, y: -10, z: -5, r: 8, color: "hsl(38, 92%, 50%)" },
    ];

    const bonds = [
      [0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [2, 7],
    ];

    const draw = () => {
      angle += 0.008;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Subtle grid pattern
      ctx.strokeStyle = "hsla(217, 33%, 25%, 0.08)";
      ctx.lineWidth = 0.5;
      for (let gx = 0; gx < w; gx += 20) {
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx, h);
        ctx.stroke();
      }
      for (let gy = 0; gy < h; gy += 20) {
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(w, gy);
        ctx.stroke();
      }

      // Center glow
      const centerGlow = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, 100);
      centerGlow.addColorStop(0, "hsla(187, 79%, 54%, 0.08)");
      centerGlow.addColorStop(1, "transparent");
      ctx.fillStyle = centerGlow;
      ctx.fillRect(0, 0, w, h);

      const cosA = Math.cos(angle);
      const sinA = Math.sin(angle);

      const projected = atoms.map((a) => {
        const rx = a.x * cosA - a.z * sinA;
        const rz = a.x * sinA + a.z * cosA;
        const scale = 300 / (300 + rz);
        return {
          px: w / 2 + rx * scale,
          py: h / 2 + a.y * scale,
          scale,
          r: a.r * scale,
          color: a.color,
          z: rz,
        };
      });

      // Draw bonds with glow
      bonds.forEach(([i, j]) => {
        // Glow layer
        ctx.beginPath();
        ctx.moveTo(projected[i].px, projected[i].py);
        ctx.lineTo(projected[j].px, projected[j].py);
        ctx.strokeStyle = "hsla(187, 79%, 54%, 0.08)";
        ctx.lineWidth = 6;
        ctx.stroke();

        // Main bond
        ctx.beginPath();
        ctx.moveTo(projected[i].px, projected[i].py);
        ctx.lineTo(projected[j].px, projected[j].py);
        ctx.strokeStyle = "hsl(217, 33%, 30%)";
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Sort by z for proper overlap
      const sorted = [...projected].sort((a, b) => a.z - b.z);
      sorted.forEach((p) => {
        // Atom glow
        ctx.beginPath();
        ctx.arc(p.px, p.py, p.r * 2, 0, Math.PI * 2);
        const glowGrad = ctx.createRadialGradient(p.px, p.py, 0, p.px, p.py, p.r * 2);
        glowGrad.addColorStop(0, p.color.replace("hsl(", "hsla(").replace(")", ", 0.2)"));
        glowGrad.addColorStop(1, "transparent");
        ctx.fillStyle = glowGrad;
        ctx.fill();

        // Atom body
        ctx.beginPath();
        ctx.arc(p.px, p.py, p.r, 0, Math.PI * 2);
        const gradient = ctx.createRadialGradient(p.px - p.r * 0.3, p.py - p.r * 0.3, 0, p.px, p.py, p.r);
        gradient.addColorStop(0, p.color);
        gradient.addColorStop(0.7, p.color);
        gradient.addColorStop(1, p.color.replace("hsl(", "hsla(").replace(")", ", 0.4)"));
        ctx.fillStyle = gradient;
        ctx.fill();

        // Specular highlight
        ctx.beginPath();
        ctx.arc(p.px - p.r * 0.25, p.py - p.r * 0.25, p.r * 0.35, 0, Math.PI * 2);
        ctx.fillStyle = "hsla(0, 0%, 100%, 0.2)";
        ctx.fill();
      });

      animFrame = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animFrame);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-6 relative overflow-hidden"
    >
      {/* Top glow line */}
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

      <h3 className="mb-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">Molecular Viewer</h3>
      <div className="flex items-center justify-center rounded-xl bg-background/30 p-4 ring-1 ring-white/5">
        <canvas ref={canvasRef} width={320} height={280} className="max-w-full" />
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-xl glass-surface p-2.5">
          <p className="text-muted-foreground">Formula</p>
          <p className="font-mono font-semibold mt-0.5">C₂₁H₃₀O₅</p>
        </div>
        <div className="rounded-xl glass-surface p-2.5">
          <p className="text-muted-foreground">MW</p>
          <p className="font-mono font-semibold mt-0.5">362.46</p>
        </div>
        <div className="rounded-xl glass-surface p-2.5">
          <p className="text-muted-foreground">LogP</p>
          <p className="font-mono font-semibold mt-0.5">2.14</p>
        </div>
      </div>
    </motion.div>
  );
}
