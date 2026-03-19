import { motion } from "framer-motion";
import { useEffect, useRef } from "react";

export default function BindingSimulation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animFrame: number;
    let t = 0;

    const draw = () => {
      t += 0.005;
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      // Subtle grid background
      ctx.strokeStyle = "hsla(217, 33%, 25%, 0.06)";
      ctx.lineWidth = 0.5;
      for (let gx = 0; gx < w; gx += 25) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      }
      for (let gy = 0; gy < h; gy += 25) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
      }

      // Protein (large blob on right)
      const proteinX = w * 0.65;
      const proteinY = h * 0.5;

      // Protein outer glow
      const pGlow = ctx.createRadialGradient(proteinX, proteinY, 60, proteinX, proteinY, 120);
      pGlow.addColorStop(0, "hsla(217, 91%, 60%, 0.08)");
      pGlow.addColorStop(1, "transparent");
      ctx.fillStyle = pGlow;
      ctx.fillRect(proteinX - 120, proteinY - 120, 240, 240);

      // Draw protein cavity
      ctx.beginPath();
      ctx.ellipse(proteinX, proteinY, 80, 70, 0, 0, Math.PI * 2);
      ctx.fillStyle = "hsl(217, 33%, 10%)";
      ctx.fill();
      ctx.strokeStyle = "hsl(217, 91%, 60%)";
      ctx.lineWidth = 2;
      ctx.stroke();

      // Active site indent
      ctx.beginPath();
      ctx.ellipse(proteinX - 50, proteinY, 25, 20, 0, 0, Math.PI * 2);
      ctx.fillStyle = "hsl(217, 91%, 60%, 0.08)";
      ctx.fill();
      ctx.strokeStyle = "hsl(217, 91%, 60%, 0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Drug molecule approaching
      const progress = Math.min(t, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      const drugStartX = w * 0.15;
      const drugTargetX = proteinX - 50;
      const drugX = drugStartX + (drugTargetX - drugStartX) * eased;
      const drugY = proteinY + Math.sin(t * 3) * (1 - eased) * 15;

      // Drug trail
      if (eased < 0.95) {
        for (let i = 0; i < 5; i++) {
          const trailX = drugX - (eased < 0.5 ? 1 : 0.5) * (i + 1) * 12;
          const trailAlpha = 0.15 - i * 0.03;
          ctx.beginPath();
          ctx.arc(trailX, drugY + Math.sin(t * 3 - i * 0.3) * 3, 4 - i * 0.5, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(187, 79%, 54%, ${trailAlpha})`;
          ctx.fill();
        }
      }

      // Drug glow (intensifies near binding)
      if (eased > 0.6) {
        const glowIntensity = (eased - 0.6) / 0.4;
        const glowRadius = 30 + glowIntensity * 60;
        const gradient = ctx.createRadialGradient(drugX, drugY, 0, drugX, drugY, glowRadius);
        gradient.addColorStop(0, `hsla(187, 79%, 54%, ${0.3 * glowIntensity})`);
        gradient.addColorStop(0.5, `hsla(187, 79%, 54%, ${0.1 * glowIntensity})`);
        gradient.addColorStop(1, "transparent");
        ctx.fillStyle = gradient;
        ctx.fillRect(drugX - glowRadius, drugY - glowRadius, glowRadius * 2, glowRadius * 2);

        // Interaction lines from drug to protein
        if (eased > 0.85) {
          const lineAlpha = (eased - 0.85) / 0.15;
          for (let li = 0; li < 3; li++) {
            const angle = (li - 1) * 0.4;
            const lineEndX = proteinX - 50 + Math.cos(angle) * 25;
            const lineEndY = proteinY + Math.sin(angle) * 20;
            ctx.beginPath();
            ctx.moveTo(drugX, drugY);
            ctx.lineTo(lineEndX, lineEndY);
            ctx.strokeStyle = `hsla(187, 79%, 54%, ${0.6 * lineAlpha})`;
            ctx.lineWidth = 1.5;
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        }
      }

      // Drug molecule
      ctx.beginPath();
      ctx.arc(drugX, drugY, 12, 0, Math.PI * 2);
      ctx.fillStyle = "hsl(187, 79%, 54%)";
      ctx.shadowColor = "hsl(187, 79%, 54%)";
      ctx.shadowBlur = 15;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Small atoms around drug
      for (let i = 0; i < 4; i++) {
        const angle = (Math.PI * 2 * i) / 4 + t * 2;
        const ax = drugX + Math.cos(angle) * 18;
        const ay = drugY + Math.sin(angle) * 18;
        ctx.beginPath();
        ctx.arc(ax, ay, 5, 0, Math.PI * 2);
        ctx.fillStyle = "hsl(142, 71%, 45%, 0.8)";
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(drugX, drugY);
        ctx.lineTo(ax, ay);
        ctx.strokeStyle = "hsl(187, 79%, 54%, 0.25)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Labels
      ctx.font = "11px 'JetBrains Mono'";
      ctx.fillStyle = "hsl(215, 20%, 55%)";
      ctx.textAlign = "center";
      ctx.fillText("Drug Molecule", drugStartX, h - 18);
      ctx.fillText("Protein Target", proteinX, h - 18);

      if (eased > 0.9) {
        ctx.fillStyle = "hsl(187, 79%, 54%)";
        ctx.font = "bold 12px 'JetBrains Mono'";
        ctx.shadowColor = "hsl(187, 79%, 54%)";
        ctx.shadowBlur = 10;
        ctx.fillText("BINDING ACTIVE", w / 2, 25);
        ctx.shadowBlur = 0;
      }

      if (t < 1.5) {
        animFrame = requestAnimationFrame(draw);
      } else {
        t = 0;
        animFrame = requestAnimationFrame(draw);
      }
    };

    draw();
    return () => cancelAnimationFrame(animFrame);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-6 relative overflow-hidden"
    >
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

      <h3 className="mb-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Binding Simulation
      </h3>
      <div className="flex items-center justify-center rounded-xl bg-background/30 p-2 ring-1 ring-white/5">
        <canvas ref={canvasRef} width={400} height={250} className="max-w-full" />
      </div>
      <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
        <div className="rounded-xl glass-surface p-2.5">
          <p className="text-muted-foreground">Binding Score</p>
          <p className="font-mono font-semibold text-quantum mt-0.5">0.94</p>
        </div>
        <div className="rounded-xl glass-surface p-2.5">
          <p className="text-muted-foreground">Site Coverage</p>
          <p className="font-mono font-semibold text-success mt-0.5">87%</p>
        </div>
        <div className="rounded-xl glass-surface p-2.5">
          <p className="text-muted-foreground">Stability</p>
          <p className="font-mono font-semibold text-primary mt-0.5">High</p>
        </div>
      </div>
    </motion.div>
  );
}
