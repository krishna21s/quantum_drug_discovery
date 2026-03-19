import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import { generateTrajectory, type MDFrame } from "@/lib/mdEngine";
import { cn } from "@/lib/utils";

/**
 * Canvas-based trajectory playback showing a protein backbone
 * with ligand position shifting over time.
 */
export default function TrajectoryPlayer() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [playing, setPlaying] = useState(false);
    const [frame, setFrame] = useState(0);
    const [speed, setSpeed] = useState(1);
    const animRef = useRef<number | null>(null);

    const trajectory = useMemo(() => generateTrajectory(100, 0.5), []);
    const totalFrames = trajectory.frames.length;

    const drawFrame = useCallback((ctx: CanvasRenderingContext2D, f: MDFrame, idx: number) => {
        const w = ctx.canvas.width;
        const h = ctx.canvas.height;
        ctx.clearRect(0, 0, w, h);

        // Background
        ctx.fillStyle = "rgba(10, 14, 28, 0.9)";
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = "rgba(57, 213, 230, 0.05)";
        ctx.lineWidth = 0.5;
        for (let x = 0; x < w; x += 30) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
        for (let y = 0; y < h; y += 30) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }

        const cx = w / 2;
        const cy = h / 2;
        const phase = idx / totalFrames;

        // Protein backbone (stylized helix)
        ctx.lineWidth = 2;
        ctx.strokeStyle = "rgba(77, 142, 247, 0.6)";
        ctx.beginPath();
        for (let i = 0; i < 40; i++) {
            const angle = (i / 40) * Math.PI * 4 + phase * 0.5;
            const x = cx + Math.cos(angle) * 60 + i * 3 - 60;
            const y = cy + Math.sin(angle) * 25;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Secondary structure spheres (residue positions)
        for (let i = 0; i < 12; i++) {
            const angle = (i / 12) * Math.PI * 4 + phase * 0.5;
            const x = cx + Math.cos(angle) * 60 + (i * 3) * (40 / 12) - 60;
            const y = cy + Math.sin(angle) * 25;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(77, 142, 247, ${0.3 + Math.sin(angle) * 0.3})`;
            ctx.fill();
        }

        // Ligand (floating molecule)
        const ligX = cx + 30 + Math.sin(phase * Math.PI * 2) * 15 + Math.sin(f.rmsd * 3) * 5;
        const ligY = cy - 5 + Math.cos(phase * Math.PI * 3) * 8;

        // Ligand glow
        const grad = ctx.createRadialGradient(ligX, ligY, 0, ligX, ligY, 20);
        grad.addColorStop(0, "rgba(57, 213, 230, 0.3)");
        grad.addColorStop(1, "rgba(57, 213, 230, 0)");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(ligX, ligY, 20, 0, Math.PI * 2);
        ctx.fill();

        // Ligand atoms
        ctx.fillStyle = "rgba(57, 213, 230, 0.9)";
        ctx.beginPath(); ctx.arc(ligX, ligY, 6, 0, Math.PI * 2); ctx.fill();
        const atomOffsets = [[-10, -5], [10, -3], [-5, 10], [8, 8]];
        atomOffsets.forEach(([dx, dy]) => {
            ctx.fillStyle = "rgba(77, 142, 247, 0.7)";
            ctx.beginPath(); ctx.arc(ligX + dx, ligY + dy, 3.5, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "rgba(57, 213, 230, 0.3)";
            ctx.lineWidth = 0.8;
            ctx.beginPath(); ctx.moveTo(ligX, ligY); ctx.lineTo(ligX + dx, ligY + dy); ctx.stroke();
        });

        // Frame info
        ctx.fillStyle = "rgba(57, 213, 230, 0.7)";
        ctx.font = "10px JetBrains Mono, monospace";
        ctx.fillText(`Frame ${idx + 1}/${totalFrames}`, 10, 16);
        ctx.fillText(`Time: ${f.time.toFixed(1)} ps`, 10, 28);
        ctx.fillText(`RMSD: ${f.rmsd.toFixed(2)} Å`, w - 90, 16);
        ctx.fillText(`T: ${f.temperature.toFixed(0)} K`, w - 90, 28);
    }, [totalFrames]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        drawFrame(ctx, trajectory.frames[frame], frame);
    }, [frame, drawFrame, trajectory]);

    useEffect(() => {
        if (!playing) return;
        let last = performance.now();
        const step = (now: number) => {
            if (now - last > 50 / speed) {
                setFrame(f => {
                    const next = f + 1;
                    if (next >= totalFrames) { setPlaying(false); return totalFrames - 1; }
                    return next;
                });
                last = now;
            }
            animRef.current = requestAnimationFrame(step);
        };
        animRef.current = requestAnimationFrame(step);
        return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
    }, [playing, speed, totalFrames]);

    return (
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card rounded-2xl p-5 relative overflow-hidden">
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">Trajectory Playback</h3>

            <div className="rounded-xl overflow-hidden ring-1 ring-white/5">
                <canvas ref={canvasRef} width={420} height={200} className="w-full" style={{ imageRendering: "auto" }} />
            </div>

            {/* Scrubber */}
            <div className="mt-3">
                <input
                    type="range"
                    min={0}
                    max={totalFrames - 1}
                    value={frame}
                    onChange={(e) => { setFrame(Number(e.target.value)); setPlaying(false); }}
                    className="w-full accent-quantum h-1"
                />
            </div>

            {/* Controls */}
            <div className="flex items-center justify-center gap-3 mt-2">
                <button onClick={() => { setFrame(0); setPlaying(false); }} className="p-1.5 rounded-lg glass-surface hover:ring-1 hover:ring-quantum/20 transition-all">
                    <SkipBack className="h-3.5 w-3.5 text-muted-foreground" />
                </button>
                <button onClick={() => setPlaying(!playing)} className="p-2.5 rounded-xl bg-quantum/10 ring-1 ring-quantum/30 hover:bg-quantum/20 transition-all">
                    {playing ? <Pause className="h-4 w-4 text-quantum" /> : <Play className="h-4 w-4 text-quantum" />}
                </button>
                <button onClick={() => setFrame(Math.min(totalFrames - 1, frame + 10))} className="p-1.5 rounded-lg glass-surface hover:ring-1 hover:ring-quantum/20 transition-all">
                    <SkipForward className="h-3.5 w-3.5 text-muted-foreground" />
                </button>
                <div className="flex items-center gap-1.5 ml-3">
                    <label className="text-xs text-muted-foreground">Speed</label>
                    <input type="range" min={0.5} max={5} step={0.5} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} className="w-14 accent-quantum h-1" />
                    <span className="text-xs font-mono text-quantum w-5">{speed}×</span>
                </div>
            </div>
        </motion.div>
    );
}
