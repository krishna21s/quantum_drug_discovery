import { useEffect, useRef } from "react";

export default function AnimatedBackground() {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (prefersReducedMotion) return;

        let animFrame: number;
        let time = 0;

        const resize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        };
        resize();
        window.addEventListener("resize", resize);

        // Floating orbs in dark blue/teal, matching health dashboard aesthetic
        const orbs = [
            { x: 0.15, y: 0.25, r: 0.38, hue: 207, sat: 100, lit: 50, speed: 0.5, amp: 0.06 },
            { x: 0.82, y: 0.7,  r: 0.3,  hue: 187, sat: 85,  lit: 55, speed: 0.4, amp: 0.05 },
            { x: 0.5,  y: 0.5,  r: 0.22, hue: 280, sat: 75,  lit: 60, speed: 0.6, amp: 0.04 },
            { x: 0.9,  y: 0.1,  r: 0.2,  hue: 350, sat: 85,  lit: 62, speed: 0.7, amp: 0.05 },
        ];

        // Tiny star particles
        const particles = Array.from({ length: 60 }, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.18,
            vy: (Math.random() - 0.5) * 0.18,
            r: Math.random() * 1.2 + 0.3,
            alpha: Math.random() * 0.25 + 0.05,
            hue: [207, 187, 280][Math.floor(Math.random() * 3)],
        }));

        const draw = () => {
            time += 0.005;
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            // Animated glowing orbs
            orbs.forEach((orb) => {
                const cx = (orb.x + Math.sin(time * orb.speed) * orb.amp) * w;
                const cy = (orb.y + Math.cos(time * orb.speed * 0.7) * orb.amp) * h;
                const radius = orb.r * Math.min(w, h);

                const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
                grad.addColorStop(0, `hsla(${orb.hue}, ${orb.sat}%, ${orb.lit}%, 0.08)`);
                grad.addColorStop(0.5, `hsla(${orb.hue}, ${orb.sat}%, ${orb.lit}%, 0.04)`);
                grad.addColorStop(1, "transparent");
                ctx.fillStyle = grad;
                ctx.fillRect(0, 0, w, h);
            });

            // Star particles
            particles.forEach((p) => {
                p.x += p.vx;
                p.y += p.vy;
                if (p.x < 0) p.x = w;
                if (p.x > w) p.x = 0;
                if (p.y < 0) p.y = h;
                if (p.y > h) p.y = 0;

                const flicker = p.alpha + Math.sin(time * 2.5 + p.x * 0.015) * 0.06;
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fillStyle = `hsla(${p.hue}, 85%, 70%, ${Math.max(0, flicker)})`;
                ctx.fill();
            });

            // Subtle near-particle connections
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 100) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `hsla(207, 100%, 60%, ${0.045 * (1 - dist / 100)})`;
                        ctx.lineWidth = 0.5;
                        ctx.stroke();
                    }
                }
            }

            animFrame = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            cancelAnimationFrame(animFrame);
            window.removeEventListener("resize", resize);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            className="fixed inset-0 -z-10 pointer-events-none"
            aria-hidden="true"
        />
    );
}
