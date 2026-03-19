/**
 * Molecular Dynamics Simulation Engine
 *
 * Generates scientifically realistic RMSD/RMSF time-series,
 * protein preprocessing pipeline, and stability metrics.
 */

export interface MDFrame {
    time: number;       // ps
    rmsd: number;       // Å
    rmsf: number;       // Å
    energy: number;     // kJ/mol
    temperature: number; // K
}

export interface MDTrajectory {
    frames: MDFrame[];
    totalTime: number;
    avgRMSD: number;
    avgRMSF: number;
    convergence: boolean;
    stabilityScore: number; // 0–1
}

export interface PrepStep {
    id: string;
    label: string;
    detail: string;
    duration: number; // simulated ms
}

export const PREP_STEPS: PrepStep[] = [
    { id: "water", label: "Remove Water Molecules", detail: "Stripped 342 crystallographic waters", duration: 800 },
    { id: "hetero", label: "Remove Heteroatoms", detail: "Removed 12 non-standard residues", duration: 600 },
    { id: "hydrogen", label: "Add Hydrogens", detail: "Added 1,247 polar + non-polar H atoms", duration: 1200 },
    { id: "charges", label: "Assign Charges", detail: "Applied AMBER ff14SB force field", duration: 900 },
    { id: "minimize", label: "Energy Minimization", detail: "Converged in 1,500 steps · ΔE = −2,847 kJ/mol", duration: 1500 },
    { id: "solvate", label: "Solvation Box", detail: "TIP3P water box · 12 Å padding · 8,432 waters", duration: 700 },
];

function clamp(v: number, lo: number, hi: number) {
    return Math.max(lo, Math.min(hi, v));
}

/** Generate a realistic MD trajectory */
export function generateTrajectory(durationPs: number = 100, dt: number = 0.5): MDTrajectory {
    const steps = Math.floor(durationPs / dt);
    const frames: MDFrame[] = [];

    let rmsd = 0.5;
    let rmsf = 0.3;
    let energy = -12500;
    const temp = 300;

    for (let i = 0; i <= steps; i++) {
        const t = i * dt;
        const phase = t / durationPs;

        // RMSD: equilibration rise then plateau with noise
        const equilibrium = 1.8 + Math.sin(phase * 2) * 0.15;
        rmsd = phase < 0.15
            ? 0.5 + (equilibrium - 0.5) * (phase / 0.15)
            : equilibrium + (Math.random() - 0.5) * 0.25;
        rmsd = clamp(rmsd, 0.3, 3.5);

        // RMSF: per-residue flexibility proxy
        rmsf = 0.6 + Math.sin(phase * 4 + 1) * 0.2 + (Math.random() - 0.5) * 0.15;
        rmsf = clamp(rmsf, 0.2, 2.0);

        // Energy: convergence toward minimum
        energy = -12500 - phase * 800 + Math.sin(phase * 10) * 150 + (Math.random() - 0.5) * 100;

        frames.push({
            time: Number(t.toFixed(2)),
            rmsd: Number(rmsd.toFixed(3)),
            rmsf: Number(rmsf.toFixed(3)),
            energy: Number(energy.toFixed(1)),
            temperature: Number((temp + (Math.random() - 0.5) * 4).toFixed(1)),
        });
    }

    const avgRMSD = frames.reduce((s, f) => s + f.rmsd, 0) / frames.length;
    const avgRMSF = frames.reduce((s, f) => s + f.rmsf, 0) / frames.length;
    const lastQuarter = frames.slice(Math.floor(frames.length * 0.75));
    const rmsdStd = Math.sqrt(lastQuarter.reduce((s, f) => s + (f.rmsd - avgRMSD) ** 2, 0) / lastQuarter.length);
    const convergence = rmsdStd < 0.4;
    const stabilityScore = clamp(1 - avgRMSD / 3, 0, 1);

    return {
        frames,
        totalTime: durationPs,
        avgRMSD: Number(avgRMSD.toFixed(3)),
        avgRMSF: Number(avgRMSF.toFixed(3)),
        convergence,
        stabilityScore: Number(stabilityScore.toFixed(3)),
    };
}

/** Subsample trajectory for chart display */
export function subsample(frames: MDFrame[], maxPoints: number = 120): MDFrame[] {
    if (frames.length <= maxPoints) return frames;
    const step = Math.ceil(frames.length / maxPoints);
    return frames.filter((_, i) => i % step === 0);
}
