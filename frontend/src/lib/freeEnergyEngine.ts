/**
 * Free Energy Estimation Engine
 *
 * MM-GBSA-like binding free energy estimation from docking scores
 * and solvation terms, with confidence indicator and decomposition.
 */

export interface EnergyDecomposition {
    vanDerWaals: number;    // kcal/mol
    electrostatic: number;  // kcal/mol
    polarSolvation: number; // kcal/mol
    nonPolarSolvation: number; // kcal/mol (SASA-based)
    entropy: number;        // −TΔS kcal/mol
}

export interface FreeEnergyResult {
    deltaG: number;         // kcal/mol
    decomposition: EnergyDecomposition;
    confidence: "High" | "Moderate" | "Low";
    confidenceScore: number; // 0–1
    dockingContribution: number;
    solvationContribution: number;
}

function clamp01(v: number) { return Math.max(0, Math.min(1, v)); }

/**
 * Estimate binding free energy from docking score and molecular descriptors.
 * Uses an approximate MM-GBSA-like approach.
 */
export function estimateFreeEnergy(
    dockingScore: number,    // kcal/mol (negative = stronger)
    mw: number,
    logP: number,
    tpsa: number,
    hbd: number,
    hba: number,
): FreeEnergyResult {
    // Van der Waals: scales with lipophilicity and molecular size
    const vdw = -Math.abs(dockingScore) * 0.45 - logP * 1.2 - mw * 0.005;

    // Electrostatic: depends on H-bond network and polarity
    const elec = -(hbd + hba) * 0.8 - tpsa * 0.02;

    // Polar solvation (desolvation penalty — opposes binding)
    const polarSolv = tpsa * 0.035 + (hbd + hba) * 0.4;

    // Non-polar solvation (hydrophobic effect — favors binding)
    const npSolv = -logP * 0.6 - 0.5;

    // Entropy penalty (conformational + translational/rotational)
    const entropy = mw * 0.003 + 2.5;

    // Total ΔG
    const deltaG = vdw + elec + polarSolv + npSolv + entropy;

    // Confidence based on how well-defined the energy landscape is
    const consistency = Math.abs(deltaG - dockingScore) / Math.abs(dockingScore);
    const confidenceScore = clamp01(1 - consistency * 0.5);
    const confidence: FreeEnergyResult["confidence"] =
        confidenceScore > 0.7 ? "High" : confidenceScore > 0.4 ? "Moderate" : "Low";

    return {
        deltaG: Number(deltaG.toFixed(2)),
        decomposition: {
            vanDerWaals: Number(vdw.toFixed(2)),
            electrostatic: Number(elec.toFixed(2)),
            polarSolvation: Number(polarSolv.toFixed(2)),
            nonPolarSolvation: Number(npSolv.toFixed(2)),
            entropy: Number(entropy.toFixed(2)),
        },
        confidence,
        confidenceScore: Number(confidenceScore.toFixed(3)),
        dockingContribution: Number((vdw + elec).toFixed(2)),
        solvationContribution: Number((polarSolv + npSolv).toFixed(2)),
    };
}

/** Demo free energy results for known molecules */
export const DEMO_FREE_ENERGIES: Record<string, FreeEnergyResult> = {
    Aspirin: estimateFreeEnergy(-8.2, 180.16, 1.24, 63.6, 1, 4),
    Cetuximab: estimateFreeEnergy(-9.4, 500.2, 2.1, 78.2, 3, 7),
    Ibuprofen: estimateFreeEnergy(-6.8, 206.28, 3.97, 37.3, 1, 2),
    Metformin: estimateFreeEnergy(-5.2, 129.16, -1.43, 91.5, 3, 5),
    Paracetamol: estimateFreeEnergy(-7.1, 151.16, 0.46, 49.3, 2, 3),
};
