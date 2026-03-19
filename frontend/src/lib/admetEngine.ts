/**
 * ADMET Prediction Engine
 *
 * Computes Absorption, Distribution, Metabolism, Excretion, and Toxicity
 * sub-scores from molecular descriptors, returning a comprehensive profile
 * used for early-stage drug candidate filtering.
 */

export interface MolDescriptors {
    mw: number;          // Molecular weight (g/mol)
    logP: number;        // Partition coefficient
    tpsa: number;        // Topological polar surface area (Å²)
    hbd: number;         // Hydrogen bond donors
    hba: number;         // Hydrogen bond acceptors
    rotBonds: number;    // Rotatable bonds
    aromaticRings: number;
    solubility: number;  // LogS (aqueous solubility)
}

export interface ADMETScores {
    absorption: number;     // 0–1
    distribution: number;   // 0–1
    metabolism: number;     // 0–1
    excretion: number;      // 0–1
    toxicity: number;       // 0–1 (higher = safer)
    overall: number;        // 0–1 weighted composite
    verdict: "Pass" | "Caution" | "Fail";
}

export interface ADMETDetail {
    scores: ADMETScores;
    absorption: {
        lipinskiViolations: number;
        bioavailability: "High" | "Moderate" | "Low";
        solubilityClass: "Good" | "Moderate" | "Poor";
        intestinalAbsorption: "High" | "Moderate" | "Low";
    };
    distribution: {
        bbbPermeant: boolean;
        vdCategory: "High" | "Moderate" | "Low";
        plasmaProteinBinding: "High" | "Moderate" | "Low";
    };
    metabolism: {
        cyp450Substrate: boolean;
        cyp450Inhibitor: boolean;
        hepaticClearance: "Fast" | "Moderate" | "Slow";
    };
    excretion: {
        renalClearance: "Fast" | "Moderate" | "Slow";
        halfLifeCategory: "Short" | "Moderate" | "Long";
        estimatedHalfLife: string;
    };
    toxicity: {
        hergRisk: "Low" | "Moderate" | "High";
        amesMutagenicity: "Negative" | "Borderline" | "Positive";
        hepatotoxicity: "Low" | "Moderate" | "High";
        cardiotoxicity: "Low" | "Moderate" | "High";
    };
}

/* ───── Scoring functions ───── */

function clamp01(v: number): number {
    return Math.max(0, Math.min(1, v));
}

function scoreAbsorption(d: MolDescriptors): { score: number; violations: number; bio: string; sol: string; absorption: string } {
    let violations = 0;
    if (d.mw > 500) violations++;
    if (d.logP > 5) violations++;
    if (d.hbd > 5) violations++;
    if (d.hba > 10) violations++;

    const lipinskiScore = clamp01(1 - violations * 0.25);
    const tpsaScore = clamp01(d.tpsa < 140 ? 1 : 1 - (d.tpsa - 140) / 60);
    const solScore = clamp01(d.solubility > -4 ? 1 : d.solubility > -6 ? 0.5 : 0.2);

    const score = clamp01(lipinskiScore * 0.4 + tpsaScore * 0.3 + solScore * 0.3);

    const bio = violations <= 1 ? "High" : violations <= 2 ? "Moderate" : "Low";
    const sol = d.solubility > -3 ? "Good" : d.solubility > -5 ? "Moderate" : "Poor";
    const absorption = score > 0.7 ? "High" : score > 0.4 ? "Moderate" : "Low";

    return { score, violations, bio, sol, absorption };
}

function scoreDistribution(d: MolDescriptors): { score: number; bbb: boolean; vd: string; ppb: string } {
    const bbb = d.tpsa < 90 && d.mw < 400;
    const bbbScore = bbb ? 0.9 : d.tpsa < 120 ? 0.5 : 0.2;

    const vdScore = d.logP > 1 && d.logP < 4 ? 0.9 : d.logP <= 1 ? 0.5 : 0.3;
    const vd = d.logP > 3 ? "High" : d.logP > 1 ? "Moderate" : "Low";

    const ppbScore = d.logP < 3 ? 0.8 : d.logP < 5 ? 0.5 : 0.2;
    const ppb = d.logP > 4 ? "High" : d.logP > 2 ? "Moderate" : "Low";

    const score = clamp01(bbbScore * 0.4 + vdScore * 0.3 + ppbScore * 0.3);
    return { score, bbb, vd, ppb };
}

function scoreMetabolism(d: MolDescriptors): { score: number; substrate: boolean; inhibitor: boolean; clearance: string } {
    const substrate = d.logP > 2.5 && d.mw > 200;
    const inhibitor = d.logP > 3.5 && d.aromaticRings >= 2;
    const clearance = d.mw < 300 ? "Fast" : d.mw < 500 ? "Moderate" : "Slow";

    let score = 0.8;
    if (substrate) score -= 0.15;
    if (inhibitor) score -= 0.25;
    if (clearance === "Slow") score -= 0.1;

    return { score: clamp01(score), substrate, inhibitor, clearance };
}

function scoreExcretion(d: MolDescriptors): { score: number; renal: string; halfLife: string; halfLifeEst: string } {
    const renal = d.mw < 350 && d.logP < 2 ? "Fast" : d.mw < 500 ? "Moderate" : "Slow";
    const renalScore = renal === "Fast" ? 0.9 : renal === "Moderate" ? 0.6 : 0.3;

    const halfLife = d.mw < 300 ? "Short" : d.mw < 500 ? "Moderate" : "Long";
    const halfLifeScore = halfLife === "Short" ? 0.8 : halfLife === "Moderate" ? 0.7 : 0.3;

    const halfLifeEst = halfLife === "Short" ? "2–6 h" : halfLife === "Moderate" ? "6–24 h" : ">24 h";

    const score = clamp01(renalScore * 0.5 + halfLifeScore * 0.5);
    return { score, renal, halfLife, halfLifeEst };
}

function scoreToxicity(d: MolDescriptors): { score: number; herg: string; ames: string; hepato: string; cardio: string } {
    // hERG inhibition risk
    const hergRisk = d.logP > 3.5 && d.mw > 400 ? "High" : d.logP > 2.5 ? "Moderate" : "Low";
    const hergScore = hergRisk === "Low" ? 0.95 : hergRisk === "Moderate" ? 0.6 : 0.2;

    // Ames mutagenicity (aromatic amines, nitro groups estimated by ring count)
    const ames = d.aromaticRings >= 3 ? "Borderline" : "Negative";
    const amesScore = ames === "Negative" ? 0.9 : 0.5;

    // Hepatotoxicity
    const hepato = d.mw > 600 || d.logP > 5 ? "High" : d.mw > 400 ? "Moderate" : "Low";
    const hepatoScore = hepato === "Low" ? 0.9 : hepato === "Moderate" ? 0.6 : 0.2;

    // Cardiotoxicity
    const cardio = hergRisk;
    const cardioScore = hergScore;

    const score = clamp01(hergScore * 0.35 + amesScore * 0.2 + hepatoScore * 0.25 + cardioScore * 0.2);
    return { score, herg: hergRisk, ames, hepato, cardio };
}

/* ───── Main entry ───── */

export function computeADMET(descriptors: MolDescriptors): ADMETDetail {
    const abs = scoreAbsorption(descriptors);
    const dist = scoreDistribution(descriptors);
    const met = scoreMetabolism(descriptors);
    const exc = scoreExcretion(descriptors);
    const tox = scoreToxicity(descriptors);

    const overall = clamp01(
        abs.score * 0.25 +
        dist.score * 0.15 +
        met.score * 0.2 +
        exc.score * 0.15 +
        tox.score * 0.25
    );

    const verdict: ADMETScores["verdict"] =
        overall > 0.7 ? "Pass" : overall > 0.45 ? "Caution" : "Fail";

    return {
        scores: {
            absorption: abs.score,
            distribution: dist.score,
            metabolism: met.score,
            excretion: exc.score,
            toxicity: tox.score,
            overall,
            verdict,
        },
        absorption: {
            lipinskiViolations: abs.violations,
            bioavailability: abs.bio as "High" | "Moderate" | "Low",
            solubilityClass: abs.sol as "Good" | "Moderate" | "Poor",
            intestinalAbsorption: abs.absorption as "High" | "Moderate" | "Low",
        },
        distribution: {
            bbbPermeant: dist.bbb,
            vdCategory: dist.vd as "High" | "Moderate" | "Low",
            plasmaProteinBinding: dist.ppb as "High" | "Moderate" | "Low",
        },
        metabolism: {
            cyp450Substrate: met.substrate,
            cyp450Inhibitor: met.inhibitor,
            hepaticClearance: met.clearance as "Fast" | "Moderate" | "Slow",
        },
        excretion: {
            renalClearance: exc.renal as "Fast" | "Moderate" | "Slow",
            halfLifeCategory: exc.halfLife as "Short" | "Moderate" | "Long",
            estimatedHalfLife: exc.halfLifeEst,
        },
        toxicity: {
            hergRisk: tox.herg as "Low" | "Moderate" | "High",
            amesMutagenicity: tox.ames as "Negative" | "Borderline" | "Positive",
            hepatotoxicity: tox.hepato as "Low" | "Moderate" | "High",
            cardiotoxicity: tox.cardio as "Low" | "Moderate" | "High",
        },
    };
}

/**
 * Compute a combined drug suitability score from binding affinity,
 * quantum energy, and ADMET profile.
 */
export function combinedScore(bindingAffinity: number, quantumEnergy: number, admetOverall: number): number {
    const bindNorm = clamp01(Math.abs(bindingAffinity) / 15);
    const quantNorm = clamp01(Math.abs(quantumEnergy) / 100);
    return clamp01(bindNorm * 0.35 + quantNorm * 0.25 + admetOverall * 0.4);
}

export interface MultiObjectiveResult {
    binding: number;
    admet: number;
    quantum: number;
    stability: number;
    freeEnergy: number;
    composite: number;
    verdict: "Excellent" | "Good" | "Moderate" | "Poor";
}

/**
 * Multi-objective scoring combining all 5 evaluation axes:
 * binding affinity, ADMET, quantum energy, MD stability, and free energy.
 */
export function multiObjectiveScore(
    bindingAffinity: number,
    admetOverall: number,
    quantumEnergy: number,
    stabilityScore: number,
    freeEnergyDeltaG: number,
): MultiObjectiveResult {
    const binding = clamp01(Math.abs(bindingAffinity) / 15);
    const quantum = clamp01(Math.abs(quantumEnergy) / 100);
    const freeEnergy = clamp01(Math.abs(freeEnergyDeltaG) / 20);

    const composite = clamp01(
        binding * 0.25 +
        admetOverall * 0.20 +
        quantum * 0.15 +
        stabilityScore * 0.20 +
        freeEnergy * 0.20
    );

    const verdict: MultiObjectiveResult["verdict"] =
        composite > 0.75 ? "Excellent" :
            composite > 0.55 ? "Good" :
                composite > 0.35 ? "Moderate" : "Poor";

    return {
        binding: Number(binding.toFixed(3)),
        admet: Number(admetOverall.toFixed(3)),
        quantum: Number(quantum.toFixed(3)),
        stability: Number(stabilityScore.toFixed(3)),
        freeEnergy: Number(freeEnergy.toFixed(3)),
        composite: Number(composite.toFixed(3)),
        verdict,
    };
}

/** Example molecular descriptor sets for demo molecules */
export const DEMO_MOLECULES: Record<string, MolDescriptors> = {
    Aspirin: { mw: 180.16, logP: 1.24, tpsa: 63.6, hbd: 1, hba: 4, rotBonds: 3, aromaticRings: 1, solubility: -1.5 },
    Cetuximab: { mw: 500.2, logP: 2.1, tpsa: 78.2, hbd: 3, hba: 7, rotBonds: 5, aromaticRings: 2, solubility: -3.2 },
    Ibuprofen: { mw: 206.28, logP: 3.97, tpsa: 37.3, hbd: 1, hba: 2, rotBonds: 4, aromaticRings: 1, solubility: -3.1 },
    Metformin: { mw: 129.16, logP: -1.43, tpsa: 91.5, hbd: 3, hba: 5, rotBonds: 2, aromaticRings: 0, solubility: 0.5 },
    Paracetamol: { mw: 151.16, logP: 0.46, tpsa: 49.3, hbd: 2, hba: 3, rotBonds: 1, aromaticRings: 1, solubility: -0.8 },
};
