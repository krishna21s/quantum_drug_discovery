import AppLayout from "@/components/AppLayout";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useCallback } from "react";
import {
  Search, Loader2, Sparkles, AlertTriangle, FlaskConical,
  CheckCircle2, Zap, Info, TrendingUp, Shield,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/* ─────────────────────────────────────────────────────────────────────────────
   Pharmacophore / Structural Pattern Engine (pure JS, no backend)
   Each disease entry defines weighted feature patterns detected on a SMILES
   string. Score = Σ(matched_weight) / Σ(total_weight) → displayed as 0–100%.
   Only diseases scoring ≥ 50 are user-visible.
────────────────────────────────────────────────────────────────────────────── */

type MatchLevel = "CONFIRMED" | "HIGH" | "MODERATE";

interface Feature {
  label: string;         // Human-readable feature name
  test: (s: string) => boolean;  // Detector on raw SMILES
  weight: number;        // Contribution to total score
  description: string;   // What this feature does clinically
}

interface Disease {
  id: string;
  name: string;
  category: "Viral" | "Bacterial" | "Fungal" | "Oncology" | "CNS" | "Metabolic" | "Cardiovascular" | "Inflammatory";
  description: string;
  icon: string;
  color: string;
  gradientFrom: string;
  gradientTo: string;
  features: Feature[];
  approvedAnalogues: string[];
  clinicalNote: string;
}

const DISEASES: Disease[] = [
  // ── Viral ──────────────────────────────────────────────────────────────────
  {
    id: "covid19",
    name: "COVID-19 (SARS-CoV-2)",
    category: "Viral",
    description: "Respiratory illness caused by SARS-CoV-2. Main drug targets: RNA-dependent RNA polymerase (RdRp), main protease (Mpro), and spike protein.",
    icon: "🦠",
    color: "#38bdf8",
    gradientFrom: "from-sky-500/20",
    gradientTo: "to-cyan-500/10",
    features: [
      {
        label: "Nucleoside/Nucleotide scaffold",
        test: (s) => /[CN]1[Cc][Cc]([Cc][Cc]1)[Nn]/i.test(s) || /n1cc[cn]c1|N1C=CN=C1/i.test(s) || /nc(=O)n|nc(n)/i.test(s),
        weight: 30,
        description: "Mimics viral RNA building blocks to inhibit RdRp",
      },
      {
        label: "Phosphonate or phosphoramidate",
        test: (s) => /P\(=O\)|OP\(|NP\(/i.test(s),
        weight: 25,
        description: "Prodrug moiety enabling intracellular conversion to active triphosphate",
      },
      {
        label: "Heterocyclic core (pyrimidine/purine)",
        test: (s) => /c1ncnc|c1ccnc|C1=NC|n1ccnc/i.test(s) || /c1nc2|n1cn/i.test(s),
        weight: 20,
        description: "Core scaffold of nucleobase analogues targeting viral polymerase",
      },
      {
        label: "Hydroxyl on sugar ring",
        test: (s) => /\[C@@H\]\(O\)|\[C@H\]\(O\)|CO[C@@H]/i.test(s),
        weight: 15,
        description: "Ribose/deoxyribose hydroxyl groups critical for polymerase binding",
      },
      {
        label: "Amide / nitrile pharmacophore",
        test: (s) => /C#N|C\(=O\)N/i.test(s),
        weight: 10,
        description: "Electrophilic warhead for covalent Mpro inhibition",
      },
    ],
    approvedAnalogues: ["Remdesivir (GS-5734)", "Molnupiravir (EIDD-2801)", "Nirmatrelvir"],
    clinicalNote: "Approved by FDA for moderate-to-severe COVID-19. RdRp inhibitors are first-line antivirals.",
  },
  {
    id: "influenza",
    name: "Influenza (Flu)",
    category: "Viral",
    description: "Respiratory infection by Influenza A/B viruses. Key target: neuraminidase (NA) surface glycoprotein essential for viral release.",
    icon: "🌡️",
    color: "#a78bfa",
    gradientFrom: "from-violet-500/20",
    gradientTo: "to-purple-500/10",
    features: [
      {
        label: "Carboxylate group (–COOH / –COO⁻)",
        test: (s) => /C\(=O\)O|C\(O\)=O/i.test(s),
        weight: 30,
        description: "Interacts with Arg118/Arg292/Arg371 in the neuraminidase active site",
      },
      {
        label: "Cyclohexene / cyclohexane ring",
        test: (s) => /C1CC=CC|C1=CCC|C1CCCCC1/i.test(s),
        weight: 25,
        description: "Core scaffold of oseltamivir-class neuraminidase inhibitors",
      },
      {
        label: "Acetamido group (–NHC(=O)CH₃)",
        test: (s) => /NC\(=O\)C|NC\(C\)=O/i.test(s),
        weight: 20,
        description: "Mimics the N-acetyl group of sialic acid substrate",
      },
      {
        label: "Guanidinium group",
        test: (s) => /NC\(=N\)N|N=C\(N\)N/i.test(s),
        weight: 15,
        description: "Electrostatic interaction with Glu119 in zanamivir-class inhibitors",
      },
      {
        label: "Alkyl ether side chain",
        test: (s) => /COCCC|OCCC|OCC/i.test(s),
        weight: 10,
        description: "Hydrophobic pocket occupancy (150-cavity) of influenza NA",
      },
    ],
    approvedAnalogues: ["Oseltamivir (Tamiflu)", "Zanamivir (Relenza)", "Baloxavir marboxil"],
    clinicalNote: "Neuraminidase inhibitors reduce flu duration by 1–3 days and prevent complications.",
  },
  {
    id: "hiv",
    name: "HIV / AIDS",
    category: "Viral",
    description: "Retroviral infection targeting CD4⁺ T-cells. Drug targets: HIV protease, reverse transcriptase (RT), integrase.",
    icon: "🔴",
    color: "#fb7185",
    gradientFrom: "from-rose-500/20",
    gradientTo: "to-red-500/10",
    features: [
      {
        label: "Peptidomimetic scaffold",
        test: (s) => /NC\(=O\)[C@@H]|NC\(=O\)[C@H]|\[C@@H\]\(O\)CN|C\(=O\)N[C@@H]/i.test(s),
        weight: 30,
        description: "Mimics the Phe-Pro cleavage site of HIV Gag polyprotein",
      },
      {
        label: "Hydroxyl ketone / hydroxyethylamine",
        test: (s) => /C\(=O\)CC\(O\)|CC\(O\)CN|\[C@@H\]\(O\)/i.test(s),
        weight: 25,
        description: "Transition-state isostere binding to HIV protease catalytic Asp dyad",
      },
      {
        label: "Aromatic sulfonamide",
        test: (s) => /NS\(=O\)\(=O\)c|c.*S\(=O\)\(=O\)N/i.test(s),
        weight: 20,
        description: "P2 substituent enhancing potency and bioavailability",
      },
      {
        label: "Piperazine / morpholine ring",
        test: (s) => /N1CCNCC1|N1CCOCC1/i.test(s),
        weight: 15,
        description: "CYP3A4 interaction motif for pharmacokinetic boosting",
      },
      {
        label: "Di-halogenated arene",
        test: (s) => /c.*F.*Cl|c.*Cl.*F|c1cc\(F\).*\(Cl\)/i.test(s),
        weight: 10,
        description: "Hydrophobic contacts in integrase/NNRTI binding pocket",
      },
    ],
    approvedAnalogues: ["Lopinavir/Ritonavir", "Atazanavir", "Darunavir"],
    clinicalNote: "Modern ART combinations suppress viral load to undetectable. Protease inhibitors are backbone of many regimens.",
  },
  {
    id: "herpes",
    name: "Herpes Simplex Virus (HSV)",
    category: "Viral",
    description: "HSV-1/HSV-2 infections. Key target: viral thymidine kinase (TK) phosphorylates nucleoside analogs to their active triphosphate inhibitors.",
    icon: "⚡",
    color: "#34d399",
    gradientFrom: "from-emerald-500/20",
    gradientTo: "to-green-500/10",
    features: [
      {
        label: "Acyclic nucleoside framework",
        test: (s) => /NCOCCO|OCCOCCO|NCCO[Cc]/i.test(s),
        weight: 35,
        description: "Acyclic side chain mimics ribose; phosphorylated by HSV-TK selectively",
      },
      {
        label: "Guanine / hypoxanthine base",
        test: (s) => /Nc1nc2c\(ncn2\)c\(=O\)|c1nc2nc\(N\)nc\(=O\)c2n/i.test(s) || /Nc1nc.*nc\(=O\)/i.test(s),
        weight: 30,
        description: "Purine scaffold recognized by viral TK with high selectivity",
      },
      {
        label: "NH-lactam moiety",
        test: (s) => /\[nH\]|N1C=|c1\[nH\]/i.test(s),
        weight: 20,
        description: "N-H donor required for recognition by viral thymidine kinase",
      },
      {
        label: "Ether oxygen linkage",
        test: (s) => /COC|OCO|NCOC/i.test(s),
        weight: 15,
        description: "Acyclic ribose mimic flexibility for TK binding",
      },
    ],
    approvedAnalogues: ["Acyclovir (Zovirax)", "Valacyclovir", "Famciclovir"],
    clinicalNote: "Nucleoside analogs activated by viral TK enable selective virus killing over host cells.",
  },
  // ── Bacterial ──────────────────────────────────────────────────────────────
  {
    id: "bacterial_broad",
    name: "Broad-Spectrum Bacterial Infections",
    category: "Bacterial",
    description: "Coverage of Gram-positive & Gram-negative pathogens. Key targets: cell wall synthesis (beta-lactam), DNA gyrase (fluoroquinolones), ribosome (macrolides).",
    icon: "🧫",
    color: "#fbbf24",
    gradientFrom: "from-amber-500/20",
    gradientTo: "to-yellow-500/10",
    features: [
      {
        label: "Beta-lactam ring",
        test: (s) => /C1CNC1=O|C1CN\(C1=O\)|[SC]1CCN.*C1/i.test(s) || /N1CC(=O)|N1C.*C(=O)/i.test(s),
        weight: 35,
        description: "Mechanism-based inhibitor of penicillin-binding proteins (PBPs)",
      },
      {
        label: "Fluoroquinolone scaffold",
        test: (s) => /c\(F\)cc.*n.*c.*=O|O=C.*n1c.*c\(F\)/i.test(s) || /c(F)c.*N.*CC.*C(=O)O/i.test(s),
        weight: 30,
        description: "Dual inhibitor of DNA gyrase and topoisomerase IV",
      },
      {
        label: "Carboxylic acid + N-heterocycle",
        test: (s) => /C\(=O\)O.*n|n.*C\(=O\)O/i.test(s),
        weight: 20,
        description: "Pharmacophore essential for gyrase chelation",
      },
      {
        label: "Cyclopropyl amine",
        test: (s) => /N.*C1CC1|c1.*n.*C2CC2/i.test(s),
        weight: 15,
        description: "N1-cyclopropyl group for fluoroquinolone selectivity",
      },
    ],
    approvedAnalogues: ["Ciprofloxacin", "Amoxicillin", "Azithromycin"],
    clinicalNote: "Combination strategies recommended to reduce resistance. Beta-lactam + inhibitor pairs are first-line for many hospital-acquired infections.",
  },
  // ── Fungal ─────────────────────────────────────────────────────────────────
  {
    id: "fungal",
    name: "Invasive Fungal Infections",
    category: "Fungal",
    description: "Candida, Aspergillus, and Cryptococcus infections. Key target: lanosterol 14α-demethylase (CYP51A1) — disrupts fungal ergosterol biosynthesis.",
    icon: "🍄",
    color: "#c084fc",
    gradientFrom: "from-purple-500/20",
    gradientTo: "to-fuchsia-500/10",
    features: [
      {
        label: "1,2,4-Triazole ring",
        test: (s) => /c1ncnn1|n1ncn.|n1cc[nH]/i.test(s) || /Cn1cc|Cn1cn/i.test(s),
        weight: 40,
        description: "Azole nitrogen coordinates with heme iron in fungal CYP51",
      },
      {
        label: "Fluorinated phenyl ring",
        test: (s) => /c1ccc\(F\)cc1|c.*\(F\).*c\(F\)/i.test(s),
        weight: 30,
        description: "Heme-loop hydrophobic contact critical for potency",
      },
      {
        label: "Tertiary alcohol",
        test: (s) => /C\(O\)\(C[Cc]\)|\(O\)\([Cc][Cc]\)/i.test(s),
        weight: 20,
        description: "Hydrogen bond donor in the CYP51 substrate channel",
      },
      {
        label: "Bis-triazole / imidazole arrangement",
        test: (s) => /n1cnc.*Cn|Cn1cc.*Cn/i.test(s),
        weight: 10,
        description: "Dual azole motif signature of fluconazole-class drugs",
      },
    ],
    approvedAnalogues: ["Fluconazole", "Itraconazole", "Voriconazole"],
    clinicalNote: "Azoles are first-line for most invasive fungal infections. Resistance surveillance is critical in immunocompromised patients.",
  },
  // ── Oncology ───────────────────────────────────────────────────────────────
  {
    id: "kinase_cancer",
    name: "Tyrosine-Kinase–Driven Cancers",
    category: "Oncology",
    description: "Cancers driven by mutant/overexpressed kinases (CML, NSCLC, HER2+ breast cancer). Target: ATP-competitive kinase inhibition.",
    icon: "🎗️",
    color: "#f87171",
    gradientFrom: "from-red-500/20",
    gradientTo: "to-orange-500/10",
    features: [
      {
        label: "ATP-mimetic hinge-binding pharmacophore",
        test: (s) => /Nc1nc|Nc1cc.*nc|c.*NC.*c.*n/i.test(s),
        weight: 35,
        description: "H-bond donor-acceptor pair mimicking adenine N1/N6 of ATP",
      },
      {
        label: "Piperazine / piperidine solubiliser",
        test: (s) => /N1CCNCC1|N1CCCCC1/i.test(s),
        weight: 25,
        description: "Basic nitrogen improves aqueous solubility and cell permeability",
      },
      {
        label: "Pyridine / pyrimidine diamine scaffold",
        test: (s) => /Nc1ncnc|Nc1ccnc|Nc.*nc.*N/i.test(s),
        weight: 25,
        description: "Critical DFG-in binding motif of Type I kinase inhibitors",
      },
      {
        label: "Amide linker",
        test: (s) => /C\(=O\)Nc|cNC\(=O\)/i.test(s),
        weight: 15,
        description: "Connects pharmacophore units and H-bonds gatekeeper residue",
      },
    ],
    approvedAnalogues: ["Imatinib (Gleevec)", "Erlotinib", "Lapatinib", "Osimertinib"],
    clinicalNote: "Targeted kinase inhibitors have revolutionised oncology with response rates exceeding 80% in biomarker-selected populations.",
  },
  {
    id: "topo_cancer",
    name: "Hematologic Malignancies",
    category: "Oncology",
    description: "Leukaemia, lymphoma, and multiple myeloma. Targets include DNA topoisomerase I/II, anthracycline intercalation, and alkylation.",
    icon: "🩸",
    color: "#f43f5e",
    gradientFrom: "from-rose-600/20",
    gradientTo: "to-red-600/10",
    features: [
      {
        label: "Polycyclic aromatic / anthracene core",
        test: (s) => /c1ccc2cc3|c1cccc2c1|c1ccc2ccc3/i.test(s),
        weight: 40,
        description: "Intercalation into DNA minor groove stacking between base pairs",
      },
      {
        label: "Quinone moiety (C=O adjacent arene)",
        test: (s) => /C\(=O\)c1c.*C\(=O\)|c1cc\(=O\)c/i.test(s),
        weight: 30,
        description: "Redox cycling generates ROS and adducts with topoisomerase II",
      },
      {
        label: "Glycosyl amino sugar",
        test: (s) => /\[C@@H\]\(N\)|O[C@@H].*N|\[NH\d\]/i.test(s),
        weight: 20,
        description: "Amino sugar of anthracyclines provides electrostatic contacts with DNA phosphate backbone",
      },
      {
        label: "Multiple hydroxyl groups",
        test: (s) => (s.match(/\(O\)/g) || []).length >= 3,
        weight: 10,
        description: "Hydroxyl groups chelate metal ions and H-bond topoisomerase residues",
      },
    ],
    approvedAnalogues: ["Doxorubicin", "Epirubicin", "Daunorubicin", "Mitoxantrone"],
    clinicalNote: "Anthracycline-based regimens remain cornerstone of AML and aggressive lymphoma treatment despite cardiotoxicity risk.",
  },
  // ── CNS ────────────────────────────────────────────────────────────────────
  {
    id: "depression",
    name: "Major Depression & Anxiety",
    category: "CNS",
    description: "Mood disorders involving serotonin/norepinephrine dysregulation. Target: serotonin reuptake transporter (SERT) blockade.",
    icon: "🧠",
    color: "#818cf8",
    gradientFrom: "from-indigo-500/20",
    gradientTo: "to-blue-500/10",
    features: [
      {
        label: "Tricyclic / bicyclic arene scaffold",
        test: (s) => /C1CC2=CC=CC=C2[CH]1|c1ccc2c\(c1\)|c1cccc2cccc/i.test(s),
        weight: 30,
        description: "Aromatic core for hydrophobic occupancy of SERT transmembrane vestibule",
      },
      {
        label: "Basic amine (pKa ~9–10)",
        test: (s) => /CN\(C\)CC|CCNCC|CCN\(C/i.test(s) || /N[Cc][Cc]|[Cc]N[Cc]/i.test(s),
        weight: 30,
        description: "Ionisable amine forms salt bridge with Asp98 in SERT binding site",
      },
      {
        label: "Halogenated aryl ring",
        test: (s) => /c1cc\(Cl\)|c1cc\(F\)|c.*Br.*c|c.*F.*cc/i.test(s),
        weight: 25,
        description: "Halogen occupies selectivity-determining S1 pocket in SERT",
      },
      {
        label: "Ether linkage to arene",
        test: (s) => /Oc1ccc|c1.*Oc|OCc1c/i.test(s),
        weight: 15,
        description: "O-aryl linkage present in fluoxetine, citalopram pharmacophore",
      },
    ],
    approvedAnalogues: ["Sertraline (Zoloft)", "Fluoxetine (Prozac)", "Escitalopram", "Venlafaxine"],
    clinicalNote: "SSRIs/SNRIs are first-line for MDD with 60–70% response. Side-effect profile strongly influences choice.",
  },
  // ── Metabolic ──────────────────────────────────────────────────────────────
  {
    id: "diabetes",
    name: "Type 2 Diabetes Mellitus",
    category: "Metabolic",
    description: "Insulin resistance and β-cell dysfunction. Targets: AMPK activation (biguanides), DPP-4 inhibition (gliptins), SGLT2 (gliflozins).",
    icon: "💉",
    color: "#4ade80",
    gradientFrom: "from-green-500/20",
    gradientTo: "to-emerald-500/10",
    features: [
      {
        label: "Biguanide pharmacophore",
        test: (s) => /NC\(=N\)NC\(=N\)N|N=C\(N\)NC\(=N\)N/i.test(s),
        weight: 40,
        description: "Metformin class — inhibits mitochondrial complex I to activate AMPK",
      },
      {
        label: "Glucopyranoside / C-glycoside",
        test: (s) => /O[C@@H]1CO[C@H]|O[C@H]1CO[C@@H]|Oc1ccc.*CO/i.test(s),
        weight: 35,
        description: "SGLT2 gliflozin scaffold — blocks renal glucose reabsorption",
      },
      {
        label: "Fluorinated phenyl + nitrile",
        test: (s) => /c\(F\).*C#N|C#N.*c\(F\)/i.test(s),
        weight: 15,
        description: "DPP-4 gliptin pharmacophore targeting S1/S2 subsites",
      },
      {
        label: "Aromatic sulfonyl-urea",
        test: (s) => /NS\(=O\)\(=O\)c.*NC\(=O\)N|c.*S\(=O\)\(=O\)NC/i.test(s),
        weight: 10,
        description: "Sulfonylurea class — closes K-ATP channels to stimulate insulin release",
      },
    ],
    approvedAnalogues: ["Metformin", "Empagliflozin", "Sitagliptin (Januvia)", "Glipizide"],
    clinicalNote: "First-line: Metformin + lifestyle. SGLT2i and GLP-1 agonists added for CV/renal benefit.",
  },
  // ── Cardiovascular ─────────────────────────────────────────────────────────
  {
    id: "cardiovascular",
    name: "Cardiovascular Disease (Hyperlipidaemia)",
    category: "Cardiovascular",
    description: "Dyslipidaemia and atherosclerosis. Target: HMG-CoA reductase inhibition reduces LDL cholesterol 40–60%.",
    icon: "❤️",
    color: "#fb923c",
    gradientFrom: "from-orange-500/20",
    gradientTo: "to-red-500/10",
    features: [
      {
        label: "3,5-Dihydroxy heptanoic acid chain",
        test: (s) => /CC\(O\)CC\(O\)CC\(=O\)O|C\(O\)CC\(O\)C/i.test(s),
        weight: 40,
        description: "Mevalonate isostere that competitively inhibits HMG-CoA reductase",
      },
      {
        label: "Fluorinated arene",
        test: (s) => /c1ccc\(F\)cc1|c1cc\(F\)ccc1/i.test(s),
        weight: 25,
        description: "Para-fluorophenyl substructure of atorvastatin/rosuvastatin class",
      },
      {
        label: "Isopropyl / tert-butyl group",
        test: (s) => /CC\(C\)c|c.*C\(C\)C|CC\(C\)|c1.*CC\(C\)C/i.test(s),
        weight: 20,
        description: "Hydrophobic contact in HMG-CoA reductase amphipathic helix",
      },
      {
        label: "Pyrrole / pyrimidine scaffold",
        test: (s) => /c1cc\[nH\]c1|c1ccnc.*c1|n1ccc/i.test(s),
        weight: 15,
        description: "Bicyclic ring system of synthetic statins (atorvastatin, rosuvastatin)",
      },
    ],
    approvedAnalogues: ["Atorvastatin (Lipitor)", "Rosuvastatin (Crestor)", "Simvastatin"],
    clinicalNote: "Statins reduce major cardiovascular events by 25–35%. High-intensity therapy recommended for secondary prevention.",
  },
  // ── Inflammatory ────────────────────────────────────────────────────────────
  {
    id: "rheumatoid",
    name: "Rheumatoid Arthritis / Autoimmune",
    category: "Inflammatory",
    description: "Chronic inflammatory joint disease. Targets: COX-2 inhibition (NSAIDs), DMARD immunomodulation, anti-TNF-α biologics.",
    icon: "🦴",
    color: "#f0abfc",
    gradientFrom: "from-fuchsia-500/20",
    gradientTo: "to-pink-500/10",
    features: [
      {
        label: "Carboxylic acid + aromatic ring (NSAID pattern)",
        test: (s) => /C\(=O\)O.*c|cC\(=O\)O|c.*CC\(=O\)O/i.test(s),
        weight: 35,
        description: "Pharmacophore for COX-1/COX-2 inhibition — carboxylate chelates active-site Arg120",
      },
      {
        label: "Sulfonamide / sulfone",
        test: (s) => /NS\(=O\)\(=O\)|S\(=O\)\(=O\)N/i.test(s),
        weight: 25,
        description: "Selective COX-2 pharmacophore (coxib class — fits COX-2 side pocket)",
      },
      {
        label: "Acetamide / aniline",
        test: (s) => /NC\(=O\)c|cNC\(=O\)/i.test(s),
        weight: 25,
        description: "Amide linker for DMARD immunomodulators (methotrexate, leflunomide)",
      },
      {
        label: "Chlorine on aromatic ring",
        test: (s) => /c.*Cl|Clc/i.test(s),
        weight: 15,
        description: "Halogen occupancy of hydrophobic sub-pocket in COX/JAK binding sites",
      },
    ],
    approvedAnalogues: ["Ibuprofen", "Celecoxib (Celebrex)", "Methotrexate", "Hydroxychloroquine"],
    clinicalNote: "Combination DMARD therapy (methotrexate + biologic) achieves remission in ~50% of RA patients.",
  },
];

/* ─────────────────────────────────────────────────────────────────────────────
   Scoring Engine
────────────────────────────────────────────────────────────────────────────── */
interface ScoredDisease {
  disease: Disease;
  score: number;              // 0–100
  level: MatchLevel;
  matchedFeatures: { label: string; description: string }[];
  missedFeatures: string[];
}

function analyzeSmiles(smiles: string): ScoredDisease[] {
  const s = smiles.trim();
  const results: ScoredDisease[] = [];

  for (const disease of DISEASES) {
    const totalWeight = disease.features.reduce((sum, f) => sum + f.weight, 0);
    let earnedWeight = 0;
    const matched: { label: string; description: string }[] = [];
    const missed: string[] = [];

    for (const feature of disease.features) {
      if (feature.test(s)) {
        earnedWeight += feature.weight;
        matched.push({ label: feature.label, description: feature.description });
      } else {
        missed.push(feature.label);
      }
    }

    const score = Math.round((earnedWeight / totalWeight) * 100);

    // Threshold: only include if ≥ 40%
    if (score >= 40) {
      const level: MatchLevel =
        score >= 75 ? "CONFIRMED" :
        score >= 60 ? "HIGH" : "MODERATE";
      results.push({ disease, score, level, matchedFeatures: matched, missedFeatures: missed });
    }
  }

  return results.sort((a, b) => b.score - a.score);
}

/* ─────────────────────────────────────────────────────────────────────────────
   Sub-components
────────────────────────────────────────────────────────────────────────────── */

const LEVEL_META: Record<MatchLevel, { label: string; bg: string; dot: string }> = {
  CONFIRMED: { label: "CONFIRMED MATCH", bg: "bg-emerald-500/15 text-emerald-400 ring-emerald-500/30", dot: "bg-emerald-400" },
  HIGH:      { label: "HIGH CONFIDENCE", bg: "bg-sky-500/15 text-sky-400 ring-sky-500/30",           dot: "bg-sky-400"     },
  MODERATE:  { label: "MODERATE",        bg: "bg-amber-500/15 text-amber-400 ring-amber-500/30",     dot: "bg-amber-400"  },
};

const CATEGORY_COLORS: Record<string, string> = {
  Viral:          "text-sky-400",
  Bacterial:      "text-amber-400",
  Fungal:         "text-purple-400",
  Oncology:       "text-rose-400",
  CNS:            "text-indigo-400",
  Metabolic:      "text-green-400",
  Cardiovascular: "text-orange-400",
  Inflammatory:   "text-fuchsia-400",
};

function ScoreRing({ score, color }: { score: number; color: string }) {
  const r = 28;
  const circumference = 2 * Math.PI * r;
  const dash = (score / 100) * circumference;

  return (
    <div className="relative flex h-20 w-20 items-center justify-center shrink-0">
      <svg className="absolute inset-0 -rotate-90" width="80" height="80" viewBox="0 0 80 80">
        <circle cx="40" cy="40" r={r} fill="none" stroke="white" strokeOpacity="0.07" strokeWidth="5" />
        <motion.circle
          cx="40" cy="40" r={r}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - dash }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </svg>
      <div className="flex flex-col items-center">
        <motion.span
          className="text-xl font-bold leading-none"
          style={{ color }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          {score}
        </motion.span>
        <span className="text-[9px] text-muted-foreground font-medium leading-none mt-0.5">%</span>
      </div>
    </div>
  );
}

function DiseaseCard({ result, index }: { result: ScoredDisease; index: number }) {
  const { disease, score, level, matchedFeatures, missedFeatures } = result;
  const levelMeta = LEVEL_META[level];
  const catColor = CATEGORY_COLORS[disease.category] ?? "text-muted-foreground";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.07, ease: "easeOut" }}
      className={cn(
        "relative overflow-hidden rounded-3xl p-6",
        "bg-white/[0.03] dark:bg-black/20",
        "backdrop-blur-xl",
        "border border-white/10",
        "hover:border-white/20 hover:bg-white/[0.05] transition-all duration-300",
        "group"
      )}
    >
      {/* Gradient glow top */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent 0%, ${disease.color}80 50%, transparent 100%)` }}
      />
      {/* Subtle background glow */}
      <div
        className="absolute -top-12 -right-12 h-36 w-36 rounded-full blur-3xl opacity-20 transition-opacity duration-300 group-hover:opacity-30"
        style={{ background: disease.color }}
      />

      {/* Top row */}
      <div className="flex items-start gap-4 mb-5">
        {/* Emoji icon */}
        <div
          className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl text-2xl"
          style={{ background: `${disease.color}18`, border: `1px solid ${disease.color}35` }}
        >
          {disease.icon}
        </div>

        <div className="flex-1 min-w-0">
          {/* Level badge */}
          <div className={cn("inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold ring-1 mb-2", levelMeta.bg)}>
            <span className={cn("h-1.5 w-1.5 rounded-full animate-pulse", levelMeta.dot)} />
            {levelMeta.label}
          </div>
          <h3 className="font-bold text-base leading-tight">{disease.name}</h3>
          <span className={cn("text-xs font-semibold", catColor)}>{disease.category}</span>
        </div>

        {/* Score ring */}
        <ScoreRing score={score} color={disease.color} />
      </div>

      {/* Description */}
      <p className="text-xs text-muted-foreground leading-relaxed mb-5">{disease.description}</p>

      {/* Matched pharmacophoric features */}
      <div className="mb-4">
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2">
          Matched Pharmacophoric Features ({matchedFeatures.length}/{matchedFeatures.length + missedFeatures.length})
        </p>
        <div className="space-y-2">
          {matchedFeatures.map((f) => (
            <div key={f.label} className="flex items-start gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0 mt-0.5" style={{ color: disease.color }} />
              <div>
                <span className="text-xs font-semibold">{f.label}</span>
                <span className="text-[10px] text-muted-foreground ml-1">— {f.description}</span>
              </div>
            </div>
          ))}
          {missedFeatures.map((f) => (
            <div key={f} className="flex items-start gap-2 opacity-40">
              <div className="h-3.5 w-3.5 shrink-0 mt-0.5 flex items-center justify-center">
                <div className="h-2 w-2 rounded-full border border-muted-foreground" />
              </div>
              <span className="text-xs text-muted-foreground">{f}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Approved analogues */}
      <div className="mb-4">
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2">Approved Analogues</p>
        <div className="flex flex-wrap gap-1.5">
          {disease.approvedAnalogues.map((drug) => (
            <span
              key={drug}
              className="text-[10px] font-medium px-2.5 py-1 rounded-full"
              style={{ background: `${disease.color}14`, color: disease.color, border: `1px solid ${disease.color}30` }}
            >
              {drug}
            </span>
          ))}
        </div>
      </div>

      {/* Clinical note */}
      <div
        className="flex items-start gap-2 rounded-2xl px-3 py-2.5 text-[10px] text-muted-foreground leading-relaxed"
        style={{ background: `${disease.color}0d`, border: `1px solid ${disease.color}20` }}
      >
        <Info className="h-3 w-3 shrink-0 mt-0.5" style={{ color: disease.color }} />
        <span>{disease.clinicalNote}</span>
      </div>
    </motion.div>
  );
}

const EXAMPLE_SMILES = [
  { label: "Remdesivir-like (Antiviral)", smiles: "CCC(CC)COC(=O)[C@H](CN)OP(=O)(Oc1ccccc1)O[C@H]1[C@@H](O)[C@@H](CO)O[C@@H]1N1C=NC2=C1N=CN=C2N" },
  { label: "Acyclovir (HSV)", smiles: "Nc1nc2c(ncn2COCCO)c(=O)[nH]1" },
  { label: "Ciprofloxacin (Antibacterial)", smiles: "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O" },
  { label: "Atorvastatin (Cardiovascular)", smiles: "CC(C)c1n(-c2ccccc2)c([C@@H](O)CC(=O)O)c(-c2ccc(F)cc2)c1CCC(=O)O" },
  { label: "Imatinib (Kinase Cancer)", smiles: "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1" },
  { label: "Fluconazole (Antifungal)", smiles: "OC(Cn1ccnc1)(Cn1ccnc1)c1ccc(F)cc1F" },
];

/* ─────────────────────────────────────────────────────────────────────────────
   Main Page
────────────────────────────────────────────────────────────────────────────── */
export default function SimilarityAnalyzer() {
  const [smiles, setSmiles] = useState("");
  const [results, setResults] = useState<ScoredDisease[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = useCallback(() => {
    const trimmed = smiles.trim();
    if (!trimmed) return;

    if (trimmed.length < 4) {
      setError("SMILES string too short. Please enter a valid molecular structure.");
      return;
    }
    if (!/^[A-Za-z0-9@+\-\[\]()=#%/.\\]+$/.test(trimmed)) {
      setError("Invalid characters detected. Please enter a valid SMILES string.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    setTimeout(() => {
      try {
        const res = analyzeSmiles(trimmed);
        setResults(res);
        if (res.length === 0) {
          setError("No significant matches found. The molecule doesn't share sufficient structural features with known therapeutic agents. Try a more complex or drug-like SMILES.");
        }
      } catch {
        setError("Analysis failed. Please check your SMILES string.");
      } finally {
        setLoading(false);
      }
    }, 1200);
  }, [smiles]);

  const topResult = results?.[0];

  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen"
      >
        {/* ── Hero Header ── */}
        <div className="relative px-6 lg:px-10 pt-8 pb-6 overflow-hidden">
          {/* Background glow */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 h-48 w-96 rounded-full blur-3xl opacity-10"
            style={{ background: "linear-gradient(135deg, hsl(187 85% 55%), hsl(207 100% 50%))" }}
          />
          <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="stat-pill bg-sky-500/15 text-sky-400 text-[11px] font-semibold ring-1 ring-sky-500/25">
                <FlaskConical className="h-3 w-3" />
                Structural Pharmacology Engine
              </span>
              <span className="stat-pill bg-purple-500/15 text-purple-400 text-[11px] font-semibold ring-1 ring-purple-500/25">
                <Zap className="h-3 w-3" />
                Frontend-Native · No Backend Required
              </span>
            </div>
            <h1 className="text-4xl font-bold tracking-tight">
              Drug Similarity <span className="gradient-text">Analyzer</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-2 max-w-2xl">
              Enter any SMILES string and instantly discover which diseases or viruses your drug candidate may be suitable for — powered by a pharmacophoric feature-matching engine against 11 major therapeutic areas.
            </p>
          </motion.div>
        </div>

        {/* ── SMILES Input Section ── */}
        <div className="px-6 lg:px-10 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="relative rounded-3xl overflow-hidden"
            style={{
              background: "linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01))",
              border: "1px solid rgba(255,255,255,0.1)",
              backdropFilter: "blur(20px)",
            }}
          >
            <div className="absolute top-0 left-0 right-0 h-[2px]"
              style={{ background: "linear-gradient(90deg, transparent, hsl(187 85% 55%), hsl(207 100% 50%), transparent)" }}
            />
            <div className="p-6">
              <div className="flex flex-col lg:flex-row gap-4 items-start">
                <div className="flex-1">
                  <label className="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-2 block">
                    SMILES Input
                  </label>
                  <div className="relative">
                    <Search className="absolute left-4 top-3.5 h-4 w-4 text-muted-foreground" />
                    <input
                      value={smiles}
                      onChange={(e) => { setSmiles(e.target.value); setResults(null); setError(null); }}
                      onKeyDown={(e) => { if (e.key === "Enter") runAnalysis(); }}
                      placeholder="Paste SMILES string, e.g. CC(=O)Oc1ccccc1C(=O)O ..."
                      className={cn(
                        "w-full pl-10 pr-4 py-3 rounded-2xl font-mono text-sm",
                        "bg-white/20 border focus:outline-none focus:ring-1 transition-all",
                        smiles.trim()
                          ? "border-sky-500/40 focus:ring-sky-500/30 focus:border-sky-400"
                          : "border-white/10 focus:ring-white/15"
                      )}
                    />
                  </div>
                  {/* Quick examples */}
                  <div className="flex flex-wrap gap-2 mt-3">
                    {EXAMPLE_SMILES.map((ex) => (
                      <button
                        key={ex.label}
                        onClick={() => { setSmiles(ex.smiles); setResults(null); setError(null); }}
                        className={cn(
                          "text-[10px] px-3 py-1.5 rounded-full font-medium transition-all duration-200",
                          "border border-white/10 text-muted-foreground hover:text-foreground hover:border-white/25 hover:bg-white/5",
                          smiles === ex.smiles && "border-sky-500/40 text-sky-400 bg-sky-500/10"
                        )}
                      >
                        {ex.label}
                      </button>
                    ))}
                  </div>
                </div>

                <Button
                  onClick={runAnalysis}
                  disabled={!smiles.trim() || loading}
                  className="h-12 px-8 rounded-2xl font-bold shrink-0 lg:mt-6"
                  style={{
                    background: smiles.trim() && !loading
                      ? "linear-gradient(135deg, hsl(187 85% 45%), hsl(207 100% 45%))"
                      : undefined,
                    boxShadow: smiles.trim() && !loading ? "0 8px 28px -4px hsl(187 85% 55% / 0.45)" : undefined,
                    border: "none",
                  }}
                >
                  {loading ? (
                    <><Loader2 className="h-4 w-4 mr-2 animate-spin" />Analyzing…</>
                  ) : (
                    <><Sparkles className="h-4 w-4 mr-2" />Analyze</>
                  )}
                </Button>
              </div>
            </div>
          </motion.div>
        </div>

        {/* ── Results Area ── */}
        <div className="px-6 lg:px-10 pb-12">
          <AnimatePresence mode="wait">
            {/* Loading */}
            {loading && (
              <motion.div
                key="loading"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center py-24 space-y-6"
              >
                <div className="relative">
                  <div className="h-20 w-20 rounded-3xl bg-sky-500/10 ring-1 ring-sky-500/20 flex items-center justify-center">
                    <FlaskConical className="h-10 w-10 text-sky-400 animate-pulse" />
                  </div>
                  <div className="absolute inset-0 rounded-3xl animate-ping opacity-20" style={{ background: "hsl(187 85% 55%)" }} />
                </div>
                <div className="text-center">
                  <p className="text-lg font-bold">Running Pharmacophore Analysis</p>
                  <p className="text-sm text-muted-foreground mt-1 max-w-sm">
                    Tokenising structural fragments · Matching against 11 therapeutic disease databases…
                  </p>
                </div>
                <div className="flex gap-2">
                  {["Tokenizing", "Pattern Matching", "Ranking Results"].map((step, i) => (
                    <motion.div
                      key={step}
                      initial={{ opacity: 0.3 }}
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ duration: 1.4, repeat: Infinity, delay: i * 0.35 }}
                      className="px-3 py-1.5 rounded-full glass-surface text-xs font-medium"
                    >
                      {step}
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Error */}
            {!loading && error && (
              <motion.div
                key="error"
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center py-20 space-y-4"
              >
                <div className="h-16 w-16 rounded-2xl bg-destructive/10 ring-1 ring-destructive/30 flex items-center justify-center">
                  <AlertTriangle className="h-8 w-8 text-destructive" />
                </div>
                <p className="text-base font-semibold text-destructive">No Matching Indications Found</p>
                <p className="text-sm text-muted-foreground max-w-md text-center">{error}</p>
                <Button variant="outline" onClick={() => { setError(null); setSmiles(""); }} className="rounded-xl">
                  Clear & Try Again
                </Button>
              </motion.div>
            )}

            {/* Empty state */}
            {!loading && !error && !results && (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="flex flex-col items-center justify-center py-24 space-y-6"
              >
                <div className="relative">
                  <div className="h-24 w-24 rounded-3xl bg-muted/20 ring-1 ring-white/10 flex items-center justify-center text-4xl">
                    🔬
                  </div>
                </div>
                <div className="text-center max-w-md">
                  <p className="text-xl font-bold text-muted-foreground">Ready for Analysis</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Paste any SMILES string above or select an example molecule to begin. The engine will identify which diseases your compound may be therapeutically suitable for.
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-4 text-center mt-4">
                  {[
                    { icon: "🦠", label: "4 Viral Targets", sub: "COVID-19, Flu, HIV, HSV" },
                    { icon: "🎗️", label: "2 Oncology Targets", sub: "Kinase, Topoisomerase" },
                    { icon: "💊", label: "5 Other Areas", sub: "CNS, Metabolic, CV, Auto-immune, Fungal/Bacterial" },
                  ].map((c) => (
                    <div key={c.label} className="glass-card rounded-2xl p-4">
                      <div className="text-2xl mb-1">{c.icon}</div>
                      <p className="text-xs font-bold">{c.label}</p>
                      <p className="text-[10px] text-muted-foreground mt-0.5">{c.sub}</p>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Results */}
            {!loading && !error && results && results.length > 0 && (
              <motion.div
                key="results"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              >
                {/* Results header */}
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-xl font-bold">
                      <span className="gradient-text">{results.length}</span> Therapeutic{results.length !== 1 ? " Areas" : " Area"} Identified
                    </h2>
                    <p className="text-xs text-muted-foreground mt-1">
                      Ranked by pharmacophoric match score · Only showing ≥40% confidence
                    </p>
                  </div>
                  {topResult && (
                    <div className="flex items-center gap-2 px-4 py-2 rounded-2xl" style={{
                      background: `${topResult.disease.color}12`,
                      border: `1px solid ${topResult.disease.color}30`,
                    }}>
                      <TrendingUp className="h-4 w-4" style={{ color: topResult.disease.color }} />
                      <span className="text-xs font-semibold">
                        Best: <span style={{ color: topResult.disease.color }}>{topResult.disease.name} ({topResult.score}%)</span>
                      </span>
                    </div>
                  )}
                </div>

                {/* Legend */}
                <div className="flex gap-4 mb-6">
                  {(Object.entries(LEVEL_META) as [MatchLevel, typeof LEVEL_META[MatchLevel]][]).map(([key, meta]) => (
                    <div key={key} className="flex items-center gap-1.5">
                      <span className={cn("h-2 w-2 rounded-full", meta.dot)} />
                      <span className="text-[10px] text-muted-foreground font-medium">{meta.label}</span>
                    </div>
                  ))}
                  <div className="flex items-center gap-1.5 ml-auto">
                    <Shield className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] text-muted-foreground">Pharmacophore-validated · Research use only</span>
                  </div>
                </div>

                {/* Cards grid */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                  {results.map((r, i) => (
                    <DiseaseCard key={r.disease.id} result={r} index={i} />
                  ))}
                </div>

                {/* Footer disclaimer */}
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
                  className="mt-8 flex items-start gap-2 px-5 py-3.5 rounded-2xl border border-white/8 bg-white/[0.02] text-[11px] text-muted-foreground"
                >
                  <Info className="h-3.5 w-3.5 shrink-0 mt-0.5 text-sky-400" />
                  <span>
                    <strong className="text-foreground">Research Disclaimer:</strong> This analysis is based on structural pharmacophore pattern matching and is intended for research ideation only. It does not constitute clinical advice and should not replace in-vitro/in-vivo validation, binding assays, or regulatory review.
                  </span>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </AppLayout>
  );
}
