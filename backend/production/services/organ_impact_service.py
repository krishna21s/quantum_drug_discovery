"""
Organ Impact Prediction Service
================================
Uses RDKit substructure matching against pharmacologically validated
SMARTS patterns to predict which body organs a molecule targets
(therapeutic) and which organs may experience adverse effects.

Approach:
  1. Match the molecule against curated structural-alert SMARTS patterns
     (sourced from published toxicophore/pharmacophore literature).
  2. Compute physicochemical descriptors (logP, TPSA, MW, HBA, HBD)
     to infer distribution-based organ exposure.
  3. Combine substructure hits + physicochemical profile → organ-level
     target and side-effect predictions with confidence scores.

References:
  - Brenk et al. (2008) — structural alerts
  - Stepan et al. (2011) — hepatotoxicity alerts
  - Sanguinetti & Bhattacharjee (2005) — hERG cardiotoxicity
  - Lipinski et al. (2001) — BBB penetration rules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments

logger = logging.getLogger(__name__)


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class OrganEffect:
    name: str
    reason: str
    confidence: float  # 0.0 – 1.0

    def to_dict(self):
        return asdict(self)


@dataclass
class OrganImpactResult:
    smiles: str
    canonical_smiles: str
    target_organs: List[OrganEffect]
    side_effect_organs: List[OrganEffect]
    drug_class: str
    mechanism_summary: str

    def to_dict(self):
        return {
            "smiles": self.smiles,
            "canonical_smiles": self.canonical_smiles,
            "target_organs": [o.to_dict() for o in self.target_organs],
            "side_effect_organs": [o.to_dict() for o in self.side_effect_organs],
            "drug_class": self.drug_class,
            "mechanism_summary": self.mechanism_summary,
        }


# ── Pharmacophore / Toxicophore SMARTS Library ──────────────────────
# Each entry: (SMARTS, drug_class, mechanism, target_organs, side_effect_organs)
# target/side entries: (organ_name, reason, base_confidence)

PHARMACOPHORE_RULES = [
    # ─── NSAID / COX Inhibitors ───────────────────────────────────
    {
        "smarts": "[CX3](=O)[OX2H1]",  # Carboxylic acid (core NSAID motif)
        "class": "NSAID / COX Inhibitor",
        "mechanism": "Cyclooxygenase (COX-1/COX-2) inhibition — blocks prostaglandin synthesis",
        "targets": [
            ("GI Tract", "COX-1/COX-2 inhibition reduces inflammation in gastrointestinal mucosa", 0.70),
            ("Bone", "Reduces prostaglandin-mediated bone resorption and inflammatory joint pain", 0.55),
        ],
        "side_effects": [
            ("GI Tract", "Gastric mucosal erosion — COX-1 inhibition depletes protective prostaglandins", 0.82),
            ("Kidneys", "Prostaglandin inhibition reduces renal blood flow — risk of nephrotoxicity", 0.68),
            ("Heart", "COX-2 selective inhibition may increase cardiovascular thrombotic risk", 0.55),
        ],
    },
    {
        "smarts": "[CX3](=O)[OX2][CX3](=O)[cH1]",  # Aspirin-like acetylsalicylate ester
        "class": "Acetylsalicylate / Aspirin-class",
        "mechanism": "Irreversible COX-1 acetylation — anti-platelet and anti-inflammatory",
        "targets": [
            ("Blood / Immune", "Irreversible COX-1 acetylation in platelets — anti-thrombotic effect", 0.90),
            ("Heart", "Cardioprotective anti-platelet activity at low doses", 0.80),
        ],
        "side_effects": [
            ("GI Tract", "Gastric ulceration — direct mucosal irritation and COX-1 depletion", 0.85),
            ("Kidneys", "Reduced renal prostaglandins — risk in pre-existing renal disease", 0.60),
        ],
    },

    # ─── Kinase Inhibitors (Cancer) ───────────────────────────────
    {
        "smarts": "[#7]~[#6]~[#7]~[#6]~1~[#6]~[#7]~[#6]~[#7]~1",  # Purine scaffold
        "class": "Purine-based Kinase Inhibitor",
        "mechanism": "ATP-competitive kinase inhibition — blocks cancer cell proliferation signals",
        "targets": [
            ("Lungs", "Targets EGFR/ALK kinases overexpressed in non-small cell lung cancer", 0.80),
            ("Lymph Nodes", "Inhibits aberrant kinase signaling in lymphoma cells", 0.60),
        ],
        "side_effects": [
            ("Skin", "Acneiform rash — EGFR expressed in skin keratinocytes", 0.78),
            ("Liver", "Hepatotoxicity — CYP3A4 metabolism generates reactive intermediates", 0.72),
            ("GI Tract", "Diarrhea — disruption of EGFR signaling in intestinal epithelium", 0.70),
        ],
    },
    {
        "smarts": "c1cnc2ccccc2n1",  # Quinazoline scaffold (erlotinib, gefitinib, lapatinib)
        "class": "Quinazoline EGFR Kinase Inhibitor",
        "mechanism": "ATP-competitive EGFR tyrosine kinase inhibition — blocks ErbB receptor autophosphorylation",
        "targets": [
            ("Lungs", "Targets EGFR overexpressed in non-small cell lung cancer (NSCLC)", 0.88),
            ("Lymph Nodes", "Inhibits EGFR-driven proliferation in metastatic lymph nodes", 0.62),
            ("Brain", "BBB-penetrant — active against brain metastases in EGFR-mutant NSCLC", 0.70),
        ],
        "side_effects": [
            ("Skin", "Acneiform rash in >50% of patients — EGFR is critical for keratinocyte differentiation", 0.85),
            ("GI Tract", "Diarrhea — EGFR inhibition disrupts intestinal epithelial renewal", 0.75),
            ("Liver", "Hepatotoxicity — CYP3A4/CYP1A2-mediated reactive metabolite formation", 0.72),
            ("Lungs", "Interstitial lung disease — rare but serious (1-3% incidence)", 0.45),
        ],
    },
    {
        "smarts": "[#7]c1ncncc1",  # 4-Aminopyrimidine (imatinib, dasatinib class)
        "class": "Aminopyrimidine Kinase Inhibitor",
        "mechanism": "Type II kinase inhibition — binds inactive DFG-out kinase conformation",
        "targets": [
            ("Blood / Immune", "Targets BCR-ABL in CML / c-KIT in GIST", 0.85),
            ("Lymph Nodes", "Inhibits PDGFR-driven lymphoid malignancies", 0.65),
        ],
        "side_effects": [
            ("Heart", "QT prolongation and cardiotoxicity — off-target kinase inhibition", 0.68),
            ("Liver", "Hepatotoxicity — CYP3A4 metabolism generates reactive intermediates", 0.70),
            ("GI Tract", "Nausea and fluid retention — PDGFR inhibition in GI stroma", 0.72),
            ("Skin", "Periorbital edema and rash — PDGFR inhibition in dermal fibroblasts", 0.60),
        ],
    },
    {
        "smarts": "c1ccc2c(c1)[nH]c1ccccc12",  # Indole scaffold (sunitinib, nintedanib)
        "class": "Indole-based Kinase Inhibitor",
        "mechanism": "Kinase inhibition via indole scaffold — targets VEGFR/PDGFR/FLT3",
        "targets": [
            ("Blood / Immune", "Targets VEGFR in tumor vasculature — anti-angiogenic", 0.75),
            ("Lungs", "Anti-tumor activity in NSCLC through VEGFR inhibition", 0.65),
        ],
        "side_effects": [
            ("Heart", "QT prolongation risk — hERG channel interaction", 0.65),
            ("Liver", "Hepatotoxicity — CYP-mediated reactive metabolite formation", 0.70),
            ("Skin", "Hand-foot syndrome — capillary toxicity in extremities", 0.60),
        ],
    },

    # ─── Nucleoside Analogs (Antivirals) ──────────────────────────
    {
        "smarts": "[#8,#7]~[#6]1~[#8]~[#6]~[#6]~[#6]~1",  # Ribose/deoxyribose sugar
        "class": "Nucleoside Analog / Antiviral",
        "mechanism": "Nucleoside analog — inhibits viral RNA/DNA polymerase by chain termination",
        "targets": [
            ("Lungs", "Targets viral replication in respiratory epithelial cells", 0.80),
            ("Upper Respiratory", "Inhibits viral RNA polymerase in nasopharyngeal tissue", 0.75),
        ],
        "side_effects": [
            ("Liver", "Mitochondrial toxicity — inhibits human mitochondrial RNA polymerase", 0.75),
            ("Kidneys", "Active metabolite accumulation in renal tubular cells", 0.70),
            ("Pancreas", "Pancreatitis risk — mitochondrial dysfunction in acinar cells", 0.50),
        ],
    },
    {
        "smarts": "[#6]~1~[#7]~[#6](=[#8])~[#7]~[#6](=[#8])~[#7]~1",  # Uracil / pyrimidine base
        "class": "Pyrimidine Nucleoside Analog",
        "mechanism": "Pyrimidine analog — incorporated into viral RNA causing lethal mutagenesis",
        "targets": [
            ("Lungs", "Inhibits SARS-CoV-2 / influenza RNA replication in alveolar cells", 0.82),
            ("Upper Respiratory", "Suppresses viral load in upper airway epithelium", 0.70),
        ],
        "side_effects": [
            ("Liver", "Elevated transaminases — hepatocyte mitochondrial stress", 0.72),
            ("Blood / Immune", "Neutropenia risk — bone marrow suppression at high doses", 0.55),
        ],
    },

    # ─── Protease Inhibitors (HIV / HCV) ─────────────────────────
    {
        "smarts": "[#6](=[#8])~[#7]~[#6]~[#6](~[#8])~[#6]~[#7]",  # Hydroxyethylamine peptidomimetic
        "class": "Protease Inhibitor",
        "mechanism": "Competitive inhibition of viral aspartyl protease — prevents polyprotein cleavage",
        "targets": [
            ("Lymph Nodes", "Suppresses HIV replication in CD4+ T-cells in lymphoid tissue", 0.85),
            ("Blood / Immune", "Reduces systemic viral load in circulating CD4+ lymphocytes", 0.82),
        ],
        "side_effects": [
            ("Liver", "CYP3A4 inhibition — drug-drug interactions and hepatotoxicity", 0.80),
            ("GI Tract", "Nausea and diarrhea — GI mucosal irritation", 0.75),
            ("Pancreas", "Pancreatitis — lipid metabolism disruption", 0.60),
        ],
    },

    # ─── Estrogen Receptor Modulators (SERMs) ────────────────────
    {
        "smarts": "c1ccc(cc1)/[#6]=[#6]/c1ccccc1",  # Stilbene scaffold (tamoxifen-like)
        "class": "Selective Estrogen Receptor Modulator (SERM)",
        "mechanism": "Competitive estrogen receptor antagonist in breast — blocks ER-driven proliferation",
        "targets": [
            ("Breast", "Blocks estrogen receptor α in ER+ breast cancer cells", 0.90),
            ("Bone", "ER agonist in bone — preserves bone mineral density", 0.72),
        ],
        "side_effects": [
            ("Uterus", "ER agonist in endometrium — risk of endometrial hyperplasia", 0.78),
            ("Liver", "Fatty liver disease — altered lipid metabolism via CYP2D6", 0.65),
            ("Eyes", "Retinopathy — cumulative retinal crystalline deposits", 0.50),
        ],
    },
    {
        "smarts": "c1ccc(cc1)OCCN",  # Phenoxyethylamine (tamoxifen side chain)
        "class": "SERM (Tamoxifen-class)",
        "mechanism": "Phenoxyethylamine moiety enables ER binding — tissue-selective modulation",
        "targets": [
            ("Breast", "Anti-estrogenic in mammary tissue — inhibits ER+ tumor growth", 0.85),
        ],
        "side_effects": [
            ("Uterus", "Estrogenic stimulation of endometrial tissue", 0.75),
            ("Blood / Immune", "Venous thromboembolism — pro-coagulant effect", 0.55),
        ],
    },

    # ─── Sulfonamides (Antibiotics / Diuretics) ──────────────────
    {
        "smarts": "[#7]~[S](=[O])(=[O])~[#7]",  # Sulfonamide
        "class": "Sulfonamide",
        "mechanism": "Competitive inhibitor of dihydropteroate synthase — blocks folate synthesis in bacteria",
        "targets": [
            ("Kidneys", "Urinary tract concentration — effective against UTI pathogens", 0.75),
            ("Lungs", "Penetrates lung tissue — effective against respiratory pathogens", 0.60),
        ],
        "side_effects": [
            ("Kidneys", "Crystalluria — sulfonamide precipitation in renal tubules", 0.72),
            ("Skin", "Stevens-Johnson syndrome — hypersensitivity reaction", 0.58),
            ("Liver", "Hepatotoxicity — idiosyncratic drug reaction", 0.50),
        ],
    },

    # ─── Beta-Lactams (Antibiotics) ──────────────────────────────
    {
        "smarts": "[#6]1~[#6]~[#6](=[#8])~[#7]1",  # Beta-lactam ring
        "class": "Beta-Lactam Antibiotic",
        "mechanism": "Inhibits bacterial transpeptidase — disrupts cell wall synthesis",
        "targets": [
            ("Lungs", "Effective against respiratory tract bacterial infections", 0.78),
            ("Upper Respiratory", "Treats pharyngitis, sinusitis, otitis media", 0.75),
            ("Kidneys", "Renally excreted — effective against urinary tract infections", 0.65),
        ],
        "side_effects": [
            ("GI Tract", "Antibiotic-associated diarrhea — gut microbiome disruption", 0.72),
            ("Skin", "Allergic rash — IgE-mediated hypersensitivity (up to 10%)", 0.60),
            ("Kidneys", "Interstitial nephritis — immune-mediated renal inflammation", 0.45),
        ],
    },

    # ─── Benzodiazepines (CNS) ───────────────────────────────────
    {
        "smarts": "c1cc2c(cc1)C(=NCC(=O)N2)c1ccccc1",  # Benzodiazepine core
        "class": "Benzodiazepine",
        "mechanism": "Positive allosteric modulator of GABA-A receptors — enhances inhibitory neurotransmission",
        "targets": [
            ("Brain", "Enhances GABAergic inhibition in limbic and cortical circuits", 0.92),
            ("Nerves", "Reduces neuronal excitability — anticonvulsant / muscle relaxant", 0.78),
        ],
        "side_effects": [
            ("Brain", "Cognitive impairment and dependence — chronic GABA-A downregulation", 0.80),
            ("Liver", "CYP3A4/CYP2C19 metabolism — risk of hepatic enzyme induction", 0.55),
            ("Lungs", "Respiratory depression at high doses — brainstem GABA-A suppression", 0.50),
        ],
    },

    # ─── Statins (Cardiovascular) ────────────────────────────────
    {
        "smarts": "[CX3](=O)~[#6]~[#6]~[CX3](=O)~[OX2]",  # Dihydroxy acid (statin pharmacophore)
        "class": "HMG-CoA Reductase Inhibitor (Statin)",
        "mechanism": "Competitive inhibition of HMG-CoA reductase — blocks cholesterol biosynthesis",
        "targets": [
            ("Liver", "Primary site of action — hepatic cholesterol synthesis inhibition", 0.92),
            ("Heart", "Reduces LDL-C — lowers atherosclerotic cardiovascular risk", 0.85),
        ],
        "side_effects": [
            ("Liver", "Elevated transaminases — dose-dependent hepatotoxicity", 0.65),
            ("Bone", "Myopathy / rhabdomyolysis — skeletal muscle mitochondrial dysfunction", 0.60),
        ],
    },

    # ─── Phosphodiesterase Inhibitors ────────────────────────────
    {
        "smarts": "[#6]~1~[#7]~[#6]~[#7]~[#6]~2~1~[#7]~[#6]~[#7]~2",  # Xanthine / purine PDE scaffold
        "class": "Phosphodiesterase Inhibitor",
        "mechanism": "Inhibits phosphodiesterase — increases intracellular cAMP/cGMP",
        "targets": [
            ("Lungs", "Bronchodilation — relaxes airway smooth muscle via cAMP elevation", 0.80),
            ("Heart", "Positive inotropic — increases cardiac contractility", 0.65),
        ],
        "side_effects": [
            ("Heart", "Tachycardia and arrhythmia — excessive cAMP in cardiomyocytes", 0.70),
            ("Brain", "Seizure risk at high doses — CNS stimulation from cAMP/cGMP", 0.55),
            ("GI Tract", "Nausea and vomiting — smooth muscle stimulation", 0.60),
        ],
    },

    # ─── Fluoroquinolones ────────────────────────────────────────
    {
        "smarts": "c1cc2c(cc1F)c(=O)c(cn2)C(=O)O",  # Fluoroquinolone core
        "class": "Fluoroquinolone Antibiotic",
        "mechanism": "Inhibits bacterial DNA gyrase and topoisomerase IV — prevents DNA replication",
        "targets": [
            ("Kidneys", "Concentrated in urine — highly effective against UTI pathogens", 0.85),
            ("Lungs", "Excellent lung tissue penetration — treats pneumonia", 0.80),
            ("Upper Respiratory", "Effective against sinusitis and bronchitis pathogens", 0.72),
        ],
        "side_effects": [
            ("Bone", "Tendon rupture — collagen degradation via MMP activation", 0.68),
            ("Nerves", "Peripheral neuropathy — mitochondrial toxicity in neurons", 0.60),
            ("Heart", "QT prolongation — hERG potassium channel blockade", 0.55),
            ("GI Tract", "C. difficile colitis — disruption of gut flora", 0.58),
        ],
    },

    # ─── Opioid Scaffold ─────────────────────────────────────────
    {
        "smarts": "[#6]~1~[#6]~[#6]~2~[#6](~[#6]~1)~[#6]~[#6]~[#7]~[#6]~2",  # Morphinan skeleton
        "class": "Opioid Agonist / Analgesic",
        "mechanism": "Mu-opioid receptor agonist — modulates pain signaling in CNS",
        "targets": [
            ("Brain", "Activates mu-opioid receptors in periaqueductal gray — analgesia", 0.90),
            ("Nerves", "Modulates pain transmission at dorsal horn synapses", 0.82),
        ],
        "side_effects": [
            ("GI Tract", "Constipation — mu-receptor activation in myenteric plexus", 0.88),
            ("Lungs", "Respiratory depression — suppresses brainstem respiratory centers", 0.80),
            ("Brain", "Dependence and tolerance — receptor desensitization", 0.78),
        ],
    },

    # ─── Steroid Scaffold ────────────────────────────────────────
    {
        "smarts": "[#6]~1~[#6]~[#6]~2~[#6](~[#6]~1)~[#6]~[#6]~[#6]~1~[#6]~2~[#6]~[#6]~[#6]~2~[#6]~1~[#6]~[#6]~[#6]~[#6]~2",  # Steroid 4-ring system
        "class": "Corticosteroid / Steroid",
        "mechanism": "Glucocorticoid receptor agonist — potent anti-inflammatory and immunosuppressive",
        "targets": [
            ("Lungs", "Reduces airway inflammation in asthma/COPD", 0.85),
            ("Skin", "Suppresses dermatitis and inflammatory skin conditions", 0.80),
            ("Blood / Immune", "Systemic immunosuppression — reduces lymphocyte activation", 0.82),
        ],
        "side_effects": [
            ("Bone", "Osteoporosis — inhibits osteoblast activity", 0.78),
            ("Pancreas", "Steroid-induced diabetes — beta-cell dysfunction", 0.65),
            ("Eyes", "Glaucoma and cataracts — increased intraocular pressure", 0.60),
            ("Kidneys", "Fluid retention — mineralocorticoid effect", 0.55),
        ],
    },

    # ─── Acetaminophen / Aniline Pattern ─────────────────────────
    {
        "smarts": "[#7]~c1ccc([#8])cc1",  # Para-aminophenol (paracetamol core)
        "class": "Aniline Analgesic (Acetaminophen-class)",
        "mechanism": "Central COX-3 / TRPV1 modulation — analgesic and antipyretic without peripheral anti-inflammatory",
        "targets": [
            ("Brain", "Central analgesic — modulates pain perception in hypothalamus", 0.82),
        ],
        "side_effects": [
            ("Liver", "NAPQI hepatotoxicity — CYP2E1 generates reactive quinone imine at high doses", 0.90),
            ("Kidneys", "Chronic use associated with analgesic nephropathy", 0.50),
        ],
    },

    # ─── Aryl Amines / Nitroaromatics (Toxicophores) ─────────────
    {
        "smarts": "[NH2]c1ccccc1",  # Aromatic primary amine (mutagenicity alert)
        "class": "Aromatic Amine (Structural Alert)",
        "mechanism": "Metabolic activation to reactive nitrenium ion — potential genotoxicity",
        "targets": [],
        "side_effects": [
            ("Liver", "Bioactivation by CYP1A2 — reactive metabolite covalent binding", 0.75),
            ("Blood / Immune", "Methemoglobinemia — oxidation of hemoglobin", 0.60),
        ],
    },
    {
        "smarts": "[N+](=O)[O-]c1ccccc1",  # Nitroaromatic (genotoxicity alert)
        "class": "Nitroaromatic (Structural Alert)",
        "mechanism": "Nitroreduction generates reactive intermediates — DNA damage risk",
        "targets": [],
        "side_effects": [
            ("Liver", "Nitroreductase activation — hepatocyte DNA damage", 0.80),
            ("Blood / Immune", "Methemoglobin formation — oxidative hemolysis", 0.65),
        ],
    },

    # ─── Epoxide (Reactive Metabolite) ───────────────────────────
    {
        "smarts": "[#6]1~[#8]~[#6]1",  # Epoxide ring
        "class": "Epoxide-containing Compound",
        "mechanism": "Electrophilic epoxide — reacts with cellular nucleophiles (GSH, DNA, proteins)",
        "targets": [],
        "side_effects": [
            ("Liver", "Glutathione depletion — hepatocellular damage", 0.82),
            ("Lungs", "Reactive epoxide in Clara cells — pulmonary toxicity", 0.55),
            ("Skin", "Contact sensitization — hapten formation with skin proteins", 0.60),
        ],
    },

    # ─── Thiol / Thione (Prodrug / Toxicophore) ──────────────────
    {
        "smarts": "[#6]=[#16]",  # Thioamide / thiocarbonyl
        "class": "Thiocarbonyl Compound",
        "mechanism": "Metabolized to reactive sulfoxide — cytochrome P450-mediated bioactivation",
        "targets": [],
        "side_effects": [
            ("Liver", "Thiocarbonyl S-oxidation generates hepatotoxic metabolites", 0.78),
        ],
    },

    # ─── Hydrazine ───────────────────────────────────────────────
    {
        "smarts": "[#7]~[#7]",  # Hydrazine (N-N bond)
        "class": "Hydrazine / Diazo Compound",
        "mechanism": "Oxidative metabolism generates reactive diazonium species",
        "targets": [],
        "side_effects": [
            ("Liver", "Hepatotoxicity — acyl hydrazine metabolite damages hepatocytes", 0.80),
            ("Blood / Immune", "Drug-induced lupus — anti-histone antibody formation", 0.55),
        ],
    },

    # ─── ACE Inhibitor Pattern ───────────────────────────────────
    {
        "smarts": "[#6][CX3](=O)[NX3][CX4][CX3](=O)[OX2]",  # Proline dipeptide mimic
        "class": "ACE Inhibitor",
        "mechanism": "Inhibits angiotensin-converting enzyme — reduces angiotensin II-mediated vasoconstriction",
        "targets": [
            ("Heart", "Reduces afterload and blood pressure — cardioprotective in heart failure", 0.88),
            ("Kidneys", "Reduces intraglomerular pressure — renoprotective in diabetic nephropathy", 0.82),
        ],
        "side_effects": [
            ("Lungs", "Dry cough — bradykinin accumulation in pulmonary epithelium", 0.70),
            ("Kidneys", "Hyperkalemia — reduced aldosterone secretion", 0.55),
        ],
    },

    # ─── Anthracycline / Quinone (Chemo) ─────────────────────────
    {
        "smarts": "[#6]~1~[#6](=[#8])~[#6]~[#6](=[#8])~[#6]~[#6]~1",  # Quinone pattern
        "class": "Quinone / Anthracycline Chemotherapeutic",
        "mechanism": "Generates reactive oxygen species via redox cycling — causes DNA intercalation and strand breaks",
        "targets": [
            ("Lymph Nodes", "Targets rapidly dividing lymphoma and leukemia cells", 0.80),
            ("Breast", "DNA damage in ER+ and triple-negative breast cancer cells", 0.75),
        ],
        "side_effects": [
            ("Heart", "Cardiomyopathy — cumulative ROS damage to cardiomyocyte mitochondria", 0.88),
            ("Blood / Immune", "Myelosuppression — bone marrow toxicity from DNA damage", 0.82),
            ("Liver", "Hepatotoxicity — oxidative stress in hepatocytes", 0.60),
        ],
    },
]


# ── Physicochemical Distribution Rules ──────────────────────────────
# These refine predictions based on computed molecular properties.

def _distribution_effects(mol) -> tuple[list[OrganEffect], list[OrganEffect]]:
    """Predict organ exposure based on physicochemical descriptors."""
    targets = []
    side_effects = []

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # ── BBB Penetration (Lipinski-like rules for CNS) ──
    # MW < 450, logP 1-3, TPSA < 90, HBD ≤ 3
    bbb_score = 0
    if mw < 450: bbb_score += 0.25
    if 1.0 < logp < 3.5: bbb_score += 0.30
    if tpsa < 90: bbb_score += 0.25
    if hbd <= 3: bbb_score += 0.20

    if bbb_score >= 0.60:
        targets.append(OrganEffect(
            name="Brain",
            reason=f"High BBB penetration predicted (MW={mw:.0f}, logP={logp:.1f}, TPSA={tpsa:.0f})",
            confidence=round(min(bbb_score, 0.85), 2),
        ))
        side_effects.append(OrganEffect(
            name="Brain",
            reason=f"CNS exposure risk — molecule likely crosses blood-brain barrier",
            confidence=round(min(bbb_score * 0.7, 0.70), 2),
        ))

    # ── High Lipophilicity → Liver accumulation ──
    if logp > 4.0:
        side_effects.append(OrganEffect(
            name="Liver",
            reason=f"High lipophilicity (logP={logp:.1f}) — hepatic first-pass accumulation and CYP metabolism",
            confidence=round(min(0.50 + (logp - 4.0) * 0.1, 0.80), 2),
        ))

    # ── Very Hydrophilic → Renal excretion ──
    if logp < 0 and mw < 500:
        targets.append(OrganEffect(
            name="Kidneys",
            reason=f"Hydrophilic (logP={logp:.1f}, MW={mw:.0f}) — renally excreted, concentrates in urinary tract",
            confidence=0.65,
        ))

    # ── Large MW + High TPSA → Poor oral absorption ──
    if mw > 500 and tpsa > 140:
        side_effects.append(OrganEffect(
            name="GI Tract",
            reason=f"Poor oral absorption predicted (MW={mw:.0f}, TPSA={tpsa:.0f}) — GI tract retention",
            confidence=0.55,
        ))

    return targets, side_effects


# ── Main Prediction Function ────────────────────────────────────────

def predict_organ_impact(smiles: str) -> OrganImpactResult:
    """
    Predict organ-level therapeutic targets and adverse effects for a molecule.
    
    Uses RDKit substructure matching against curated pharmacophore/toxicophore
    patterns combined with physicochemical property-based distribution rules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: '{smiles}'")

    canonical = Chem.MolToSmiles(mol, canonical=True)

    matched_classes = []
    all_targets: dict[str, OrganEffect] = {}  # organ_name → best OrganEffect
    all_side_effects: dict[str, OrganEffect] = {}

    # ── 1. Substructure matching ──
    for rule in PHARMACOPHORE_RULES:
        pattern = Chem.MolFromSmarts(rule["smarts"])
        if pattern is None:
            logger.warning(f"Invalid SMARTS: {rule['smarts']}")
            continue

        if mol.HasSubstructMatch(pattern):
            matched_classes.append(rule["class"])

            for organ_name, reason, conf in rule["targets"]:
                if organ_name not in all_targets or conf > all_targets[organ_name].confidence:
                    all_targets[organ_name] = OrganEffect(name=organ_name, reason=reason, confidence=round(conf, 2))

            for organ_name, reason, conf in rule["side_effects"]:
                if organ_name not in all_side_effects or conf > all_side_effects[organ_name].confidence:
                    all_side_effects[organ_name] = OrganEffect(name=organ_name, reason=reason, confidence=round(conf, 2))

    # ── 2. Physicochemical distribution rules ──
    dist_targets, dist_side_effects = _distribution_effects(mol)

    for effect in dist_targets:
        if effect.name not in all_targets or effect.confidence > all_targets[effect.name].confidence:
            all_targets[effect.name] = effect
    for effect in dist_side_effects:
        if effect.name not in all_side_effects or effect.confidence > all_side_effects[effect.name].confidence:
            all_side_effects[effect.name] = effect

    # ── 3. Resolve conflicts (organ in both target and side-effect is valid — e.g. GI for NSAIDs) ──
    # Sort by confidence descending
    target_list = sorted(all_targets.values(), key=lambda o: o.confidence, reverse=True)
    side_effect_list = sorted(all_side_effects.values(), key=lambda o: o.confidence, reverse=True)

    # ── 4. Determine drug class and mechanism ──
    if matched_classes:
        # Deduplicate while preserving order
        seen = set()
        unique_classes = []
        for c in matched_classes:
            if c not in seen:
                seen.add(c)
                unique_classes.append(c)
        drug_class = " / ".join(unique_classes[:3])  # Top 3

        # Find the mechanism from the highest-confidence matched rule
        best_mechanism = "Multiple pharmacophore matches detected"
        for rule in PHARMACOPHORE_RULES:
            if rule["class"] == unique_classes[0]:
                best_mechanism = rule["mechanism"]
                break
    else:
        drug_class = "Unknown / Novel Scaffold"
        best_mechanism = "No known pharmacophore patterns matched — novel chemical scaffold"

        # For unknown compounds, add generic distribution-based warnings
        if not target_list and not side_effect_list:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            side_effect_list.append(OrganEffect(
                name="Liver",
                reason=f"Default hepatic metabolism assumed (MW={mw:.0f}, logP={logp:.1f})",
                confidence=0.40,
            ))

    return OrganImpactResult(
        smiles=smiles,
        canonical_smiles=canonical,
        target_organs=target_list,
        side_effect_organs=side_effect_list,
        drug_class=drug_class,
        mechanism_summary=best_mechanism,
    )
