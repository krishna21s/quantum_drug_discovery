import AppLayout from "@/components/AppLayout";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useCallback } from "react";
import {
  Search, Loader2, Sparkles, AlertTriangle, FlaskConical,
  CheckCircle2, Zap, Info, TrendingUp, Shield, Dna,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/* ─────────────────────────────────────────────────────────────────────────────
   Types mirroring backend response
────────────────────────────────────────────────────────────────────────────── */

interface MolecularProfile {
  molecular_weight: number;
  logp: number;
  tpsa: number;
  hba: number;
  hbd: number;
  rotatable_bonds: number;
  num_rings: number;
  formula: string;
}

interface DrugMatch {
  name: string;
  similarity: number;
  indication: string;
  category: string;
  target_protein: string;
  mechanism: string;
}

interface DiseaseResult {
  indication: string;
  category: string;
  max_similarity: number;
  avg_similarity: number;
  match_level: "CONFIRMED" | "HIGH" | "MODERATE";
  matched_drugs: DrugMatch[];
  top_target: string;
  top_mechanism: string;
}

interface SimilarityResponse {
  query_smiles: string;
  canonical_smiles: string;
  molecular_profile: MolecularProfile;
  diseases: DiseaseResult[];
  total_diseases_found: number;
  total_drugs_matched: number;
}

/* ─────────────────────────────────────────────────────────────────────────────
   Constants & UI helpers
────────────────────────────────────────────────────────────────────────────── */

const LEVEL_META: Record<string, { label: string; bg: string; dot: string }> = {
  CONFIRMED: { label: "CONFIRMED MATCH", bg: "bg-emerald-500/15 text-emerald-400 ring-emerald-500/30", dot: "bg-emerald-400" },
  HIGH:      { label: "HIGH CONFIDENCE", bg: "bg-sky-500/15 text-sky-400 ring-sky-500/30",           dot: "bg-sky-400"     },
  MODERATE:  { label: "MODERATE",        bg: "bg-amber-500/15 text-amber-400 ring-amber-500/30",     dot: "bg-amber-400"  },
};

const CATEGORY_META: Record<string, { color: string; icon: string; textColor: string }> = {
  Viral:          { color: "#38bdf8", icon: "🦠", textColor: "text-sky-400" },
  Bacterial:      { color: "#fbbf24", icon: "🧫", textColor: "text-amber-400" },
  Fungal:         { color: "#c084fc", icon: "🍄", textColor: "text-purple-400" },
  Oncology:       { color: "#f87171", icon: "🎗️", textColor: "text-rose-400" },
  CNS:            { color: "#818cf8", icon: "🧠", textColor: "text-indigo-400" },
  Metabolic:      { color: "#4ade80", icon: "💉", textColor: "text-green-400" },
  Cardiovascular: { color: "#fb923c", icon: "❤️", textColor: "text-orange-400" },
  Inflammatory:   { color: "#f0abfc", icon: "🦴", textColor: "text-fuchsia-400" },
  Respiratory:    { color: "#2dd4bf", icon: "🫁", textColor: "text-teal-400" },
  GI:             { color: "#a3e635", icon: "🫄", textColor: "text-lime-400" },
  Dermatology:    { color: "#fca5a5", icon: "🧴", textColor: "text-red-300" },
  Endocrine:      { color: "#93c5fd", icon: "⚗️", textColor: "text-blue-300" },
  Infectious:     { color: "#fdba74", icon: "🦟", textColor: "text-orange-300" },
  Immunology:     { color: "#d8b4fe", icon: "🛡️", textColor: "text-violet-300" },
};

const DEFAULT_CATEGORY = { color: "#94a3b8", icon: "💊", textColor: "text-slate-400" };

function getCatMeta(cat: string) {
  return CATEGORY_META[cat] ?? DEFAULT_CATEGORY;
}

const EXAMPLE_SMILES = [
  { label: "Imatinib (CML)", smiles: "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1" },
  { label: "Aspirin (Pain)", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O" },
  { label: "Fluoxetine (Depression)", smiles: "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1" },
  { label: "Metformin (Diabetes)", smiles: "CN(C)C(=N)NC(=N)N" },
  { label: "Atorvastatin (Cholesterol)", smiles: "CC(C)c1n(CC(O)CC(=O)O)c(c(-c2ccc(F)cc2)c1-c1ccccc1)C(=O)Nc1ccccc1" },
  { label: "Acyclovir (Herpes)", smiles: "Nc1nc2c(ncn2COCCO)c(=O)[nH]1" },
];

/* ─────────────────────────────────────────────────────────────────────────────
   Sub-components
────────────────────────────────────────────────────────────────────────────── */

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

function DiseaseCard({ result, index }: { result: DiseaseResult; index: number }) {
  const { indication, category, max_similarity, match_level, matched_drugs, top_target, top_mechanism } = result;
  const levelMeta = LEVEL_META[match_level] ?? LEVEL_META.MODERATE;
  const catMeta = getCatMeta(category);
  const score = Math.round(max_similarity * 100);

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
        style={{ background: `linear-gradient(90deg, transparent 0%, ${catMeta.color}80 50%, transparent 100%)` }}
      />
      <div
        className="absolute -top-12 -right-12 h-36 w-36 rounded-full blur-3xl opacity-20 transition-opacity duration-300 group-hover:opacity-30"
        style={{ background: catMeta.color }}
      />

      {/* Top row */}
      <div className="flex items-start gap-4 mb-5">
        <div
          className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl text-2xl"
          style={{ background: `${catMeta.color}18`, border: `1px solid ${catMeta.color}35` }}
        >
          {catMeta.icon}
        </div>

        <div className="flex-1 min-w-0">
          <div className={cn("inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold ring-1 mb-2", levelMeta.bg)}>
            <span className={cn("h-1.5 w-1.5 rounded-full animate-pulse", levelMeta.dot)} />
            {levelMeta.label}
          </div>
          <h3 className="font-bold text-base leading-tight">{indication}</h3>
          <span className={cn("text-xs font-semibold", catMeta.textColor)}>{category}</span>
        </div>

        <ScoreRing score={score} color={catMeta.color} />
      </div>

      {/* Mechanism */}
      <div className="mb-4">
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-1">Primary Mechanism</p>
        <p className="text-xs text-muted-foreground leading-relaxed">{top_mechanism}</p>
        <p className="text-[10px] text-muted-foreground mt-1">Target: <span className="font-semibold text-foreground/80">{top_target}</span></p>
      </div>

      {/* Matched reference drugs */}
      <div className="mb-4">
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground mb-2">
          Similar Approved Drugs ({matched_drugs.length})
        </p>
        <div className="space-y-2">
          {matched_drugs.map((drug) => (
            <div key={drug.name} className="flex items-center gap-2">
              <CheckCircle2 className="h-3.5 w-3.5 shrink-0" style={{ color: catMeta.color }} />
              <span className="text-xs font-semibold flex-1">{drug.name}</span>
              <span className="text-[10px] font-bold px-2 py-0.5 rounded-full"
                style={{ background: `${catMeta.color}14`, color: catMeta.color, border: `1px solid ${catMeta.color}30` }}>
                {(drug.similarity * 100).toFixed(0)}% match
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Mechanism note */}
      <div
        className="flex items-start gap-2 rounded-2xl px-3 py-2.5 text-[10px] text-muted-foreground leading-relaxed"
        style={{ background: `${catMeta.color}0d`, border: `1px solid ${catMeta.color}20` }}
      >
        <Info className="h-3 w-3 shrink-0 mt-0.5" style={{ color: catMeta.color }} />
        <span>Most similar to <strong>{matched_drugs[0]?.name}</strong> — {matched_drugs[0]?.mechanism}</span>
      </div>
    </motion.div>
  );
}

function MolecularProfileCard({ profile }: { profile: MolecularProfile }) {
  const lipinski = [
    { rule: "MW ≤ 500", pass: profile.molecular_weight <= 500 },
    { rule: "logP ≤ 5", pass: profile.logp <= 5 },
    { rule: "HBD ≤ 5", pass: profile.hbd <= 5 },
    { rule: "HBA ≤ 10", pass: profile.hba <= 10 },
  ];
  const passCount = lipinski.filter(r => r.pass).length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
      className="rounded-3xl overflow-hidden border border-white/10 bg-white/[0.03] backdrop-blur-xl p-5"
    >
      <div className="flex items-center gap-2 mb-4">
        <Dna className="h-4 w-4 text-sky-400" />
        <h3 className="text-sm font-bold">Molecular Profile</h3>
        <span className="text-[10px] font-mono text-muted-foreground ml-auto">{profile.formula}</span>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-4">
        {[
          { label: "MW", value: profile.molecular_weight.toFixed(1), unit: "g/mol" },
          { label: "logP", value: profile.logp.toFixed(2), unit: "" },
          { label: "TPSA", value: profile.tpsa.toFixed(1), unit: "Å²" },
          { label: "Rings", value: profile.num_rings, unit: "" },
        ].map(d => (
          <div key={d.label} className="text-center bg-white/[0.03] rounded-xl p-2.5 border border-white/[0.06]">
            <p className="text-lg font-bold leading-tight">{d.value}</p>
            <p className="text-[9px] text-muted-foreground font-semibold uppercase">{d.label} {d.unit}</p>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 mb-2">
        <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
          Lipinski's Rule of Five
        </p>
        <span className={cn("text-[10px] font-bold px-2 py-0.5 rounded-full",
          passCount === 4 ? "bg-emerald-500/15 text-emerald-400" : passCount >= 3 ? "bg-amber-500/15 text-amber-400" : "bg-red-500/15 text-red-400"
        )}>
          {passCount}/4 Pass
        </span>
      </div>
      <div className="flex gap-2">
        {lipinski.map(r => (
          <div key={r.rule} className={cn("text-[10px] font-medium px-2.5 py-1 rounded-full border",
            r.pass ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/10" : "border-red-500/30 text-red-400 bg-red-500/10"
          )}>
            {r.pass ? "✓" : "✗"} {r.rule}
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main Page
────────────────────────────────────────────────────────────────────────────── */
export default function SimilarityAnalyzer() {
  const [smiles, setSmiles] = useState("");
  const [data, setData] = useState<SimilarityResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runAnalysis = useCallback(async () => {
    const trimmed = smiles.trim();
    if (!trimmed) return;

    if (trimmed.length < 4) {
      setError("SMILES string too short. Please enter a valid molecular structure.");
      return;
    }

    setLoading(true);
    setError(null);
    setData(null);

    try {
      const res = await fetch("/api/similarity/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: trimmed }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(body.detail || `API error: ${res.status}`);
      }

      const result: SimilarityResponse = await res.json();
      setData(result);

      if (result.diseases.length === 0) {
        setError("No significant matches found. The molecule doesn't share sufficient structural similarity with known therapeutic agents. Try a more complex or drug-like SMILES.");
      }
    } catch (e: any) {
      setError(e.message || "Analysis failed. Please check your SMILES string.");
    } finally {
      setLoading(false);
    }
  }, [smiles]);

  const topResult = data?.diseases?.[0];

  return (
    <AppLayout>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="min-h-screen"
      >
        {/* ── Hero Header ── */}
        <div className="relative px-6 lg:px-10 pt-8 pb-6 overflow-hidden">
          <div className="absolute top-0 left-1/2 -translate-x-1/2 h-48 w-96 rounded-full blur-3xl opacity-10"
            style={{ background: "linear-gradient(135deg, hsl(187 85% 55%), hsl(207 100% 50%))" }}
          />
          <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="stat-pill bg-sky-500/15 text-sky-400 text-[11px] font-semibold ring-1 ring-sky-500/25">
                <FlaskConical className="h-3 w-3" />
                RDKit Morgan Fingerprint Engine
              </span>
              <span className="stat-pill bg-purple-500/15 text-purple-400 text-[11px] font-semibold ring-1 ring-purple-500/25">
                <Zap className="h-3 w-3" />
                Tanimoto Similarity · {data ? `${data.total_drugs_matched} drugs matched` : "144 Reference Drugs"}
              </span>
            </div>
            <h1 className="text-4xl font-bold tracking-tight">
              Drug Similarity <span className="gradient-text">Analyzer</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-2 max-w-2xl">
              Enter any SMILES string and discover which diseases your drug candidate may treat — powered by Morgan fingerprint (ECFP4) similarity search against 144 approved drugs across 40+ therapeutic indications.
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
                      onChange={(e) => { setSmiles(e.target.value); setData(null); setError(null); }}
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
                  <div className="flex flex-wrap gap-2 mt-3">
                    {EXAMPLE_SMILES.map((ex) => (
                      <button
                        key={ex.label}
                        onClick={() => { setSmiles(ex.smiles); setData(null); setError(null); }}
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
                  <p className="text-lg font-bold">Running Similarity Analysis</p>
                  <p className="text-sm text-muted-foreground mt-1 max-w-sm">
                    Computing Morgan fingerprints · Searching 144 approved drugs · Ranking by Tanimoto similarity…
                  </p>
                </div>
                <div className="flex gap-2">
                  {["Fingerprinting", "Tanimoto Search", "Ranking"].map((step, i) => (
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
            {!loading && error && !data && (
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
            {!loading && !error && !data && (
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
                    Paste any SMILES string above or select an example molecule. The engine will compute RDKit Morgan fingerprints and find structurally similar approved drugs to predict therapeutic indications.
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-4 text-center mt-4">
                  {[
                    { icon: "🧬", label: "ECFP4 Fingerprints", sub: "2048-bit Morgan (radius=2)" },
                    { icon: "📊", label: "Tanimoto Similarity", sub: "Gold-standard metric" },
                    { icon: "💊", label: "144 Approved Drugs", sub: "40+ disease indications" },
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
            {!loading && data && data.diseases.length > 0 && (
              <motion.div
                key="results"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
              >
                {/* Molecular Profile */}
                <div className="mb-6">
                  <MolecularProfileCard profile={data.molecular_profile} />
                </div>

                {/* Results header */}
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-xl font-bold">
                      <span className="gradient-text">{data.total_diseases_found}</span> Therapeutic{data.total_diseases_found !== 1 ? " Areas" : " Area"} Identified
                    </h2>
                    <p className="text-xs text-muted-foreground mt-1">
                      Ranked by Tanimoto similarity · {data.total_drugs_matched} drug matches across {data.total_diseases_found} indications
                    </p>
                  </div>
                  {topResult && (
                    <div className="flex items-center gap-2 px-4 py-2 rounded-2xl" style={{
                      background: `${getCatMeta(topResult.category).color}12`,
                      border: `1px solid ${getCatMeta(topResult.category).color}30`,
                    }}>
                      <TrendingUp className="h-4 w-4" style={{ color: getCatMeta(topResult.category).color }} />
                      <span className="text-xs font-semibold">
                        Best: <span style={{ color: getCatMeta(topResult.category).color }}>{topResult.indication} ({Math.round(topResult.max_similarity * 100)}%)</span>
                      </span>
                    </div>
                  )}
                </div>

                {/* Legend */}
                <div className="flex gap-4 mb-6">
                  {(Object.entries(LEVEL_META) as [string, typeof LEVEL_META[string]][]).map(([key, meta]) => (
                    <div key={key} className="flex items-center gap-1.5">
                      <span className={cn("h-2 w-2 rounded-full", meta.dot)} />
                      <span className="text-[10px] text-muted-foreground font-medium">{meta.label}</span>
                    </div>
                  ))}
                  <div className="flex items-center gap-1.5 ml-auto">
                    <Shield className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] text-muted-foreground">RDKit-validated · Research use only</span>
                  </div>
                </div>

                {/* Cards grid */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                  {data.diseases.map((r, i) => (
                    <DiseaseCard key={r.indication} result={r} index={i} />
                  ))}
                </div>

                {/* Footer disclaimer */}
                <motion.div
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
                  className="mt-8 flex items-start gap-2 px-5 py-3.5 rounded-2xl border border-white/8 bg-white/[0.02] text-[11px] text-muted-foreground"
                >
                  <Info className="h-3.5 w-3.5 shrink-0 mt-0.5 text-sky-400" />
                  <span>
                    <strong className="text-foreground">Research Disclaimer:</strong> This analysis uses RDKit Morgan fingerprint (ECFP4) Tanimoto similarity against a curated database of approved drugs. Results are for research ideation only and do not constitute clinical advice.
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
