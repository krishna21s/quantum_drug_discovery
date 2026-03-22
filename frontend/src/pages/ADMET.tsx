import AppLayout from "@/components/AppLayout";
import ADMETPanel from "@/components/ADMETPanel";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Database, Loader2, Sparkles, RefreshCw, FlaskConical } from "lucide-react";
import { useState, useEffect, useMemo, useCallback } from "react";
import { computeADMET, combinedScore, DEMO_MOLECULES, type ADMETDetail } from "@/lib/admetEngine";
import { fetchDBCandidates, type DBCandidate } from "@/lib/dbApi";
import { type Candidate } from "@/lib/drugApi";
import { generateADMET, fillMissingADMET, getADMETByCandidate, type ADMETData } from "@/lib/admetApi";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Demo molecules remain available as fallback
const demoMoleculeNames = Object.keys(DEMO_MOLECULES);

const bindingData: Record<string, number> = {
    Aspirin: -8.2,
    Cetuximab: -9.4,
    Ibuprofen: -6.8,
    Metformin: -5.2,
    Paracetamol: -7.1,
};

const quantumData: Record<string, number> = {
    Aspirin: -75.3,
    Cetuximab: -82.1,
    Ibuprofen: -68.5,
    Metformin: -55.2,
    Paracetamol: -71.8,
};

type ViewMode = "database" | "demo";

export default function ADMET() {
    const [viewMode, setViewMode] = useState<ViewMode>("database");

    // Database state
    const [dbCandidates, setDbCandidates] = useState<Candidate[]>([]);
    const [loadingDb, setLoadingDb] = useState(true);
    const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
    const [admetData, setAdmetData] = useState<ADMETData | null>(null);
    const [loadingAdmet, setLoadingAdmet] = useState(false);
    const [admetError, setAdmetError] = useState<string | null>(null);
    const [fillingMissing, setFillingMissing] = useState(false);
    const [fillResult, setFillResult] = useState<string | null>(null);
    const [customSmiles, setCustomSmiles] = useState("");

    // Demo state
    const [selectedDemo, setSelectedDemo] = useState(demoMoleculeNames[0]);

    // Load DB candidates 
    useEffect(() => {
        fetchDBCandidates()
            .then(res => setDbCandidates(res.candidates))
            .catch(err => console.error("Failed to load DB candidates:", err))
            .finally(() => setLoadingDb(false));
    }, []);

    // Load ADMET for selected DB candidate
    const loadAdmetForCandidate = useCallback(async (candidateId: number) => {
        setLoadingAdmet(true);
        setAdmetError(null);
        setAdmetData(null);
        try {
            const data = await getADMETByCandidate(candidateId);
            setAdmetData(data);
        } catch {
            // ADMET not yet generated for this candidate
            setAdmetError("No ADMET data found. Click 'Generate' to create it.");
        } finally {
            setLoadingAdmet(false);
        }
    }, []);

    // When user selects a DB candidate
    const handleCandidateSelect = (idStr: string) => {
        const id = parseInt(idStr);
        setSelectedCandidateId(id);
        loadAdmetForCandidate(id);
    };

    // Generate ADMET for selected candidate
    const handleGenerateAdmet = useCallback(async () => {
        if (!selectedCandidateId) return;
        const candidate = dbCandidates.find(c => c.rank === selectedCandidateId);
        if (!candidate) return;

        setLoadingAdmet(true);
        setAdmetError(null);
        try {
            const data = await generateADMET(candidate.smiles);
            setAdmetData(data);
        } catch (e: unknown) {
            setAdmetError(e instanceof Error ? e.message : "Generation failed");
        } finally {
            setLoadingAdmet(false);
        }
    }, [selectedCandidateId, dbCandidates]);

    // Generate ADMET from manual SMILES input
    const handleCustomSmilesGenerate = useCallback(async () => {
        if (!customSmiles.trim()) return;
        setSelectedCandidateId(null);
        setLoadingAdmet(true);
        setAdmetError(null);
        setAdmetData(null);
        try {
            const data = await generateADMET(customSmiles.trim());
            setAdmetData(data);
        } catch (e: unknown) {
            setAdmetError(e instanceof Error ? e.message : "Generation failed");
        } finally {
            setLoadingAdmet(false);
        }
    }, [customSmiles]);

    // Fill missing ADMET for all candidates
    const handleFillMissing = useCallback(async () => {
        setFillingMissing(true);
        setFillResult(null);
        try {
            const result = await fillMissingADMET();
            setFillResult(result.message);
        } catch (e: unknown) {
            setFillResult(e instanceof Error ? e.message : "Batch processing failed");
        } finally {
            setFillingMissing(false);
        }
    }, []);

    // Demo ADMET computation
    const demoAdmet = useMemo(() => computeADMET(DEMO_MOLECULES[selectedDemo]), [selectedDemo]);
    const demoCs = useMemo(
        () => combinedScore(bindingData[selectedDemo], quantumData[selectedDemo], demoAdmet.scores.overall),
        [selectedDemo, demoAdmet.scores.overall]
    );

    // Convert DB ADMET data to ADMETDetail format for ADMETPanel
    const dbAdmetDetail: ADMETDetail | null = useMemo(() => {
        if (!admetData) return null;
        return {
            scores: {
                absorption: admetData.absorption,
                distribution: admetData.distribution,
                metabolism: admetData.metabolism,
                excretion: admetData.excretion,
                toxicity: 1 - (1 - admetData.overall), // derive from overall
                overall: admetData.overall,
                verdict: admetData.verdict as "Pass" | "Caution" | "Fail",
            },
            absorption: {
                lipinskiViolations: 0,
                bioavailability: admetData.absorption > 0.7 ? "High" : admetData.absorption > 0.4 ? "Moderate" : "Low",
                solubilityClass: admetData.absorption > 0.6 ? "Good" : admetData.absorption > 0.3 ? "Moderate" : "Poor",
                intestinalAbsorption: admetData.absorption > 0.7 ? "High" : admetData.absorption > 0.4 ? "Moderate" : "Low",
            },
            distribution: {
                bbbPermeant: admetData.distribution > 0.5,
                vdCategory: admetData.distribution > 0.7 ? "High" : admetData.distribution > 0.4 ? "Moderate" : "Low",
                plasmaProteinBinding: admetData.distribution > 0.6 ? "High" : admetData.distribution > 0.3 ? "Moderate" : "Low",
            },
            metabolism: {
                cyp450Substrate: admetData.metabolism < 0.5,
                cyp450Inhibitor: admetData.metabolism < 0.4,
                hepaticClearance: admetData.metabolism > 0.7 ? "Fast" : admetData.metabolism > 0.4 ? "Moderate" : "Slow",
            },
            excretion: {
                renalClearance: admetData.excretion > 0.7 ? "Fast" : admetData.excretion > 0.4 ? "Moderate" : "Slow",
                halfLifeCategory: admetData.excretion > 0.7 ? "Short" : admetData.excretion > 0.4 ? "Moderate" : "Long",
                estimatedHalfLife: admetData.excretion > 0.7 ? "2–6 h" : admetData.excretion > 0.4 ? "6–24 h" : ">24 h",
            },
            toxicity: {
                hergRisk: admetData.overall > 0.7 ? "Low" : admetData.overall > 0.4 ? "Moderate" : "High",
                amesMutagenicity: "Negative",
                hepatotoxicity: admetData.overall > 0.7 ? "Low" : admetData.overall > 0.4 ? "Moderate" : "High",
                cardiotoxicity: admetData.overall > 0.7 ? "Low" : "Moderate",
            },
        };
    }, [admetData]);

    // Demo mode table data
    const allDemoResults = useMemo(
        () =>
            demoMoleculeNames.map((name) => {
                const d = computeADMET(DEMO_MOLECULES[name]);
                return { name, scores: d.scores, cs: combinedScore(bindingData[name], quantumData[name], d.scores.overall) };
            }),
        []
    );

    return (
        <AppLayout>
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-8 space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between flex-wrap gap-4">
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-2">
                            <Shield className="h-6 w-6 text-quantum" />
                            ADMET Analysis
                        </h1>
                        <p className="text-muted-foreground mt-1">
                            Absorption, Distribution, Metabolism, Excretion &amp; Toxicity profiling
                        </p>
                    </div>

                    {/* Mode toggle */}
                    <div className="flex items-center gap-2 glass-surface rounded-xl p-1">
                        <button
                            onClick={() => setViewMode("database")}
                            className={cn(
                                "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                                viewMode === "database" ? "bg-quantum/20 text-quantum ring-1 ring-quantum/30" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            <Database className="h-4 w-4" /> Real Data
                        </button>
                        <button
                            onClick={() => setViewMode("demo")}
                            className={cn(
                                "px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2",
                                viewMode === "demo" ? "bg-purple-500/20 text-purple-400 ring-1 ring-purple-400/30" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            <FlaskConical className="h-4 w-4" /> Demo
                        </button>
                    </div>
                </div>

                <AnimatePresence mode="wait">
                    {viewMode === "database" ? (
                        <motion.div key="db" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                                {/* Left: Controls */}
                                <div className="lg:col-span-2 space-y-4">
                                    {/* Candidate selector + Actions */}
                                    <div className="glass-card rounded-2xl p-6 relative overflow-hidden">
                                        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />
                                        <div className="flex items-center justify-between mb-4">
                                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Select Candidate</h3>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                onClick={handleFillMissing}
                                                disabled={fillingMissing}
                                                className="rounded-xl text-xs"
                                            >
                                                {fillingMissing ? (
                                                    <><Loader2 className="h-3 w-3 mr-1 animate-spin" /> Processing...</>
                                                ) : (
                                                    <><RefreshCw className="h-3 w-3 mr-1" /> Fill Missing ADMET</>
                                                )}
                                            </Button>
                                        </div>

                                        {fillResult && (
                                            <div className="mb-4 p-3 rounded-xl glass-surface text-xs text-muted-foreground">
                                                {fillResult}
                                            </div>
                                        )}

                                        {loadingDb ? (
                                            <div className="flex items-center text-sm text-muted-foreground py-4">
                                                <Loader2 className="h-4 w-4 mr-2 animate-spin" /> Loading candidates...
                                            </div>
                                        ) : dbCandidates.length > 0 ? (
                                            <div className="flex gap-3">
                                                <Select onValueChange={handleCandidateSelect}>
                                                    <SelectTrigger className="flex-1 rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm px-3 py-5 text-sm font-mono">
                                                        <SelectValue placeholder="-- Select a candidate --" />
                                                    </SelectTrigger>
                                                    <SelectContent className="max-h-60 rounded-xl bg-background border-white/10">
                                                        {dbCandidates.map((c) => (
                                                            <SelectItem key={c.rank} value={String(c.rank)} className="font-mono text-xs cursor-pointer focus:bg-quantum/20">
                                                                #{c.rank} — pIC₅₀ {c.xgb_pic50.toFixed(2)} — {c.smiles.substring(0, 40)}...
                                                            </SelectItem>
                                                        ))}
                                                    </SelectContent>
                                                </Select>
                                                {selectedCandidateId && admetError && (
                                                    <Button
                                                        onClick={handleGenerateAdmet}
                                                        disabled={loadingAdmet}
                                                        className="rounded-xl"
                                                        style={{
                                                            background: "linear-gradient(135deg, hsl(270 70% 55%), hsl(290 80% 60%))",
                                                            border: "none",
                                                        }}
                                                    >
                                                        {loadingAdmet ? <Loader2 className="h-4 w-4 animate-spin" /> : <><Sparkles className="h-4 w-4 mr-1" /> Generate</>}
                                                    </Button>
                                                )}
                                            </div>
                                        ) : (
                                            <p className="text-sm text-muted-foreground">No candidates in DB. Seed the database first.</p>
                                        )}
                                    </div>

                                    {/* Manual SMILES Input */}
                                    <div className="glass-card rounded-2xl p-6 relative overflow-hidden">
                                        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-purple-500/40 to-transparent" />
                                        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Or Enter SMILES Manually</h3>
                                        <div className="flex gap-3">
                                            <textarea
                                                value={customSmiles}
                                                onChange={(e) => setCustomSmiles(e.target.value)}
                                                placeholder="Enter SMILES string, e.g. CC(=O)OC1=CC=CC=C1C(=O)O"
                                                className={cn(
                                                    "flex-1 rounded-xl border bg-muted/20 backdrop-blur-sm px-4 py-3",
                                                    "text-sm font-mono focus:outline-none focus:ring-1 transition-all resize-none h-[52px]",
                                                    customSmiles.trim()
                                                        ? "border-purple-400/30 focus:ring-purple-400/40 focus:border-purple-400"
                                                        : "border-white/10 focus:ring-white/20"
                                                )}
                                            />
                                            <Button
                                                onClick={handleCustomSmilesGenerate}
                                                disabled={!customSmiles.trim() || loadingAdmet}
                                                className="rounded-xl h-[52px] px-5"
                                                style={{
                                                    background: "linear-gradient(135deg, hsl(270 70% 55%), hsl(290 80% 60%))",
                                                    border: "none",
                                                    opacity: !customSmiles.trim() || loadingAdmet ? 0.5 : 1,
                                                }}
                                            >
                                                {loadingAdmet ? <Loader2 className="h-4 w-4 animate-spin" /> : <><Sparkles className="h-4 w-4 mr-1" /> Predict</>}
                                            </Button>
                                        </div>
                                    </div>

                                    {/* Status / Results area */}
                                    {loadingAdmet && (
                                        <div className="glass-card rounded-2xl p-8 text-center space-y-3">
                                            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-quantum/10 ring-1 ring-quantum/20">
                                                <FlaskConical className="h-7 w-7 text-quantum animate-pulse" />
                                            </div>
                                            <p className="text-sm font-semibold">Running ADMET-AI Model...</p>
                                            <p className="text-xs text-muted-foreground">Predicting across 41 ADMET datasets</p>
                                        </div>
                                    )}

                                    {admetError && !loadingAdmet && (
                                        <div className="glass-card rounded-2xl p-6 text-center space-y-3 ring-1 ring-warning/30">
                                            <p className="text-sm text-warning font-medium">{admetError}</p>
                                            {selectedCandidateId && (
                                                <Button
                                                    onClick={handleGenerateAdmet}
                                                    className="rounded-xl"
                                                    style={{
                                                        background: "linear-gradient(135deg, hsl(270 70% 55%), hsl(290 80% 60%))",
                                                        border: "none",
                                                    }}
                                                >
                                                    <Sparkles className="h-4 w-4 mr-2" /> Generate ADMET Now
                                                </Button>
                                            )}
                                        </div>
                                    )}

                                    {admetData && !loadingAdmet && (
                                        <div className="glass-card rounded-2xl p-6 relative overflow-hidden">
                                            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-success/40 to-transparent" />
                                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">ADMET Scores</h3>
                                            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                                                {[
                                                    { label: "Absorption", value: admetData.absorption, color: "from-blue-500 to-cyan-500" },
                                                    { label: "Distribution", value: admetData.distribution, color: "from-purple-500 to-pink-500" },
                                                    { label: "Metabolism", value: admetData.metabolism, color: "from-amber-500 to-orange-500" },
                                                    { label: "Excretion", value: admetData.excretion, color: "from-emerald-500 to-green-500" },
                                                    { label: "Overall", value: admetData.overall, color: "from-quantum to-cyan-400" },
                                                    { label: "Verdict", value: null as unknown as number, color: "" },
                                                ].map((item) => (
                                                    <div key={item.label} className="glass-surface rounded-xl p-4 text-center">
                                                        <p className="text-xs text-muted-foreground mb-1">{item.label}</p>
                                                        {item.label === "Verdict" ? (
                                                            <span className={cn(
                                                                "inline-block rounded-full px-3 py-1 text-sm font-bold ring-1",
                                                                admetData.verdict === "Pass" ? "bg-success/10 text-success ring-success/30" :
                                                                    admetData.verdict === "Caution" ? "bg-warning/10 text-warning ring-warning/30" :
                                                                        "bg-destructive/10 text-destructive ring-destructive/30"
                                                            )}>
                                                                {admetData.verdict}
                                                            </span>
                                                        ) : (
                                                            <p className="font-mono font-bold text-lg">
                                                                <span className={cn("bg-clip-text text-transparent bg-gradient-to-r", item.color)}>
                                                                    {(item.value * 100).toFixed(0)}%
                                                                </span>
                                                            </p>
                                                        )}
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Right: ADMET Panel */}
                                <div>
                                    {dbAdmetDetail ? (
                                        <ADMETPanel data={dbAdmetDetail} combinedScore={admetData?.overall} />
                                    ) : (
                                        <div className="glass-card rounded-2xl p-8 text-center space-y-3">
                                            <Shield className="h-10 w-10 text-muted-foreground mx-auto" />
                                            <p className="text-sm text-muted-foreground">
                                                Select a candidate and generate ADMET data to see the profile panel.
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    ) : (
                        /* Demo mode - same as before */
                        <motion.div key="demo" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
                            <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                                <div className="lg:col-span-2 space-y-4">
                                    {/* Molecule selector */}
                                    <div className="glass-card rounded-2xl p-4 relative overflow-hidden">
                                        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
                                        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">Select Molecule</h3>
                                        <div className="flex flex-wrap gap-2">
                                            {demoMoleculeNames.map((name) => (
                                                <button
                                                    key={name}
                                                    onClick={() => setSelectedDemo(name)}
                                                    className={cn(
                                                        "relative rounded-xl px-4 py-2 text-sm font-medium transition-all duration-300",
                                                        selectedDemo === name
                                                            ? "text-quantum"
                                                            : "text-muted-foreground hover:text-foreground"
                                                    )}
                                                >
                                                    {selectedDemo === name && (
                                                        <motion.div
                                                            layoutId="admet-mol-select"
                                                            className="absolute inset-0 rounded-xl glass-surface ring-1 ring-quantum/30"
                                                            transition={{ type: "spring", stiffness: 350, damping: 30 }}
                                                        />
                                                    )}
                                                    <span className="relative z-10">{name}</span>
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Comparison Table */}
                                    <div className="glass-card rounded-2xl overflow-hidden relative">
                                        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />
                                        <div className="p-4 border-b border-white/5">
                                            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Candidate Comparison</h3>
                                        </div>
                                        <table className="w-full text-sm">
                                            <thead>
                                                <tr className="border-b border-white/5">
                                                    <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Molecule</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">A</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">D</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">M</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">E</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">T</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">Overall</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">Verdict</th>
                                                    <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">Combined</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {allDemoResults.map((r, i) => (
                                                    <motion.tr
                                                        key={r.name}
                                                        initial={{ opacity: 0, x: -8 }}
                                                        animate={{ opacity: 1, x: 0 }}
                                                        transition={{ delay: i * 0.05 }}
                                                        onClick={() => setSelectedDemo(r.name)}
                                                        className={cn(
                                                            "border-b border-white/3 cursor-pointer transition-colors",
                                                            selectedDemo === r.name ? "bg-quantum/5" : "hover:bg-muted/20"
                                                        )}
                                                    >
                                                        <td className="px-4 py-3 font-medium">{r.name}</td>
                                                        <ScoreCell score={r.scores.absorption} />
                                                        <ScoreCell score={r.scores.distribution} />
                                                        <ScoreCell score={r.scores.metabolism} />
                                                        <ScoreCell score={r.scores.excretion} />
                                                        <ScoreCell score={r.scores.toxicity} />
                                                        <ScoreCell score={r.scores.overall} bold />
                                                        <td className="px-4 py-3 text-center">
                                                            <span className={cn(
                                                                "inline-block rounded-full px-2 py-0.5 text-xs font-semibold ring-1",
                                                                r.scores.verdict === "Pass" ? "bg-success/10 text-success ring-success/30" :
                                                                    r.scores.verdict === "Caution" ? "bg-warning/10 text-warning ring-warning/30" :
                                                                        "bg-destructive/10 text-destructive ring-destructive/30"
                                                            )}>
                                                                {r.scores.verdict}
                                                            </span>
                                                        </td>
                                                        <ScoreCell score={r.cs} bold />
                                                    </motion.tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>

                                    {/* Descriptors */}
                                    <div className="glass-card rounded-2xl p-4 relative overflow-hidden">
                                        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />
                                        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                                            Molecular Descriptors — {selectedDemo}
                                        </h3>
                                        <div className="grid grid-cols-4 gap-3">
                                            {Object.entries(DEMO_MOLECULES[selectedDemo]).map(([key, val]) => (
                                                <div key={key} className="glass-surface rounded-xl p-3 text-center">
                                                    <p className="text-xs text-muted-foreground capitalize">{key}</p>
                                                    <p className="font-mono font-semibold text-sm mt-0.5">{typeof val === "number" ? val.toFixed(2) : val}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* Right: Demo ADMET Panel */}
                                <div>
                                    <ADMETPanel
                                        data={demoAdmet}
                                        bindingAffinity={bindingData[selectedDemo]}
                                        quantumEnergy={quantumData[selectedDemo]}
                                        combinedScore={demoCs}
                                    />
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </AppLayout>
    );
}

function ScoreCell({ score, bold }: { score: number; bold?: boolean }) {
    const color = score > 0.7 ? "text-success" : score > 0.45 ? "text-warning" : "text-destructive";
    return (
        <td className={cn("px-4 py-3 text-center font-mono text-xs", color, bold && "font-bold")}>
            {(score * 100).toFixed(0)}%
        </td>
    );
}
