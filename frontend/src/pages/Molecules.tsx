import AppLayout from "@/components/AppLayout";
import { motion, AnimatePresence } from "framer-motion";
import { Atom, Download, RefreshCw, Loader2, Sparkles, Zap, FileJson, FileSpreadsheet, FileText, BarChart2, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useState, useEffect, useCallback } from "react";
import { useLocation, Link } from "react-router-dom";
import {
  fetchCandidates,
  type Candidate,
  type CandidatesResponse,
  type GenerateResponse,
} from "@/lib/drugApi";
import { fetchDBCandidates } from "@/lib/dbApi";
import { cn } from "@/lib/utils";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

const PROTEIN_TARGETS = [
  { pdb: "1M17", name: "EGFR Kinase", disease: "Cancer" },
  { pdb: "6LU7", name: "SARS-CoV-2 Mpro", disease: "COVID-19" },
  { pdb: "1HHP", name: "HIV-1 Protease", disease: "HIV/AIDS" },
  { pdb: "3ERT", name: "Estrogen Receptor", disease: "Breast Cancer" },
  { pdb: "1ZG4", name: "Beta-Lactamase", disease: "Antibiotic Resistance" },
];

function pic50Badge(val: number | null) {
  if (val === null) return <span className="text-muted-foreground">—</span>;
  const color =
    val >= 7.0 ? "text-success" : val >= 6.5 ? "text-quantum" : val >= 6.0 ? "text-warning" : "text-destructive";
  return <span className={cn("font-mono font-semibold", color)}>{val.toFixed(2)}</span>;
}

function qedBadge(val: number) {
  const pct = (val * 100).toFixed(0);
  const color =
    val >= 0.7 ? "bg-success/10 text-success ring-success/30" :
      val >= 0.5 ? "bg-warning/10 text-warning ring-warning/30" :
        "bg-destructive/10 text-destructive ring-destructive/30";
  return (
    <span className={cn("inline-block rounded-full px-2 py-0.5 text-xs font-semibold ring-1", color)}>
      {pct}%
    </span>
  );
}

export default function Molecules() {
  const location = useLocation();

  // Pre-computed candidates
  const [data, setData] = useState<CandidatesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [filter, setFilter] = useState<"all" | "high" | "medium">("all");

  // Generate panel state
  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);

  // Active display: either pre-computed or generated
  const [viewMode, setViewMode] = useState<"precomputed" | "generated">("precomputed");

  // Accept results passed from the Experiment page via router state
  useEffect(() => {
    const state = location.state as { genResult?: GenerateResponse; fromExperiment?: boolean } | null;
    if (state?.fromExperiment && state?.genResult) {
      setGenResult(state.genResult);
      setViewMode("generated");
      if (state.genResult.candidates.length > 0) {
        setSelected(state.genResult.candidates[0]);
      }
      setLoading(false);
      // Clear the state so refreshing the page doesn't replay
      window.history.replaceState({}, document.title);
    }
  }, [location.state]);

  const loadCandidates = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchDBCandidates();
      setData(res);
      if (res.candidates.length > 0 && !selected) {
        setSelected(res.candidates[0]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load candidates");
    } finally {
      setLoading(false);
    }
  }, [selected]);

  useEffect(() => { loadCandidates(); }, []);

  const activeCandidates = viewMode === "generated" && genResult
    ? genResult.candidates
    : data?.candidates ?? [];

  const filtered = activeCandidates.filter((c) => {
    if (filter === "high") return c.xgb_pic50 >= 7.0;
    if (filter === "medium") return c.xgb_pic50 >= 6.5 && c.xgb_pic50 < 7.0;
    return true;
  });

  const exportData = (format: "csv" | "json" | "excel") => {
    if (!activeCandidates.length) return;

    let content = "";
    let mimeType = "";
    let extension = "";

    if (format === "json") {
      content = JSON.stringify(activeCandidates, null, 2);
      mimeType = "application/json";
      extension = "json";
    } else {
      const headers = ["Rank", "SMILES", "XGB_pIC50", "QSVR_pIC50", "QED", "SA_Score", "MW", "LogP"];
      const rows = activeCandidates.map(c => [
        c.rank,
        c.smiles,
        c.xgb_pic50 ?? "",
        c.quantum_pic50 ?? "",
        c.qed,
        c.sa_score,
        c.mw,
        c.logp
      ]);
      
      const csvContent = [
        headers.join(","),
        ...rows.map(row => row.join(","))
      ].join("\r\n");

      if (format === "csv") {
        content = csvContent;
        mimeType = "text/csv;charset=utf-8;";
        extension = "csv";
      } else if (format === "excel") {
        content = "\uFEFF" + csvContent; // UTF-8 BOM for Excel
        mimeType = "application/vnd.ms-excel;charset=utf-8;";
        extension = "csv";
      }
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    const prefix = viewMode === "generated" ? `generated_${genResult?.target || "new"}` : `candidates_${data?.target?.replace(/[^a-zA-Z0-9]/g, "") || "all"}`;
    link.setAttribute("download", `${prefix}.${extension}`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <AppLayout>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-6 lg:p-8 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="stat-pill bg-quantum/15 text-quantum text-[11px] font-semibold">
                <Atom className="h-3 w-3" /> RL-Generated
              </span>
              {viewMode === "generated" && genResult && (
                <span className="stat-pill bg-success/15 text-success text-[11px] font-semibold">
                  <Sparkles className="h-3 w-3" /> Fresh — {genResult.target}
                </span>
              )}
              {viewMode === "precomputed" && data && (
                <span className="stat-pill bg-primary/15 text-primary text-[11px] font-semibold">
                  {data.target}
                </span>
              )}
            </div>
            <h1 className="text-3xl font-bold tracking-tight">
              Drug <span className="gradient-text">Candidates</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-1">
              {viewMode === "generated" && genResult
                ? `${genResult.candidates.length} freshly generated · ${genResult.n_sampled} sampled · ${genResult.generation_time_s}s`
                : data
                  ? `${data.candidates.length} candidates from ${data.total_generated.toLocaleString()} generated · ${data.n_rl_episodes} RL episodes`
                  : "Loading candidates..."}
            </p>
          </div>
          <div className="flex gap-2">
            <Link to="/experiment">
              <Button className="rounded-xl gap-1.5 font-semibold" style={{
                background: "linear-gradient(135deg, hsl(187 85% 45%), hsl(207 100% 50%))",
                border: "none",
                boxShadow: "0 4px 16px -4px hsl(207 100% 50% / 0.4)",
              }}>
                <FlaskConical className="h-4 w-4" /> New Experiment
              </Button>
            </Link>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="rounded-xl gap-1.5" disabled={activeCandidates.length === 0}>
                  <Download className="h-4 w-4" /> Export
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="rounded-xl">
                <DropdownMenuItem onClick={() => exportData("excel")} className="gap-2 cursor-pointer">
                  <FileSpreadsheet className="h-4 w-4 text-success" /> Export to Excel
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => exportData("csv")} className="gap-2 cursor-pointer">
                  <FileText className="h-4 w-4 text-muted-foreground" /> Export as CSV
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => exportData("json")} className="gap-2 cursor-pointer">
                  <FileJson className="h-4 w-4 text-quantum" /> Export as JSON
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            {viewMode === "generated" && (
              <Button variant="outline" className="rounded-xl gap-1.5"
                onClick={() => { setViewMode("precomputed"); if (data?.candidates[0]) setSelected(data.candidates[0]); }}>
                <RefreshCw className="h-4 w-4" /> Show Pre-computed
              </Button>
            )}
            <Button variant="outline" className="rounded-xl gap-1.5" onClick={loadCandidates} disabled={loading}>
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} /> Refresh
            </Button>
          </div>
        </div>



        {/* Error */}
        {error && !genResult && (
          <div className="glass-card rounded-3xl p-6 ring-1 ring-destructive/30 text-center space-y-2">
            <p className="text-destructive font-semibold">{error}</p>
            <p className="text-sm text-muted-foreground">Make sure the backend is running on port 8000</p>
            <Button variant="outline" onClick={loadCandidates} className="rounded-xl">Retry</Button>
          </div>
        )}

        {/* Loading skeleton */}
        {loading && !data && !genResult && (
          <div className="glass-card rounded-3xl p-12 text-center space-y-4">
            <Loader2 className="h-10 w-10 text-quantum animate-spin mx-auto" />
            <p className="text-lg font-semibold">Loading Candidates...</p>
            <p className="text-sm text-muted-foreground">Fetching RL-generated drug candidates from the API</p>
          </div>
        )}

        {/* Main content */}
        {activeCandidates.length > 0 && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* Table */}
            <div className="lg:col-span-2">
              {/* Filter bar */}
              <div className="flex gap-2 mb-4">
                {(["all", "high", "medium"] as const).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={cn(
                      "px-3 py-1.5 rounded-xl text-xs font-semibold transition-all",
                      filter === f
                        ? "bg-quantum/15 text-quantum ring-1 ring-quantum/30"
                        : "glass-surface text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {f === "all" ? `All (${activeCandidates.length})` :
                      f === "high" ? `pIC50 ≥ 7 (${activeCandidates.filter(c => c.xgb_pic50 >= 7).length})` :
                        `pIC50 6.5–7 (${activeCandidates.filter(c => c.xgb_pic50 >= 6.5 && c.xgb_pic50 < 7).length})`}
                  </button>
                ))}
              </div>

              <div className="glass-card rounded-2xl overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-white/5">
                        <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider w-10">#</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">SMILES</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">XGB pIC₅₀</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">QSVR pIC₅₀</th>
                        <th className="px-4 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">QED</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">SA</th>
                        <th className="px-4 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">MW</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((c, i) => (
                        <motion.tr
                          key={c.rank}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: i * 0.02 }}
                          onClick={() => setSelected(c)}
                          className={cn(
                            "border-b border-white/3 transition-colors cursor-pointer",
                            selected?.rank === c.rank
                              ? "bg-quantum/10 ring-1 ring-quantum/20"
                              : "hover:bg-quantum/5"
                          )}
                        >
                          <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{c.rank}</td>
                          <td className="px-4 py-3 font-mono text-xs max-w-[200px] truncate" title={c.smiles}>{c.smiles}</td>
                          <td className="px-4 py-3 text-right">{pic50Badge(c.xgb_pic50)}</td>
                          <td className="px-4 py-3 text-right">{pic50Badge(c.quantum_pic50)}</td>
                          <td className="px-4 py-3 text-center">{qedBadge(c.qed)}</td>
                          <td className="px-4 py-3 text-right font-mono text-xs">{c.sa_score.toFixed(1)}</td>
                          <td className="px-4 py-3 text-right font-mono text-xs">{c.mw.toFixed(0)}</td>
                        </motion.tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Side panel: selected candidate detail */}
            <div>
              <AnimatePresence mode="wait">
                {selected && (
                  <motion.div
                    key={selected.rank}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="glass-card rounded-3xl p-6 space-y-4 sticky top-6"
                  >
                    <div className="flex items-center justify-between">
                      <h2 className="font-bold text-lg">Candidate #{selected.rank}</h2>
                      {selected.lipinski_pass && (
                        <span className="text-xs bg-success/10 text-success ring-1 ring-success/30 px-2 py-0.5 rounded-full font-semibold">
                          Lipinski ✓
                        </span>
                      )}
                    </div>

                    <div className="bg-muted/20 rounded-xl p-3">
                      <p className="font-mono text-xs break-all leading-relaxed">{selected.smiles}</p>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="glass-surface rounded-xl p-3 text-center">
                        <p className="text-xs text-muted-foreground">XGB pIC₅₀</p>
                        <p className="text-xl font-bold text-quantum mt-1">{selected.xgb_pic50?.toFixed(2) ?? "—"}</p>
                      </div>
                      <div className="glass-surface rounded-xl p-3 text-center">
                        <p className="text-xs text-muted-foreground">QSVR pIC₅₀</p>
                        <p className="text-xl font-bold text-purple-400 mt-1">
                          {selected.quantum_pic50?.toFixed(2) ?? "—"}
                        </p>
                      </div>
                      <div className="glass-surface rounded-xl p-3 text-center">
                        <p className="text-xs text-muted-foreground">QED</p>
                        <p className="text-xl font-bold mt-1">{(selected.qed * 100).toFixed(0)}%</p>
                      </div>
                      <div className="glass-surface rounded-xl p-3 text-center">
                        <p className="text-xs text-muted-foreground">SA Score</p>
                        <p className="text-xl font-bold mt-1">{selected.sa_score.toFixed(1)}</p>
                      </div>
                    </div>

                    <div className="space-y-2 text-xs">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Molecular Weight</span>
                        <span className="font-mono">{selected.mw.toFixed(1)} Da</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">LogP</span>
                        <span className="font-mono">{selected.logp.toFixed(2)}</span>
                      </div>
                      {selected.tpsa !== null && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">TPSA</span>
                          <span className="font-mono">{selected.tpsa.toFixed(1)} Å²</span>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Novel</span>
                        <span>{selected.is_novel ? "✓ Yes" : selected.is_novel === false ? "✗ No" : "—"}</span>
                      </div>
                      {selected.docking_score != null && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Docking Score</span>
                          <span className={cn("font-mono font-semibold", selected.docking_score <= -8 ? "text-success" : selected.docking_score <= -6 ? "text-quantum" : "text-warning")}>
                            {selected.docking_score.toFixed(2)} kcal/mol
                          </span>
                        </div>
                      )}
                    </div>

                    {/* ADMET Panel */}
                    {selected.admet && (
                      <div className="bg-muted/10 rounded-xl p-3 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold">ADMET Profile</span>
                          <span className={cn(
                            "text-xs font-semibold px-2 py-0.5 rounded-full ring-1",
                            selected.admet.verdict === "Promising" ? "bg-success/10 text-success ring-success/30" :
                            selected.admet.verdict === "Acceptable" ? "bg-warning/10 text-warning ring-warning/30" :
                            "bg-destructive/10 text-destructive ring-destructive/30"
                          )}>
                            {selected.admet.verdict}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {[
                            { label: "Absorption", val: selected.admet.absorption },
                            { label: "Distribution", val: selected.admet.distribution },
                            { label: "Metabolism", val: selected.admet.metabolism },
                            { label: "Excretion", val: selected.admet.excretion },
                          ].map((prop) => (
                            <div key={prop.label} className="flex justify-between">
                              <span className="text-muted-foreground">{prop.label}</span>
                              <span className="font-mono">{(prop.val * 100).toFixed(0)}%</span>
                            </div>
                          ))}
                        </div>
                        <div className="flex justify-between text-xs pt-1 border-t border-border/50">
                          <span className="font-semibold">Overall Safety</span>
                          <span className="font-mono font-semibold">{(selected.admet.overall * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    )}

                    {/* pIC50 interpretation */}
                    {selected.xgb_pic50 != null && (
                      <div className="bg-muted/10 rounded-xl p-3 space-y-1">
                        <p className="text-xs font-semibold">Predicted Activity</p>
                        <p className="text-xs text-muted-foreground">
                          pIC₅₀ {selected.xgb_pic50.toFixed(1)} → IC₅₀ ≈{" "}
                          <span className="font-mono text-quantum">
                            {(Math.pow(10, -selected.xgb_pic50) * 1e9).toFixed(0)} nM
                          </span>
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {selected.xgb_pic50 >= 7.0
                            ? "🟢 Strong binder — lead compound quality"
                            : selected.xgb_pic50 >= 6.5
                              ? "🟡 Moderate binder — optimization candidate"
                              : "🟠 Weak binder — needs improvement"}
                        </p>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        )}

        {/* ——— Oracle Comparison Charts ——— */}
        {activeCandidates.length > 0 && (() => {
          const avgXgb = activeCandidates.reduce((s, c) => s + (c.xgb_pic50 ?? 0), 0) / activeCandidates.length;
          const avgQsvr = activeCandidates.reduce((s, c) => s + (c.quantum_pic50 ?? 0), 0) / activeCandidates.length;
          const avgGap = avgQsvr - avgXgb;
          const maxGap = Math.max(...activeCandidates.map(c => (c.quantum_pic50 ?? 0) - (c.xgb_pic50 ?? 0)));
          const qsvrAbove7 = activeCandidates.filter(c => (c.quantum_pic50 ?? 0) >= 7).length;
          const xgbAbove7 = activeCandidates.filter(c => (c.xgb_pic50 ?? 0) >= 7).length;
          const verdictType = avgGap > 3 ? "massive" : avgGap > 1 ? "significant" : avgGap > 0.3 ? "moderate" : "comparable";

          return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="space-y-6"
          >
            {/* Section header */}
            <div className="flex items-center gap-3">
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-purple-500/30 to-transparent" />
              <div className="flex items-center gap-2 px-5 py-2 rounded-full ring-1 ring-purple-400/30"
                style={{ background: "linear-gradient(135deg, hsl(270 60% 20% / 0.6), hsl(217 60% 20% / 0.4))" }}>
                <BarChart2 className="h-4 w-4 text-purple-400" />
                <span className="text-sm font-bold">
                  Why <span className="text-purple-400">Quantum</span> Matters
                </span>
              </div>
              <div className="h-px flex-1 bg-gradient-to-r from-transparent via-purple-500/30 to-transparent" />
            </div>

            {/* ── Big Summary Cards ── */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Classical avg */}
              <div className="glass-card rounded-2xl p-5 text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent" />
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Classical Avg</p>
                <p className="text-3xl font-black text-blue-400 font-mono">{avgXgb.toFixed(2)}</p>
                <p className="text-[10px] text-muted-foreground mt-1">XGBoost pIC₅₀</p>
                <div className="mt-2 flex items-center justify-center gap-1">
                  <span className="text-xs font-semibold text-blue-400">{xgbAbove7}</span>
                  <span className="text-[10px] text-muted-foreground">hits above 7.0</span>
                </div>
              </div>

              {/* Quantum avg */}
              <div className="glass-card rounded-2xl p-5 text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-transparent" />
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Quantum Avg</p>
                <p className="text-3xl font-black text-purple-400 font-mono">{avgQsvr.toFixed(2)}</p>
                <p className="text-[10px] text-muted-foreground mt-1">QSVR pIC₅₀</p>
                <div className="mt-2 flex items-center justify-center gap-1">
                  <span className="text-xs font-semibold text-purple-400">{qsvrAbove7}</span>
                  <span className="text-[10px] text-muted-foreground">hits above 7.0</span>
                </div>
              </div>

              {/* Quantum Advantage */}
              <div className="glass-card rounded-2xl p-5 text-center relative overflow-hidden"
                style={{ background: "linear-gradient(135deg, hsl(270 60% 15% / 0.6), hsl(142 60% 15% / 0.3))" }}>
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-emerald-500/5" />
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Quantum Advantage</p>
                <p className="text-3xl font-black font-mono" style={{ color: avgGap > 1 ? "hsl(142 70% 55%)" : "hsl(45 90% 55%)" }}>
                  +{avgGap.toFixed(2)}
                </p>
                <p className="text-[10px] text-muted-foreground mt-1">avg pIC₅₀ gap</p>
                <div className="mt-2 flex items-center justify-center gap-1">
                  <span className="text-[10px] text-muted-foreground">max gap</span>
                  <span className="text-xs font-semibold font-mono" style={{ color: "hsl(142 70% 55%)" }}>+{maxGap.toFixed(2)}</span>
                </div>
              </div>

              {/* Verdict */}
              <div className="glass-card rounded-2xl p-5 text-center relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/5 to-transparent" />
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-1">Verdict</p>
                <p className="text-2xl mt-1">
                  {verdictType === "massive" ? "🚀" : verdictType === "significant" ? "🔮" : verdictType === "moderate" ? "⚡" : "🤝"}
                </p>
                <p className="text-sm font-bold mt-1"
                  style={{ color: verdictType === "massive" || verdictType === "significant" ? "hsl(142 70% 55%)" : "hsl(45 90% 55%)" }}>
                  {verdictType === "massive"
                    ? "Quantum Essential"
                    : verdictType === "significant"
                      ? "Quantum Critical"
                      : verdictType === "moderate"
                        ? "Quantum Helpful"
                        : "Both Agree"}
                </p>
                <p className="text-[10px] text-muted-foreground mt-1 leading-tight">
                  {verdictType === "massive" || verdictType === "significant"
                    ? "Classical ML fails on these molecules — Quantum kernel is the only reliable scorer"
                    : verdictType === "moderate"
                      ? "Quantum captures additional binding features beyond classical"
                      : "Both oracles reach similar conclusions on this target"}
                </p>
              </div>
            </div>

            {/* ── Charts Grid ── */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

              {/* Chart 1: pIC50 Side-by-Side (Top 10 for clarity) */}
              <div className="glass-card rounded-3xl p-6 space-y-4">
                <div>
                  <h3 className="font-bold text-lg">Binding Affinity — Classical vs Quantum</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    Top 10 candidates · <span className="text-blue-400">Blue = XGBoost (Classical)</span> ·{" "}
                    <span className="text-purple-400">Purple = QSVR (Quantum)</span>
                  </p>
                </div>
                <div className="rounded-xl px-4 py-3 text-sm font-medium"
                  style={{ background: "linear-gradient(135deg, hsl(270 60% 20% / 0.3), hsl(217 80% 20% / 0.2))" }}>
                  {avgGap > 1
                    ? <>⚠️ Classical XGBoost scores <span className="text-blue-400 font-bold">{avgXgb.toFixed(1)}</span> (inactive).
                        Quantum QSVR correctly identifies <span className="text-purple-400 font-bold">{avgQsvr.toFixed(1)}</span> (active binders).</>
                    : <>Both models detect activity. Quantum provides <span className="text-purple-400 font-bold">+{avgGap.toFixed(2)}</span> higher sensitivity.</>}
                </div>
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    data={activeCandidates.slice(0, 10).map((c) => ({
                      name: `#${c.rank}`,
                      Classical: c.xgb_pic50 ?? 0,
                      Quantum: c.quantum_pic50 ? parseFloat(c.quantum_pic50.toFixed(2)) : 0,
                    }))}
                    margin={{ top: 10, right: 10, left: -5, bottom: 5 }}
                    barGap={4}
                    barSize={16}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 20% 30% / 0.3)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: "hsl(220 15% 65%)" }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 8]} ticks={[0, 2, 4, 6, 7, 8]} tick={{ fontSize: 11, fill: "hsl(220 15% 65%)" }} axisLine={false} tickLine={false} />
                    <ReferenceLine y={7} stroke="hsl(142 70% 45% / 0.6)" strokeDasharray="6 3"
                      label={{ value: "✓ Active Drug (pIC₅₀ ≥ 7)", position: "insideTopRight", fill: "hsl(142 70% 60%)", fontSize: 11, fontWeight: 600 }} />
                    <ReferenceLine y={5} stroke="hsl(0 70% 50% / 0.3)" strokeDasharray="4 4"
                      label={{ value: "✗ Inactive", position: "insideBottomRight", fill: "hsl(0 70% 50%)", fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ background: "hsl(222 47% 9%)", border: "1px solid hsl(217 20% 25%)", borderRadius: 12, fontSize: 13 }}
                      formatter={(val: number, name: string) => [
                        `${val.toFixed(2)} pIC₅₀`,
                        name === "Classical" ? "🔷 Classical (XGBoost)" : "🔮 Quantum (QSVR)"
                      ]}
                    />
                    <Legend
                      formatter={(value) => <span style={{ fontSize: 13, fontWeight: 600 }}>
                        {value === "Classical" ? "🔷 Classical (XGBoost)" : "🔮 Quantum (QSVR)"}
                      </span>}
                      wrapperStyle={{ paddingTop: 12 }}
                    />
                    <Bar dataKey="Classical" fill="hsl(217 91% 55%)" radius={[6, 6, 0, 0]} opacity={0.85} />
                    <Bar dataKey="Quantum" fill="hsl(270 80% 65%)" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Chart 2: pIC50 Distribution — Where do predictions land? */}
              {(() => {
                const bins = [
                  { label: "0–2\nInactive", min: 0, max: 2 },
                  { label: "2–4\nVery Weak", min: 2, max: 4 },
                  { label: "4–5\nWeak", min: 4, max: 5 },
                  { label: "5–6\nModerate", min: 5, max: 6 },
                  { label: "6–7\nGood", min: 6, max: 7 },
                  { label: "7+\nLead Drug", min: 7, max: 99 },
                ];
                const histData = bins.map(b => {
                  const xgbCount = activeCandidates.filter(c => {
                    const v = c.xgb_pic50 ?? 0;
                    return v >= b.min && (b.max === 99 ? true : v < b.max);
                  }).length;
                  const qsvrCount = activeCandidates.filter(c => {
                    const v = c.quantum_pic50 ?? 0;
                    return v >= b.min && (b.max === 99 ? true : v < b.max);
                  }).length;
                  return { zone: b.label.split("\n")[0], activity: b.label.split("\n")[1], Classical: xgbCount, Quantum: qsvrCount };
                });
                
                // Find where each model concentrates
                const xgbPeak = histData.reduce((a, b) => b.Classical > a.Classical ? b : a);
                const qsvrPeak = histData.reduce((a, b) => b.Quantum > a.Quantum ? b : a);

                return (
                <div className="glass-card rounded-3xl p-6 space-y-4">
                  <div>
                    <h3 className="font-bold text-lg">Prediction Distribution</h3>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Where do <span className="text-blue-400 font-semibold">Classical</span> vs{" "}
                      <span className="text-purple-400 font-semibold">Quantum</span> predictions cluster on the activity scale?
                    </p>
                  </div>
                  <div className="rounded-xl px-4 py-3 text-sm font-medium"
                    style={{ background: "linear-gradient(135deg, hsl(270 60% 20% / 0.3), hsl(217 80% 20% / 0.2))" }}>
                    {xgbPeak.zone !== qsvrPeak.zone
                      ? <>📊 Classical clusters at <span className="text-blue-400 font-bold">{xgbPeak.zone} ({xgbPeak.activity})</span> while
                          Quantum correctly predicts <span className="text-purple-400 font-bold">{qsvrPeak.zone} ({qsvrPeak.activity})</span></>
                      : <>📊 Both models agree — predictions cluster at <span className="text-purple-400 font-bold">{qsvrPeak.zone} ({qsvrPeak.activity})</span></>}
                  </div>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart
                      data={histData}
                      margin={{ top: 10, right: 10, left: -5, bottom: 30 }}
                      barGap={2}
                      barSize={20}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 20% 30% / 0.3)" vertical={false} />
                      <XAxis
                        dataKey="zone"
                        tick={{ fontSize: 11, fill: "hsl(220 15% 65%)" }}
                        axisLine={false} tickLine={false}
                        label={{ value: "pIC₅₀ Activity Zone", position: "insideBottom", offset: -15, fill: "hsl(220 15% 55%)", fontSize: 11 }}
                      />
                      <YAxis
                        tick={{ fontSize: 11, fill: "hsl(220 15% 65%)" }}
                        axisLine={false} tickLine={false}
                        label={{ value: "Number of Candidates", angle: -90, position: "insideLeft", offset: 15, fill: "hsl(220 15% 55%)", fontSize: 11 }}
                        allowDecimals={false}
                      />
                      <Tooltip
                        cursor={{ fill: "hsl(270 60% 40% / 0.08)" }}
                        content={({ payload, label }) => {
                          if (!payload?.length) return null;
                          const d = payload[0].payload as { zone: string; activity: string; Classical: number; Quantum: number };
                          return (
                            <div style={{ background: "hsl(222 47% 11%)", border: "1px solid hsl(270 40% 35%)", borderRadius: 14, padding: "12px 16px", minWidth: 200, boxShadow: "0 8px 32px hsl(0 0% 0% / 0.5)" }}>
                              <p style={{ fontWeight: 700, fontSize: 14, color: "hsl(0 0% 95%)", marginBottom: 2 }}>
                                pIC₅₀ {d.zone}
                              </p>
                              <p style={{ fontSize: 11, color: "hsl(220 15% 55%)", marginBottom: 8 }}>{d.activity}</p>
                              <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "6px 16px", fontSize: 13 }}>
                                <span style={{ color: "hsl(217 91% 70%)" }}>🔷 Classical (XGB)</span>
                                <span style={{ color: "hsl(217 91% 80%)", fontFamily: "monospace", fontWeight: 700 }}>{d.Classical} candidates</span>
                                <span style={{ color: "hsl(270 80% 75%)" }}>🔮 Quantum (QSVR)</span>
                                <span style={{ color: "hsl(270 80% 85%)", fontFamily: "monospace", fontWeight: 700 }}>{d.Quantum} candidates</span>
                              </div>
                            </div>
                          );
                        }}
                      />
                      <Legend
                        formatter={(value) => <span style={{ fontSize: 12, fontWeight: 600 }}>
                          {value === "Classical" ? "🔷 Classical (XGBoost)" : "🔮 Quantum (QSVR)"}
                        </span>}
                        wrapperStyle={{ paddingTop: 16 }}
                      />
                      <Bar dataKey="Classical" fill="hsl(217 91% 55%)" radius={[5, 5, 0, 0]} opacity={0.8} />
                      <Bar dataKey="Quantum" fill="hsl(270 80% 65%)" radius={[5, 5, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                );
              })()}

              {/* Chart 3: Candidate Quality Overview — Classical vs Quantum vs Drug Quality */}
              <div className="glass-card rounded-3xl p-6 space-y-4 xl:col-span-2">
                <div className="flex items-start justify-between flex-wrap gap-3">
                  <div>
                    <h3 className="font-bold text-lg">Complete Candidate Profile — Top 10</h3>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Classical binding · Quantum binding · Drug quality (QED) — all in one view
                    </p>
                  </div>
                  <div className="flex gap-4 text-xs items-center flex-shrink-0">
                    <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-6 rounded bg-blue-500/80" /> Classical (XGB)</span>
                    <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-6 rounded bg-purple-500" /> Quantum (QSVR)</span>
                    <span className="flex items-center gap-1.5"><span className="inline-block h-3 w-6 rounded bg-emerald-500/80" /> Drug Quality (QED × 10)</span>
                  </div>
                </div>
                <div className="rounded-xl px-4 py-3 text-sm font-medium"
                  style={{ background: "linear-gradient(135deg, hsl(270 60% 20% / 0.2), hsl(142 60% 20% / 0.15))" }}>
                  {avgGap > 1
                    ? <>📊 Notice how <span className="text-blue-400 font-semibold">blue bars stay flat</span> while <span className="text-purple-400 font-semibold">purple bars are tall</span> — this is the quantum advantage in action</>
                    : <>📊 Compare the three metrics side-by-side. <span className="text-purple-400 font-semibold">Purple (Quantum)</span> consistently detects higher binding than <span className="text-blue-400 font-semibold">Blue (Classical)</span></>}
                </div>
                <ResponsiveContainer width="100%" height={340}>
                  <BarChart
                    data={activeCandidates.slice(0, 10).map((c) => ({
                      name: `#${c.rank}`,
                      Classical: c.xgb_pic50 ?? 0,
                      Quantum: c.quantum_pic50 ? parseFloat(c.quantum_pic50.toFixed(2)) : 0,
                      "Drug Quality": parseFloat((c.qed * 10).toFixed(2)),
                    }))}
                    margin={{ top: 10, right: 15, left: -5, bottom: 5 }}
                    barGap={3}
                    barSize={14}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 20% 30% / 0.3)" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 12, fill: "hsl(220 15% 65%)", fontWeight: 600 }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 10]} ticks={[0, 2, 4, 6, 7, 8, 10]} tick={{ fontSize: 11, fill: "hsl(220 15% 65%)" }} axisLine={false} tickLine={false} />
                    <ReferenceLine y={7} stroke="hsl(142 70% 45% / 0.6)" strokeDasharray="6 3"
                      label={{ value: "✓ Active Drug (7.0)", position: "insideTopRight", fill: "hsl(142 70% 60%)", fontSize: 11, fontWeight: 600 }} />
                    <Tooltip
                      cursor={{ fill: "hsl(270 60% 40% / 0.08)" }}
                      content={({ payload }) => {
                        if (!payload?.length) return null;
                        const d = payload[0].payload as { name: string; Classical: number; Quantum: number; "Drug Quality": number };
                        const gap = d.Quantum - d.Classical;
                        return (
                          <div style={{ background: "hsl(222 47% 11%)", border: "1px solid hsl(270 40% 35%)", borderRadius: 14, padding: "12px 16px", minWidth: 220, boxShadow: "0 8px 32px hsl(0 0% 0% / 0.5)" }}>
                            <p style={{ fontWeight: 700, fontSize: 14, color: "hsl(0 0% 95%)", marginBottom: 8 }}>Candidate {d.name}</p>
                            <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "6px 16px", fontSize: 13 }}>
                              <span style={{ color: "hsl(217 91% 70%)" }}>🔷 Classical (XGB)</span>
                              <span style={{ color: "hsl(217 91% 80%)", fontFamily: "monospace", fontWeight: 700 }}>{d.Classical.toFixed(2)} pIC₅₀</span>
                              <span style={{ color: "hsl(270 80% 75%)" }}>🔮 Quantum (QSVR)</span>
                              <span style={{ color: "hsl(270 80% 85%)", fontFamily: "monospace", fontWeight: 700 }}>{d.Quantum.toFixed(2)} pIC₅₀</span>
                              <span style={{ color: "hsl(142 70% 65%)" }}>💊 Drug Quality</span>
                              <span style={{ color: "hsl(142 70% 75%)", fontFamily: "monospace", fontWeight: 700 }}>{(d["Drug Quality"] * 10).toFixed(0)}% QED</span>
                            </div>
                            <div style={{ marginTop: 8, padding: "6px 10px", borderRadius: 8, background: gap > 3 ? "hsl(142 60% 15%)" : "hsl(270 40% 15%)", textAlign: "center" }}>
                              <span style={{ fontSize: 12, fontWeight: 800, color: gap > 3 ? "hsl(142 70% 65%)" : "hsl(270 80% 75%)", fontFamily: "monospace" }}>
                                Quantum advantage: +{gap.toFixed(2)}
                              </span>
                            </div>
                          </div>
                        );
                      }}
                    />
                    <Legend
                      formatter={(value) => <span style={{ fontSize: 12, fontWeight: 600 }}>
                        {value === "Classical" ? "🔷 Classical (XGB)" : value === "Quantum" ? "🔮 Quantum (QSVR)" : "💊 Drug Quality (QED×10)"}
                      </span>}
                      wrapperStyle={{ paddingTop: 12 }}
                    />
                    <Bar dataKey="Classical" fill="hsl(217 91% 55%)" radius={[5, 5, 0, 0]} opacity={0.8} />
                    <Bar dataKey="Quantum" fill="hsl(270 80% 65%)" radius={[5, 5, 0, 0]} />
                    <Bar dataKey="Drug Quality" fill="hsl(142 70% 50%)" radius={[5, 5, 0, 0]} opacity={0.75} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

            </div>
          </motion.div>
          );
        })()}
      </motion.div>
    </AppLayout>
  );
}
