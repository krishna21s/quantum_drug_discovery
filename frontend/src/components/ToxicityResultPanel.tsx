import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { PredictResponse } from "@/lib/toxicityApi";
import {
  ShieldAlert, ShieldCheck, Zap, Cpu, Layers,
  Clock, Activity, TrendingUp,
} from "lucide-react";

interface Props {
  result: PredictResponse;
}

function ProbGauge({ label, value, icon: Icon, color, delay }: {
  label: string; value: number; icon: typeof Cpu; color: string; delay: number;
}) {
  const pct = Math.round(value * 100);
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="glass-surface rounded-2xl p-5 text-center space-y-3"
    >
      <div
        className="mx-auto flex h-11 w-11 items-center justify-center rounded-2xl ring-1"
        style={{ background: `${color}15`, boxShadow: `0 4px 16px ${color}25`, borderColor: `${color}30` }}
      >
        <Icon style={{ color, width: 20, height: 20 }} />
      </div>
      <p className="text-xs text-muted-foreground font-semibold uppercase tracking-wider">{label}</p>
      <motion.p
        initial={{ scale: 0.6 }}
        animate={{ scale: 1 }}
        transition={{ delay: delay + 0.2, type: "spring", stiffness: 200 }}
        className="text-3xl font-bold font-mono"
        style={{ color }}
      >
        {pct}<span className="text-lg text-muted-foreground">%</span>
      </motion.p>
      <div className="h-1.5 rounded-full bg-muted/30 overflow-hidden ring-1 ring-white/5">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ delay: delay + 0.1, duration: 0.8 }}
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${color}90, ${color})` }}
        />
      </div>
    </motion.div>
  );
}

export default function ToxicityResultPanel({ result }: Props) {
  const isToxic = result.ensemble_probability > 0.5;

  return (
    <div className="space-y-5">
      {/* Verdict Banner */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className={cn(
          "rounded-3xl p-6 text-center relative overflow-hidden",
          isToxic
            ? "bg-destructive/8 ring-1 ring-destructive/30"
            : "bg-success/8 ring-1 ring-success/30"
        )}
      >
        <div className={cn(
          "absolute top-0 left-6 right-6 h-[2px] rounded-full",
        )} style={{
          background: isToxic
            ? "linear-gradient(90deg, transparent, hsl(0 72% 51%), transparent)"
            : "linear-gradient(90deg, transparent, hsl(145 63% 49%), transparent)",
        }} />

        <div className="flex items-center justify-center gap-3">
          {isToxic ? (
            <ShieldAlert className="h-8 w-8 text-destructive" />
          ) : (
            <ShieldCheck className="h-8 w-8 text-success" />
          )}
          <div>
            <p className={cn("text-xl font-bold", isToxic ? "text-destructive" : "text-success")}>
              {result.verdict}
            </p>
            <p className="text-sm text-muted-foreground mt-0.5">
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </motion.div>

      {/* 3 Probability Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <ProbGauge
          label="Classical (XGBoost)"
          value={result.classical_probability}
          icon={Cpu}
          color="hsl(207, 100%, 50%)"
          delay={0.1}
        />
        <ProbGauge
          label="Quantum (QSVM)"
          value={result.quantum_probability}
          icon={Zap}
          color="hsl(280, 80%, 65%)"
          delay={0.2}
        />
        <ProbGauge
          label="Ensemble (Hybrid)"
          value={result.ensemble_probability}
          icon={Layers}
          color={isToxic ? "hsl(0, 72%, 51%)" : "hsl(145, 63%, 49%)"}
          delay={0.3}
        />
      </div>

      {/* Timings & Metadata */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card rounded-2xl p-5 relative overflow-hidden"
      >
        <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent" />

        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Clock className="h-3.5 w-3.5" /> Performance & Metadata
        </h3>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="XGBoost Latency" value={`${result.timings.xgb_ms.toFixed(1)} ms`} />
          <MetricCard label="Quantum Latency" value={`${result.timings.quantum_s.toFixed(2)} s`} />
          <MetricCard label="Baseline Score" value={`${(result.baseline_score * 100).toFixed(0)}%`} />
          <MetricCard label="Mode" value={result.mode === "full" ? "Full (CI)" : "Fast"} />
        </div>

        {result.canonical_smiles && (
          <div className="mt-3 glass-surface rounded-xl px-4 py-2">
            <p className="text-xs text-muted-foreground">Canonical SMILES</p>
            <p className="font-mono text-xs mt-0.5 break-all">{result.canonical_smiles}</p>
          </div>
        )}
      </motion.div>

      {/* CI Panel (if available) */}
      {result.ci && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="glass-card rounded-2xl p-5 relative overflow-hidden"
        >
          <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/30 to-transparent" />

          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 flex items-center gap-1.5">
            <Activity className="h-3.5 w-3.5" /> Confidence Interval (Shot-Based)
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <MetricCard label="Shot Probability" value={`${(result.ci.probability * 100).toFixed(1)}%`} />
            <MetricCard label="Std Dev" value={result.ci.std.toFixed(4)} />
            <MetricCard
              label="95% CI"
              value={`[${(result.ci.ci_lower * 100).toFixed(1)}%, ${(result.ci.ci_upper * 100).toFixed(1)}%]`}
            />
            <MetricCard label="Bootstrap Runs" value={String(result.ci.n_bootstrap)} />
          </div>
        </motion.div>
      )}
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass-surface rounded-xl p-3 text-center">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono font-semibold text-sm mt-0.5">{value}</p>
    </div>
  );
}
