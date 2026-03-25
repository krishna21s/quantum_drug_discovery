import AppLayout from "@/components/AppLayout";
import StatCard from "@/components/StatCard";
import { motion } from "framer-motion";
import {
  FlaskConical, Atom, Zap, Target, ArrowRight,
  Activity, TrendingUp, Clock, Sparkles, CheckCircle2,
  Circle, Loader2, ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip as ReTooltip, ResponsiveContainer,
} from "recharts";
import { MagicCard } from "@/components/ui/magic-card";
import { useEffect, useState, useCallback } from "react";

const API_BASE = "http://localhost:8000";

interface DashboardStats {
  experiments_count: number;
  experiments_this_week: number;
  molecules_count: number;
  molecules_ai_generated: number;
  quantum_runs: number;
  active_candidates: number;
  high_confidence_candidates: number;
  recent_experiments: {
    id: string;
    name: string;
    protein: string;
    status: "completed" | "running" | "queued";
    score: number | null;
    date: string;
    n_candidates: number;
  }[];
  activity_chart: { day: string; score: number }[];
  trend_pct: number;
}

const statusConfig = {
  completed: { icon: CheckCircle2, label: "Complete", color: "text-success", dot: "bg-success" },
  running: { icon: Loader2, label: "Running", color: "text-primary", dot: "bg-primary" },
  queued: { icon: Clock, label: "Queued", color: "text-muted-foreground", dot: "bg-muted-foreground" },
};

const quickActions = [
  { label: "New Experiment", icon: FlaskConical, href: "/experiment" },
  { label: "Quantum Lab", icon: Zap, href: "/quantum" },
  { label: "Molecules", icon: Atom, href: "/molecules" },
  { label: "ADMET Screen", icon: Activity, href: "/admet" },
];

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.08 } } },
  item: { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0 } },
};

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchStats = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/dashboard/stats`);
      if (!res.ok) throw new Error("Failed to fetch dashboard stats");
      const data = await res.json();
      setStats(data);
    } catch (e) {
      console.error("Dashboard stats fetch failed:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  // Derived values
  const experimentsCount = stats?.experiments_count ?? 0;
  const experimentsThisWeek = stats?.experiments_this_week ?? 0;
  const moleculesCount = stats?.molecules_count ?? 0;
  const moleculesAI = stats?.molecules_ai_generated ?? 0;
  const quantumRuns = stats?.quantum_runs ?? 0;
  const activeCandidates = stats?.active_candidates ?? 0;
  const highConfidence = stats?.high_confidence_candidates ?? 0;
  const recentExperiments = stats?.recent_experiments ?? [];
  const activityData = stats?.activity_chart ?? [];
  const trendPct = stats?.trend_pct ?? 0;

  // Progress bar calculations
  const expProgress = experimentsCount > 0 ? Math.min(Math.round((experimentsThisWeek / experimentsCount) * 100), 100) : 0;
  const molProgress = moleculesCount > 0 ? Math.min(Math.round((moleculesAI / moleculesCount) * 100), 100) : 0;
  const qRunProgress = quantumRuns > 0 ? Math.min(Math.round((quantumRuns / (quantumRuns + 10)) * 100), 100) : 0;
  const candProgress = activeCandidates > 0 ? Math.min(Math.round((highConfidence / activeCandidates) * 100), 100) : 0;

  return (
    <AppLayout>
      <div className="min-h-screen p-6 space-y-6">

        {/* ── Header ── */}
        <motion.div
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="flex items-center justify-between"
        >
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="stat-pill-primary">
                <Sparkles className="h-3 w-3" />
                Quantum–AI Platform
              </span>
            </div>
            <h1 className="text-3xl font-bold tracking-tight">
              Welcome back, <span className="gradient-text">Researcher</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-1">Your drug discovery command center</p>
          </div>

          <div className="flex items-center gap-3">
            <Link to="/experiment">
              <Button
                className="rounded-2xl font-semibold bg-primary text-primary-foreground hover:opacity-90 transition-all"
                style={{
                  boxShadow: "0 8px 24px -4px hsl(var(--primary) / 0.4)",
                  border: "none",
                }}
              >
                <FlaskConical className="h-4 w-4 mr-2" />
                New Experiment
              </Button>
            </Link>
          </div>
        </motion.div>

        {/* ── Stats Row ── */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            title="Experiments"
            value={loading ? "—" : String(experimentsCount)}
            subtitle={loading ? "loading…" : `${experimentsThisWeek} this week`}
            icon={FlaskConical}
            variant="default"
            index={0}
            progress={expProgress}
            trend={experimentsThisWeek > 0 ? `${experimentsThisWeek} new this week` : undefined}
            trendUp={experimentsThisWeek > 0}
          />
          <StatCard
            title="Molecules Tested"
            value={loading ? "—" : moleculesCount.toLocaleString()}
            subtitle={loading ? "loading…" : `${moleculesAI} AI-generated`}
            icon={Atom}
            variant="quantum"
            index={1}
            progress={molProgress}
          />
          <StatCard
            title="Quantum Runs"
            value={loading ? "—" : String(quantumRuns)}
            subtitle="VQE + VQC combined"
            icon={Zap}
            variant="warning"
            index={2}
            progress={qRunProgress}
          />
          <StatCard
            title="Active Candidates"
            value={loading ? "—" : String(activeCandidates)}
            subtitle={loading ? "loading…" : `${highConfidence} high confidence`}
            icon={Target}
            variant="success"
            index={3}
            progress={candProgress}
          />
        </div>

        {/* ── Main Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

          {/* ── Left: Activity Chart ── */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="lg:col-span-2 relative h-full"
          >
          <MagicCard className="rounded-3xl p-5 h-full w-full bg-card border border-border">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="font-semibold text-sm">Discovery Activity</h2>
                <p className="text-xs text-muted-foreground mt-0.5">Avg binding score (pIC50) trend — last 7 days</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 text-xs font-medium text-foreground">
                  <TrendingUp className="h-3.5 w-3.5" />
                  <span>{trendPct > 0 ? "+" : ""}{trendPct}%</span>
                </div>
                <div className="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11px] font-semibold bg-primary/15 text-primary-foreground dark:text-primary">This Week</div>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={activityData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                <defs>
                  <linearGradient id="activityGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(var(--foreground))" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="hsl(var(--foreground))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} domain={['auto', 'auto']} />
                <ReTooltip
                  contentStyle={{
                    background: "hsl(var(--card))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "0.75rem",
                    fontSize: "12px",
                    color: "hsl(var(--foreground))",
                    boxShadow: "var(--shadow-card)",
                  }}
                  formatter={(val: number) => [val.toFixed(2), "Avg pIC50"]}
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  stroke="hsl(var(--foreground))"
                  strokeWidth={2}
                  fill="url(#activityGrad)"
                  dot={{ fill: "hsl(var(--foreground))", r: 3, strokeWidth: 0 }}
                  activeDot={{ r: 5, fill: "hsl(var(--primary))", stroke: "hsl(var(--foreground))", strokeWidth: 1.5 }}
                />
              </AreaChart>
            </ResponsiveContainer>
           </MagicCard>
          </motion.div>

          {/* ── Right: Quick Actions ── */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="relative h-full"
          >
          <MagicCard className="rounded-3xl p-5 h-full w-full bg-card border border-border">
            <h2 className="font-semibold text-sm mb-1">Quick Actions</h2>
            <p className="text-xs text-muted-foreground mb-4">Jump to a module</p>
            <div className="grid grid-cols-2 gap-2">
              {quickActions.map((a, i) => (
                <motion.div
                  key={a.href}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.35 + i * 0.07 }}
                >
                  <Link to={a.href}>
                    <div className="bg-muted/15 dark:bg-muted/10 border border-border/40 rounded-2xl p-3 flex flex-col items-center gap-2 text-center group transition-all duration-300 hover:-translate-y-1 hover:shadow-lg cursor-pointer max-h-[80px]">
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 transition-transform group-hover:scale-110">
                        <a.icon className="h-4 w-4 text-foreground" />
                      </div>
                      <span className="text-[11px] font-semibold leading-tight text-foreground">{a.label}</span>
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>

            {/* Mini stat */}
            <div className="mt-4 pt-3 border-t border-border/40">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Backend API</span>
                <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold ${stats ? "bg-primary/15 text-foreground" : "bg-red-500/15 text-red-400"}`}>
                  {stats ? "Online" : "Offline"}
                </span>
              </div>
              <div className="mt-2 progress-bar">
                <div className="progress-fill" style={{ width: stats ? "100%" : "0%", background: "hsl(var(--primary))" }} />
              </div>
              <p className="text-[11px] text-muted-foreground mt-1">
                {stats ? `${experimentsCount} experiments stored` : "Connecting…"}
              </p>
            </div>
           </MagicCard>
          </motion.div>
        </div>

        {/* ── Experiments Table ── */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.35 }}
          className="relative mt-5 h-full"
        >
         <MagicCard className="rounded-3xl p-5 h-full w-full bg-card border border-border">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="font-semibold text-base">Recent Experiments</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Latest drug discovery runs from database</p>
            </div>
            <Link to="/molecules" className="flex items-center gap-1 text-xs font-semibold text-primary hover:underline">
              View all <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          <div className="space-y-3">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                <span className="ml-2 text-sm text-muted-foreground">Loading experiments…</span>
              </div>
            ) : recentExperiments.length === 0 ? (
              <div className="text-center py-8">
                <FlaskConical className="h-8 w-8 text-muted-foreground/40 mx-auto mb-2" />
                <p className="text-sm text-muted-foreground">No experiments yet</p>
                <p className="text-xs text-muted-foreground mt-1">Run your first experiment to see data here</p>
              </div>
            ) : (
              recentExperiments.map((exp, i) => {
                const cfg = statusConfig[exp.status];
                return (
                  <motion.div
                    key={exp.id}
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + i * 0.07 }}
                    className="bg-muted/10 border border-border/30 rounded-2xl px-4 py-3 flex items-center gap-3 group hover:scale-[1.01] transition-all duration-200 cursor-default"
                  >
                    {/* Index */}
                    <div className="h-8 w-8 rounded-xl bg-muted/40 flex items-center justify-center text-xs font-bold text-muted-foreground flex-shrink-0">
                      {String(i + 1).padStart(2, "0")}
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-semibold truncate">{exp.name}</p>
                      <p className="text-xs text-muted-foreground">{exp.protein} · {exp.date} · {exp.n_candidates} candidates</p>
                    </div>

                    {/* Score */}
                    {exp.score !== null ? (
                      <div className="text-right">
                        <p className="text-sm font-bold">{exp.score.toFixed(2)}</p>
                        <p className="text-xs text-muted-foreground">best pIC50</p>
                      </div>
                    ) : (
                      <div className="text-right">
                        <p className="text-xs text-muted-foreground">—</p>
                      </div>
                    )}

                    {/* Status */}
                    <div className={`flex items-center gap-1.5 text-xs font-semibold w-24 justify-end ${cfg.color}`}>
                      <div className={`h-1.5 w-1.5 rounded-full ${cfg.dot} ${exp.status === "running" ? "animate-pulse" : ""}`} />
                      {cfg.label}
                    </div>

                    <ChevronRight className="h-4 w-4 text-muted-foreground/40 group-hover:text-muted-foreground transition-colors" />
                  </motion.div>
                );
              })
            )}
          </div>
         </MagicCard>
        </motion.div>

      </div>
    </AppLayout>
  );
}
