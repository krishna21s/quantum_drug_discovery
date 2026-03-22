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

const activityData = [
  { day: "Mon", score: 0.62 },
  { day: "Tue", score: 0.71 },
  { day: "Wed", score: 0.68 },
  { day: "Thu", score: 0.80 },
  { day: "Fri", score: 0.76 },
  { day: "Sat", score: 0.89 },
  { day: "Sun", score: 0.94 },
];

const recentExperiments = [
  { id: "1", name: "SARS-CoV-2 Mpro Inhibitor", protein: "6LU7", status: "completed" as const, score: 0.94, date: "Feb 18" },
  { id: "2", name: "EGFR Kinase Blocker", protein: "1M17", status: "running" as const, score: undefined, date: "Feb 19" },
  { id: "3", name: "HIV-1 Protease Drug", protein: "1HHP", status: "queued" as const, score: undefined, date: "Feb 19" },
  { id: "4", name: "Beta-Lactamase Inhibitor", protein: "1ZG4", status: "completed" as const, score: 0.78, date: "Feb 17" },
];

const statusConfig = {
  completed: { icon: CheckCircle2, label: "Complete", color: "text-success", dot: "bg-success" },
  running: { icon: Loader2, label: "Running", color: "text-primary", dot: "bg-primary" },
  queued: { icon: Clock, label: "Queued", color: "text-muted-foreground", dot: "bg-muted-foreground" },
};

const quickActions = [
  { label: "New Experiment", icon: FlaskConical, href: "/experiment", color: "hsl(207 100% 50%)" },
  { label: "Quantum Lab", icon: Zap, href: "/quantum", color: "hsl(38 95% 56%)" },
  { label: "3D Viewer", icon: Atom, href: "/visualization", color: "hsl(187 85% 55%)" },
  { label: "ADMET Screen", icon: Activity, href: "/admet", color: "hsl(350 85% 62%)" },
];

const stagger = {
  container: { animate: { transition: { staggerChildren: 0.08 } } },
  item: { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0 } },
};

export default function Dashboard() {
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
                className="rounded-2xl font-semibold"
                style={{
                  background: "linear-gradient(135deg, hsl(207 100% 50%), hsl(187 85% 55%))",
                  boxShadow: "0 8px 24px -4px hsl(207 100% 50% / 0.5)",
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
          <StatCard title="Experiments" value="24" subtitle="8 this week" icon={FlaskConical} variant="default" index={0} progress={67} trend="↑ 33% from last week" trendUp />
          <StatCard title="Molecules Tested" value="1,247" subtitle="312 AI-generated" icon={Atom} variant="quantum" index={1} progress={82} />
          <StatCard title="Quantum Runs" value="89" subtitle="VQE + VQC combined" icon={Zap} variant="warning" index={2} progress={56} />
          <StatCard title="Active Candidates" value="7" subtitle="3 high confidence" icon={Target} variant="success" index={3} progress={44} />
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
           <MagicCard className="rounded-3xl p-6 glow-line-top h-full w-full">
            <div className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
              style={{ background: "linear-gradient(90deg, transparent, hsl(207 100% 50%), transparent)" }}
            />
            <div className="flex items-center justify-between mb-6">
              <div>
                <h2 className="font-semibold text-base">Discovery Activity</h2>
                <p className="text-xs text-muted-foreground mt-0.5">Binding score trend this week</p>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex items-center gap-1.5 text-xs font-medium text-success">
                  <TrendingUp className="h-3.5 w-3.5" />
                  <span>+12.4%</span>
                </div>
                <div className="stat-pill-primary text-[11px]">This Week</div>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={activityData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                <defs>
                  <linearGradient id="activityGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="hsl(207 100% 50%)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="hsl(207 100% 50%)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }} domain={[0.5, 1]} />
                <ReTooltip
                  contentStyle={{
                    background: "var(--background)",
                    border: "1px solid var(--border)",
                    borderRadius: "1rem",
                    backdropFilter: "blur(20px)",
                    fontSize: "12px",
                    color: "hsl(var(--foreground))",
                    boxShadow: "0 8px 32px -8px rgba(0,0,0,0.4)",
                  }}
                  formatter={(val: number) => [val.toFixed(2), "Binding Score"]}
                />
                <Area
                  type="monotone"
                  dataKey="score"
                  stroke="hsl(207 100% 50%)"
                  strokeWidth={2.5}
                  fill="url(#activityGrad)"
                  dot={{ fill: "hsl(207 100% 50%)", r: 4, strokeWidth: 0 }}
                  activeDot={{ r: 6, fill: "hsl(207 100% 60%)", stroke: "hsl(207 100% 80%)", strokeWidth: 1.5 }}
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
           <MagicCard className="rounded-3xl p-6 h-full w-full">
            <div className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
              style={{ background: "linear-gradient(90deg, transparent, hsl(187 85% 55%), transparent)" }}
            />
            <h2 className="font-semibold text-base mb-1">Quick Actions</h2>
            <p className="text-xs text-muted-foreground mb-5">Jump to a module</p>
            <div className="grid grid-cols-2 gap-3">
              {quickActions.map((a, i) => (
                <motion.div
                  key={a.href}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.35 + i * 0.07 }}
                >
                  <Link to={a.href}>
                    <div
                      className="bg-muted/20 dark:bg-muted/10 border border-border/50 rounded-2xl p-4 flex flex-col items-center gap-2 text-center group transition-all duration-300 hover:scale-105 hover:shadow-card cursor-pointer"
                      style={{ "--action-color": a.color } as React.CSSProperties}
                    >
                      <div
                        className="flex h-10 w-10 items-center justify-center rounded-2xl transition-transform group-hover:scale-110"
                        style={{ background: `${a.color}20`, boxShadow: `0 4px 12px ${a.color}30` }}
                      >
                        <a.icon style={{ color: a.color, width: 18, height: 18 }} />
                      </div>
                      <span className="text-xs font-semibold leading-tight">{a.label}</span>
                    </div>
                  </Link>
                </motion.div>
              ))}
            </div>

            {/* Mini stat */}
            <div className="mt-5 pt-4 border-t border-border/40">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">GPU Cluster</span>
                <span className="stat-pill-success">Online</span>
              </div>
              <div className="mt-2 progress-bar">
                <div className="progress-fill" style={{ width: "78%", background: "linear-gradient(90deg, hsl(145 63% 49%), hsl(187 85% 55%))" }} />
              </div>
              <p className="text-xs text-muted-foreground mt-1">78% utilisation</p>
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
         <MagicCard className="rounded-3xl p-6 h-full w-full">
          <div className="absolute top-0 left-6 right-6 h-[2px] rounded-full"
            style={{ background: "linear-gradient(90deg, transparent, hsl(280 75% 60%), transparent)" }}
          />
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="font-semibold text-base">Recent Experiments</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Latest drug discovery runs</p>
            </div>
            <Link to="/results" className="flex items-center gap-1 text-xs font-semibold text-primary hover:underline">
              View all <ArrowRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          <div className="space-y-3">
            {recentExperiments.map((exp, i) => {
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
                    <p className="text-xs text-muted-foreground">{exp.protein} · {exp.date}</p>
                  </div>

                  {/* Score */}
                  {exp.score !== undefined ? (
                    <div className="text-right">
                      <p className="text-sm font-bold">{exp.score.toFixed(2)}</p>
                      <p className="text-xs text-muted-foreground">score</p>
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
            })}
          </div>
         </MagicCard>
        </motion.div>

      </div>
    </AppLayout>
  );
}
