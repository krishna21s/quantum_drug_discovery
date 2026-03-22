import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown } from "lucide-react";
import { MagicCard } from "./ui/magic-card";

interface StatCardProps {
  title: string;
  value: string;
  subtitle: string;
  icon: LucideIcon;
  trend?: string;
  trendUp?: boolean;
  variant?: "default" | "quantum" | "success" | "warning" | "danger";
  index?: number;
  progress?: number; // 0-100
  unit?: string;
}

const variantConfig = {
  default:  { from: "hsl(207 100% 50%)", to: "hsl(207 100% 35%)", text: "text-primary",    bg: "bg-primary/12",  ring: "ring-primary/25"  },
  quantum:  { from: "hsl(187 85% 55%)",  to: "hsl(187 85% 35%)",  text: "text-quantum",    bg: "bg-quantum/12",  ring: "ring-quantum/25"  },
  success:  { from: "hsl(145 63% 49%)",  to: "hsl(145 63% 30%)",  text: "text-success",    bg: "bg-success/12",  ring: "ring-success/25"  },
  warning:  { from: "hsl(38 95% 56%)",   to: "hsl(38 95% 38%)",   text: "text-warning",    bg: "bg-warning/12",  ring: "ring-warning/25"  },
  danger:   { from: "hsl(350 85% 62%)",  to: "hsl(350 85% 42%)",  text: "text-destructive",bg: "bg-destructive/12",ring:"ring-destructive/25"},
};

export default function StatCard({
  title, value, subtitle, icon: Icon, trend, trendUp = true,
  variant = "default", index = 0, progress, unit,
}: StatCardProps) {
  const cfg = variantConfig[variant];

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1, ease: [0.22, 1, 0.36, 1] }}
      whileHover={{ y: -6, transition: { duration: 0.25 } }}
      className="h-full cursor-default"
    >
      <MagicCard 
        className="rounded-3xl p-5"
        gradientColor={`${cfg.from.replace(')', ' / 0.15)')}`}
      >
        {/* Top accent bar */}
        <div
          className="absolute top-0 left-5 right-5 h-[2px] rounded-full opacity-70"
          style={{ background: `linear-gradient(90deg, transparent, ${cfg.from}, transparent)` }}
        />

        <div className="flex items-start justify-between relative z-10">
          <div className="flex-1">
            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">{title}</p>
            <div className="mt-2 flex items-end gap-1.5">
              <p className="text-3xl font-bold tracking-tight leading-none">{value}</p>
              {unit && <span className="text-sm text-muted-foreground mb-0.5 font-medium">{unit}</span>}
            </div>
            <p className="mt-1.5 text-xs text-muted-foreground">{subtitle}</p>
          </div>

          {/* Icon pill */}
          <div
            className={cn(
              "flex h-11 w-11 items-center justify-center rounded-2xl ring-1 transition-all duration-300 group-hover:scale-110 flex-shrink-0",
              cfg.bg, cfg.ring
            )}
          >
            <Icon className={cn("h-5 w-5", cfg.text)} />
          </div>
        </div>

        {/* Progress bar */}
        {progress !== undefined && (
          <div className="mt-4 progress-bar">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 1, delay: index * 0.1 + 0.3, ease: "easeOut" }}
              className="progress-fill"
              style={{ background: `linear-gradient(90deg, ${cfg.from}, ${cfg.to})` }}
            />
          </div>
        )}

        {/* Trend */}
        {trend && (
          <div className={cn(
            "mt-3 inline-flex items-center gap-1.5 text-xs font-semibold",
            trendUp ? "text-success" : "text-destructive"
          )}>
            {trendUp ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
            {trend}
          </div>
        )}
      </MagicCard>
    </motion.div>
  );
}
