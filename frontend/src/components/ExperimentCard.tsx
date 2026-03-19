import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ExperimentCardProps {
  id: string;
  name: string;
  protein: string;
  status: "running" | "completed" | "queued";
  score?: number;
  date: string;
  index?: number;
}

const statusStyles = {
  running: "bg-quantum/10 text-quantum ring-1 ring-quantum/30",
  completed: "bg-success/10 text-success ring-1 ring-success/30",
  queued: "bg-warning/10 text-warning ring-1 ring-warning/30",
};

export default function ExperimentCard({ name, protein, status, score, date, index = 0 }: ExperimentCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08, ease: "easeOut" }}
      whileHover={{ y: -2, transition: { duration: 0.2 } }}
      className="glass-card rounded-2xl p-5 cursor-pointer transition-all duration-300 hover:glow-cyan group relative overflow-hidden"
    >
      {/* Hover border glow */}
      <div className="absolute inset-0 rounded-2xl border border-transparent group-hover:border-quantum/20 transition-colors duration-300" />

      <div className="relative z-10">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold">{name}</h3>
          <span className={cn(
            "rounded-full px-2.5 py-0.5 text-xs font-medium",
            statusStyles[status],
            status === "running" && "animate-pulse-glow"
          )}>
            {status}
          </span>
        </div>
        <p className="mt-1 text-sm text-muted-foreground">
          Target: <span className="font-mono text-quantum/80">{protein}</span>
        </p>
        <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
          <span>{date}</span>
          {score !== undefined && (
            <span className="font-mono text-quantum font-semibold">Score: {score.toFixed(2)}</span>
          )}
        </div>
      </div>
    </motion.div>
  );
}
