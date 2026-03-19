import { motion } from "framer-motion";
import { Shield, Filter } from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";

interface ADMETFilterBarProps {
    counts: { pass: number; caution: number; fail: number };
    onFilterChange?: (filter: "all" | "pass" | "caution" | "fail") => void;
}

const pills = [
    { key: "all" as const, label: "All", color: "" },
    { key: "pass" as const, label: "Pass", color: "text-success" },
    { key: "caution" as const, label: "Caution", color: "text-warning" },
    { key: "fail" as const, label: "Fail", color: "text-destructive" },
];

export default function ADMETFilterBar({ counts, onFilterChange }: ADMETFilterBarProps) {
    const [active, setActive] = useState<"all" | "pass" | "caution" | "fail">("all");

    const handleClick = (key: typeof active) => {
        setActive(key);
        onFilterChange?.(key);
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card rounded-2xl px-4 py-3 flex items-center gap-3"
        >
            <div className="flex items-center gap-1.5 text-muted-foreground">
                <Shield className="h-4 w-4 text-quantum" />
                <span className="text-xs font-semibold uppercase tracking-wider">ADMET</span>
            </div>

            <div className="h-4 w-px bg-white/10" />

            <div className="flex items-center gap-1.5">
                {pills.map((pill) => {
                    const count = pill.key === "all" ? counts.pass + counts.caution + counts.fail : counts[pill.key];
                    return (
                        <button
                            key={pill.key}
                            onClick={() => handleClick(pill.key)}
                            className={cn(
                                "relative rounded-xl px-3 py-1.5 text-xs font-medium transition-all duration-300",
                                active === pill.key
                                    ? "text-foreground"
                                    : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            {active === pill.key && (
                                <motion.div
                                    layoutId="admet-filter"
                                    className="absolute inset-0 rounded-xl glass-surface ring-1 ring-quantum/20"
                                    transition={{ type: "spring", stiffness: 350, damping: 30 }}
                                />
                            )}
                            <span className="relative z-10 flex items-center gap-1">
                                {pill.label}
                                <span className={cn("font-mono", pill.color)}>{count}</span>
                            </span>
                        </button>
                    );
                })}
            </div>

            <div className="ml-auto">
                <Filter className="h-3.5 w-3.5 text-muted-foreground" />
            </div>
        </motion.div>
    );
}
