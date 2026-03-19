import { motion } from "framer-motion";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function MoleculeStatsPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-5 space-y-4 relative overflow-hidden"
    >
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

      <div className="flex items-center justify-between">
        <div>
          <span className="text-4xl font-bold gradient-text">541</span>
          <p className="text-sm text-quantum font-medium mt-1">Generated</p>
        </div>
        <Button variant="outline" size="sm" className="gap-1.5 rounded-xl">
          Download results <Download className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
        <StatRow label="AVG Molar mass" value="379.278 g·mol⁻¹" />
        <StatRow label="AVG LogP" value="3.74" />
        <StatRow label="AVG Binding Energy" value="-6.5" />
        <StatRow label="Drug-likeness" value="0.82" />
      </div>

      {/* Molecule detail */}
      <div className="rounded-xl glass-surface p-3.5 space-y-2">
        <p className="text-xs text-muted-foreground font-mono leading-relaxed">
          1,2-Dimethoxy-12-methyl-9H-[1,3] benzodioxolo[5,6-c]phenanthridin-12-ium
        </p>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="py-0.5">
            <p className="text-muted-foreground">Chemical formula</p>
            <p className="font-mono font-semibold mt-0.5">C₂₁H₁₈NO₄</p>
          </div>
          <div className="py-0.5">
            <p className="text-muted-foreground">Molar mass</p>
            <p className="font-mono font-semibold mt-0.5">348.378 g·mol⁻¹</p>
          </div>
          <div className="py-0.5">
            <p className="text-muted-foreground">LogP</p>
            <p className="font-mono font-semibold mt-0.5">2.85</p>
          </div>
          <div className="py-0.5">
            <p className="text-muted-foreground">Binding Energy</p>
            <p className="font-mono font-semibold text-quantum mt-0.5">-12.5</p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground text-xs">{label}</span>
      <span className="font-mono font-semibold text-sm">{value}</span>
    </div>
  );
}
