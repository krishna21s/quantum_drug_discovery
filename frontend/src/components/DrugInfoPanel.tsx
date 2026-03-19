import { motion } from "framer-motion";
import { Heart, ExternalLink } from "lucide-react";

interface DrugInfo {
  name: string;
  genericName: string;
  brandName: string;
  group: string;
  target: string;
  mechanism: string;
  administration: string;
  chemicalFormula: string;
  proteinAverage: string;
  weight: string;
  estimatedClearance: string;
  halfLife: string;
  proteinBinding: string;
}

const drugData: DrugInfo = {
  name: "Cetuximab",
  genericName: "Erbitux",
  brandName: "Cetuximab",
  group: "Approved",
  target: "EGFR",
  mechanism: "Monoclonal Antibody",
  administration: "Intravenous",
  chemicalFormula: "C₆₄₈₄H₁₀₀₄₂N₁₇₃₂O₂₀",
  proteinAverage: "145781.6 Da",
  weight: "23536",
  estimatedClearance: "0.103 L/h",
  halfLife: "112 hours",
  proteinBinding: "N/A",
};

export default function DrugInfoPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="glass-card rounded-2xl p-5 space-y-4 relative overflow-hidden"
    >
      {/* Top glow line */}
      <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">{drugData.name}</h3>
        <div className="flex items-center gap-2">
          <button className="p-1.5 rounded-xl hover:bg-muted/30 transition-all duration-300 group">
            <Heart className="h-4 w-4 text-muted-foreground group-hover:text-quantum transition-colors" />
          </button>
          <button className="flex items-center gap-1 text-xs text-quantum hover:underline font-medium">
            Pathway <ExternalLink className="h-3 w-3" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Basic Info */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <span className="h-px flex-1 bg-gradient-to-r from-primary/30 to-transparent" />
            Drug Basic Information
            <span className="h-px flex-1 bg-gradient-to-l from-primary/30 to-transparent" />
          </h4>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            <InfoRow label="Target" value={drugData.target} />
            <InfoRow label="Generic Name" value={drugData.genericName} />
            <InfoRow label="Brand Name" value={drugData.brandName} />
            <InfoRow label="Group" value={drugData.group} highlight />
            <InfoRow label="Mechanism" value={drugData.mechanism} />
            <InfoRow label="Administration" value={drugData.administration} />
          </div>
        </div>

        {/* Physicochemical */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-2">
            <span className="h-px flex-1 bg-gradient-to-r from-quantum/30 to-transparent" />
            Drug Physicochemical Property
            <span className="h-px flex-1 bg-gradient-to-l from-quantum/30 to-transparent" />
          </h4>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
            <InfoRow label="Chemical Formula" value={drugData.chemicalFormula} />
            <InfoRow label="Protein Average" value={drugData.proteinAverage} />
            <InfoRow label="Weight" value={drugData.weight} />
            <InfoRow label="Half Life" value={drugData.halfLife} />
            <InfoRow label="Est. Clearance" value={drugData.estimatedClearance} />
            <InfoRow label="Protein Binding" value={drugData.proteinBinding} />
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function InfoRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="py-0.5">
      <p className="text-muted-foreground">{label}</p>
      <p className={`font-medium ${highlight ? "text-quantum" : ""}`}>{value}</p>
    </div>
  );
}
