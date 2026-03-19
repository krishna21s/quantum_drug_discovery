import AppLayout from "@/components/AppLayout";
import MoleculeViewer from "@/components/MoleculeViewer";
import ADMETFilterBar from "@/components/ADMETFilterBar";
import { motion } from "framer-motion";
import { Atom, Download, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useMemo, useState } from "react";
import { computeADMET, DEMO_MOLECULES, type MolDescriptors } from "@/lib/admetEngine";
import { cn } from "@/lib/utils";

const moleculeDescriptors: Record<string, MolDescriptors> = {
  Aspirin: DEMO_MOLECULES.Aspirin,
  Paracetamol: DEMO_MOLECULES.Paracetamol,
  Cetuximab: DEMO_MOLECULES.Cetuximab,
  Ibuprofen: DEMO_MOLECULES.Ibuprofen,
  Metformin: DEMO_MOLECULES.Metformin,
};

const molecules = [
  { id: 1, name: "Aspirin", formula: "C₉H₈O₄", mw: 180.16, logP: 1.24, score: -8.2, status: "Active" },
  { id: 2, name: "Paracetamol", formula: "C₈H₉NO₂", mw: 151.16, logP: 0.46, score: -7.1, status: "Active" },
  { id: 3, name: "Cetuximab", formula: "C₆₄₈₄H₁₀₀₄₂N₁₇₃₂O₂₀", mw: 145781.6, logP: 0.0, score: -9.4, status: "Active" },
  { id: 4, name: "Ibuprofen", formula: "C₁₃H₁₈O₂", mw: 206.28, logP: 3.97, score: -6.8, status: "Moderate" },
  { id: 5, name: "Metformin", formula: "C₄H₁₁N₅", mw: 129.16, logP: -1.43, score: -5.2, status: "Weak" },
];

const statusColors = {
  "Active": "text-success",
  "Moderate": "text-warning",
  "Weak": "text-destructive",
};

function admetVerdict(name: string): "Pass" | "Caution" | "Fail" {
  const desc = moleculeDescriptors[name];
  if (!desc) return "Caution";
  return computeADMET(desc).scores.verdict;
}

export default function Molecules() {
  const [filter, setFilter] = useState<"all" | "pass" | "caution" | "fail">("all");

  const withVerdicts = useMemo(() =>
    molecules.map((mol) => ({ ...mol, admetVerdict: admetVerdict(mol.name), admetScore: computeADMET(moleculeDescriptors[mol.name]).scores.overall })),
    []
  );

  const counts = useMemo(() => ({
    pass: withVerdicts.filter((m) => m.admetVerdict === "Pass").length,
    caution: withVerdicts.filter((m) => m.admetVerdict === "Caution").length,
    fail: withVerdicts.filter((m) => m.admetVerdict === "Fail").length,
  }), [withVerdicts]);

  const filtered = filter === "all" ? withVerdicts : withVerdicts.filter((m) => m.admetVerdict.toLowerCase() === filter);

  return (
    <AppLayout>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <Atom className="h-6 w-6 text-quantum" />
              Molecule Library
            </h1>
            <p className="text-muted-foreground mt-1">{molecules.length} compounds in database</p>
          </div>
          <Button variant="outline" className="rounded-xl gap-1.5">
            <Download className="h-4 w-4" /> Export CSV
          </Button>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Table */}
          <div className="lg:col-span-2">
            {/* ADMET Filter Bar */}
            <div className="mb-4">
              <ADMETFilterBar counts={counts} onFilterChange={setFilter} />
            </div>

            {/* Search */}
            <div className="liquid-glass rounded-2xl p-4 mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search molecules by name, formula, or PDB ID…"
                  className="w-full rounded-xl border border-white/10 bg-muted/20 backdrop-blur-sm pl-10 pr-4 py-2.5 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary/30 transition-all"
                />
              </div>
            </div>

            <div className="liquid-glass rounded-2xl overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/5">
                    <th className="px-5 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Name</th>
                    <th className="px-5 py-3 text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider">Formula</th>
                    <th className="px-5 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">MW</th>
                    <th className="px-5 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">LogP</th>
                    <th className="px-5 py-3 text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider">Binding</th>
                    <th className="px-5 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">ADMET</th>
                    <th className="px-5 py-3 text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((mol, i) => (
                    <motion.tr
                      key={mol.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.06 }}
                      className="border-b border-white/3 hover:bg-quantum/5 transition-colors cursor-pointer"
                    >
                      <td className="px-5 py-3.5 font-medium">{mol.name}</td>
                      <td className="px-5 py-3.5 font-mono text-xs text-muted-foreground">{mol.formula}</td>
                      <td className="px-5 py-3.5 text-right font-mono text-xs">{mol.mw.toLocaleString()}</td>
                      <td className="px-5 py-3.5 text-right font-mono text-xs">{mol.logP}</td>
                      <td className="px-5 py-3.5 text-right font-mono text-xs text-quantum">{mol.score}</td>
                      <td className="px-5 py-3.5 text-center">
                        <span className={cn(
                          "inline-block rounded-full px-2 py-0.5 text-xs font-semibold ring-1",
                          mol.admetVerdict === "Pass" ? "bg-success/10 text-success ring-success/30" :
                            mol.admetVerdict === "Caution" ? "bg-warning/10 text-warning ring-warning/30" :
                              "bg-destructive/10 text-destructive ring-destructive/30"
                        )}>
                          {(mol.admetScore * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className={`px-5 py-3.5 text-center text-xs font-semibold ${statusColors[mol.status as keyof typeof statusColors]}`}>{mol.status}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Side: Molecule Viewer */}
          <div>
            <MoleculeViewer />
          </div>
        </div>
      </motion.div>
    </AppLayout>
  );
}
