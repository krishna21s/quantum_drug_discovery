import AppLayout from "@/components/AppLayout";
import { motion } from "framer-motion";
import { FileText, Download, Calendar, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const reports = [
  { id: 1, title: "SARS-CoV-2 Mpro Drug Discovery Report", date: "Feb 18, 2026", pages: 24, status: "Final" },
  { id: 2, title: "EGFR Kinase Domain Analysis", date: "Feb 19, 2026", pages: 18, status: "Draft" },
  { id: 3, title: "HIV-1 Protease Inhibitor Screening", date: "Feb 17, 2026", pages: 31, status: "Final" },
  { id: 4, title: "Beta-Lactamase Resistance Study", date: "Feb 16, 2026", pages: 15, status: "Review" },
  { id: 5, title: "Quantum-Classical Hybrid Benchmark", date: "Feb 15, 2026", pages: 12, status: "Final" },
];

const statusStyles = {
  Final: "bg-success/10 text-success ring-1 ring-success/30",
  Draft: "bg-warning/10 text-warning ring-1 ring-warning/30",
  Review: "bg-primary/10 text-primary ring-1 ring-primary/30",
};

export default function Reports() {
  return (
    <AppLayout>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold flex items-center gap-2">
              <FileText className="h-6 w-6 text-quantum" />
              Research Reports
            </h1>
            <p className="text-muted-foreground mt-1">{reports.length} reports generated</p>
          </div>
          <Button variant="hero" className="rounded-xl gap-1.5">
            Generate Report
          </Button>
        </div>

        <div className="space-y-3">
          {reports.map((report, i) => (
            <motion.div
              key={report.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08, duration: 0.4 }}
              className="glass-card rounded-2xl p-5 flex items-center justify-between transition-all duration-300 hover:glow-cyan cursor-pointer group"
            >
              <div className="flex items-center gap-4">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 ring-1 ring-primary/20 group-hover:bg-quantum/10 group-hover:ring-quantum/20 transition-colors duration-300">
                  <FileText className="h-5 w-5 text-primary group-hover:text-quantum transition-colors duration-300" />
                </div>
                <div>
                  <h3 className="font-semibold group-hover:text-quantum transition-colors duration-300">{report.title}</h3>
                  <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1"><Calendar className="h-3 w-3" /> {report.date}</span>
                    <span>{report.pages} pages</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <span className={`rounded-full px-3 py-1 text-xs font-medium ${statusStyles[report.status as keyof typeof statusStyles]}`}>
                  {report.status}
                </span>
                <Button variant="ghost" size="icon" className="rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  <Download className="h-4 w-4" />
                </Button>
                <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-quantum transition-colors duration-300" />
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </AppLayout>
  );
}
