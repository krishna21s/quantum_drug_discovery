import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/components/ThemeProvider";
import { LiquidBackground } from "@/components/LiquidBackground";
import { ExperimentProvider } from "@/context/ExperimentContext";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import QuantumLab from "./pages/QuantumLab";
import Experiment from "./pages/Experiment";
import Molecules from "./pages/Molecules";
import Reports from "./pages/Reports";
import ADMETPage from "./pages/ADMET";
import Visualization from "./pages/Visualization";
import Analysis from "./pages/Analysis";
import ToxicityScreening from "./pages/ToxicityScreening";
import Refinement from "./pages/Refinement";
import ExperimentResults from "./pages/ExperimentResults";
import VqcCircuits from "./pages/VqcCircuits";
import SimilarityAnalyzer from "./pages/SimilarityAnalyzer";

const queryClient = new QueryClient();

const App = () => (
  <ThemeProvider>
    <QueryClientProvider client={queryClient}>
      <ExperimentProvider>
        <TooltipProvider>
          <LiquidBackground />
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/experiment" element={<Experiment />} />
              <Route path="/experiment/results" element={<ExperimentResults />} />
              <Route path="/quantum" element={<QuantumLab />} />
              <Route path="/quantum/vqc" element={<VqcCircuits />} />
              <Route path="/molecules" element={<Molecules />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/admet" element={<ADMETPage />} />
              <Route path="/visualization" element={<Visualization />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/toxicity" element={<ToxicityScreening />} />
              <Route path="/refinement" element={<Refinement />} />
              <Route path="/similarity" element={<SimilarityAnalyzer />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </BrowserRouter>
        </TooltipProvider>
      </ExperimentProvider>
    </QueryClientProvider>
  </ThemeProvider>
);

export default App;
