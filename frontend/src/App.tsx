import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "@/components/ThemeProvider";
import { LiquidBackground } from "@/components/LiquidBackground";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import QuantumLab from "./pages/QuantumLab";
import Experiment from "./pages/Experiment";
import Results from "./pages/Results";
import Molecules from "./pages/Molecules";
import Reports from "./pages/Reports";
import ADMETPage from "./pages/ADMET";
import Visualization from "./pages/Visualization";
import Simulation from "./pages/Simulation";
import Analysis from "./pages/Analysis";
import ToxicityScreening from "./pages/ToxicityScreening";

const queryClient = new QueryClient();

const App = () => (
  <ThemeProvider>
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <LiquidBackground />
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/experiment" element={<Experiment />} />
            <Route path="/quantum" element={<QuantumLab />} />
            <Route path="/results" element={<Results />} />
            <Route path="/molecules" element={<Molecules />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/admet" element={<ADMETPage />} />
            <Route path="/visualization" element={<Visualization />} />
            <Route path="/simulation" element={<Simulation />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/toxicity" element={<ToxicityScreening />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  </ThemeProvider>
);

export default App;
