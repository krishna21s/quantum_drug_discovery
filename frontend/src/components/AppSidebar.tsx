import { Link, useLocation } from "react-router-dom";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useTheme } from "./ThemeProvider";
import {
  LayoutDashboard,
  FlaskConical,
  Atom,
  Activity,
  FileText,
  Settings,
  Zap,
  Sun,
  Moon,
  Microscope,
  Shield,
  Beaker,
  PieChart,
  AlertTriangle,
  Sparkles,
} from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const navItems = [
  { path: "/",              label: "Dashboard",     icon: LayoutDashboard, color: "hsl(207 100% 50%)" },
  { path: "/experiment",   label: "New Experiment", icon: FlaskConical,    color: "hsl(187 85% 55%)" },
  { path: "/molecules",    label: "Molecules",      icon: Atom,            color: "hsl(280 80% 65%)" },
  { path: "/quantum",      label: "Quantum Lab",    icon: Zap,             color: "hsl(38 95% 56%)"  },
  { path: "/results",      label: "Results",        icon: Activity,        color: "hsl(145 63% 49%)" },
  { path: "/visualization",label: "3D Viewer",      icon: Microscope,      color: "hsl(187 85% 55%)" },
  { path: "/admet",        label: "ADMET",          icon: Shield,          color: "hsl(350 85% 62%)" },
  { path: "/simulation",   label: "Simulation",     icon: Beaker,          color: "hsl(25 95% 60%)"  },
  { path: "/analysis",     label: "Analysis",       icon: PieChart,        color: "hsl(207 100% 50%)" },
  { path: "/toxicity",     label: "Toxicity",       icon: AlertTriangle,   color: "hsl(0 72% 51%)"   },
  { path: "/refinement",  label: "Lead Optimize",  icon: Sparkles,        color: "hsl(270 70% 60%)" },
  { path: "/reports",      label: "Reports",        icon: FileText,        color: "hsl(280 80% 65%)" },
];

export default function AppSidebar() {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  return (
    <aside
      className="fixed left-0 top-0 z-40 flex h-screen w-[72px] flex-col items-center py-4 gap-2"
      style={{
        background: "var(--glass-bg)",
        backdropFilter: "blur(var(--glass-blur))",
        WebkitBackdropFilter: "blur(var(--glass-blur))",
        borderRight: "1px solid var(--glass-border)",
      }}
    >
      {/* Logo */}
      <div className="flex h-12 w-12 items-center justify-center mb-2">
        <div className="relative flex items-center justify-center w-10 h-10 rounded-2xl bg-gradient-to-br from-primary to-quantum/80 shadow-glow-sm">
          <img src="/QpharmXlogo.png" alt="Q-PharmX" width={22} height={22} />
        </div>
      </div>

      {/* Divider */}
      <div className="w-8 h-px bg-border/60 mb-1" />

      {/* Nav */}
      <nav className="flex flex-1 flex-col items-center gap-1.5 w-full px-2">
        {navItems.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Tooltip key={item.path} delayDuration={0}>
              <TooltipTrigger asChild>
                <Link
                  to={item.path}
                  className={cn(
                    "relative flex h-11 w-11 items-center justify-center rounded-2xl transition-all duration-300",
                    isActive ? "text-white" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-pill"
                      className="absolute inset-0 rounded-2xl"
                      style={{
                        background: `linear-gradient(135deg, ${item.color}, ${item.color}cc)`,
                        boxShadow: `0 8px 24px -4px ${item.color}70`,
                      }}
                      transition={{ type: "spring", stiffness: 400, damping: 32 }}
                    />
                  )}
                  {!isActive && (
                    <div className="absolute inset-0 rounded-2xl opacity-0 hover:opacity-100 transition-opacity duration-200 bg-muted/30" />
                  )}
                  <span className="relative z-10">
                    <item.icon className="h-4.5 w-4.5" style={{ width: 18, height: 18 }} />
                  </span>
                </Link>
              </TooltipTrigger>
              <TooltipContent side="right" className="font-medium">
                {item.label}
              </TooltipContent>
            </Tooltip>
          );
        })}
      </nav>

      {/* Bottom Controls */}
      <div className="flex flex-col items-center gap-2 mt-auto">
        {/* Theme toggle */}
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <button
              onClick={toggleTheme}
              className="flex h-10 w-10 items-center justify-center rounded-2xl text-muted-foreground hover:text-foreground hover:bg-muted/30 transition-all duration-300"
            >
              {theme === "dark" ? (
                <Sun style={{ width: 16, height: 16 }} />
              ) : (
                <Moon style={{ width: 16, height: 16 }} />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">
            {theme === "dark" ? "Light Mode" : "Dark Mode"}
          </TooltipContent>
        </Tooltip>

        {/* User avatar */}
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <button className="flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-primary/30 to-quantum/30 text-xs font-bold text-primary ring-1 ring-white/10 transition-all duration-300 hover:ring-primary/40 hover:scale-105">
              R
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">
            <p className="font-medium">Researcher</p>
            <p className="text-xs text-muted-foreground">researcher@lab.edu</p>
          </TooltipContent>
        </Tooltip>

        {/* Settings */}
        <Tooltip delayDuration={0}>
          <TooltipTrigger asChild>
            <button className="flex h-10 w-10 items-center justify-center rounded-2xl text-muted-foreground hover:text-foreground hover:bg-muted/30 transition-all duration-300">
              <Settings style={{ width: 16, height: 16 }} />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">Settings</TooltipContent>
        </Tooltip>
      </div>
    </aside>
  );
}
