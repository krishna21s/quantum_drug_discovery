import { ReactNode } from "react";
import AppSidebar from "./AppSidebar";
import AnimatedBackground from "./AnimatedBackground";

export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <div
      className="min-h-screen relative"
      style={{ background: "var(--gradient-glow)", backgroundColor: "hsl(var(--background))" }}
    >
      <AnimatedBackground />
      <AppSidebar />
      {/* Offset from slim 72px sidebar */}
      <main className="ml-[72px] min-h-screen relative z-10">
        {children}
      </main>
    </div>
  );
}
