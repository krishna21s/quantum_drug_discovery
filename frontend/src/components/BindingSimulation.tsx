export default function BindingSimulation() {
  return (
    <div className="relative overflow-hidden h-full flex flex-col justify-center">
      <h3 className="mb-4 text-sm font-semibold text-muted-foreground uppercase tracking-wider">
        Binding Analysis
      </h3>
      
      <div className="grid grid-cols-3 gap-3 text-center">
        <div className="rounded-2xl bg-background border border-border/50 p-4 transition-all duration-300">
          <p className="text-xs text-muted-foreground mb-1">Binding Score</p>
          <p className="font-mono font-bold text-lg text-foreground">0.94</p>
        </div>
        <div className="rounded-2xl bg-background border border-border/50 p-4 transition-all duration-300">
          <p className="text-xs text-muted-foreground mb-1">Site Coverage</p>
          <p className="font-mono font-bold text-lg text-foreground">87%</p>
        </div>
        <div className="rounded-2xl bg-background border border-border/50 p-4 transition-all duration-300">
          <p className="text-xs text-muted-foreground mb-1">Stability</p>
          <p className="font-mono font-bold text-lg text-foreground">High</p>
        </div>
      </div>
    </div>
  );
}
