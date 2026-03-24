import AppLayout from "@/components/AppLayout";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { Microscope, Activity, Heart, Shield, Droplets, Target, User, Syringe, Clock, FileText, Share2, Printer, ActivitySquare, BadgeAlert, ShieldAlert, ShieldCheck } from "lucide-react";

const pdbOptions = [
    { id: "1M17", name: "EGFR Kinase" },
    { id: "6LU7", name: "SARS-CoV-2 Mpro" },
    { id: "1HHP", name: "HIV-1 Protease" },
    { id: "3ERT", name: "Estrogen Receptor" },
];

const clinicalTimeline = [
    {
        year: "2024",
        disease: "Non-small Cell Lung Cancer",
        status: "Active Tracking",
        events: [
            { type: "doc", month: "03" },
            { type: "visit", month: "06" },
            { type: "med", name: "Erlotinib", month: "09" },
            { type: "visit", month: "12" },
        ]
    },
    {
        year: "2019",
        disease: "Glioblastoma Multiforme",
        status: "Target Identified",
        events: [
            { type: "doc", month: "04" },
            { type: "lab", month: "08" },
            { type: "doc", month: "11" },
        ]
    },
    {
        year: "2015",
        disease: "Breast Cancer (HER2+)",
        status: "Clinical Phase III",
        events: [
            { type: "med", name: "Lapatinib", month: "02" },
            { type: "visit", month: "05" },
            { type: "doc", month: "10" },
        ]
    }
];

export default function Visualization() {
    const [pdbId, setPdbId] = useState<string>("1M17");

    return (
        <AppLayout>
            <div className="min-h-screen bg-background p-4 md:p-6 lg:p-8 flex flex-col font-sans">
                
                {/* ── Top Header / Navigation ── */}
                <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-6">
                    <div className="flex items-center gap-4 bg-card border border-border px-6 py-3 rounded-full shadow-sm w-full max-w-2xl overflow-x-auto">
                        <button className="flex items-center gap-2 text-sm font-bold bg-primary/10 text-primary px-4 py-2 rounded-full whitespace-nowrap">
                            <Activity className="h-4 w-4" /> Overview
                        </button>
                        <button className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground px-4 py-2 whitespace-nowrap transition-colors">
                            <FileText className="h-4 w-4" /> Clinical Data
                        </button>
                        <button className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground px-4 py-2 whitespace-nowrap transition-colors">
                            <Microscope className="h-4 w-4" /> Assays
                        </button>
                        <button className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground px-4 py-2 whitespace-nowrap transition-colors">
                            <Target className="h-4 w-4" /> Ligands
                        </button>
                    </div>

                    <div className="flex items-center gap-3">
                        <div className="hidden sm:flex rounded-full bg-card border border-border p-1 gap-1 shadow-sm h-[52px]">
                            {pdbOptions.map((opt) => (
                                <button
                                    key={opt.id}
                                    onClick={() => setPdbId(opt.id)}
                                    className={cn(
                                        "px-4 py-2 text-xs font-bold rounded-full transition-all duration-200 uppercase tracking-widest",
                                        pdbId === opt.id
                                            ? "bg-foreground text-background shadow-md"
                                            : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                                    )}
                                >
                                    {opt.id}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* ── Main Dashboard Layout ── */}
                <div className="flex-1 grid grid-cols-1 xl:grid-cols-[320px_1fr_450px] gap-6 items-start h-full">
                    
                    {/* ── Left Sidebar (Target Profile / Vitals) ── */}
                    <div className="space-y-4">
                        {/* Profile Card */}
                        <div className="bg-card border border-border rounded-[32px] p-6 shadow-sm relative overflow-hidden flex flex-col">
                            {/* Accent blur behind profile */}
                            <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 rounded-full blur-[40px] -mr-8 -mt-8 pointer-events-none" />
                            
                            <div className="flex justify-between items-start mb-6 align-top">
                                <div className="h-20 w-20 rounded-2xl bg-muted/50 border border-border flex items-center justify-center p-2 relative z-10 overflow-hidden shadow-inner">
                                    <Target className="h-10 w-10 text-foreground/20 absolute" />
                                    <img src="/anatomy.png" alt="Target" className="h-full w-full object-contain filter drop-shadow-sm opacity-80" />
                                </div>
                                <div className="bg-background border border-border px-3 py-1.5 rounded-full text-[10px] font-bold text-foreground shadow-sm">
                                    PDB: {pdbId}
                                </div>
                            </div>
                            
                            <div className="relative z-10 mb-6">
                                <h1 className="text-2xl font-black text-foreground leading-tight tracking-tight">
                                    Epidermal Growth<br/>Factor Receptor
                                </h1>
                                <p className="text-sm font-semibold text-muted-foreground mt-1 flex items-center gap-1.5">
                                    <User className="h-4 w-4" /> Homo Sapiens
                                </p>
                            </div>

                            <div className="grid grid-cols-2 gap-4 border-t border-border pt-6 pb-2">
                                <div>
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Class</p>
                                    <p className="font-semibold text-sm">Kinase</p>
                                </div>
                                <div>
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Resolution</p>
                                    <p className="font-semibold text-sm">2.60 Å</p>
                                </div>
                                <div>
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Ligand</p>
                                    <p className="font-semibold text-sm">Erlotinib</p>
                                </div>
                                <div>
                                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Weight</p>
                                    <p className="font-semibold text-sm">134 kDa</p>
                                </div>
                            </div>
                        </div>

                        {/* Physiological Metrics (Vitals style) */}
                        <div className="bg-card border border-border rounded-[32px] p-6 shadow-sm">
                            <h3 className="text-sm font-black mb-6 flex items-center gap-2">
                                <ActivitySquare className="h-5 w-5 text-muted-foreground" /> Pharmacodynamics
                            </h3>
                            
                            <div className="space-y-6">
                                {/* Binding Affinity */}
                                <div>
                                    <div className="flex justify-between items-baseline mb-2">
                                        <p className="text-xs font-bold text-muted-foreground">Binding Affinity</p>
                                        <div className="flex items-end gap-1">
                                            <p className="text-2xl font-black">-9.2</p>
                                            <p className="text-[10px] font-bold text-muted-foreground leading-5">kcal/mol</p>
                                        </div>
                                    </div>
                                    <div className="h-14 bg-muted/20 border border-border rounded-xl w-full relative overflow-hidden flex items-end px-2 pt-2">
                                          {/* Mock chart */}
                                          <div className="w-1/5 h-[40%] bg-foreground/20 rounded-t-sm mx-1"></div>
                                          <div className="w-1/5 h-[60%] bg-foreground/40 rounded-t-sm mx-1"></div>
                                          <div className="w-1/5 h-[30%] bg-foreground/30 rounded-t-sm mx-1"></div>
                                          <div className="w-1/5 h-[90%] bg-primary rounded-t-sm mx-1"></div>
                                          <div className="w-1/5 h-[70%] bg-foreground/50 rounded-t-sm mx-1"></div>
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    {/* Clearance */}
                                    <div>
                                        <p className="text-xs font-bold text-muted-foreground mb-1">Clearance Rate</p>
                                        <div className="flex items-baseline gap-1">
                                            <p className="text-xl font-black">12.4</p>
                                            <p className="text-[10px] font-bold text-muted-foreground">L/hr</p>
                                        </div>
                                        <div className="flex items-center gap-1 mt-2">
                                            <div className="h-2 w-2 rounded-full bg-success"></div>
                                            <p className="text-[10px] font-bold">Optimal</p>
                                        </div>
                                    </div>
                                    
                                    {/* Half-life */}
                                    <div>
                                        <p className="text-xs font-bold text-muted-foreground mb-1">Half Life (t½)</p>
                                        <div className="flex items-baseline gap-1">
                                            <p className="text-xl font-black">36.2</p>
                                            <p className="text-[10px] font-bold text-muted-foreground">hrs</p>
                                        </div>
                                        <div className="flex items-center gap-1 mt-2">
                                            <div className="h-2 w-2 rounded-full bg-warning"></div>
                                            <p className="text-[10px] font-bold">Extended</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Alerts Box */}
                        <div className="bg-card border border-border rounded-3xl p-4 shadow-sm flex items-center justify-between">
                            <span className="text-sm font-bold flex items-center gap-2">
                                <ShieldAlert className="h-4 w-4 text-warning" /> Off-Target Risk
                            </span>
                            <div className="flex gap-2">
                                <span className="h-6 w-6 rounded-full bg-muted border border-border flex items-center justify-center text-xs font-bold">1</span>
                                <span className="h-6 w-6 rounded-full bg-muted border border-border flex items-center justify-center text-xs font-bold">0</span>
                            </div>
                        </div>
                    </div>

                    {/* ── Center Area (3D Anatomy Image) ── */}
                    <div className="relative h-[600px] xl:h-full w-full flex items-center justify-center bg-muted/5 dark:bg-background rounded-[40px] border border-border/50 shadow-inner overflow-hidden">
                        {/* Scale controls */}
                        <div className="absolute left-6 top-1/2 -translate-y-1/2 flex flex-col gap-2 bg-card border border-border rounded-full p-2 shadow-sm z-20">
                            <button className="h-8 w-8 rounded-full hover:bg-muted flex items-center justify-center text-lg font-medium transition-colors">+</button>
                            <div className="w-full h-[1px] bg-border my-1"></div>
                            <button className="h-8 w-8 rounded-full hover:bg-muted flex items-center justify-center text-lg font-medium transition-colors">−</button>
                        </div>

                        {/* Re-center control */}
                        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 bg-card border border-border rounded-full p-2 shadow-sm z-20 flex items-center px-4 hover:bg-muted cursor-pointer transition-colors">
                            <span className="text-xs font-bold tracking-widest uppercase">Center View</span>
                        </div>

                        {/* The Anatomy Image */}
                        <img 
                            src="/anatomy.png" 
                            alt="3D Anatomy Viewer" 
                            className="h-[90%] w-auto object-contain drop-shadow-2xl mix-blend-multiply dark:mix-blend-normal z-10 transition-transform duration-700 ease-in-out hover:scale-[1.02]"
                            onError={(e) => {
                                // Fallback if anatomy.png doesn't exist
                                e.currentTarget.style.display = 'none';
                                e.currentTarget.parentElement?.classList.add('bg-muted/10');
                                e.currentTarget.parentElement?.setAttribute('data-error', 'Image not found - add anatomy.png to public/');
                            }}
                        />

                        {/* Overlay nodes pointing to organs - using absolute positioning */}
                        <div className="absolute inset-0 z-20 pointer-events-none flex items-center justify-center">
                            {/* Head Node */}
                            <div className="absolute top-[18%] ml-[-30px] flex items-center pointer-events-auto group cursor-pointer">
                                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary text-[9px] font-bold text-primary-foreground shadow-md ring-4 ring-primary/20 z-10 group-hover:scale-110 transition-transform">01</span>
                                <div className="h-[2px] w-12 bg-primary/20 -ml-2 group-hover:bg-primary transition-colors"></div>
                                <div className="bg-card border border-border px-3 py-1.5 rounded-xl shadow-sm opacity-0 group-hover:opacity-100 transition-opacity ml-1">
                                    <p className="text-[10px] font-bold">CNS Penetration</p>
                                </div>
                            </div>

                            {/* Heart Lungs Node */}
                            <div className="absolute top-[35%] ml-[30px] flex items-center pointer-events-auto group cursor-pointer flex-row-reverse">
                                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-foreground text-[9px] font-bold text-background shadow-md ring-4 ring-foreground/10 z-10 group-hover:scale-110 transition-transform">02</span>
                                <div className="h-[2px] w-16 bg-foreground/10 -mr-2 group-hover:bg-foreground/50 transition-colors"></div>
                                <div className="bg-card border border-border px-3 py-1.5 rounded-xl shadow-sm opacity-0 group-hover:opacity-100 transition-opacity mr-1">
                                    <p className="text-[10px] font-bold">Cardiotoxicity Risk</p>
                                </div>
                            </div>
                            
                            {/* Liver Node */}
                            <div className="absolute top-[50%] ml-[-20px] flex items-center pointer-events-auto group cursor-pointer">
                                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-warning text-[10px] font-bold text-warning-foreground shadow-md ring-4 ring-warning/20 z-10 group-hover:scale-110 transition-transform">!</span>
                                <div className="h-[2px] w-20 bg-warning/30 -ml-2 group-hover:bg-warning transition-colors"></div>
                                <div className="bg-card border-warning border px-3 py-1.5 rounded-xl shadow-sm ml-1">
                                    <p className="text-[10px] font-bold text-warning-foreground">CYP450 Metabolism</p>
                                </div>
                            </div>
                        </div>

                    </div>

                    {/* ── Right Sidebar (Clinical Timeline) ── */}
                    <div className="space-y-6">
                        
                        {/* Tabs */}
                        <div className="flex bg-card border border-border rounded-full p-1.5 shadow-sm">
                            <button className="flex-1 bg-foreground text-background text-xs font-bold py-2.5 rounded-full shadow-md">Clinical Timeline</button>
                            <button className="flex-1 text-muted-foreground hover:text-foreground text-xs font-bold py-2.5 rounded-full transition-colors">Adverse Events</button>
                            <button className="flex-1 text-muted-foreground hover:text-foreground text-xs font-bold py-2.5 rounded-full transition-colors">Efficacy</button>
                        </div>

                        {/* Timeline Track */}
                        <div className="relative pt-4 pl-4 pb-20 overflow-hidden">
                            {/* Vertical Line */}
                            <div className="absolute left-[38px] top-6 bottom-0 w-[2px] bg-border z-0"></div>

                            {clinicalTimeline.map((item, idx) => (
                                <div key={idx} className="relative z-10 mb-8 pl-12">
                                    {/* Timeline dot */}
                                    <div className="absolute left-[-5px] top-2 flex h-5 w-5 items-center justify-center rounded-full bg-card border-4 border-background ring-2 ring-primary">
                                        <div className="h-1.5 w-1.5 rounded-full bg-primary"></div>
                                    </div>
                                    
                                    {/* Diagnosis Card */}
                                    <div className={cn(
                                        "rounded-3xl p-5 shadow-sm transition-transform hover:-translate-y-1 relative mb-4",
                                        idx === 0 ? "bg-foreground text-background" : "bg-card border border-border"
                                    )}>
                                        <div className="flex justify-between items-start mb-6">
                                            <h3 className="font-bold text-base leading-tight pr-4">{item.disease}</h3>
                                            <span className={cn(
                                                "text-[10px] font-mono whitespace-nowrap",
                                                idx === 0 ? "text-background/70" : "text-muted-foreground"
                                            )}>{item.year}</span>
                                        </div>
                                        <div className="flex items-center gap-4 text-xs font-semibold">
                                            <div className="flex items-center gap-1.5">
                                                <Activity className="h-3.5 w-3.5 opacity-70" /> {item.status}
                                            </div>
                                            <div className="flex items-center gap-1.5">
                                                <Target className="h-3.5 w-3.5 opacity-70" /> EGFR targeted
                                            </div>
                                        </div>
                                        
                                        {/* Connector line from card to events */}
                                        <div className="absolute bottom-[-20px] left-[20px] w-[2px] h-[20px] bg-border hidden sm:block"></div>
                                    </div>

                                    {/* Event Dots Row */}
                                    <div className="flex items-center gap-4 relative ml-4 overflow-x-auto pb-4 pt-2 -mx-4 px-4 scrollbar-none">
                                        {/* Horizontal Track Line */}
                                        <div className="absolute left-0 right-[-100px] top-1/2 h-[1px] bg-border -translate-y-[1px] -z-10"></div>
                                        
                                        {item.events.map((ev, i) => (
                                            <div key={i} className="flex flex-col items-center gap-2 group flex-shrink-0 cursor-pointer">
                                                <div className={cn(
                                                    "h-10 w-10 rounded-full flex items-center justify-center shadow-sm border transition-transform group-hover:scale-110",
                                                    ev.type === "visit" ? "bg-background border-border text-foreground" :
                                                    ev.type === "med" ? "bg-foreground text-background border-transparent" :
                                                    "bg-muted border-border text-foreground/70"
                                                )}>
                                                    {ev.type === "doc" ? <FileText className="h-4 w-4" /> :
                                                     ev.type === "lab" ? <Microscope className="h-4 w-4" /> :
                                                     ev.type === "med" ? <Syringe className="h-4 w-4" /> :
                                                     <User className="h-4 w-4" />}
                                                </div>
                                                <span className="text-[10px] font-bold text-muted-foreground">Month {ev.month}</span>
                                                {ev.name && (
                                                    <span className="absolute top-[48px] text-[9px] font-bold bg-muted px-2 py-0.5 rounded-md border border-border whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
                                                        {ev.name}
                                                    </span>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>

                    </div>

                </div>
            </div>
        </AppLayout>
    );
}
