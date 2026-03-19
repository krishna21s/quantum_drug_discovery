import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Microscope } from "lucide-react";
import { cn } from "@/lib/utils";

type RenderStyle = "cartoon" | "surface" | "stick";

interface Protein3DViewerProps {
    pdbId?: string;
}

export default function Protein3DViewer({ pdbId = "1M17" }: Protein3DViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<any>(null);
    const [style, setStyle] = useState<RenderStyle>("cartoon");
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;
        let cancelled = false;

        const init = async () => {
            try {
                setLoading(true);
                setError(null);

                // Dynamically import 3Dmol
                const $3Dmol = await import("3dmol");

                if (cancelled || !containerRef.current) return;

                // Clear previous viewer
                containerRef.current.innerHTML = "";

                const viewer = $3Dmol.createViewer(containerRef.current, {
                    backgroundColor: "transparent",
                    antialias: true,
                });

                viewerRef.current = viewer;

                // Fetch PDB from RCSB
                const res = await fetch(`https://files.rcsb.org/download/${pdbId}.pdb`);
                if (!res.ok) throw new Error(`PDB ${pdbId} not found`);
                const pdbData = await res.text();

                if (cancelled) return;

                viewer.addModel(pdbData, "pdb");
                applyStyle(viewer, style);
                viewer.zoomTo();
                viewer.spin("y", 0.5);
                viewer.render();

                setLoading(false);
            } catch (e: any) {
                if (!cancelled) {
                    setError(e.message || "Failed to load structure");
                    setLoading(false);
                }
            }
        };

        init();

        return () => {
            cancelled = true;
            if (viewerRef.current) {
                try { viewerRef.current.clear(); } catch { }
            }
        };
    }, [pdbId]);

    // Update style when toggle changes
    useEffect(() => {
        if (!viewerRef.current || loading) return;
        applyStyle(viewerRef.current, style);
        viewerRef.current.render();
    }, [style, loading]);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="glass-card rounded-2xl p-5 relative overflow-hidden flex flex-col"
        >
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-quantum/40 to-transparent" />

            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Microscope className="h-4 w-4 text-quantum" />
                    <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Protein 3D</h3>
                    <span className="font-mono text-xs text-quantum ml-1">{pdbId}</span>
                </div>

                {/* Render mode pills */}
                <div className="flex rounded-xl glass-surface p-0.5">
                    {(["cartoon", "surface", "stick"] as RenderStyle[]).map((m) => (
                        <button
                            key={m}
                            onClick={() => setStyle(m)}
                            className={cn(
                                "px-2.5 py-1 text-xs rounded-lg transition-all duration-300 capitalize",
                                style === m ? "bg-primary text-primary-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                            )}
                        >
                            {m}
                        </button>
                    ))}
                </div>
            </div>

            {/* 3D viewer container */}
            <div className="relative flex-1 min-h-[350px] rounded-xl bg-background/30 ring-1 ring-white/5 overflow-hidden">
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center z-10">
                        <div className="flex flex-col items-center gap-2">
                            <div className="h-8 w-8 border-2 border-quantum/30 border-t-quantum rounded-full animate-spin" />
                            <p className="text-xs text-muted-foreground">Loading {pdbId}…</p>
                        </div>
                    </div>
                )}
                {error && (
                    <div className="absolute inset-0 flex items-center justify-center z-10">
                        <p className="text-xs text-destructive">{error}</p>
                    </div>
                )}
                <div
                    ref={containerRef}
                    className="w-full h-full min-h-[350px]"
                    style={{ position: "relative" }}
                />
            </div>

            <p className="text-xs text-muted-foreground mt-3 text-center">
                Rotate: drag · Zoom: scroll · Pan: right-drag
            </p>
        </motion.div>
    );
}

function applyStyle(viewer: any, style: RenderStyle) {
    viewer.setStyle({}, {});
    switch (style) {
        case "cartoon":
            viewer.setStyle({}, {
                cartoon: {
                    color: "spectrum",
                    opacity: 0.9,
                },
            });
            // Highlight active site residues (common active site range for demo)
            viewer.setStyle({ resi: ["145", "41", "166", "187", "189"] }, {
                cartoon: { color: "spectrum", opacity: 0.9 },
                stick: { colorscheme: "cyanCarbon", radius: 0.15 },
            });
            break;
        case "surface":
            viewer.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.3 } });
            viewer.addSurface(
                "VDW",
                {
                    opacity: 0.7,
                    color: "white",
                    colorscheme: { prop: "b", gradient: "roygb", min: 0, max: 100 },
                },
                {}
            );
            break;
        case "stick":
            viewer.setStyle({}, {
                stick: { colorscheme: "Jmol", radius: 0.15 },
                sphere: { colorscheme: "Jmol", scale: 0.25 },
            });
            break;
    }
}
