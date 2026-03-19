import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Atom } from "lucide-react";

const DEMO_SDF = `
     RDKit          3D

 21 22  0  0  0  0  0  0  0  0999 V2000
    1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124   -0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.2124    0.7000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.4000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.1560    1.2400    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.1560   -1.2400    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -2.4800    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1560   -1.2400    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.1560    1.2400    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    2.4800    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    3.6800    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    2.5000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
    3.5000    0.8000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    3.5000   -0.8000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -2.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5000    0.8000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.5000   -0.8000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -4.5000   -0.8000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.5000 F   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
  1  7  1  0
  2  8  1  0
  3  9  1  0
  4 10  1  0
  5 11  1  0
  6 12  1  0
 12 13  1  0
  1 14  1  0
 14 15  1  0
 14 16  1  0
  5 17  1  0
 17 18  2  0
 17 19  1  0
 19 20  1  0
  3 21  1  0
  4 21  1  0
M  END
$$$$`;

interface Ligand3DViewerProps {
    name?: string;
}

export default function Ligand3DViewer({ name = "Drug Candidate" }: Ligand3DViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const viewerRef = useRef<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (!containerRef.current) return;
        let cancelled = false;

        const init = async () => {
            try {
                setLoading(true);
                const $3Dmol = await import("3dmol");
                if (cancelled || !containerRef.current) return;

                containerRef.current.innerHTML = "";

                const viewer = $3Dmol.createViewer(containerRef.current, {
                    backgroundColor: "transparent",
                    antialias: true,
                });

                viewerRef.current = viewer;

                viewer.addModel(DEMO_SDF, "sdf");
                viewer.setStyle({}, {
                    stick: { colorscheme: "cyanCarbon", radius: 0.15 },
                    sphere: { colorscheme: "Jmol", scale: 0.3 },
                });

                viewer.zoomTo();
                viewer.spin("y", 0.8);
                viewer.render();

                setLoading(false);
            } catch {
                setLoading(false);
            }
        };

        init();

        return () => {
            cancelled = true;
            if (viewerRef.current) {
                try { viewerRef.current.clear(); } catch { }
            }
        };
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="glass-card rounded-2xl p-5 relative overflow-hidden flex flex-col"
        >
            <div className="absolute top-0 left-4 right-4 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent" />

            <div className="flex items-center gap-2 mb-3">
                <Atom className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider">Ligand 3D</h3>
                <span className="ml-1 text-xs text-primary font-mono">{name}</span>
            </div>

            <div className="relative flex-1 min-h-[220px] rounded-xl bg-background/30 ring-1 ring-white/5 overflow-hidden">
                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center z-10">
                        <div className="h-6 w-6 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />
                    </div>
                )}
                <div
                    ref={containerRef}
                    className="w-full h-full min-h-[220px]"
                    style={{ position: "relative" }}
                />
            </div>

            {/* Properties overlay */}
            <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="glass-surface rounded-xl p-2">
                    <p className="text-muted-foreground">MW</p>
                    <p className="font-mono font-semibold mt-0.5">362.46</p>
                </div>
                <div className="glass-surface rounded-xl p-2">
                    <p className="text-muted-foreground">LogP</p>
                    <p className="font-mono font-semibold mt-0.5">2.14</p>
                </div>
                <div className="glass-surface rounded-xl p-2">
                    <p className="text-muted-foreground">TPSA</p>
                    <p className="font-mono font-semibold mt-0.5">78.2 Å²</p>
                </div>
            </div>
        </motion.div>
    );
}
