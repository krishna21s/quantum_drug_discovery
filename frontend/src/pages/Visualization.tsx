import AppLayout from "@/components/AppLayout";
import React, { useState, useRef, useEffect, Suspense, useCallback, useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  Microscope, Activity, Heart, Target, User, Syringe,
  FileText, ActivitySquare, ShieldAlert,
  RotateCcw, ZoomIn, ZoomOut, Bone, Brain, ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
  Waves, Eye, Loader2, FlaskConical, Pill, AlertTriangle, CheckCircle2,
  Sparkles, MonitorX, Search, Send
} from "lucide-react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF, Html, GizmoHelper, GizmoViewport } from "@react-three/drei";
import * as THREE from "three";
import { predictToxicity, type PredictResponse } from "@/lib/toxicityApi";

/* ================================================================
   DATA & CONFIG
   ================================================================ */

const anatomySystems = [
  { id: "skeleton", name: "Skeletal", file: "/skeleton.glb", icon: Bone, color: "#ffeccd", description: "Bones & Joints" },
  { id: "vascular_system", name: "Vascular", file: "/vascular_system.glb", icon: Heart, color: "#8c9eb5", description: "Heart & Blood Vessels" },
  { id: "visceral_system", name: "Visceral", file: "/visceral_system.glb", icon: Waves, color: "#d4a89c", description: "Internal Organs" },
  { id: "nervous_system", name: "Nervous", file: "/nervous_system.glb", icon: Brain, color: "#ffd966", description: "Brain & Nerves" },
];

/**
 * Mesh-keyword mapping for drug targeting on 3D models.
 * When showTargeting is ON, meshes matching targetOrgans keywords → GREEN,
 * meshes matching sideEffectOrgans keywords → RED, rest → system default color.
 */
const organMeshKeywords: Record<string, string[]> = {
  "Lungs": ["lung", "bronch", "pulmon", "thorax", "alveol"],
  "Tumor Site": ["tumor", "cancer", "mass"],
  "Skin": ["skin", "dermis", "epiderm", "integu"],
  "Liver": ["liver", "hepat", "hepatic"],
  "GI Tract": ["intestin", "colon", "stomach", "gastri", "bowel", "digest"],
  "Upper Respiratory": ["pharynx", "trachea", "larynx", "nasal"],
  "Kidneys": ["kidney", "renal", "nephro"],
  "Heart": ["heart", "cardiac", "ventricle", "atrium", "myocard"],
  "Lymph Nodes": ["lymph", "node", "thymus"],
  "Blood / Immune": ["blood", "vascular", "vein", "artery", "spleen"],
  "Pancreas": ["pancrea"],
  "Breast": ["breast", "mammary", "pectoral", "chest"],
  "Bone": ["bone", "femur", "tibia", "spine", "vertebra", "skeletal"],
  "Uterus": ["uter", "endometr", "uterine", "pelvi"],
  "Eyes": ["eye", "ocular", "orbit", "optic"],
  "Brain": ["brain", "cerebr", "cortex"],
  "Nerves": ["nerve", "spinal", "neural"],
};

const drugCompounds = [
  {
    id: "erlotinib", name: "Erlotinib", pdb: "1M17", targetProtein: "EGFR Kinase",
    disease: "Non-small Cell Lung Cancer", mechanism: "Tyrosine kinase inhibitor that blocks EGFR signaling",
    bindingAffinity: -9.2, smiles: "C=C1C=CC=C1C#CC2=CC3=C(C=C2)N=CN=C3NC4=CC(=CC=C4)C#C",
    targetOrgans: [
      { name: "Lungs", reason: "EGFR overexpressed in NSCLC tumors" },
      { name: "Tumor Site", reason: "Directly inhibits cancer cell proliferation via EGFR" },
    ],
    sideEffectOrgans: [
      { name: "Skin", reason: "Rash — EGFR expressed in skin keratinocytes" },
      { name: "Liver", reason: "Hepatotoxicity — CYP3A4/CYP1A2 metabolism" },
      { name: "GI Tract", reason: "Diarrhea — EGFR disruption in intestinal epithelium" },
    ],
    admet: { absorption: "60%", distribution: "Vd = 232 L", metabolism: "CYP3A4, CYP1A2", excretion: "Fecal (83%)", halfLife: "36.2 hrs" },
  },
  {
    id: "remdesivir", name: "Remdesivir", pdb: "6LU7", targetProtein: "SARS-CoV-2 Mpro",
    disease: "COVID-19", mechanism: "Nucleotide analog inhibiting viral RNA-dependent RNA polymerase",
    bindingAffinity: -7.8, smiles: "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3",
    targetOrgans: [
      { name: "Lungs", reason: "SARS-CoV-2 replication in alveolar cells" },
      { name: "Upper Respiratory", reason: "Inhibits viral replication in nasopharyngeal epithelium" },
    ],
    sideEffectOrgans: [
      { name: "Liver", reason: "Elevated ALT/AST — hepatic metabolism" },
      { name: "Kidneys", reason: "Nephrotoxicity — vehicle accumulates in renal tubules" },
      { name: "Heart", reason: "Bradycardia — slows sinus node conduction" },
    ],
    admet: { absorption: "IV only", distribution: "Vd = 75 L", metabolism: "Hydrolase-mediated", excretion: "Renal (74%)", halfLife: "1 hr" },
  },
  {
    id: "ritonavir", name: "Ritonavir", pdb: "1HHP", targetProtein: "HIV-1 Protease",
    disease: "HIV/AIDS", mechanism: "Protease inhibitor preventing viral polyprotein cleavage",
    bindingAffinity: -10.1, smiles: "CC(C)C(NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)NC(CC2=CC=CC=C2)CC(O)C(CC3=CC=CC=C3)NC(=O)OCC4=CN=CS4",
    targetOrgans: [
      { name: "Lymph Nodes", reason: "HIV replicates in CD4+ T-cells in lymphoid tissue" },
      { name: "Blood / Immune", reason: "Suppresses viral load in CD4+ T-lymphocytes" },
    ],
    sideEffectOrgans: [
      { name: "Liver", reason: "CYP3A4 inhibitor — hepatotoxicity risk" },
      { name: "GI Tract", reason: "Nausea, diarrhea — GI mucosal irritation" },
      { name: "Pancreas", reason: "Pancreatitis — lipid metabolism disruption" },
    ],
    admet: { absorption: "Oral (65%)", distribution: "98-99% protein bound", metabolism: "CYP3A4, CYP2D6", excretion: "Fecal (86%)", halfLife: "3-5 hrs" },
  },
  {
    id: "tamoxifen", name: "Tamoxifen", pdb: "3ERT", targetProtein: "Estrogen Receptor",
    disease: "Breast Cancer (ER+)", mechanism: "SERM — blocks estrogen binding to receptor",
    bindingAffinity: -8.5, smiles: "CC/C(=C(\\C1=CC=CC=C1)/C2=CC=C(C=C2)OCCN(C)C)/C3=CC=CC=C3",
    targetOrgans: [
      { name: "Breast", reason: "Blocks estrogen receptor in ER+ breast cancer cells" },
      { name: "Bone", reason: "Agonist — preserves bone density" },
    ],
    sideEffectOrgans: [
      { name: "Uterus", reason: "Endometrial hyperplasia — agonist on uterine ER" },
      { name: "Liver", reason: "Fatty liver — CYP2D6 metabolism" },
      { name: "Eyes", reason: "Retinopathy — cumulative retinal toxicity" },
    ],
    admet: { absorption: "Oral", distribution: "Vd = 50-60 L/kg", metabolism: "CYP2D6, CYP3A4", excretion: "Fecal", halfLife: "5-7 days" },
  },
];

const clinicalTimeline = [
  { year: "2024", disease: "Non-small Cell Lung Cancer", status: "Active Tracking",
    events: [{ type: "doc", month: "03" }, { type: "visit", month: "06" }, { type: "med", name: "Erlotinib", month: "09" }, { type: "visit", month: "12" }] },
  { year: "2019", disease: "Glioblastoma Multiforme", status: "Target Identified",
    events: [{ type: "doc", month: "04" }, { type: "lab", month: "08" }, { type: "doc", month: "11" }] },
  { year: "2015", disease: "Breast Cancer (HER2+)", status: "Clinical Phase III",
    events: [{ type: "med", name: "Lapatinib", month: "02" }, { type: "visit", month: "05" }, { type: "doc", month: "10" }] },
];

/* ================================================================
   WEBGL DETECTION
   ================================================================ */
function detectWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl") || c.getContext("experimental-webgl"));
  } catch { return false; }
}

/* ================================================================
   3D MODEL — CLONES scene to fix black screen on re-switch
   Applies green/red targeting when showTargeting is on.
   ================================================================ */
interface ModelComponentProps {
  modelPath: string;
  controlsRef: React.RefObject<any>;
  systemColor: string;
  showTargeting: boolean;
  targetOrganNames: string[];
  riskOrganNames: string[];
}

function Model({ modelPath, controlsRef, systemColor, showTargeting, targetOrganNames, riskOrganNames }: ModelComponentProps) {
  const gltf = useGLTF(modelPath);
  const { camera } = useThree();
  const groupRef = useRef<THREE.Group>(null);

  const clonedScene = useMemo(() => gltf.scene.clone(true), [gltf.scene, modelPath]);

  // Build flat keyword lists from organ names
  const targetKeywords = useMemo(() =>
    targetOrganNames.flatMap(name => (organMeshKeywords[name] || [name.toLowerCase()])),
    [targetOrganNames]
  );
  const riskKeywords = useMemo(() =>
    riskOrganNames.flatMap(name => (organMeshKeywords[name] || [name.toLowerCase()])),
    [riskOrganNames]
  );

  useEffect(() => {
    const box = new THREE.Box3().setFromObject(clonedScene);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());

    clonedScene.position.sub(center);

    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = (camera as THREE.PerspectiveCamera).fov * (Math.PI / 180);
    let cameraZ = Math.abs((maxDim / 2) / Math.tan(fov / 2));
    cameraZ *= 1.5;

    camera.position.set(0, 0, cameraZ);
    (camera as THREE.PerspectiveCamera).near = cameraZ / 100;
    (camera as THREE.PerspectiveCamera).far = cameraZ * 100;
    (camera as THREE.PerspectiveCamera).updateProjectionMatrix();

    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.userData = { ...controlsRef.current.userData, initialCameraZ: cameraZ };
      controlsRef.current.update();
    }

    clonedScene.traverse((child: any) => {
      if (child.isMesh) {
        const meshName = (child.name || "").toLowerCase();
        const isTarget = showTargeting && targetKeywords.some(k => meshName.includes(k));
        const isRisk = showTargeting && riskKeywords.some(k => meshName.includes(k));

        if (isTarget) {
          child.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color("#00ff88"), emissive: new THREE.Color("#00ff88"),
            emissiveIntensity: 0.3, roughness: 0.4, metalness: 0.1,
          });
        } else if (isRisk) {
          child.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color("#ff3344"), emissive: new THREE.Color("#ff3344"),
            emissiveIntensity: 0.3, roughness: 0.4, metalness: 0.1,
          });
        } else {
          child.material = new THREE.MeshStandardMaterial({ color: systemColor, roughness: 1, metalness: 0 });
        }
      }
    });

    if (groupRef.current) {
      while (groupRef.current.children.length) groupRef.current.remove(groupRef.current.children[0]);
      groupRef.current.add(clonedScene);
    }
  }, [clonedScene, camera, controlsRef, systemColor, showTargeting, targetKeywords, riskKeywords]);

  return <group ref={groupRef} />;
}

/* ================================================================
   FALLBACKS
   ================================================================ */
function LoadingFallback() {
  return (<Html center><div className="flex flex-col items-center gap-3 text-white/70"><Loader2 className="h-8 w-8 animate-spin" /><p className="text-sm font-semibold">Loading Model…</p></div></Html>);
}

function ModelErrorFallback({ systemName }: { systemName: string }) {
  return (<Html center><div className="flex flex-col items-center gap-3 text-center px-6 max-w-xs"><div className="h-16 w-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center"><Eye className="h-8 w-8 text-white/30" /></div><p className="text-sm font-bold text-white/60">Model Not Found</p><p className="text-xs text-white/40">Place <code className="text-cyan-400/80 bg-white/5 px-1.5 py-0.5 rounded text-[11px]">{systemName}.glb</code> in <code className="text-cyan-400/80 bg-white/5 px-1.5 py-0.5 rounded text-[11px]">public/</code></p></div></Html>);
}

function ModelWithFallback({ modelPath, controlsRef, systemColor, systemId, showTargeting, targetOrganNames, riskOrganNames }: { modelPath: string; controlsRef: React.RefObject<any>; systemColor: string; systemId: string; showTargeting: boolean; targetOrganNames: string[]; riskOrganNames: string[] }) {
  const [hasError, setHasError] = useState(false);
  useEffect(() => { setHasError(false); useGLTF.preload(modelPath); }, [modelPath]);
  if (hasError) return <ModelErrorFallback systemName={systemId} />;
  return (<ErrorBoundaryWrapper onError={() => setHasError(true)}><Model modelPath={modelPath} controlsRef={controlsRef} systemColor={systemColor} showTargeting={showTargeting} targetOrganNames={targetOrganNames} riskOrganNames={riskOrganNames} /></ErrorBoundaryWrapper>);
}

class ErrorBoundaryWrapper extends React.Component<{ children: React.ReactNode; onError: () => void }, { hasError: boolean }> {
  constructor(props: any) { super(props); this.state = { hasError: false }; }
  static getDerivedStateFromError() { return { hasError: true }; }
  componentDidCatch() { this.props.onError(); }
  render() { if (this.state.hasError) return null; return this.props.children; }
}

/* ================================================================
   MAIN PAGE
   ================================================================ */
export default function Visualization() {
  const [selectedDrug, setSelectedDrug] = useState(drugCompounds[0]);
  const [activeSystem, setActiveSystem] = useState("skeleton");
  const [showTargeting, setShowTargeting] = useState(false);
  const [webGLSupported] = useState(() => detectWebGL());
  const controlsRef = useRef<any>(null);

  // SMILES prediction state
  const [smilesInput, setSmilesInput] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [predError, setPredError] = useState("");

  const currentSystem = anatomySystems.find(s => s.id === activeSystem) || anatomySystems[0];

  const handleRecenter = useCallback(() => {
    if (controlsRef.current) { 
      controlsRef.current.target.set(0, 0, 0); 
      if (controlsRef.current.userData?.initialCameraZ) {
        controlsRef.current.object.position.set(0, 0, controlsRef.current.userData.initialCameraZ);
      }
      controlsRef.current.update(); 
    }
  }, []);

  const handleZoom = useCallback((inward: boolean) => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;
    
    // Lerp camera distance smoothly
    const distance = controls.object.position.distanceTo(controls.target);
    const scale = inward ? 0.75 : 1.33;
    
    // Limit zoom to prevent passing through the model or getting too far
    const finalDistance = Math.max(1, Math.min(50, distance * scale));
    
    controls.object.position.lerp(controls.target, 1 - (finalDistance / distance));
    controls.update();
  }, []);

  const handlePan = useCallback((dx: number, dy: number) => {
    if (!controlsRef.current) return;
    const controls = controlsRef.current;
    const camera = controls.object;
    
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
    
    const move = new THREE.Vector3();
    move.addScaledVector(right, dx);
    move.addScaledVector(up, dy);
    
    controls.target.add(move);
    camera.position.add(move);
    controls.update();
  }, []);

  const handlePredict = useCallback(async () => {
    if (!smilesInput.trim()) return;
    setPredicting(true); setPredError(""); setPrediction(null);
    try {
      const res = await predictToxicity(smilesInput.trim());
      setPrediction(res);
    } catch (e: any) {
      setPredError(e.message || "Prediction failed");
    } finally { setPredicting(false); }
  }, [smilesInput]);

  return (
    <AppLayout>
      <div className="min-h-screen bg-background p-4 md:p-6 lg:p-8 flex flex-col font-sans">

        {/* Top Header */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-6">
          <div className="flex items-center gap-4 bg-card border border-border px-6 py-3 rounded-full shadow-sm w-full max-w-2xl overflow-x-auto">
            <button className="flex items-center gap-2 text-sm font-bold bg-primary/10 text-primary px-4 py-2 rounded-full whitespace-nowrap"><Activity className="h-4 w-4" /> Overview</button>
            <button className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground px-4 py-2 whitespace-nowrap transition-colors"><FileText className="h-4 w-4" /> Clinical Data</button>
            <button className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground px-4 py-2 whitespace-nowrap transition-colors"><Microscope className="h-4 w-4" /> Assays</button>
            <button className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground px-4 py-2 whitespace-nowrap transition-colors"><Target className="h-4 w-4" /> Ligands</button>
          </div>

          <div className="flex items-center gap-2.5 px-5 py-3 rounded-full border bg-card border-border shadow-sm shrink-0">
            <Pill className="h-4 w-4 text-primary" />
            <span className="text-sm font-bold">{selectedDrug.name}</span>
            <span className="text-[10px] text-muted-foreground font-semibold">→ {selectedDrug.disease}</span>
          </div>
        </div>

        {/* Main Dashboard — center column sticky, sidebars scroll */}
        <div className="flex-1 grid grid-cols-1 xl:grid-cols-[320px_1fr_420px] gap-6 items-start" style={{ minHeight: 0 }}>

          {/* LEFT SIDEBAR — scrollable */}
          <div className="space-y-4 xl:max-h-[calc(100vh-140px)] xl:overflow-y-auto xl:pr-1 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
            {/* Active Drug Card */}
            <div className="bg-card border border-border rounded-[32px] p-6 shadow-sm relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-primary/10 rounded-full blur-[40px] -mr-8 -mt-8 pointer-events-none" />
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center gap-3">
                  <div className="h-12 w-12 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center"><FlaskConical className="h-6 w-6 text-primary" /></div>
                  <div>
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Active Compound</p>
                    <h2 className="text-lg font-black">{selectedDrug.name}</h2>
                  </div>
                </div>
                <div className="bg-background border border-border px-3 py-1.5 rounded-full text-[10px] font-bold shadow-sm">PDB: {selectedDrug.pdb}</div>
              </div>
              <p className="text-xs font-semibold text-muted-foreground flex items-center gap-1.5 mb-2"><Target className="h-3.5 w-3.5" /> {selectedDrug.targetProtein}</p>
              <p className="text-xs text-muted-foreground leading-relaxed bg-muted/30 rounded-xl p-3 border border-border/50 mb-4">{selectedDrug.mechanism}</p>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-emerald-500/5 border border-emerald-500/10 rounded-xl p-3">
                  <p className="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest mb-0.5">Disease</p>
                  <p className="font-semibold text-xs">{selectedDrug.disease}</p>
                </div>
                <div className="bg-blue-500/5 border border-blue-500/10 rounded-xl p-3">
                  <p className="text-[10px] font-bold text-blue-600 dark:text-blue-400 uppercase tracking-widest mb-0.5">Affinity</p>
                  <p className="font-semibold text-xs">{selectedDrug.bindingAffinity} kcal/mol</p>
                </div>
              </div>
            </div>

            {/* Target Zones */}
            {showTargeting && (
              <>
                <div className="bg-card border border-border rounded-[28px] p-5 shadow-sm">
                  <h3 className="text-sm font-black mb-1 flex items-center gap-2"><CheckCircle2 className="h-4 w-4 text-emerald-500" /><span className="text-emerald-600 dark:text-emerald-400">Predicted Target Zones</span></h3>
                  <p className="text-[10px] text-muted-foreground mb-4">Where Q-PharmX predicts this drug acts</p>
                  <div className="space-y-3">
                    {selectedDrug.targetOrgans.map((organ, i) => (
                      <div key={i} className="bg-emerald-500/[0.04] border border-emerald-500/10 rounded-xl p-3.5">
                        <div className="flex items-center gap-2 mb-1.5"><div className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(0,255,136,0.5)]" /><p className="text-xs font-bold">{organ.name}</p></div>
                        <p className="text-[11px] text-muted-foreground leading-relaxed pl-5">{organ.reason}</p>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-card border border-border rounded-[28px] p-5 shadow-sm">
                  <h3 className="text-sm font-black mb-1 flex items-center gap-2"><AlertTriangle className="h-4 w-4 text-red-500" /><span className="text-red-600 dark:text-red-400">Predicted Adverse Effects</span></h3>
                  <p className="text-[10px] text-muted-foreground mb-4">ADMET toxicity predictions</p>
                  <div className="space-y-3">
                    {selectedDrug.sideEffectOrgans.map((organ, i) => (
                      <div key={i} className="bg-red-500/[0.04] border border-red-500/10 rounded-xl p-3.5">
                        <div className="flex items-center gap-2 mb-1.5"><div className="h-2.5 w-2.5 rounded-full bg-red-500 shadow-[0_0_6px_rgba(255,51,68,0.5)]" /><p className="text-xs font-bold">{organ.name}</p></div>
                        <p className="text-[11px] text-muted-foreground leading-relaxed pl-5">{organ.reason}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* ADMET */}
            <div className="bg-card border border-border rounded-[28px] p-5 shadow-sm">
              <h3 className="text-sm font-black mb-4 flex items-center gap-2"><ActivitySquare className="h-5 w-5 text-muted-foreground" /> ADMET Profile</h3>
              <div className="space-y-3">
                {Object.entries(selectedDrug.admet).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center">
                    <p className="text-xs font-bold text-muted-foreground capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</p>
                    <p className="text-xs font-semibold bg-muted/50 px-2.5 py-1 rounded-lg">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* CENTER — 3D VIEWER */}
          <div className="relative w-full flex flex-col bg-[#0a0a0f] rounded-[40px] border border-white/[0.06] shadow-2xl overflow-hidden xl:sticky xl:top-6" style={{ height: 'calc(100vh - 140px)', minHeight: '500px' }}>

            {/* System Selector — horizontal scroll */}
            <div className="relative z-30 flex items-center justify-between px-5 pt-4 pb-3 gap-3">
              <div className="flex items-center gap-1.5 bg-white/[0.04] border border-white/[0.08] rounded-full p-1 backdrop-blur-xl overflow-x-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
                {anatomySystems.map((sys) => {
                  const Icon = sys.icon;
                  const isActive = activeSystem === sys.id;
                  return (
                    <button key={sys.id} onClick={() => setActiveSystem(sys.id)} className={cn("flex items-center gap-2 px-4 py-2 rounded-full text-xs font-bold transition-all duration-300 whitespace-nowrap shrink-0", isActive ? "bg-white text-black shadow-lg shadow-white/10" : "text-white/50 hover:text-white/80 hover:bg-white/[0.06]")}>
                      <Icon className="h-3.5 w-3.5" />{sys.name}
                    </button>
                  );
                })}
              </div>

              {/* Show Target/Effect toggle */}
              <button
                onClick={() => setShowTargeting(!showTargeting)}
                title={showTargeting ? "Disable Targeting Colors" : "Enable Targeting Colors"}
                className={cn("flex items-center justify-center p-2.5 rounded-full transition-all duration-300 border shrink-0", showTargeting ? "bg-emerald-500/20 border-emerald-500/30 text-emerald-400" : "bg-white/[0.04] border-white/[0.08] text-white/50 hover:text-white/80")}
              >
                <Sparkles className="h-4 w-4" />
              </button>
            </div>

            {/* Legend when targeting is on */}
            {showTargeting && (
              <div className="relative z-30 flex items-center gap-4 px-6 pb-2">
                <div className="flex items-center gap-1.5">
                  <div className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(0,255,136,0.5)]" />
                  <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">Target</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="h-2.5 w-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(255,51,68,0.5)]" />
                  <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">Side Effect</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="h-2.5 w-2.5 rounded-full bg-white/30" />
                  <span className="text-[10px] font-bold text-white/50 uppercase tracking-widest">Neutral</span>
                </div>
              </div>
            )}

            {/* 3D Canvas or WebGL Fallback */}
            <div className="flex-1 relative">
              {webGLSupported ? (
                <>
                  <Canvas camera={{ fov: 40 }} gl={{ antialias: true, toneMapping: THREE.NoToneMapping, outputColorSpace: THREE.SRGBColorSpace }}>
                    <color attach="background" args={["#000000"]} />
                    <ambientLight intensity={0.4} />
                    <directionalLight position={[5, 5, 5]} intensity={0.8} />
                    <directionalLight position={[-5, -5, -5]} intensity={0.4} />
                    <Suspense fallback={<LoadingFallback />}>
                      <ModelWithFallback modelPath={currentSystem.file} controlsRef={controlsRef} systemColor={currentSystem.color} systemId={currentSystem.id} showTargeting={showTargeting} targetOrganNames={selectedDrug.targetOrgans.map(o => o.name)} riskOrganNames={selectedDrug.sideEffectOrgans.map(o => o.name)} />
                    </Suspense>
                    <OrbitControls ref={controlsRef} makeDefault enableDamping dampingFactor={0.05} minDistance={1} maxDistance={50} />
                    <GizmoHelper alignment="bottom-right" margin={[80, 80]}>
                      <GizmoViewport axisColors={['#ff3653', '#0adb71', '#2c8fec']} labelColor="white" hideNegativeAxes />
                    </GizmoHelper>
                  </Canvas>
                  
                  {/* Zoom Controls Overlay */}
                  <div className="absolute left-5 top-1/2 -translate-y-1/2 flex flex-col gap-1.5 bg-white/[0.06] border border-white/[0.1] rounded-2xl p-1.5 backdrop-blur-xl z-20">
                    <button onClick={() => handleZoom(true)} className="h-9 w-9 rounded-xl hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors" title="Zoom In"><ZoomIn className="h-4 w-4" /></button>
                    <div className="w-full h-[1px] bg-white/10" />
                    <button onClick={() => handleZoom(false)} className="h-9 w-9 rounded-xl hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors" title="Zoom Out"><ZoomOut className="h-4 w-4" /></button>
                  </div>

                  {/* Directional Pad Overlay */}
                  <div className="absolute left-5 bottom-8 flex flex-col items-center gap-1 bg-white/[0.06] border border-white/[0.1] rounded-3xl p-2.5 backdrop-blur-xl z-20">
                    <button onClick={() => handlePan(0, 0.1)} className="h-8 w-8 rounded-full hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors" title="Pan Up"><ArrowUp className="h-4 w-4" /></button>
                    <div className="flex gap-1">
                      <button onClick={() => handlePan(-0.1, 0)} className="h-8 w-8 rounded-full hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors" title="Pan Left"><ArrowLeft className="h-4 w-4" /></button>
                      <div className="h-8 w-8 flex items-center justify-center"><div className="h-1.5 w-1.5 rounded-full bg-white/20" /></div>
                      <button onClick={() => handlePan(0.1, 0)} className="h-8 w-8 rounded-full hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors" title="Pan Right"><ArrowRight className="h-4 w-4" /></button>
                    </div>
                    <button onClick={() => handlePan(0, -0.1)} className="h-8 w-8 rounded-full hover:bg-white/10 flex items-center justify-center text-white/60 hover:text-white transition-colors" title="Pan Down"><ArrowDown className="h-4 w-4" /></button>
                  </div>

                  <button onClick={handleRecenter} className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-white/[0.06] border border-white/[0.1] rounded-full py-2.5 px-5 backdrop-blur-xl z-20 flex items-center gap-2 hover:bg-white/10 transition-colors group">
                    <RotateCcw className="h-3.5 w-3.5 text-white/50 group-hover:text-white transition-colors" />
                    <span className="text-[11px] font-bold tracking-widest uppercase text-white/50 group-hover:text-white transition-colors">Total Zoom / Fit</span>
                  </button>
                </>
              ) : (
                <div className="flex-1 flex items-center justify-center h-full">
                  <div className="flex flex-col items-center gap-5 text-center px-8 max-w-md">
                    <div className="h-20 w-20 rounded-3xl bg-red-500/10 border border-red-500/20 flex items-center justify-center"><MonitorX className="h-10 w-10 text-red-400" /></div>
                    <h3 className="text-xl font-black text-white">WebGL Not Available</h3>
                    <p className="text-sm text-white/50 leading-relaxed">Enable hardware acceleration in browser settings, update GPU drivers, or try Chrome/Edge.</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* RIGHT SIDEBAR */}
          <div className="space-y-6 xl:max-h-[calc(100vh-140px)] xl:overflow-y-auto xl:pl-1 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">

            {/* SMILES Input — New Drug Prediction */}
            <div className="bg-card border border-border rounded-[28px] p-5 shadow-sm">
              <h3 className="text-sm font-black mb-1 flex items-center gap-2">
                <Search className="h-4 w-4 text-primary" /> Predict New Drug
              </h3>
              <p className="text-[10px] text-muted-foreground mb-4">Enter a SMILES string to predict toxicity via Q-PharmX pipeline</p>
              <div className="flex gap-2 mb-3">
                <input
                  value={smilesInput}
                  onChange={(e) => setSmilesInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handlePredict()}
                  placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O"
                  className="flex-1 text-xs px-3 py-2.5 rounded-xl bg-background border border-border focus:border-primary outline-none font-mono"
                />
                <button
                  onClick={handlePredict}
                  disabled={!smilesInput.trim() || predicting}
                  className="px-4 py-2.5 rounded-xl bg-primary text-primary-foreground text-xs font-bold disabled:opacity-40 hover:opacity-90 transition-opacity flex items-center gap-1.5 shrink-0"
                >
                  {predicting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                  Predict
                </button>
              </div>

              {/* Quick examples */}
              <div className="flex flex-wrap gap-1.5 mb-3">
                {[{ name: "Aspirin", smiles: "CC(=O)OC1=CC=CC=C1C(=O)O" }, { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(cc1)C(C)C(=O)O" }, { name: "Paracetamol", smiles: "CC(=O)Nc1ccc(O)cc1" }].map((ex) => (
                  <button key={ex.name} onClick={() => setSmilesInput(ex.smiles)} className={cn("text-[10px] font-bold px-2.5 py-1 rounded-lg border transition-colors", smilesInput === ex.smiles ? "bg-primary/10 border-primary/30 text-primary" : "bg-muted/50 border-border text-muted-foreground hover:text-foreground")}>
                    {ex.name}
                  </button>
                ))}
              </div>

              {/* Prediction Result */}
              {prediction && (
                <div className="bg-muted/30 border border-border rounded-2xl p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">Prediction Result</p>
                    <span className={cn("text-[10px] font-black px-2.5 py-1 rounded-full", prediction.verdict === "TOXIC" ? "bg-red-500/15 text-red-500" : "bg-emerald-500/15 text-emerald-500")}>
                      {prediction.verdict}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-background rounded-xl p-3 border border-border">
                      <p className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Toxicity Prob</p>
                      <p className="text-lg font-black">{(prediction.ensemble_probability * 100).toFixed(1)}%</p>
                    </div>
                    <div className="bg-background rounded-xl p-3 border border-border">
                      <p className="text-[9px] font-bold text-muted-foreground uppercase mb-1">Confidence</p>
                      <p className="text-lg font-black">{(prediction.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                  {prediction.canonical_smiles && (
                    <p className="text-[10px] font-mono text-muted-foreground bg-background px-3 py-2 rounded-lg border border-border truncate">
                      {prediction.canonical_smiles}
                    </p>
                  )}
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div><p className="text-[9px] text-muted-foreground font-bold">Classical</p><p className="text-xs font-black">{(prediction.classical_probability * 100).toFixed(1)}%</p></div>
                    <div><p className="text-[9px] text-muted-foreground font-bold">Quantum</p><p className="text-xs font-black">{(prediction.quantum_probability * 100).toFixed(1)}%</p></div>
                    <div><p className="text-[9px] text-muted-foreground font-bold">Ensemble</p><p className="text-xs font-black">{(prediction.ensemble_probability * 100).toFixed(1)}%</p></div>
                  </div>
                </div>
              )}
              {predError && <p className="text-xs text-red-500 font-semibold mt-2">{predError}</p>}
            </div>

            {/* Drug Selector */}
            <div className="bg-card border border-border rounded-[28px] p-5 shadow-sm">
              <h3 className="text-sm font-black mb-4 flex items-center gap-2"><Pill className="h-4 w-4 text-primary" /> Select Drug Compound</h3>
              <div className="space-y-2">
                {drugCompounds.map((drug) => (
                  <button key={drug.id} onClick={() => setSelectedDrug(drug)} className={cn("w-full flex items-center gap-3 p-3 rounded-2xl text-left transition-all duration-300 border", selectedDrug.id === drug.id ? "bg-primary/10 border-primary/30 shadow-sm" : "bg-transparent border-transparent hover:bg-muted/50 hover:border-border")}>
                    <div className={cn("h-10 w-10 rounded-xl flex items-center justify-center text-xs font-black shrink-0 transition-colors", selectedDrug.id === drug.id ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground")}>{drug.pdb.slice(0, 2)}</div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-bold truncate">{drug.name}</p>
                      <p className="text-[10px] text-muted-foreground font-semibold truncate">{drug.disease}</p>
                    </div>
                    <div className="flex gap-1.5 shrink-0">
                      <div className="flex items-center gap-1"><div className="h-1.5 w-1.5 rounded-full bg-emerald-400" /><span className="text-[9px] font-bold text-muted-foreground">{drug.targetOrgans.length}</span></div>
                      <div className="flex items-center gap-1"><div className="h-1.5 w-1.5 rounded-full bg-red-500" /><span className="text-[9px] font-bold text-muted-foreground">{drug.sideEffectOrgans.length}</span></div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Timeline */}
            <div>
              <div className="flex bg-card border border-border rounded-full p-1.5 shadow-sm mb-4">
                <button className="flex-1 bg-foreground text-background text-xs font-bold py-2.5 rounded-full shadow-md">Clinical Timeline</button>
                <button className="flex-1 text-muted-foreground hover:text-foreground text-xs font-bold py-2.5 rounded-full transition-colors">Adverse Events</button>
                <button className="flex-1 text-muted-foreground hover:text-foreground text-xs font-bold py-2.5 rounded-full transition-colors">Efficacy</button>
              </div>
              <div className="relative pt-4 pl-4 pb-20 overflow-hidden">
                <div className="absolute left-[38px] top-6 bottom-0 w-[2px] bg-border z-0"></div>
                {clinicalTimeline.map((item, idx) => (
                  <div key={idx} className="relative z-10 mb-8 pl-12">
                    <div className="absolute left-[-5px] top-2 flex h-5 w-5 items-center justify-center rounded-full bg-card border-4 border-background ring-2 ring-primary"><div className="h-1.5 w-1.5 rounded-full bg-primary"></div></div>
                    <div className={cn("rounded-3xl p-5 shadow-sm transition-transform hover:-translate-y-1 relative mb-4", idx === 0 ? "bg-foreground text-background" : "bg-card border border-border")}>
                      <div className="flex justify-between items-start mb-6">
                        <h3 className="font-bold text-base leading-tight pr-4">{item.disease}</h3>
                        <span className={cn("text-[10px] font-mono whitespace-nowrap", idx === 0 ? "text-background/70" : "text-muted-foreground")}>{item.year}</span>
                      </div>
                      <div className="flex items-center gap-4 text-xs font-semibold">
                        <div className="flex items-center gap-1.5"><Activity className="h-3.5 w-3.5 opacity-70" /> {item.status}</div>
                        <div className="flex items-center gap-1.5"><Target className="h-3.5 w-3.5 opacity-70" /> EGFR targeted</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 relative ml-4 overflow-x-auto pb-4 pt-2 -mx-4 px-4 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
                      <div className="absolute left-0 right-[-100px] top-1/2 h-[1px] bg-border -translate-y-[1px] -z-10"></div>
                      {item.events.map((ev, i) => (
                        <div key={i} className="flex flex-col items-center gap-2 group flex-shrink-0 cursor-pointer">
                          <div className={cn("h-10 w-10 rounded-full flex items-center justify-center shadow-sm border transition-transform group-hover:scale-110", ev.type === "visit" ? "bg-background border-border text-foreground" : ev.type === "med" ? "bg-foreground text-background border-transparent" : "bg-muted border-border text-foreground/70")}>
                            {ev.type === "doc" ? <FileText className="h-4 w-4" /> : ev.type === "lab" ? <Microscope className="h-4 w-4" /> : ev.type === "med" ? <Syringe className="h-4 w-4" /> : <User className="h-4 w-4" />}
                          </div>
                          <span className="text-[10px] font-bold text-muted-foreground">Month {ev.month}</span>
                          {ev.name && (<span className="absolute top-[48px] text-[9px] font-bold bg-muted px-2 py-0.5 rounded-md border border-border whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">{ev.name}</span>)}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

        </div>
      </div>
    </AppLayout>
  );
}
