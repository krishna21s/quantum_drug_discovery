# 🧬 Q-PharmX — Frontend Features Documentation

> **Stack**: React + TypeScript · Vite · TailwindCSS · Framer Motion · Recharts · Shadcn/UI  
> **Routes**: 10 pages with a collapsible icon sidebar  

---

## 🗺️ Application Architecture

### Layout & Navigation
- **Fixed Icon Sidebar** (`AppSidebar.tsx`): 72px-wide glassmorphism sidebar with tooltip labels on hover and animated Framer Motion route-indicator pill that slides between active items.
- **Dark / Light Theme Toggle** (`ThemeProvider.tsx`): Persisted theme switch (Sun/Moon) at the bottom of the sidebar.
- **User Avatar** (bottom of sidebar): Shows researcher identity (`R` avatar with email tooltip).
- **Settings Button**: Placeholder icon button at the bottom of the sidebar.
- **Liquid Animated Background** (`LiquidBackground.tsx`): Full-viewport animated canvas background on all pages.
- **Toast Notifications**: Both shadcn `Toaster` and Sonner `Toaster` wired globally.
- **React Query**: Global `QueryClientProvider` for data-fetching infrastructure.
- **Responsive Grid Layouts**: All pages use `grid-cols-1 / lg:grid-cols-3` breakpoints.

---

## 📄 Pages & Features

### 1. 🏠 Dashboard (`/`)
*The command centre overview.*

| Feature | Details |
|---|---|
| **Stat Cards** | 4 animated cards: Experiments (24), Molecules Tested (1,247), Quantum Runs (89), Active Candidates (7) — each with progress bar and trend label |
| **Discovery Activity Chart** | Area chart (Recharts) showing binding-score trend over 7-day week |
| **Quick Actions Panel** | 2×2 grid of shortcut buttons to: New Experiment, Quantum Lab, 3D Viewer, ADMET Screen |
| **GPU Cluster Status** | Live status pill ("Online") with 78% utilisation progress bar |
| **Recent Experiments Feed** | List of 4 recent runs with status badges (Completed / Running-pulse / Queued) and binding scores |

---

### 2. 🧪 New Experiment (`/experiment`)
*Guided 4-step wizard to set up and launch a drug discovery pipeline.*

| Step | Feature |
|---|---|
| **Step 1 – Protein Target** | PDB ID text input with "Load" button; preset library cards: 5 protein targets (6LU7, 1M17, 1HHP, 1ZG4, 3ERT) with disease labels |
| **Step 2 – Molecule Source** | 4 source modes: Drug Database search, File Upload (SDF/MOL2/PDB), Molecular Editor (draw), AI Generate; AI mode expands a form with candidate count, max MW, and optimisation goal selector |
| **Step 3 – Configure Analysis** | Quantum Parameters panel: VQE Optimizer (COBYLA/SPSA/L-BFGS-B), max iterations; AI Config panel: Docking engine (AutoDock Vina/DiffDock), ADMET prediction toggle |
| **Step 4 – Run Pipeline** | Summary pre-launch screen → animated 4-stage progress tracker (Molecular Docking → VQE Energy → VQC Prediction → Binding Simulation) → auto-navigates to `/results` |
| **Step Indicator** | Animated numbered stepper with spring-transition active ring; Previous/Next nav buttons |

---

### 3. ⚡ Quantum Lab (`/quantum`)
*All-in-one quantum chemistry workspace.*

| Component | Feature |
|---|---|
| **Drug Info Banner** (`DrugInfoPanel`) | Summary card for the selected drug candidate |
| **Protein Target Map** (`ProteinTargetMap`) | Interactive binding site map with target hotspots |
| **Disease Panel** (`DiseasePanel`) | Disease indication and disease area metadata |
| **Molecule Stats Panel** (`MoleculeStatsPanel`) | Molecular property summary (MW, HBD, HBA, etc.) |
| **Physico-Chemical Radar** (`PhysicoChemicalRadar`) | SVG radar chart of Lipinski/physico-chemical properties |
| **Molecule Viewer** (`MoleculeViewer`) | 2D structural representation of the selected molecule |
| **VQC Circuit Diagram** (`QuantumCircuitDiagram`) | Animated SVG circuit with H, Ry, Rz, CNOT, Measure gates; traveling pulse dot on qubit wires |
| **Binding Simulation Canvas** (`BindingSimulation`) | Real-time Canvas 2D animation of drug approaching protein cavity with trail, glow, and interaction line effects |
| **Quantum Output Panel** (`QuantumOutputPanel`) | VQE/VQC result metrics display |
| **ADMET Panel** (`ADMETPanel`) | Compact ADMET summary with binding affinity, quantum energy, and combined score |
| **Quantum Chemistry Panel** (`QuantumChemPanel`) | Quantum energy breakdown and orbital/Hamiltonian data |

---

### 4. 🔬 Molecules (`/molecules`)
*Searchable, filterable compound library.*

| Feature | Details |
|---|---|
| **ADMET Filter Bar** (`ADMETFilterBar`) | Tab-style filter: All / Pass / Caution / Fail with live counts |
| **Search Input** | Molecule name / formula / PDB ID search box |
| **Compound Table** | Columns: Name, Formula, MW, LogP, Binding Score, ADMET%, Status (Active/Moderate/Weak) |
| **Molecule Viewer Sidebar** | 2D structure viewer for selected compound |
| **Export CSV Button** | Download compound list (UI only) |
| **ADMET Auto-scoring** | Each molecule's ADMET verdict computed client-side via `admetEngine.ts` |

---

### 5. 🛡️ ADMET Analysis (`/admet`)
*Full ADMET profiling and multi-molecule comparison.*

| Feature | Details |
|---|---|
| **Molecule Selector** | Animated pill-selector for 5 demo molecules (Aspirin, Cetuximab, Ibuprofen, Metformin, Paracetamol) |
| **Candidate Comparison Table** | A / D / M / E / T individual scores + Overall % + Pass/Caution/Fail verdict + Combined score for all molecules |
| **Molecular Descriptors Grid** | All raw descriptor values (MW, HBD, HBA, LogP, TPSA, etc.) in card grid |
| **ADMET Panel** (`ADMETPanel`) | Radar chart, per-category scores, binding affinity, quantum energy, and combined drug suitability score |
| **ADMET Radar Chart** (`ADMETRadarChart`) | Animated 5-axis Spider/Radar chart for A/D/M/E/T |

---

### 6. 🔭 3D Viewer (`/visualization`)
*Multi-modal 3D structural visualization.*

| Feature | Details |
|---|---|
| **PDB Selector** | Spring-animated pill selector for 4 structures: 1M17, 6LU7, 1HHP, 3ERT |
| **3D Body/Organ Viewer** (`BodyPartViewer`) | Human anatomy viewer showing organ-level drug distribution |
| **Protein 3D Viewer** (`Protein3DViewer`) | 3D protein spine/structure rendering with PDB id |
| **Structure Info Card** | Metadata: PDB ID, Resolution, Chains, Method, Source, Organism |
| **Ligand 3D Viewer** (`Ligand3DViewer`) | 3D ligand structure for the selected drug (e.g., Cetuximab) |
| **Binding Cinematic** (`BindingCinematic`) | Full cinematic binding animation sequence |

---

### 7. 🧫 Simulation Studio (`/simulation`)
*Molecular dynamics (MD) preparation and execution.*

| Component | Feature |
|---|---|
| **Protein Prep Panel** (`ProteinPrepPanel`) | Protein setup steps: add hydrogens, assign charges, solvate |
| **MD Simulation Panel** (`MDSimulationPanel`) | Run/Re-run/Reset MD simulation; animated progress bar during simulation; on completion renders: RMSD vs Time SVG chart, Potential Energy vs Time SVG chart, metric cards (Avg RMSD, Avg RMSF, Convergence, Stability %) |
| **Trajectory Player** (`TrajectoryPlayer`) | Playback controls (Play/Pause/Reset) for MD trajectory frames |
| **Free Energy Panel** (`FreeEnergyPanel`) | Free energy landscape and ΔG estimation |

---

### 8. 📊 Detailed Analysis (`/analysis`)
*Deep-dive molecular interaction profiling.*

| Component | Feature |
|---|---|
| **Interaction Analysis Panel** (`InteractionAnalysisPanel`) | Tables for Hydrogen Bonds (donor, acceptor, distance, angle, strength), Hydrophobic Contacts (ligand atom → residue, type), π-Stacking Interactions (ring ↔ residue, type, distance); Interaction Quality Score |
| **Residue Contact Map** (`ResidueContactMap`) | Protein residue-level contact heatmap |
| **Quantum Chemistry Panel** (`QuantumChemPanel`) | Quantum energy, orbital contributions, Hamiltonian terms |
| **Multi-Objective Score Panel** (`MultiObjectiveScorePanel`) | Animated radar chart with 5 objectives (Binding 25%, ADMET 20%, Quantum 15%, MD Stability 20%, Free Energy 20%); Composite % score; Verdict badge (Excellent/Good/Moderate/Poor); Per-axis progress bars |

---

### 9. 📋 Experiment Results (`/results`)
*Overview of all experiment runs.*

| Feature | Details |
|---|---|
| **Experiment Card Grid** (`ExperimentCard`) | 6 experiment cards with protein ID, date, status (completed/running/queued), binding score |
| **Quantum Output Sidebar** | VQE/VQC output metrics for selected experiment |
| **ADMET Sidebar Panel** | Compact ADMET + combined score for selected molecule |

---

### 10. 📁 Research Reports (`/reports`)
*Report management and generation.*

| Feature | Details |
|---|---|
| **Report List** | 5 research reports with title, date, page count, and status badges (Final/Draft/Review) |
| **Hover Actions** | Download button appears on card hover |
| **Generate Report Button** | Hero CTA button to create new report |

---

## 🧩 Shared Components

| Component | Purpose |
|---|---|
| `StatCard` | Animated metric card with progress bar, trend, and variant styling (default/quantum/warning/success) |
| `AnimatedBackground` | Particle / wave animated canvas backdrop |
| `ADMETFilterBar` | Reusable ADMET verdict filter tabs with counts |
| `ADMETRadarChart` | Reusable 5-axis A/D/M/E/T radar SVG chart |
| `PhysicoChemicalRadar` | Lipinski property radar chart |

---

## 📦 Client-Side Engine Libraries (`src/lib/`)

| Library | Purpose |
|---|---|
| `admetEngine.ts` | Computes ADMET scores from molecular descriptors; calculates combined drug suitability score; `multiObjectiveScore()` for radar panel |
| `mdEngine.ts` | Generates synthetic MD trajectory frames (RMSD, energy); subsampling utility for chart rendering |
| `interactionEngine.ts` | Generates hydrogen bond, hydrophobic contact, π-interaction, and salt bridge data |
| `freeEnergyEngine.ts` | Free energy landscape estimation and ΔG data generation |

---

## 🖥️ UI / Design System

| Aspect | Detail |
|---|---|
| **Glassmorphism** | `glass-card`, `glass-surface`, `liquid-glass` CSS classes with `backdrop-filter: blur` |
| **Animations** | Framer Motion throughout: `layoutId` shared transitions, stagger children, spring physics |
| **Color Palette** | Custom HSL tokens: `--quantum` (cyan), `--primary` (blue), `--success` (green), `--warning` (amber), `--destructive` (red) |
| **Typography** | JetBrains Mono for code/numeric; system sans for UI text |
| **Charts** | Recharts (AreaChart on Dashboard), custom SVG (MD, Radar, Circuit, Contact Map) |
| **Canvas Animations** | Raw `requestAnimationFrame` Canvas 2D for BindingSimulation |
| **Responsive** | Tailwind breakpoints: `sm:`, `md:`, `lg:` column spanning |
| **Component Library** | Shadcn/UI (Button, Progress, Tooltip, Toaster, Sonner) |
| **Routing** | React Router v6 (`BrowserRouter`, `Routes`, `Route`) |
