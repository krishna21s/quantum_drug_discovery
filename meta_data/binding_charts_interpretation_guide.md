# Understanding the Quantum Advantage Visualizations

The **Oracle Comparison Charts** section on the Molecules page is designed to visually prove the superiority of the Quantum Support Vector Regression (QSVR) model over classical Machine Learning (XGBoost) models, particularly when discovering novel drugs.

This guide explains how to read and interpret the three primary charts in this section.

---

## 1. Binding Affinity — Classical vs Quantum (Side-by-Side Bar Chart)

### What it shows
This chart compares the predicted pIC₅₀ binding affinity scores of the top 10 candidates side-by-side using two models:
*   **🔷 Blue Bars:** Classical Machine Learning (XGBoost) prediction.
*   **🔮 Purple Bars:** Quantum Machine Learning (QSVR) prediction.

### Key Reference Lines
*   **Green dashed line (7.0):** The threshold for an "Active Drug". Molecules scoring above this are considered highly potent binders.
*   **Red dashed line (5.0):** The threshold for inactivity. Molecules scoring below this are functionally inactive.

### How to interpret it
*   **On known targets (like EGFR):** You will typically see both the blue and purple bars reaching similar heights, often crossing the green line. This indicates both models recognize the active binding patterns found in the training data.
*   **On novel targets (like HIV-1 Protease, when trained on EGFR):** You will see a dramatic discrepancy. The blue bars will typically hover around 2.0 (the baseline for inactivity), showing that the classical model is completely "blind" to the novel binding domain. In contrast, the purple bars will reach up to 7.0+, indicating that the quantum model correctly generalizes its physical understanding of molecular binding to a completely unseen protein target.
*   **The Insight:** The difference in height between the blue and purple pairs for a single candidate is the visual representation of the **Quantum Advantage** for that specific molecule.

---

## 2. Prediction Distribution (Grouped Histogram)

### What it shows
Instead of looking at individual candidates, this chart takes *all* active candidates and groups them into "Activity Zones." It shows exactly **where** each model's predictions tend to cluster. 

The X-axis represents distinct zones of binding potency:
*   **0–2 (Inactive):** The model predicts no binding.
*   **2–4 (Very Weak):** The model predicts barely measurable interaction.
*   **4–5 (Weak):** Low binding activity.
*   **5–6 (Moderate):** Meaningful but sub-optimal binding.
*   **6–7 (Good):** Strong binding, typical of compounds ready for lead optimization.
*   **7+ (Lead Drug):** Highly potent binders.

The Y-axis represents the **Number of Candidates** that fell into that zone.

### How to interpret it
*   **Look for the peaks:** You are looking for which bucket the tall blue bar is in, versus which bucket the tall purple bar is in.
*   **When both agree:** On familiar targets, both the tall blue bar and the tall purple bar will exist in the "Good" or "Lead Drug" buckets. 
*   **The Quantum Proof:** When generating candidates for a novel target, the classical model will fail. You will see a massive blue spike in the **"0–2 Inactive"** zone, meaning classical ML thinks *every* generated molecule is useless. Simultaneously, you will see a massive purple spike in the **"6–7 Good"** zone, proving the quantum model accurately identifies the potent candidates in the batch.
*   **The Insight:** This chart proves that the classical model's failure isn't isolated to a few edge cases; it represents a systemic failure across the entire generated dataset, contrasting with a systemic success by the quantum model.

---

## 3. Complete Candidate Profile (Triple Bar Chart)

### What it shows
This chart provides a holistic view of the top 10 candidates by combining binding predictions with drug-likeness quality in a single view. For each candidate (X-axis), you see three bars:
*   **🔷 Blue Bar (Classical Binding):** XGBoost predicted pIC₅₀.
*   **🔮 Purple Bar (Quantum Binding):** QSVR predicted pIC₅₀.
*   **🟩 Green Bar (Drug Quality):** Quantitative Estimate of Drug-likeness (QED), scaled by 10 to fit the same 0-10 axis (e.g., a QED of 0.85 shows as a bar of height 8.5).

### How to interpret it
*   This chart answers the ultimate question: *"Are the molecules the quantum model found actually good drugs?"*
*   You want to see candidates that have **tall purple bars AND tall green bars**, while ignoring the blue bars. 
*   A tall purple bar (above 7.0) means it binds well. A tall green bar (above 7.0) means it is biologically viably as a drug (good absorption, distribution, metabolism, excretion, and toxicity properties, adhering to Lipinski's Rule of Five).
*   **The Insight:** On complex targets, you will frequently see a pattern: `[Tiny Blue Bar] - [Tall Purple Bar] - [Tall Green Bar]`. This tells a complete story: "This is a high-quality, viable drug candidate that binds strongly to the target, and classical computing entirely missed it."
