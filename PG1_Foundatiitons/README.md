# Prime Geometry I: Foundations  
### The Prime Triangle Identity, Energy, Curvature, and Dimensionless Structure

This folder contains *Prime Geometry I: The Prime Triangle Identity and Its Consequences* —  
the foundational document of the Prime Geometry research program.

Prime Geometry I establishes the **core algebraic–geometric identity** relating
consecutive primes to the geometry of two right triangles formed from
\((p_n, p_{n+1})\) and \((p_{n+1}, p_{n+2})\).  
From this identity follow a sequence of derived invariants that define the
energy, curvature, and dimensionless structure of the prime-gap landscape.

This document is the **conceptual and mathematical spine** on which all later
Prime Geometry work (Prime Geometry II: Dynamics and Phenomena, and
Prime Geometry III: Applications and Analysis) is built.

---

## Contents

### **1. The Prime Triangle Identity (One Equation Core)**  
For consecutive primes \(p_n, p_{n+1}, p_{n+2}\) with hypotenuse lengths
\[
C_1 = \sqrt{p_n^2 + p_{n+1}^2},\qquad
C_2 = \sqrt{p_{n+1}^2 + p_{n+2}^2},
\]
the following exact identity holds:
\[
(C_2 - C_1)(C_1 + C_2) = p_{n+2}^2 - p_n^2.
\]
This is the structural backbone of Prime Geometry and the source of all
subsequent invariants.

---

### **2. Prime Triangle Energy \(E_n\)**  
Defined by:
\[
E_n = C_2 - C_1,
\]
with large-prime expansion:
\[
E_n \approx \frac{\sqrt{2}}{2}\,(p_{n+2} - p_n).
\]

**Gap–Energy Constant:**  
\[
E_n / G_n \to \sqrt{2}/2.
\]

---

### **3. PSD and the Prime Scale Law**  
The skip-one square difference:
\[
p_{n+2}^2 - p_n^2 = 12\,\text{PSD}_n
\]
admits the expansion:
\[
\text{PSD}_n \approx \frac{p_n G_n}{6}.
\]

**PSD Scale Constant:**  
\[
\text{PSD}_n /(p_n G_n) \to 1/6.
\]

Exact bias term:
\[
\frac{1}{6} + \frac{G_n}{12p_n}.
\]

---

### **4. Energy Curvature \(K_n\)**  
Defined by the second difference:
\[
K_n = E_{n+1} - E_n.
\]
First-order expansion:
\[
K_n \approx (\sqrt{2}/2)(g_{n+2} - g_n).
\]

---

### **5. Dimensionless Shape Curvature \(\chi_n\)**  
A pure geometric invariant:
\[
\chi_n = \frac{K_n}{E_n}
\approx
\frac{g_{n+2} - g_n}{g_n + g_{n+1}}.
\]

---

### **6. Lagrangian and Action**
The natural action density:
\[
\mathcal{L}_n = \chi_n^2,
\]
and the associated **Prime Action**:
\[
S = \sum_n \mathcal{L}_n.
\]

This describes the total curvature of the prime-gap landscape.

---

## Role of This Document

Prime Geometry I defines the **fundamental mathematical structure**:
- the exact identity,
- first-order constants,
- bias corrections,
- geometric energy,
- curvature,
- and the scale-free Lagrangian framework.

It contains *no empirical analysis* or comparisons to random models  
(those belong to Prime Geometry II).

This document should be cited as the source of the **Prime Triangle Identity**  
and the **energy–curvature–action hierarchy**.

---

## Files in This Folder

- `PG1_PrimeTriangleIdentity.pdf` — main document  
- `PG1_PrimeTriangleIdentity.tex` — Overleaf-compatible source  
- `figures/` — any supporting diagrams  
- `README.md` — this file

---

## Citation

If you use this material in derivative work, please cite the DOI associated with
this release.
https://zenodo.org/badge/1107934958.svg

---

Prime Geometry I forms the foundation.  
Prime Geometry II (Dynamics and Phenomena) will build on this framework with
empirical analysis, curvature distributions, action comparisons, and connections
to Δα and the Prime–Zero Ratio.
