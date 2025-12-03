# Prime Geometry II: Curvature Dynamics of Prime Gaps

[![DOI](https://zenodo.org/badge/1107934958.svg)](https://doi.org/10.5281/zenodo.17795904)

This folder contains all materials for **Prime Geometry II**, the second paper in the Prime Geometry series.  
PG2 introduces a geometric–dynamical perspective on the prime gap sequence, based on the discrete curvature

\[
\chi_n = \frac{g_{n+2} - g_n}{g_n + g_{n+1}}, \qquad L_n = \chi_n^2,
\]

where \(g_n = p_{n+1} - p_n\) are consecutive prime gaps and \(L_n\) acts as a local action density.

---

## 📄 Paper

**PG2_CurvatureDynamics_PrimeGaps.pdf**  
A full, self-contained research note describing:

- the curvature and action framework,  
- large-scale statistical experiments (50,000 primes),  
- comparison to random permutations and the Cramér model,  
- sliding-window and multi-scale coherence phenomena,  
- geometric return-map constraints.

---

## 🔍 Summary of Main Results

PG2 shows that the prime gap sequence exhibits *dynamical structure* not captured by classical randomness models:

### **1. Least-Action Structure**
The total curvature action  
\[
S = \sum_n L_n
\]  
for true primes lies in the bottom **~1%** compared to random permutations of the same gap multiset.

### **2. Suppressed Curvature Distribution**
The curvature values \(\chi_n\) for primes show dramatically lighter tails than both:
- random permutations  
- Cramér pseudo-primes  

Prime gaps avoid violent curvature spikes.

### **3. Sliding-Window Coherence**
Moving averages of \(L_n\) reveal long coherent intervals where curvature is unusually low or high.  
Random permutations show no such structure.

### **4. Multi-Scale Heatmap Structure**
Heatmaps of mean \(\chi_n\) and \(L_n\) across window sizes \(W \in \{200,500,1000,2000,5000\}\) show persistent vertical bands —  
evidence of *true multi-scale coherence*.

### **5. Return-Map Geometry**
The return maps  
\[
(g_n, g_{n+1}) \quad \text{and} \quad (\chi_n, \chi_{n+1})
\]  
show tight geometric constraints unique to real primes.

Cramér and permuted gaps scatter broadly and fail to reproduce this structure.

---

## 📁 Contents

- paper
- code
- figures