# Prime Geometry III: A Unified Action–Curvature Theory of Prime Gap Dynamics

**Version:** 1.0  
**Author:** Allen Proxmire  
**Series:** Prime Geometry (PG1–PG3)  
**Status:** Complete  
**PDF:** `PG3.pdf`  
**Experiments:** `/Experiments/` folder (Python scripts)

---

## Overview

**Prime Geometry III (PG3)** develops a unified action–curvature framework for the dynamics of prime gaps.  
It synthesizes the geometric identity introduced in **PG1** and the empirical curvature phenomena revealed in **PG2**, and shows that prime gaps evolve along a *low-curvature, near–least-action trajectory*.

PG3 introduces and unifies:

- The **Prime Dynamical Law** derived from the Prime Triangle identity  
- The **curvature invariant**  
  \[
  \chi_n = \frac{g_{n+2} - g_n}{g_n + g_{n+1}}
  \]
- The **action functional**  
  \[
  S = \sum_n \chi_n^2
  \]
- The **curvature attractor** in $(\chi_n,\chi_{n+1})$-space  
- Large-scale computational experiments up to **1,000,000 primes**

PG3 demonstrates that the prime gap sequence follows a highly non-random, geometrically constrained path through gap space.

---

## Major Results

### **1. Extreme Least-Action Behavior**
The true prime sequence has dramatically lower action than random permutations of the same gaps:

- For $N = 1{,}000{,}000$ primes:
  - True action: \(S_{\text{true}} = 9.4165 \times 10^5\)
  - Random permutation mean: \(9.7474 \times 10^5\)
  - Over **6σ** separation

The action gap increases with $N$, strengthening the Prime Action Principle.

---

### **2. Perturbation Fragility**
Tiny local changes to the gap sequence:

- swapping two adjacent gaps  
- adjusting one gap by only 5  
- randomizing a small window  

produce enormous distortions in curvature structure despite tiny changes in $S$.

---

### **3. Attractor Geometry**
The $(\chi_n,\chi_{n+1})$ return map reveals a compact, nearly isotropic attractor:

- Max radius: $\approx 14$  
- Mean radius: $\approx 0.89$  
- PCA eigenvalues: $0.8033$ and $0.6943$  
- Slight anti-diagonal orientation

---

### **4. Multi-scale Curvature Coherence**
Sliding-window averages of $\chi_n$ with

- $W = 500$  
- $W = 2000$  
- $W = 5000$  
- $W = 10000$  

all produce **a single positive curvature phase** with **zero negative runs**, across 200,000 primes.

This is a striking sign of long-range organization in the prime gap sequence.

---

## Citation

If you use this material in derivative work, please cite the DOI associated with
this release.
[![DOI](https://zenodo.org/badge/1107934958.svg)](https://doi.org/10.5281/zenodo.17795904)

---

## Repository Structure

