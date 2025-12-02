# Prime Geometry  
### A Geometric Framework for Consecutive Primes

Prime Geometry is a research program exploring the geometric structure hidden
inside consecutive prime triples.  
At its core is an exact algebraic–geometric identity relating consecutive primes
to the hypotenuse lengths of two right triangles.  
This identity gives rise to a hierarchy of geometric invariants:
energy, curvature, shape curvature, and a natural Lagrangian describing the
local variation of prime gaps.

Prime Geometry is organized into three developing components:

- **Prime Geometry I: Foundations**  
  (core identity, energy, curvature, invariants)

- **Prime Geometry II: Dynamics and Phenomena**  
  (empirical behavior, action distributions, comparisons to random models)

- **Prime Geometry III: Applications and Analytical Directions**  
  (connections to Δα, geometric scaling, and the Prime–Zero Ratio)

This repository collects the documents, code, and figures associated with each
stage of the framework.

---

## 1. The Prime Triangle Identity (Core of the Framework)

Given three consecutive primes \(p_n < p_{n+1} < p_{n+2}\), define the 
hypotenuse lengths
\[
C_1 = \sqrt{p_n^2 + p_{n+1}^2},\qquad
C_2 = \sqrt{p_{n+1}^2 + p_{n+2}^2}.
\]

These satisfy the exact identity:
\[
\boxed{
(C_2 - C_1)(C_1 + C_2) = p_{n+2}^2 - p_n^2.
}
\]

Expressed in terms of the Prime Square-Difference factor:
\[
\mathrm{PSD}_n = \frac{p_{n+2}^2 - p_n^2}{12},
\]
the identity becomes:
\[
\boxed{
\mathrm{PSD}_n = \frac{(C_2 - C_1)(C_1 + C_2)}{12}.
}
\]

This “Prime Triangle Identity” is the spine of Prime Geometry.

---

## 2. Derived Invariants

From the Prime Triangle Identity arise several structural invariants of
consecutive prime triples:

- **Prime Triangle Energy**  
  \[
  E_n = C_2 - C_1 \approx \frac{\sqrt{2}}{2}(p_{n+2}-p_n)
  \]

- **Gap–Energy Constant**  
  \[
  E_n / G_n \to \sqrt{2}/2
  \]

- **PSD Scale Constant**  
  \[
  \mathrm{PSD}_n /(p_n G_n) \to 1/6
  \]

- **Energy Curvature**  
  \[
  K_n = E_{n+1} - E_n \approx (\sqrt{2}/2)(g_{n+2}-g_n)
  \]

- **Dimensionless Shape Curvature**  
  \[
  \chi_n = \frac{K_n}{E_n}
  = \frac{g_{n+2} - g_n}{g_n + g_{n+1}}
  \]

- **Prime Geometry Lagrangian**  
  \[
  \mathcal{L}_n = \chi_n^2
  \]

- **Prime Action**  
  \[
  S = \sum_n \mathcal{L}_n
  \]

These provide a geometric and dynamical description of how the prime-gap
landscape bends and varies.

---

## 3. Repository Structure

