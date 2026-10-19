---
title: 'CryoSPH-CUDA: GPU multiphase smoothed particle hydrodynamics for cryogenic propellant tanks in microgravity'
tags:
  - C++
  - CUDA
  - Python
  - smoothed particle hydrodynamics
  - multiphase flow
  - phase change
  - sloshing
  - cryogenic propellants
  - microgravity
authors:
  - name: Abdelrahman Shaltout
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: AFFILIATION
    index: 1
date: 18 October 2026
bibliography: paper.bib
---

# Summary

Long-duration storage of cryogenic propellants (liquid hydrogen, oxygen,
methane) is a key enabling technology for in-space refuelling and deep-space
missions. In microgravity the liquid position is set by surface tension,
wetting and small residual accelerations rather than by gravity. Heat leaks
drive evaporation and self-pressurisation, and every manoeuvre makes the
propellant slosh against the tank walls. `CryoSPH-CUDA` simulates these coupled
processes with a weakly compressible, multiphase smoothed particle
hydrodynamics (SPH) method. It runs on NVIDIA GPUs, and an identical
OpenMP CPU build is generated from the same source. It is a C++20 library
with a Python interface (`CryoSPH`), example drivers, a browser-based particle
viewer and a test suite that compares the solver against analytic solutions
and laboratory data.

# Statement of need

Established open-source SPH codes such as DualSPHysics [@Crespo2015] target
free-surface hydrodynamics at normal gravity. Propellant-management studies
need a different combination of physics in one Lagrangian solver:

* two phases at a large density ratio (≈ 50–180 for cryogens);
* surface tension and contact angles that dominate at low Bond number;
* heat conduction and evaporation/condensation with exact mass and energy
  bookkeeping, plus the resulting tank pressurisation;
* the inertial loads of an accelerating, rotating spacecraft, together with
  the slosh forces and moments the liquid exerts back on the tank.

`CryoSPH-CUDA` provides this combination. It is aimed at researchers and
engineers who need fast, reproducible what-if studies of propellant
positioning, boil-off and slosh loads, and at students learning the
numerical methods involved.

# Method

The fluid solver uses the multiphase SPH formulation of @HuAdams2006 with a
Wendland C2 kernel [@Wendland1995]. The liquid follows the Tait equation of
state and the vapour a linearised ideal gas. Walls follow the generalised
boundary condition of @Adami2012, with pressure extrapolated separately for
each phase. A transport-velocity correction [@Adami2013], restricted to the
liquid–vapour interface band, keeps thin films regular. Optional
$\delta$-SPH density diffusion [@MolteniColagrossi2009] and particle shifting
[@Sun2017] are available.

Surface tension uses the continuum surface force model [@Brackbill1992], with
a wall-adhesion contact-angle condition. A contact-line force [@Huber2016] is
available as an option.

Heat conduction follows @ClearyMonaghan1999. Phase change uses the Lee model
[@Lee1980], with conservative particle splitting and merging. A vapour mass
rebalancing step keeps the vapour particle masses uniform at strongly
evaporating interfaces.

Closed tanks evolve a lumped ullage pressure and the corresponding saturation
temperature. Tank motion is prescribed as a time table of linear acceleration
and angular velocity. The liquid is integrated in the tank frame, including
the Euler, centrifugal and Coriolis accelerations, and the solver reports the
resulting slosh force and moment on the tank [@Dodge2000].

GPU runs are bitwise reproducible: stream ordering is explicit, slot
assignment is deterministic, and the CPU reductions are ordered.

# Validation

The test suite (32 C++ and 9 Python tests, run on both backends) checks the
solver against known answers:

* hydrostatic pressure;
* the Laplace pressure jump of a static drop;
* conduction between two slabs;
* mass and energy conservation through evaporation, condensation and venting;
* buoyant bubble rise;
* sessile-drop contact angles (within 9° of the prescribed 60°, 90° and
  120°);
* rigid-body loads and the centrifugal pressure field in an accelerating or
  rotating tank, which converges with resolution (7.1 %, 3.7 % and 1.2 %
  error at 5, 7.5 and 10 particles per radius).

Against laboratory data, the code reproduces the SPHERIC Test 2 dam break with
an obstacle (MARIN; @Kleefsman2005). At a coarse resolution of 80 k particles (18 across the water depth), the normalised RMS errors of the 20 ms averaged signals are 0.066 and 0.135 for the water heights H4 and H2, and 0.106 and 0.150 for the pressures P1 and P3 on the front of the obstacle. The impact arrival matches to within 0.06 s, while the impact peak itself and the thin splashing sheets over the obstacle remain noisy, as is typical of weakly compressible SPH.

High-evaporation cases (liquid nitrogen, Lee coefficient up to 20 s$^{-1}$,
5 K superheat) keep an intact interface. Mass errors stay below
$3\times10^{-6}$ and energy errors below 1 % of the transferred latent heat.

# Acknowledgements

We thank the SPHERIC ERCOFTAC Special Interest Group and MARIN for making the
dam-break benchmark data available.

# References
