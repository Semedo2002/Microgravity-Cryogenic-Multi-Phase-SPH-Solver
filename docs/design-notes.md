# Design notes

Why the solver is built the way it is. Most of these choices came out of a
failing test or a measurement; the numbers below are what was measured on a
Quadro P1000 (sm_61) and an i7-8750H.

## Numerical choices

Several choices were made deliberately to keep the large density ratio (≈ 175
for LN2, ≈ 53 for LH2) stable in weakly-compressible SPH:

* **Transport-velocity formulation** (Adami, Hu & Adams 2013). Particles are
  advected with ṽ = v + Δt/2·a_b, where a_b = −(p_b/m_i) Σ (V_i² + V_j²) ∇W_ij
  comes from a constant background pressure p_b = f·ρ0·c² of each phase. The
  momentum equation sees only the physical velocity and pressure, so particles
  stay regular without a large background pressure suppressing buoyancy.
  Implementation details, each backed by a measurement:
  * **f = 2** (`transport_background_factor`). With f = 0.01 the correction has
    no visible effect. f = 1 still lets a 1.4 dx film break up. f = 5 (Adami's
    5·p0) over-damps a 1 g bubble rise by about 25 %. f = 2 gives a stable film
    and a 1 g rise of 55 mm in 0.2 s, against roughly 45–50 mm from Davies–Taylor
    with wall correction.
  * **Interface band only** (`transport_interface_only`). The correction is
    applied only where the other phase is inside the kernel. In the bulk of a
    stratified liquid the discrete background term follows density gradients
    (about f times the pressure acceleration). In a 1 g water column it raised
    the residual velocity tenfold, to ≈ 0.2 m/s.
  * **Geometric volumes** 1/Σ W in the sum. Wall particles continue the fluid's
    volume field via their extrapolated pressure, so walls exert no spurious
    thrust on a compressed fluid.
  * **No extra stress term by default** (`transport_stress`). The term
    A = ρ v (ṽ − v) destabilised an evaporating interface (|v|max 2.9 m/s with
    it, 0.55 m/s without), and Adami et al. note it is negligible at moderate
    Reynolds number.
* **Vapour equation of state and background pressure.** The mechanical vapour
  EOS is a linearised ideal gas with its own stiffness,
  p = p_bg + K (ρT/(ρ_v0 T_ref) − 1) with K = ρ_v0 c0² (`vapor_stiffness`), and
  the background pressure defaults to p_bg = 0.1 K. With p_bg = K this is
  exactly p = ρRT. The sound speed and thermal expansion are the same either
  way, but SPH's zeroth-order pressure-gradient error scales with the
  *absolute* pressure. A quiescent ideal-gas vapour box therefore moved at
  v99 ≈ 1.4 cm/s indefinitely, while at 0.1 K it stays at rest. That was the
  entire two-phase velocity noise floor; a liquid-only box was always still.
  The equal-stiffness alternative B = ρ_l c0²/γ (Colagrossi & Landrini 2003) can
  still be set via `background_pressure`, but it freezes buoyancy-driven motion.
* **Per-phase wall pressure.** Wall particles extrapolate one pressure from
  liquid neighbours and one from vapour neighbours; each fluid particle uses its
  own phase's value.
* **Consistent initial state.** The wall volume is the lattice volume 1/Σ W.
  Fluid masses are set to ρ/Σ W at start-up, which removes a ≈ 1 % lattice-sum
  error that the stiff Tait EOS would turn into a pressure jump at the interface.
* **Phase change at constant density.** In summation mode the liquid density is
  m0·Σ W with a per-particle reference mass. Evaporation shrinks the liquid by
  dm/ρ_l (neglected, since it is 1/175 of the vapour's dm/ρ_v) instead of
  putting the liquid under tension.
* **Closed-tank pressure** is lumped: the ullage volume is the fixed fluid
  volume minus m_l/ρ_l, so the saturation state is not affected by the
  artificial compressibility.
* **Vapour artificial viscosity** (α = 0.2 vs 0.02 for the liquid) damps the
  acoustics that mass injection pumps into the light, nearly inviscid vapour.
* **Damped relaxation** (`Solver.relax`) removes the one-off burst released when
  the cubic start lattice rearranges. Relax for several acoustic crossing times
  of the domain (L/c0), not a fixed number of steps. For a 5 cm LN2 drop at
  c0 = 0.65 m/s, 0.05 s left v99 ≈ 3 cm/s, while 0.5 s brought it down to
  ≈ 1.3 cm/s.
* **Vapour mass rebalancing** (`vapor_mass_rebalance`, β = 1). Evaporated mass
  lands in the first vapour layer. Those particles then grow heavy and spread
  apart, so the liquid's kernel sum (which counts all neighbours) drops and the
  stiff liquid is pulled into tension until the interface tears. Each step,
  vapour particles exchange mass with their vapour neighbours,
  Δm_i = β Σ_j V0 W_ij (m_j − m_i), with enthalpy and momentum carried upwind.
  The exchange is pairwise antisymmetric, so mass, energy and momentum are
  conserved. It plays the role of the physical outflow and keeps vapour
  particle masses near m_ref. It has no effect where vapour masses are equal.
  β = 0.2 and 0.5 still tore in some cases at the lower background pressure;
  β = 1 was clean in all sixteen cases tested (Lee 2–20 s⁻¹, 3–5 K, with and
  without clamping). The update keeps a non-negative self-weight
  (β Σ_j V0 W_ij ≈ 0.7 < 1).
* **CSF and wetting defaults.** The default is the classic density-scaled CSF
  along the wall-rotated colour gradient (Brackbill normal rotation), without
  normal smoothing. The contact angle is imposed wherever a wall is within the
  kernel (`contact_angle_wall_weight = 0.01`), and wall particles enter the
  colour gradient with an extrapolated colour (`wall_colour`). Contact angles
  were measured with a sessile drop and a sphere fit to interface midpoints,
  which reads +0.6…+1.8° high on exact lattice caps:

  | Wetting model (current EOS) | 60° | 90° | 120° | Worst error |
  |---|---|---|---|---|
  | **normal rotation (default)** | 64.9 | 97.0 | 128.0 | 8.0 |
  | contact-line force (Huber et al. 2016) | 67.4 | 100.7 | 126.7 | 10.7 |
  | both | 65.1 | 95.8 | 130.1 | 10.1 |

  The balanced-force CSF (Adami, Hu & Adams 2010; `csf_balanced`) and normal
  smoothing (`csf_smooth_normals`) remain available. Neither reduced the
  velocity noise, and both weaken wall adhesion.
* **Particle shifting** (Sun et al. 2017 δ⁺-SPH; `particle_shifting`,
  `shifting_strength`) is implemented with interface-tangential projection,
  free-surface suppression and wall mirroring, but is **off by default**. It
  did not lower the noise in any test (the floor was the vapour EOS, see above),
  and at A = 0.25 it destabilises a 1 g hydrostatic column (|v|max 0.41 m/s vs
  0.03 m/s without).
* **Phase-change flux limiter** (`phase_change_density_limit`, 1 % per step). An
  evaporating particle's emission is capped so that no receiving vapour particle
  grows in density by more than this fraction per step. The same cap applies to
  a condensing vapour particle's mass loss. Excess superheat carries over to
  later steps, so mass and energy remain exactly conserved.
* **Spacing-aware CFL.** The acoustic time step shrinks by 0.5 dx / r_min when
  any pair of particles comes closer than half the lattice spacing, e.g. a pair
  just created by splitting. In ordinary flows it has no effect.

## Determinism

GPU runs are bitwise reproducible. Two defects used to break this:

* **Stream ordering.** The solver stream was created non-blocking, while device
  memsets and memcpys ran on the legacy default stream. These were therefore
  unordered with the solver's kernels. In particular, the per-step clear of the
  neighbour-grid cell tables could land after the kernel that fills them, so
  particles silently lost neighbours for a step. This was the likely cause of
  the rare hydrostatic-test failure. The compute stream is now blocking. Only
  the telemetry DMA stream, which touches its own buffers only, stays
  non-blocking. With the old setting the determinism test failed in 4 of 5 runs.
* **Particle splitting** assigned slots with `atomicAdd`, so particle order
  depended on thread scheduling. Slots now come from an exclusive prefix sum.
* **CPU reductions.** OpenMP `reduction(+)` combines per-thread partial sums in
  an unspecified order, so floating-point totals (and, through the closed-tank
  pressure, the phase change) differed between runs. `reduce_sum` now sums
  fixed per-thread blocks in thread order.

The neighbour search (stable `thrust::sort_by_key` / counting sort, gathers
only) and all reductions (`thrust::transform_reduce`, ordered CPU sums) were
audited and are deterministic. No kernel uses shared memory or atomics.

## Known limitations

* **WCSPH velocity noise scales with the artificial sound speed** (≈ 0.7–1 % of
  c0 near interfaces with the current defaults). The automatic c0 (10× the
  larger of the gravity and capillary velocity scales) keeps it at mm/s in
  milli-g. Setting `velocity_scale` higher raises the noise proportionally.
* **Velocity noise.** In a static LN2 drop at zero g (c0 = 0.65 m/s, after
  0.5 s of relaxation), the 99th-percentile speed over all fluid particles is
  0.44–0.48 cm/s for t ≥ 1 s, against 1.3–1.6 cm/s before the vapour EOS change,
  and the vapour alone is at ≈ 0.25 cm/s. The liquid alone is still at ≈ 0.7 cm/s,
  with peaks of ≈ 1 cm/s. These are residual parasitic currents at the
  interface: neither more liquid viscosity, the balanced CSF, normal smoothing
  nor particle shifting lowered them.
* **High evaporation rates** are stable up to Lee 20 s⁻¹ with 5 K superheat in
  the tests above. Beyond that it is untested. The per-step flux limiter
  (`phase_change_density_limit`) never engaged in these cases; it remains as a
  safeguard.
* **Thin films.** The transport-velocity formulation keeps films down to about
  1.4 dx stable in the cases tested (LH2 capsule, 70 % and 85 % fill, CSF at 2°,
  0.05 m/s² and 1 g). Films thinner than that are not resolved. A short-range
  wall guard is still in place as a last line of defence.
* **Contact angles** are reproduced within ≈ 8° (60° → 65°, 90° → 97°,
  120° → 128°), an almost constant positive offset. The curvature of interface
  particles touching the wall is underestimated by ≈ 25–35 %, so even a neutral
  90° drop bulges. Extrapolated wall colour, a contact-line force, and mirrored
  curvature neighbours were tried; the last made it worse and was removed.
  Brackbill's normal rotation also does not converge with resolution: at 60°
  it gave 70°, 74° and 79° at dx = R0/8, R0/10 and R0/12. The test tolerance is
  9°; 5° remains the target.
* Constant fluid properties, a lumped (spatially uniform) saturation state, and
  static walls only.
* The **Akinci** model is restricted to free-surface (liquid-only) runs.

## Performance

`cryosph_perf` (half-filled LN2 cylinder, all physics on), as measured on a
Quadro P1000 (sm_61, 4 GB) against 6 cores of an i7-8750H:

| particles | GPU ms/step | CPU ms/step | GPU Mparticle-updates/s |
|---:|---:|---:|---:|
| 10 008 | 9.2 | 27.3 | 1.09 |
| 26 360 | 17.1 | 62.8 | 1.54 |
| 55 800 | 34.1 | 132.4 | 1.63 |
| 89 352 | 49.0 | – | 1.82 |

Both columns were measured on the current code, each on an otherwise idle
machine; the GPU is 3.0–3.9× faster. Against the previous release a step costs
≈ 14 % more on the GPU (42.9 → 49.0 ms at 89 k particles) and 17–85 % more on
the CPU, mostly from the vapour mass rebalancing pass (an extra neighbour loop). Capturing a telemetry
frame every 10 steps costs ≤ 2 % (AsyncCopy 48.9 ms, ZeroCopy 49.8 ms at
89 k particles).
