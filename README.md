# DR-MMSE TAC — Distributionally Robust Kalman Update

A drop-in **measurement-update wrapper** that makes any Kalman / EKF / ESKF
filter robust to misspecified noise covariances. Pure C++ / Eigen with a
thin Python layer. No CUDA, no PyTorch, no MOSEK / SDP solver.

| Property | Value |
|---|---|
| Algorithm | Frank-Wolfe over Bures-Wasserstein balls |
| C++ deps | Eigen ≥ 3.4 (header-only, auto-fetched if not on the system) |
| Python deps | NumPy, pybind11 |
| Per-call latency | tens to a few hundred microseconds for typical filter sizes (see §6) |
| Targets | any modern Linux / macOS, x86_64 or aarch64 |

> **Status: ready to integrate.** The solver is the same code that produced
> the published benchmark results in the parent project (UTIL UWB indoor
> localization, i2Nav-Robot GNSS/INS) — only the packaging is new.

---

## 1. What this solves

Standard Kalman filters assume the process noise covariance `Q` and the
measurement noise covariance `R` are **known exactly**. In practice they
aren't: GNSS receiver-reported σ underestimates urban multipath, IMU bias
drift breaks Q, UWB anchors get blocked, etc.

DR-MMSE treats `(Q, R)` as **uncertain within a ball**. At every measurement
it solves a per-step minimax:

```
min  ||x − x̂_post||²
 K
s.t. (Σ_w, Σ_v) ∈ argmax_{Σ_w, Σ_v}  trace(P_post(K, Σ_w, Σ_v))
                  s.t.  BW(Σ_w, Q̂) ≤ θ_w
                        BW(Σ_v, R̂) ≤ θ_v
```

where `BW(·, ·)` is the Bures-Wasserstein distance between PSD matrices.
The adversary picks the noise that *maximizes* the posterior trace; the
Kalman gain is computed against that worst case.

The two radii `(θ_w, θ_v)` are the only hyperparameters you tune. `(0, 0)`
recovers the standard Kalman update exactly — use this as a parity check.

The TAC ("test-and-adapt covariance") decomposition splits the prior as
```
P_pri = APA + Σ_w
APA = Φ_accum · P_post_prev · Φ_accumᵀ
```
so the BW ball acts on `Σ_w` (process noise accumulated since the last
measurement) instead of the full prior. This makes the optimization
well-conditioned even for INS-style filters where `Σ_w` lives in a
6-dimensional subspace of a 21-state filter.

The C++ solver implements **Frank-Wolfe with an exact Bures-Wasserstein
oracle** (8 outer iterations is typically enough; warm-starting from the
previous step's solution drops it to 1–3). No SDP / MOSEK is used or
required.

---

## 2. Build

### Linux (Ubuntu / Debian, x86_64 or aarch64)
```bash
sudo apt install build-essential cmake libeigen3-dev python3-dev
pip install pybind11 numpy
./build.sh
```

The CMake build picks `-mcpu=native` on aarch64 (enables NEON / SVE) and
`-march=native` on x86_64. Eigen is found via the system package, or
fetched at version 3.4.0 if missing.

### macOS
```bash
brew install cmake eigen python
pip install pybind11 numpy
./build.sh
```

The build script writes `python/dr_mmse/dr_mmse_cpp*.so` next to the
Python package, so `import dr_mmse` works from the source tree without
an install step. To install system-wide: `pip install -e python/` after
building.

### Verify the build
```bash
PYTHONPATH=python python3 tests/test_basic.py
PYTHONPATH=python python3 examples/01_minimal_linear.py
PYTHONPATH=python python3 scripts/bench.py
```

The first thing the smoke test checks is that θ=0 reproduces the standard
Kalman posterior to machine precision. If that fails, the build is broken;
don't continue.

---

## 3. The 30-second quick start

```python
from dr_mmse import solve_dr_mmse_tac

# Inputs you compute from your filter at every measurement:
#   APA         = Φ_accum @ P_post_prev @ Φ_accumᵀ      (nx, nx)
#   Σ_w_hat     = P_pri − APA  (or your filter's Q_d)   (nx, nx, PSD)
#   H, R_hat    = your measurement Jacobian + nominal R
#   theta_w, theta_v = your tuned BW radii
res = solve_dr_mmse_tac(APA, H, Sigma_w_hat, R_hat, theta_w, theta_v)

# Worst-case Kalman gain
P_pri_dr = res.wc_Xprior        # = APA + res.wc_Sigma_w
S = H @ P_pri_dr @ H.T + res.wc_Sigma_v
K = np.linalg.solve(S, H @ P_pri_dr).T

# Apply
innov = z − h(x_pri)
x_post = x_pri + K @ innov
P_post = res.wc_Xpost
```

That's the whole integration. Three lines for the solve, four lines for
the gain. Everything else is standard Kalman.

If your filter only exposes `(P_pri, Phi_per_step)` and not APA, the
`DRWrapper` helper accumulates APA for you — see
`examples/02_ekf_apa_tracking.py`.

---

## 4. The TAC contract: what your filter must expose

To wire DR into a filter you don't own, you need three things at every
measurement step:

| Quantity | Shape | Where it comes from |
|---|---|---|
| `APA` | (nx, nx) | Accumulate `Φ_k @ Φ_accum` across propagation steps since the last measurement; then `APA = Φ_accum @ P_post_prev @ Φ_accumᵀ`. |
| `Σ_w_hat` | (nx, nx, PSD) | Either your filter's `Q_d` accumulated, or `P_pri - APA`. The latter is exact for any filter that does `P_pri = Φ P_post Φᵀ + Q_d`. |
| `H, R_hat` | (ny, nx), (ny, ny) | The measurement Jacobian and nominal measurement noise (`H = ∂h/∂x` for an EKF). |

Most filters give you `Phi_k` per propagation step. If yours doesn't, you
can read it back from the source (e.g. KF-GINS exposes `propagationRecords()`)
or bookkeep it yourself.

After the DR solve you must **inject the worst-case gain back into your
filter's state and covariance**:
```
x_post = x_pri + K_dr @ innov
P_post = res.wc_Xpost
```
For Joseph-form filters, the wrapper gives you `P_post` directly — you do
NOT also apply your filter's own update. The DR solve already accounts for
the measurement.

### "I have a 21-state INS where Σ_w is rank-6"
Use `solve_dr_mmse_tac_factored(APA, H, G_list, Q_hat, R_hat, theta_w, theta_v)`.
`G_list[k]` is `nx × nw` and `Σ_w = Σ_k G_k @ Q_w @ G_kᵀ`. The BW ball acts on
the well-conditioned `Q_w (nw × nw)` matrix instead of the rank-deficient `Σ_w`.

### "My filter has a sign convention `dz = h(x) − z`"
KF-GINS uses this. The DR solver doesn't care about innovation sign — it
only operates on covariances. But if your downstream code expects a
specific sign, apply it consistently between the innovation, the gain,
and any μ_v override you might have on top.

---

## 5. Tuning `(θ_w, θ_v)`

`θ_w, θ_v ≥ 0` are the only knobs. Both default to 0 (= no DR). Practical
heuristics from the parent project's deployments:

* `θ_v` (measurement-noise robustness) usually **matters less** than `θ_w`.
  Differences across `θ_v ∈ {0.001, 0.01, 0.1, 1.0}` are typically <0.5 m
  RMSE on GNSS at the seq level.

* `θ_w` (process-noise robustness) is the **primary knob**.
  - `θ_w = 0.001` → essentially vanilla Kalman (DR is too weak to act).
  - `θ_w = 0.5` → conservative default; performs well across mixed environments.
  - `θ_w = 1.0` → aggressive; helps in harsh urban canyons, hurts in clean.
  - `θ_w = 5.0` → only useful when GNSS is intermittently lost and you
    want the filter to stay tight to the few good fixes you do get.

* **Tune offline by grid sweep.** A 6×6 grid of `(θ_w, θ_v) ∈ {0.001, 0.01,
  0.05, 0.1, 0.5, 1.0}` is enough for most datasets. There is no good
  online θ-selection method in this codebase — `(θ_w, θ_v)` are constants
  per-deployment. If your robot operates in mixed regimes (highway +
  tunnel + parking), pick θ for the *worst* regime; it is conservative on
  the easy regimes by ≤ 1–2 m.

* **Diagnostic for "θ too aggressive": NEES blows up.**
  Compute `NEES = err_postᵀ · P_post⁻¹ · err_post` and compare to χ²(dof)
  thresholds. Mean NEES > 2× dof means the filter is over-inflating.

* **Diagnostic for "θ too weak": filter ignores DR.**
  If `res.iterations == 0` or `res.iterations == 1` for a typical step,
  the BW oracle is not finding any direction to inflate. Crank θ up.

A grid-sweep harness pattern:
```python
for theta_w in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    for theta_v in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        run_filter(theta_w, theta_v)  # log RMSE + NEES
```
Pick the `(θ_w, θ_v)` with best RMSE subject to NEES within `[0.5×dof, 2×dof]`.

---

## 6. Performance + numerics tips

### Latency expectations
The dominant cost is two `SelfAdjointEigenSolver` calls per FW iteration
(one per BW oracle). Eigen's solver is `O(d³)`. Indicative numbers from
benchmarking on a modern desktop x86_64 (warm-started, fw_exact, 8 max
iterations):

| Problem | Typical |
|---|---:|
| (nx, ny) = (9, 1)  — ESKF + scalar UWB     | ~30–80 µs  |
| (nx, ny) = (15, 3) — PVA INS + 3D pos       | ~80–250 µs |
| (nx, ny) = (21, 3) — 21-state INS + 3D GNSS | ~150–500 µs |

Embedded ARM SoCs (e.g. automotive / robotic compute modules) typically
land 2–4× slower than desktop x86 at the same problem size. Run
`scripts/bench.py` on the actual hardware for an honest number — laptop
benchmarks are not representative.

### Warm-start aggressively
Pass `warm_Sigma_w` and `warm_Sigma_v` from the previous step's result.
This drops typical FW iterations from ~6 cold to ~1–2 warm — a ~3× speedup.

```python
warm_w = None; warm_v = None
for k in range(N):
    res = solve_dr_mmse_tac(
        APA_k, H_k, Sigma_w_k, R_k, theta_w, theta_v,
        warm_Sigma_w=warm_w, warm_Sigma_v=warm_v,
    )
    warm_w, warm_v = res.wc_Sigma_w, res.wc_Sigma_v
```
The `DRWrapper` helper does this automatically.

### Use `solver='fw_exact'` (the default)
The two solvers in the C++ are:
* `fw_exact` — exact BW oracle via 1D bisection, 8 FW iterations. Default.
  Tighter solution, slightly slower per-iteration but converges in fewer.
* `fw` — approximate BW oracle, up to 50 FW iterations. Faster per
  iteration but may need many more iterations on hard problems.

Empirically `fw_exact` is the right default for online filters: lower
worst-case latency, tighter posterior. Switch to `fw` only if you have a
hard real-time deadline that 8 exact iterations can't hit.

### Symmetrize defensively
The solver returns symmetric outputs, but if you `np.linalg.solve(S, ...)`
with a slightly asymmetric `S`, you can leak ε-level asymmetry into the
gain. The `DRWrapper` does `0.5 * (P + P.T)` after every step. If you
write your own integration, do the same.

### PSD numerics for non-trivial state vectors
For the 21-state INS path the parent project hit one numerical pitfall:
clipping individual diagonal entries of an LDL-form covariance can
violate PSD. **Don't clip diagonals.** If you need to ceiling the
worst-case `R`, scale the *whole matrix* down by the largest-axis ratio:
```python
ratios = np.diag(res.wc_Sigma_v) / np.diag(R_hat)
scale_down = max(1.0, ratios.max())
R_apply = res.wc_Sigma_v / scale_down
```
This preserves the off-diagonal structure (correlations) while bounding
the diagonal.

---

## 7. Layout

```
dr_mmse_pkg/
  CMakeLists.txt              # build the C++ pybind11 module
  build.sh                    # one-shot build helper
  pyproject.toml              # metadata for `pip install -e python/`
  cpp/
    include/dr_mmse/          # public headers
      types.h                 # Eigen typedefs + sym, inner, fro_norm2
      kalman_utils.h          # posterior_cov, kalman_stats
      fw_oracle.h             # BW-ball linear oracle
      dr_mmse_tac.h           # standalone solver API
    src/
      kalman_utils.cpp
      fw_oracle.cpp           # exact + bisection oracles
      dr_mmse_tac.cpp         # FW solver (TAC + factored TAC)
      bindings.cpp            # pybind11 module: dr_mmse_cpp
  python/
    dr_mmse/
      __init__.py             # re-exports + import-safety message
      wrapper.py              # DRWrapper helper class + dr_kalman_update
  examples/
    01_minimal_linear.py      # raw API on a 2-state random walk
    02_ekf_apa_tracking.py    # 9-state ESKF with the wrapper
  tests/
    test_basic.py             # parity, PSD, warm-start, wrapper tests
  scripts/
    bench.py                  # latency benchmark
```

---

## 8. Reference deployments from the parent project

The same C++ solver was used in two pipelines that reached publication:

* **UTIL UWB indoor localization** (Crazyflie 2.1, ~8m × 8m, Vicon GT)
  — DR-ESKF wrapped a 9-state Python ESKF with scalar TDoA measurements.
  Best fixed-θ DR improved RMSE by 19% on the hardest constellation
  (corner geometry, multi-path heavy).

* **i2Nav-Robot GNSS/INS** (8 sequences, F9P GNSS, ADIS-16465 IMU)
  — DR-KFGINS wrapped a 21-state C++ INS via override hooks. DR alone
  recovered 47% on parking00 (heavy intermittent GNSS) over the vanilla
  KF-GINS baseline.

Both deployments shared the same FW-exact solver in this repo. The
parent project also shipped a learned noise-prediction head ("MUSE
adapter") that fed `(μ_v, R̂)` to the filter; that adapter is a separate
research artifact and is NOT part of this delivery. **DR-MMSE on its own
is enough to be a useful robustifier** — the adapter is icing.

---

## 9. What this package is and isn't

**Is**: a dataset-agnostic, filter-agnostic per-step DR Kalman update
that compiles cleanly on common targets and gives you correct numerics
out of the box. The hard part of the implementation (the BW-oracle math,
the APA decomposition, PSD safeguarding) is done.

**Isn't**:

* A full filter. You bring your own EKF / ESKF / INS. This package
  replaces only the measurement-update step.
* A learned adapter. `(μ_v, R̂)` prediction from raw sensor features
  needs training data and is out of scope.
* An online θ-selector. `(θ_w, θ_v)` are tuned offline.
* A real-time guarantee. Worst-case latency is bounded by `fw_exact_iters`
  but not deterministic at the OS level — pin to a real-time priority
  if your stack needs hard deadlines.

---

## 10. Citation

If this code helps your research, cite the original method paper and the
parent project (the algorithmic contribution is the FW-with-exact-BW-oracle
solver and the TAC formulation, not this packaging).

```
% TODO: add citation block once the parent paper is on arxiv
```

License: MIT — see [LICENSE](LICENSE).
