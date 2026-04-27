# DR-MMSE TAC — Distributionally Robust Kalman Update

A drop-in **measurement-update wrapper** that makes any Kalman / EKF / ESKF
filter robust to misspecified noise covariances. Pure C++ / Eigen with a
thin Python layer. No CUDA, no PyTorch, no MOSEK / SDP solver.

| Property | Value |
|---|---|
| Algorithm | Frank-Wolfe over Bures-Wasserstein balls |
| C++ deps | Eigen ≥ 3.4 (header-only, auto-fetched if not on the system) |
| Python deps | NumPy, pybind11 |
| Per-call latency | tens to a few hundred microseconds for typical filter sizes (see §7) |
| Targets | any modern Linux / macOS, x86_64 or aarch64 |

The solver is **filter-agnostic and dataset-agnostic** — it consumes a few
matrices per measurement step and returns a worst-case posterior. You bring
your own EKF / ESKF / INS; this package replaces only the gain computation.

---

## 1. The intuition

A standard Kalman filter assumes you know two things exactly:

* `Q` — the process-noise covariance (how uncertain your dynamics are).
* `R` — the measurement-noise covariance (how noisy your sensor is).

In practice, `(Q, R)` are usually only approximately known. They get
tuned on benchtop data, on a clean stretch of trajectory, or directly
from a sensor datasheet — and then conditions change at runtime. A few
illustrative cases (GNSS / IMU here are just examples; the same idea
applies to UWB, lidar, vision, encoders, magnetometers, etc.):

* The receiver-reported σ for a GNSS fix can underestimate the true
  error in environments where multipath or NLOS dominate.
* The `Q` calibrated for an IMU at one operating point can become
  inaccurate when temperature shifts or biases drift.
* Range / TDoA sensors (UWB, acoustic) report nominal noise that does
  not capture occasional blockage or NLOS bounces.
* Vision / lidar front-ends sometimes pass through correspondences
  whose true error is far larger than the assumed pixel- or
  point-level noise.

When `(Q, R)` are inaccurate, the Kalman gain weighs the dynamics and
the measurement by the wrong relative uncertainty, and the posterior
tends to become over-confident or slightly biased. Typical symptoms
include the filter overreacting to a noisy measurement, or staying
locked into dead-reckoning when a good measurement arrives; NEES
exceeds the expected χ² envelope, and RMSE on harder segments grows
disproportionately compared to clean segments.

DR-MMSE fixes this by treating `(Q, R)` as **uncertain within a ball.**
At every measurement, it solves a per-step minimax:

```
min  ||x − x̂_post||²
 K
s.t. (Σ_w, Σ_v) ∈ argmax_{Σ_w, Σ_v}  trace(P_post(K, Σ_w, Σ_v))
                  s.t.  BW(Σ_w, Q̂) ≤ θ_w
                        BW(Σ_v, R̂) ≤ θ_v
```

where `BW(·, ·)` is the Bures-Wasserstein distance between PSD matrices.
The adversary picks the noise that *maximizes* the posterior error trace;
the Kalman gain is then computed against that worst case. Intuitively:
**hedge against the worst noise within a believable distance of the one
you tuned.**

The two radii `(θ_w, θ_v)` are the only hyperparameters you tune. `(0, 0)`
recovers the standard Kalman update exactly — use this as a parity check
when integrating.

The TAC ("test-and-adapt covariance") decomposition splits the prior as

```
P_pri = APA + Σ_w
APA   = Φ_accum · P_post_prev · Φ_accumᵀ
```

so the BW ball acts on `Σ_w` (process noise accumulated since the last
measurement) instead of the full prior. This makes the optimization
well-conditioned even for INS-style filters where `Σ_w` is structurally
rank-deficient (e.g. process noise injected only into the bias states
of a 21-state filter).

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

The first thing the smoke test checks is that θ=0 reproduces the
standard Kalman posterior to machine precision. If that check fails,
something is off with the build — investigate before relying on the
solver outputs.

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
can either patch the filter to expose it or bookkeep `Phi_accum` yourself
(it's just `Phi_k @ Phi_accum` accumulated in user code).

After the DR solve, inject the worst-case gain back into your filter's
state and covariance:
```
x_post = x_pri + K_dr @ innov
P_post = res.wc_Xpost
```
For Joseph-form filters, the wrapper returns `P_post` directly. Do not
also run your filter's own update on top — the DR solve already
accounts for the measurement.

### "I have an INS where Σ_w is structurally low-rank"

For example, a 21-state INS that injects process noise only on the 6
bias states leaves `Σ_w` in a 6-dimensional subspace of a 21-state
filter, so `Σ_w` is rank-deficient. In that case:

```python
solve_dr_mmse_tac_factored(APA, H, G_list, Q_hat, R_hat, theta_w, theta_v)
```

`G_list[k]` is `nx × nw` (the per-propagation-step noise input matrix)
and `Σ_w = Σ_k G_k @ Q_w @ G_kᵀ`. The BW ball acts on the
well-conditioned `Q_w (nw × nw)` matrix instead of the rank-deficient
`Σ_w` — usually cleaner numerics and faster.

### Innovation sign convention

Some filters define innovation as `dz = z − h(x)`, others as
`dz = h(x) − z`. The DR solver is unaffected by the choice — it only
operates on covariances. The downstream code should apply the chosen
sign consistently between the innovation and the gain.

---

## 5. Mapping your filter onto the API

The solver is dimension-agnostic. The main effort when integrating is
mapping your own state vector and propagation onto
`(APA, Σ_w_hat, H, R_hat)`. The patterns below are illustrative — the
solver itself does not assume any particular sensor or state layout.

### A. Example: loosely-coupled GNSS/INS (15-state error-state)

A common inertial-navigation error state:
```
δx = [δp (3), δv (3), δθ (3), δb_a (3), δb_g (3)]   # nx = 15
```
* `Φ_k` comes from IMU mechanization linearization (kinematic coupling
  between attitude error → velocity error → position error, plus bias
  feed-through).
* `Σ_w` is non-zero only on the `(δv, δθ, δb_a, δb_g)` block — the
  position rows/cols are zero. The structural rank is ≤ 12 in a 15-state
  filter, so the factored API is a good fit.
* When a GNSS position fix arrives in a loosely-coupled setup,
  `H = [I_3, 0, 0, 0, 0]` selects the position rows and `R_hat` is the
  receiver-reported position covariance. (For tightly-coupled
  pseudoranges, each measurement contributes a 1×nx Jacobian whose
  position block is the line-of-sight unit vector.)
* `θ_w` here mainly hedges against IMU bias / mechanization mismatch;
  `θ_v` hedges against measurement-side errors such as multipath in
  challenging environments.

### B. Example: IMU + UWB indoor localization (9-state ESKF)

A leaner error state when biases are estimated separately or considered
small:
```
δx = [δp (3), δv (3), δθ (3)]   # nx = 9
```
* `Φ_k` follows the same skeleton as above, without the bias rows.
* Each UWB measurement is a scalar TDoA or range to a known anchor:
  `H = ∂(‖p − a_i‖)/∂p` plugged into a 1×9 row, `R_hat = σ_uwb²`.
* `θ_v` tends to be the dominant knob here — it hedges against a
  partially-blocked anchor or an NLOS bounce that pushes the residual
  outside what `σ_uwb` would predict.

### C. Example: visual-inertial / lidar-inertial (15- to 21-state)

Same skeleton as A, optionally extended with accelerometer/gyroscope
scale-factor or lever-arm states (→ 21-state). Measurements are
landmark reprojection residuals (VIO) or scan-to-map residuals (LIO):
* `H` is the Jacobian of the residual w.r.t. the error state (typically
  involves the rotation Jacobian and the camera/lidar projection
  Jacobian).
* `R_hat` is the per-feature pixel variance (VIO) or per-point
  point-to-plane variance (LIO).
* `θ_v` helps when feature-matching or correspondence outliers slip
  past the front-end's gating.

### D. Other architectures

The pattern is invariant to the specific state vector and sensor
combination. As long as your filter does
`P_pri = Φ P_post Φᵀ + Q_d` at each propagation and exposes
`(Φ_k, Q_d, H, R_hat)`, DR-MMSE plugs in — wheel odometry + IMU + GNSS,
attitude-only filters, multi-IMU setups, etc.

---

## 6. Tuning `(θ_w, θ_v)`

`θ_w, θ_v ≥ 0` are the only knobs. Both default to 0 (= no DR).
Practical heuristics:

* `θ_v` (measurement-noise robustness) tends to matter less than `θ_w`
  on dense, well-behaved measurement streams, and tends to matter more
  on intermittent or outlier-prone measurements (e.g. NLOS UWB,
  multipath-heavy GNSS, occluded feature tracks).

* `θ_w` (process-noise robustness) is typically the primary knob when
  the dynamics model is the weaker side (poorly characterized IMU,
  unmodeled vehicle dynamics, drifting biases).
  - `θ_w ≈ 0.001` → effectively vanilla Kalman (DR is too small to
    have an effect).
  - `θ_w ≈ 0.5` → a reasonable default; tends to perform well across
    mixed conditions.
  - `θ_w ≈ 1.0` → more conservative; can help in challenging or
    outlier-prone regimes, and may slightly degrade performance in
    clean ones.
  - `θ_w ≈ 5.0` → mainly useful when measurements are intermittently
    lost and you want the filter to stay close to the few good fixes
    that do arrive.

* **Tune offline by grid sweep.** A 6×6 grid of `(θ_w, θ_v) ∈ {0.001,
  0.01, 0.05, 0.1, 0.5, 1.0}` is typically enough. This codebase does
  not include an online θ-selection method — `(θ_w, θ_v)` are constants
  per-deployment. If the robot operates in mixed regimes (e.g. open sky,
  tunnel, parking garage), tuning θ for the worst regime is a
  reasonable starting point; the cost on easier regimes is usually
  small.

* **Diagnostic for "θ too aggressive": NEES grows beyond expected
  bounds.** Compute `NEES = err_postᵀ · P_post⁻¹ · err_post` and
  compare to χ²(dof) thresholds. A mean NEES that exceeds roughly
  2× dof suggests the filter is over-inflating uncertainty.

* **Diagnostic for "θ too weak": DR has minimal effect.**
  If `res.iterations == 0` or `res.iterations == 1` for a typical step,
  the BW oracle is not finding a direction to inflate. Try increasing
  θ.

A grid-sweep harness pattern:
```python
for theta_w in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    for theta_v in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
        run_filter(theta_w, theta_v)  # log RMSE + NEES
```
Pick the `(θ_w, θ_v)` with best RMSE subject to NEES within
`[0.5×dof, 2×dof]`.

---

## 7. Performance + numerics tips

### Latency expectations
The dominant cost is two `SelfAdjointEigenSolver` calls per FW iteration
(one per BW oracle). Eigen's solver is `O(d³)`. Indicative numbers from
benchmarking on a modern desktop x86_64 (warm-started, fw_exact, 8 max
iterations):

| Problem | Typical |
|---|---:|
| `(nx, ny) = (9, 1)`  — 9-state ESKF + scalar range  | ~30–80 µs  |
| `(nx, ny) = (15, 3)` — 15-state INS + 3D position    | ~80–250 µs |
| `(nx, ny) = (21, 3)` — 21-state INS + 3D position    | ~150–500 µs |

Embedded ARM SoCs (e.g. automotive / robotic compute modules) tend to
run roughly 2–4× slower than desktop x86 at the same problem size. For
a representative number, run `scripts/bench.py` on the target hardware;
laptop measurements may differ noticeably from the deployment
environment.

### Warm-start across steps
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
  iteration but may need many more on hard problems.

Empirically `fw_exact` is the recommended default for online filters:
lower worst-case latency and a tighter posterior. Consider switching to
`fw` if a hard real-time deadline cannot accommodate 8 exact
iterations.

### Symmetrize defensively
The solver returns symmetric outputs, but if you `np.linalg.solve(S, ...)`
with a slightly asymmetric `S`, you can leak ε-level asymmetry into the
gain. The `DRWrapper` does `0.5 * (P + P.T)` after every step. If you
write your own integration, do the same.

### PSD numerics for high-dimensional state vectors
For larger INS-style filters, one numerical pitfall worth flagging:
clipping individual diagonal entries of an LDL-form covariance can
break PSD-ness. It is safer to avoid per-entry diagonal clipping. If
you need to bound the worst-case `R`, scale the *whole matrix* down by
the largest-axis ratio:
```python
ratios = np.diag(res.wc_Sigma_v) / np.diag(R_hat)
scale_down = max(1.0, ratios.max())
R_apply = res.wc_Sigma_v / scale_down
```
This preserves the off-diagonal structure (correlations) while bounding
the diagonal.

---

## 8. Layout

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

## 9. What this package is and isn't

**Is**: a dataset-agnostic, filter-agnostic per-step DR Kalman update
that compiles cleanly on common targets and produces correct numerics
by default. The implementation work (the BW-oracle math, the APA
decomposition, PSD safeguarding) is in place.

**Isn't**:

* A full filter. You bring your own EKF / ESKF / INS. This package
  replaces only the measurement-update step.
* An online θ-selector. `(θ_w, θ_v)` are tuned offline.
* A real-time guarantee. Worst-case latency is bounded by
  `fw_exact_iters` but is not deterministic at the OS level — pin to
  a real-time priority if your stack requires hard deadlines.

---

## 10. Citation & license

If this code helps your research, please cite the DR-MMSE / TAC method
paper. The algorithmic contribution is the FW-with-exact-BW-oracle
solver and the TAC formulation.

```
% TODO: add citation block once the paper is on arxiv
```

License: MIT — see [LICENSE](LICENSE).
