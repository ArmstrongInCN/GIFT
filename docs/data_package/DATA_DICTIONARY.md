# Data dictionary / 数据字典

`vorticity` is float32 `[trajectory,time,y,x]`; `time` is float64 `[time]`;
`trajectory_index` is int64 `[trajectory]`. The spatial axes are y,x on the
periodic domain `[0,2π)²`, without a repeated endpoint. No SI conversion is
specified. NaN and Inf are invalid observations, not missing-data sentinels.

Prediction containers have the attribute `trajectory_id_scheme=canonical`.
Training IDs are 0–999, validation IDs 1000–1039, independent test IDs 1040–1219.
Groups called `validation_aux` contain validation IDs 1020–1039 at the saved
evaluation time range; they are not additional independent test trajectories.
The complete partitions and per-file groups are in `splits.json`.

| Prediction file | Observation groups / saved times |
| --- | --- |
| `fno/fno1000_n64_t0_t10_dt0p02.h5` | 1,000 training and 20 full-interval validation trajectories; N64, 0–10, dt=0.02 |
| `fair_short_horizon/fno1000_n64_t0_t10_dt0p1.h5` | Same training/full-interval validation identities; N64, 0–10, dt=0.1 |
| `prediction/standard_ns_n64_full_spectrum.h5` | GIFT-Lite's 50 training and 20 full-interval validation trajectories; 180 test and 20 auxiliary validation trajectories at 5–10, dt=0.5 |
| `fair_short_horizon/standard_ns_n64_full_spectrum_test_t4p1_t6p0_dt0p1.h5` | 180 test and 20 auxiliary validation trajectories; N64, 4.1–6, dt=0.1 |
| `cross_resolution/cross_resolution_n96_n128_t4p1_t6p0_dt0p1.h5` | Paired N96/N128 test and auxiliary validation groups; 4.1–6, dt=0.1 |
| `fno/fno_test_n64_n96_n128_dt0p02.h5` | Paired test and auxiliary validation groups; N64 4.1–8, N96/N128 4.1–6, dt=0.02 |

The canonical-ID export only renumbers identities and separates validation from
test rows. It does not interpolate, smooth, rescale, round or regenerate fields.
`schema.json` records decoded scientific-array hashes; `manifest.json` binds all
files. Models' independent training records also bind the observed fields and
canonical identities separately from storage-container metadata.

## Identification inputs / 方程识别输入

`standard_ns_n64_full_spectrum.h5` is the unchanged M1 clean observation container.
`m1_parameter_identification/noise_001.h5` and `noise_010.h5` contain paired 1%
and 10% noise. Their local training IDs are 0–49 and local validation IDs 50–69.
These identifiers belong to M1's local protocol, not the prediction-ID space.
The clean container's additional fields are preserved but are not M1 training.

For each group, noise is `ωη = ω + η std(ω,ddof=0) Z`, with NumPy `RandomState(0)`.
The two conditions share the same standard-normal direction. Only vorticity is
perturbed; velocity is reconstructed from that observation by Biot–Savart.

`auxiliary/pinn_sampling/noise_*_seed1234.npz` stores 30,000 measured `(x,y,t)`
locations and `(u,v,ω)` targets, observation row indices split into 24,000 training
and 6,000 validation rows, plus 60,000 LHS physics locations. All belong to local
trajectory 0. These row indices are not trajectory IDs or independent trajectories.

`initial_conditions/baseline_extra950.json` contains prediction training IDs
50–999 and the unchanged integration parameters used by the independent data
generator. Each parameter array has four rows `[circulation,sigma,cx,cy]`.

## Numerical source / 数值来源

The simulated equation is `ω_t + u·∇ω = 0.01 Δω − 4 cos(4y)`. Full-spectrum
Fourier pseudospectral ETDRK4 uses internal dt=0.005 and padding grids 99,147,195
for N64,N96,N128. State spectral modes are not truncated. All noise/grid variants
of a given initial condition must stay in the same prediction split.
