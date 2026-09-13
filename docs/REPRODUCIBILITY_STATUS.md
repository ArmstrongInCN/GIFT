# Reproducibility status / 复现状态

Status: **private candidate; not yet release-ready**. No successful full new
training campaign or complete six-experiment acceptance is claimed for this
candidate. This file separates existing evidence from work still to do.

## Preserved reference / 保留的参照

`EXPERIMENTS.md` is copied without changing a byte from the authoritative project.
SHA256: `874859f3b40565c6ee252ced471b507f475421213a89e682863db410275cfa5c`.
The tables and figures under `results/formal/` are the archived reference,
not outputs produced by these candidate commands. They are not training inputs.

## Current verification / 当前验证

Portable summaries and execution bindings for completed quick checks are in
[verification evidence](../validation/README.md). They are separate from the
unchanged archived reference tables and do not establish fresh training.

| Item / 项目 | Evidence / 状态 |
| --- | --- |
| First-party GIFT model and correction tests | 17 original tests passed in the candidate |
| Shared resume journal | 3 CPU tests passed: replayed update is byte-identical; changed identity, corrupt data and stale concurrent writer rejected |
| Dataset package | Separate 13.03 GB package; full checksum and schema checks passed; owner selected CC BY 4.0 for own data, with third-party input exceptions |
| External baseline loading | All four model types passed strict original-state loading and CPU forward checks; FNO GPU first-forward and N64/N96/N128 normalization lifting exactly matched original implementation |
| Low-frequency GIFT resume | 7 CPU fixture tests passed, including three interruption points across two phases |
| High-frequency GIFT resume | Derivative, short-rollout and long-rollout interruptions exactly replayed CPU fixture history/selection/state (3 tests) |
| Independent baseline resume | Four real models passed short CPU continuous/resume comparisons; final U-NO import-side-effect fix and checkpoint-interval regression passed separately |
| Complete-journal export recovery | Added an independent CPU-only [recovery tool](CHECKPOINT_EXPORT_RECOVERY.md) for interruption after the complete journal is committed but before weights export finishes. Six targeted tests passed in 6.02 s, including an actual synthetic-tensor save/load subprocess, input-SHA preservation and no-overwrite behavior. Synthetic metadata is explicitly not training evidence. Unrecorded provenance stays unknown; real full-model recovery and scientific acceptance remain separate. Existing training source files were not changed |
| Controlled first training update | FNO-2D's full 150-step objective and FNO-3D's full 150-frame block matched the current formal functions exactly. U-Net's complete physical batch 20 / four-step first update matched the original B7 control, including all 36 gradient tensors and 60 updated state tensors byte-exact; see [formal first-batch proof](../validation/unet_formal4_20260911/README.md). This is not current-source 500-epoch acceptance; the historical B7 full-training evidence remains separate |
| Baseline normalizer reduction | Input position 0 and output position 0, each over all 62,000 training windows, produced exactly equal mean/std; 2 of 196 positions checked |
| M1 independent quick jobs | Three conditions × five methods: 15 separate jobs, 45 parameter rows and all 180 numeric values passed the unchanged tolerance. Original nine GIFT/PDE jobs retain runner SHA prefix `27AEF71B`; six later PINN coefficient-readout jobs retain `3D95B708` (their 72 values are exact). This is not a single-source or fresh-training campaign; see [comparison](../validation/quick_20260910/M1/comparison.json) and [per-job provenance](../validation/quick_20260910/M1/provenance.json) |
| Data generation controls | 17 bounded CPU tests passed; noise/sampling equivalence and clean integration schema checked. Full clean dataset regeneration and training on that collection remain unverified |
| Regenerated baseline input profile | Default released-file SHA retained; 29 bounded metadata/plumbing and rejection tests passed. Successful fixtures are not full generated-data acceptance |
| Experiment continuation | Six CPU receipt tests passed; full M2 then reused all seven completed numerical calls in 31.37 s with a byte-identical metrics CSV, without re-inference. This tests completed-call recovery, not an actual power failure inside a unit |
| Optional plotting failure | 2 tests passed: numerical evidence persists and failure/skipped status is explicit |
| Recorded default test suite after GPU02 integration | The September 11 CPU run had 143 passed, 25 optional external-data/real-runtime checks skipped, 3 CPU pin-memory warnings, and 24 separately reported subtests passed in 17.72 s, rc0. Its JUnit has 168 actual testcase records, zero errors/failures and 25 skips; do not add subtests again to the passed count. Seven bound source/test files were unchanged. See [test evidence](../validation/environment_20260911/run_02/README.md); this historical run and the earlier 139-pass record remain unchanged |
| September 13 300-file index-export CPU check | All 300 staged blobs, candidate work files and new exported files matched byte-for-byte (215,907,464 bytes). The exported default CPU suite had **149 passed, 25 skipped, 3 warnings in 21.78 s**, plus 24 separately reported subtests; JUnit has 174 records and zero failures/errors. Seventy-five exported source files executed, with zero origin violations and no CUDA initialization or TensorFlow import. Checkpoint verification still **FAILED**, 15/16 passing with the U-NO Release asset missing. See [snapshot evidence](../validation/index_export_300_20260913/README.md). This fixed snapshot predates this evidence package/status update; it is an existing-environment offline export, not a clean installation, GitHub clone or scientific acceptance proof |
| M2 full checkpoint inference | Completed in 271.19 s; 200 trajectories, all five methods/three GIFT seeds; 49 rows, 245 values and 98 counts passed the original criterion. Original trained weights, no fresh training; plots deliberately separate |
| M3 full checkpoint inference | Completed in 680.75 s; N64/N96/N128, all formal cohorts and methods; 1,870 rows, 9,350 values and 3,740 counts passed the unchanged criterion; no new training |
| S1 full checkpoint inference | Completed in 747.27 s; all 180 designated values and 360 counts over 180 rows passed the unchanged criterion; no new training |
| S2 full checkpoint inference | Completed in 716.90 s; 84 rows, 420 designated values and 168 counts passed the unchanged criterion; both correction settings/cohorts and all three seeds; no new training |
| S3 full checkpoint inference | Completed in 654.81 s; 33 rows, 132 designated values and 66 counts passed the unchanged criterion; no new training |
| Candidate M1 complete acceptance | Numeric quick checks for all five methods passed as above; six recovered PINN source chains passed 384/384 hash checks. Full fresh-training and structure/scientific-assertion acceptance remain unverified; quick readout does not establish sparse structure recovery |
| PINN-KC bounded GPU validation | GPU02 passed two tiny cases with all 26,047 gradient elements and updates exact; a complete tiny FSM/L-BFGS new-process resume with all 81 variables and scientific traces exact; and one full 24,000-observation / 84,000-physics-row update at chunk 32,768, exact in 19.396 s. The sole computational correction from GPU01 fixed KC mask static shape to `[4,1]`; no tolerance or precision change. These are not full GPU training or M1 acceptance; see [gate identities and device instructions](adapters/PINN_ENVIRONMENT.md#kc-gpu-三项有界验证与独立启动) |
| PINN-KC real GPU tiny CLI and completed-run reuse | A new tiny job completed with rc0 in 17.756 s; its subsequent completed-run resume returned rc0 in 10.204 s. Actual `already_complete` changed false→true; the job's 17 NAdam updates, six STRidge rounds and terminal hashes were verified, and its entire output inventory stayed byte-exact after reuse. No training was repeated. Validation SHA `D845FD7DDDC8CB43B92DDA4CD9733EF3E1CB58B3E892F06E7543E96267D5BFF3`; see [portable candidate CLI evidence](../validation/pinn_gpu_20260911/candidate_cli/README.md). Diagnostic-only, not full acceptance |
| PINN-KC full-budget GPU training | The separate fresh noise001/known run completed with process rc0 in 3,433.60 s: 5,000 pre-NAdam updates, the original L-BFGS budget, six 1,000-step ADO rounds and 20,000 post-NAdam updates at chunk 32,768. Source/input/terminal and all budget checks passed, but the original three-row comparison **FAILED 2/12 values**, both gamma-derived error columns; all three parameter estimates themselves passed. This is completed training, not numerical acceptance. See [full-run FAIL evidence](../validation/fresh_20260910/pinn_known_noise_001/README.md). The old 2,000-step CPU run remains paused and preserved; neither it nor the tiny GPU job was resumed into this run |
| Candidate full-budget GIFT low training | Fresh clean, 1% and 10% low generators completed all 12,000 steps in 359.47 / 418.15 / 486.19 s using 16 CPU threads: terminal/input/budget checks passed, with all 21 state tensors and model configuration exact in each case. The initial two-thread runs are retained as diagnostics; their pre-training CPU normalization differed. All three used released inputs |
| M1 readout from freshly trained GIFT lows | Three independent CPU readouts used only those new weights: 9 parameter rows / 36 comparisons passed the original criterion, of which 9 values were exact. This is the GIFT subset, not full fresh M1 acceptance; see [readout evidence](../validation/fresh_20260910/m1_gift_readout/README.md) |
| Candidate full-budget GIFT high training | Seeds 20260820, 20260821 and 20260822 each completed the full 18+2+2 schedule using the newly trained low prerequisite. Terminal/provenance/budget checks and qualification passed for all three, but each has only 5/27 bit-exact state tensors; training-history differences are present from epoch 1. These are valid completed runs, not bitwise reproduction. Differences remain in the [20260820](../validation/fresh_20260910/gift_high_20260820/verification.json), [20260821](../validation/fresh_20260910/gift_high_20260821/verification.json) and [20260822](../validation/fresh_20260910/gift_high_20260822/verification.json) evidence |
| M2 using freshly trained GIFT and reference baselines | Completed numerical inference in 426.18 s: all 49 rows, 245 numeric values and 98 counts passed the unchanged criterion. GIFT uses the new low model and all three new high branches; FNO-2D/FNO-3D/U-NO/U-Net still use reference weights. This is not all-model from-scratch acceptance. See the [portable M2 evidence](../validation/fresh_20260910/experiments/M2/README.md), [actual metrics comparison](../validation/fresh_20260910/experiments/M2/comparison.json) and [executed input/source bindings](../validation/fresh_20260910/experiments/M2/provenance.json); high-weight differences remain explicit |
| M3 using freshly trained GIFT and reference FNO baselines | Completed numerical inference in 1,006.35 s: all 1,870 rows, 9,350 numeric values and 3,740 counts passed the unchanged criterion. GIFT uses the new low model and all three new high branches; only FNO-2D/FNO-3D use reference weights (no U-NO/U-Net in M3). This is not all-model from-scratch acceptance. See the [portable M3 evidence](../validation/fresh_20260910/experiments/M3/README.md), [actual metrics comparison](../validation/fresh_20260910/experiments/M3/comparison.json) and [executed input/source bindings](../validation/fresh_20260910/experiments/M3/provenance.json); high-weight differences remain explicit |
| S1 using freshly trained GIFT only | Completed numerical inference in 1,236.85 s: all 180 rows, 180 designated numeric values and 360 counts passed the unchanged criterion. All models used in this experiment are the new GIFT low/high models, not reference baseline weights; high-weight differences remain explicit. This does not establish all-project retraining or full six-experiment acceptance. See the [portable S1 evidence](../validation/fresh_20260910/experiments/S1/README.md), [actual metrics comparison](../validation/fresh_20260910/experiments/S1/comparison.json) and [executed input/source bindings](../validation/fresh_20260910/experiments/S1/provenance.json) |
| S2 using freshly trained GIFT only | Computation completed with process rc0, but numerical comparison **FAILED 6/420**; the other 414 values passed, and all 84 rows / 168 counts matched. All failures are mean, sample standard deviation and maximum at leads 2.5 and 3 for independent holdout, correction disabled, seed 20260821. At lead 3, maximum is 2,285,930.9059321247 versus reference 3,601,342.762978083. The saved per-trajectory metrics identify ID1278 as the dominant contribution, not the first differing floating-point operation. Original tolerance and all observations are retained; no retraining or outlier removal. See [FAIL evidence](../validation/fresh_20260910/experiments/S2/README.md), [trajectory attribution](../validation/fresh_20260910/experiments/S2/trajectory_diagnostic.json) and [execution identity comparison](../validation/fresh_20260910/experiments/S2/identity_comparison.json). The earlier reference-weight S2 PASS is separate evidence |
| S3 using freshly trained GIFT only | Completed with process rc0: all 33 rows, 132 numeric values and 66 counts passed the unchanged criterion. Receipt SHA is `B198B55E96B65EDAE59C1BD23388184DE32A86FF4F6D7A6C076BA18901FA0F45`. See [portable S3 evidence](../validation/fresh_20260910/experiments/S3/README.md) and [comparison](../validation/fresh_20260910/experiments/S3/comparison.json). Its PASS does not erase fresh S2's six failures or high-weight differences; the earlier reference-weight S3 quick check remains separate evidence |
| FNO-2D / FNO-3D formal two-epoch execution and own-run resume | Each fresh run completed epoch 1, then resumed its own checkpoint in a new process to epoch 2, retaining the formal 500-epoch budget and 1,000 trajectories per epoch. Actual CPU audits passed source/input/runtime, saved-state and history checks: 100 cumulative updates for [FNO-2D](../validation/fresh_20260910/fno2d_epoch2/README.md), 200 for [FNO-3D](../validation/fresh_20260910/fno3d_epoch2/README.md); FNO-3D's four normalizer arrays were preserved bit-exactly across resume. Both aggregate losses in both epochs exactly matched their original CSV rows. This is partial-budget, aggregate-only evidence—not 500-epoch acceptance or proof of reference gradients/weights/trajectory |
| Exact-file portable checkout test | An earlier offline copy of 187 proposed Git work files passed 131 default tests, with 25 skipped and 3 CPU warnings, in 33.32 s using the existing Python environment. Code-origin guards rejected access to author source paths. Artifact verification correctly failed on the sole missing U-NO release asset (15/16 files passed). This predates the new staged-byte tests and later evidence; it is not a current complete-candidate exported checkout, a Git staged-blob checkout, a GitHub clone or a clean-environment test |
| Later 249-file Git-index export snapshot | Actual staged blobs, work files and a new offline export matched byte-for-byte; the exported snapshot passed 143 default CPU tests with 25 skipped in the existing environment. Asset verification **failed** with 15/16 passing: the U-NO release asset was missing. Audit result SHA `1FEE11485727ED183E8E6D2B5367885CD1A9ACC17A1B30316F035C1D99C5839A`. This historical snapshot predates later documentation/evidence packages; it is not validation of today's complete checkout, a GitHub clone or a clean environment |
| September 10 environment installation attempts (historical) | Official cu126 wheel and all nine direct pins resolved successfully (32 packages). Those actual installation attempts were interrupted by download timeout and Windows file-in-use error; those attempts remain failures, not an installation pass |
| September 13 fresh Windows venv installation and CPU verification | A new CPython 3.10.19 venv with no shared system site-packages installed all 32 recorded runtime distributions, the verified Torch2.10.0+cu126 wheel and the actual GitHub clone at commit `c6ae4d0268f782d64603bab29049b70e87a95b2f` editable. The first editable build failed at a 262-character Windows temporary path; a short per-process temporary directory resolved it without project/system changes. All runtime versions and imported dependencies came from the new venv; both `pip check` calls passed. Default CPU suite: **149 passed, 25 skipped, 3 warnings in 21.77 s**, with 24 separately reported subtests; 174 JUnit records, zero failures/errors, 75 clone source files, no origin violations or CUDA/TF initialization. All 306 tracked files, HEAD and validation pins remained unchanged. See [fresh-install evidence](../validation/fresh_environment_20260913/README.md). The tested commit predates this evidence/documentation update; this uses an existing Python installation and does not prove a fresh OS, another device, GPU/full training or resolution of scientific failures |
| U-NO formal first epoch | A fresh 150-epoch-budget run paused after epoch 1 with all 1,000 trajectories and 63 native-backward updates. The actual CPU audit passed source/data/runtime, initialization and saved-state integrity. Its per-trajectory loss summed over 20 steps was 11.766310146331787; converting to the archived per-step reporting unit gives 0.5883155073165893 versus 0.5883155637741089, an absolute difference of 5.64575e-8. This is not bit-exact, a newly invented training tolerance, or full 150-epoch acceptance. The native CUDA nondeterminism warning and incomplete historical environment provenance remain explicit; see [first-epoch evidence](../validation/uno_epoch1_20260911/README.md) |
| U-NO complete formal training with own-run continuation | All 150 epochs / 9,450 updates completed. The CPU terminal audit passed source/input/runtime, budget, history and optimizer/RNG integrity; the export matches its own final journal in 36/36 tensors. Against the published reference, 0/36 tensors and 0/150 historical losses are bit-exact; differences are retained, not given a new training tolerance. The resumed process took 8,401.320 s, plus the separate first-epoch process's 66.069 s. See [full-training evidence](../validation/uno_full_20260911/README.md). Complete training does not imply M2 acceptance |
| M2 using freshly trained GIFT and full fresh U-NO | Inference completed in 282.54 s, but the unchanged comparison **FAILED 29/245 values**, all belonging to U-NO; 216 values passed and all 49 row keys / 98 counts matched. At t=8 (lead 3), U-NO mean relative error is 0.6600432087826834 versus reference 0.41606485124340836. FNO-2D/FNO-3D/U-Net remain reference weights; this is a mixed-origin experiment, not all-model fresh acceptance. See [M2 FAIL evidence](../validation/fresh_uno_M2_20260911/README.md). No tolerance, reference table or outlier selection was changed |
| FNO-3D complete formal training with own-run continuation | All 500 epochs / 50,000 updates completed. The CPU terminal audit passed full source/input/runtime, history, optimizer/RNG and terminal-export integrity. All 1,000 historical loss comparisons, all 50 model-state tensors and all four normalization tensors match the published reference exactly, without a tolerance. The final own-epoch-2-to-500 invocation took 41,151.544 s; separate earlier invocations are not included in that duration. See [full-training evidence](../validation/fno3d_full_20260911/README.md). This establishes the checked-host full training result, not uninterrupted-vs-resumed equivalence or M2/M3 experiment acceptance |
| FNO-2D complete formal training with own-run continuation | All 500 epochs / 25,000 updates completed. The CPU terminal audit passed full source/input/runtime, history, optimizer/RNG and terminal-export integrity. All 1,000 historical loss comparisons and all 42 model-state tensors match the published reference exactly, without a tolerance; the export also matches its own final journal in 42/42 tensors. No published weights were used for initialization or continuation. The final resumed invocation took 76,985.728 s; earlier invocations and pauses are not included in that duration. See [full-training evidence](../validation/fno2d_full_20260913/README.md). This establishes the checked-host full training result, not uninterrupted-vs-resumed equivalence; independent M2/M3 results follow below |
| M2 using freshly trained GIFT, both FNOs and U-NO | Independent inference completed in 333.05 s using both audited fresh FNO terminal exports and the full fresh U-NO; only U-Net uses reference weights. The unchanged comparison **FAILED 29/245 numeric values**, all U-NO; 216 values passed and all 49 rows / 98 counts matched. Neither fresh FNO added a failed item. This does not establish bit-exact inference or all-model acceptance. See [complete FAIL evidence](../validation/fresh_fno_M2_20260913/README.md), [comparison](../validation/fresh_fno_M2_20260913/comparison.json) and [input/execution provenance](../validation/fresh_fno_M2_20260913/provenance.json). Earlier U-NO/GIFT failures remain preserved |
| M3 using freshly trained GIFT and both FNOs | Independent inference completed in 686.55 s using the new GIFT low/high models and both audited fresh FNO terminal exports; no U-NO/U-Net is used. All 1,870 rows, 9,350 numeric values and 3,740 counts passed the original criterion. See [PASS evidence](../validation/fresh_fno_M3_20260913/README.md), [comparison](../validation/fresh_fno_M3_20260913/comparison.json) and [input/execution provenance](../validation/fresh_fno_M3_20260913/provenance.json). This is this experiment's numeric acceptance, not bit-exact inference or whole-project acceptance; M2, S2 and PINN failures are not erased |
| Private GitHub distribution and actual clone validation | Uploaded to [ArmstrongInCN/GIFT](https://github.com/ArmstrongInCN/GIFT), with PRIVATE visibility independently verified. Actual GitHub clone/fetch of commit `0a9848bea07c8fd7f21f2a44069f28e41c5d091e` matched all 303 committed files (215,915,082 bytes) exactly. The separately authenticated U-NO download matched its catalog and GitHub digest; **16/16 checkpoint files passed**. The existing-environment CPU suite had **149 passed, 25 skipped, 3 warnings in 21.48 s**, plus 24 separately reported subtests; zero JUnit failures/errors, 75 clone source files exercised, zero origin violations and no CUDA/TF initialization. See [scoped clone evidence](../validation/github_clone_20260913/README.md), including the retained initial harness-only failure. This tested commit predates the evidence/status update; no clean installation, new training or scientific acceptance is inferred |

## Known original-record and prior reproduction issues / 已知问题

An earlier independent full training campaign—not this candidate—compared 10,507
designated scalar values over 2,261 rows. It had 299 out-of-tolerance values:
M1 2; M2 29; M3 253; S1 9; S2 6; S3 0. Row keys, counts and input-integrity checks
passed. This is a failed full numerical acceptance, not a result that can be
labelled “100% reproduced”.

- **U-NO:** a custom deterministic interpolation backward pass had replaced the
  original native backward. Losses differed already in epoch 1, before the
  restart. The published-compatible path must retain the native operator; own-run
  checkpoint resume and loading published weights are distinct operations.
  The current original training CLI also imports an unexported `LpLoss` and
  fails before training. The controlled audit therefore isolated unchanged
  training functions; it did not pretend that the broken CLI executed. Native
  GPU backward is not bit-repeatable even against itself: measured parameter
  differences after one update were up to about 5.84e-6 in the self-control,
  versus 4.75e-6 between candidate and formal-function paths. This is evidence
  against promising device-independent bitwise identity, not a full explanation
  of the older campaign's large error differences.
- **FNO:** the training data, key settings and normalization statistics matched,
  but losses differed in epoch 1. The archived evidence does not establish the
  first differing numerical operation. Check initialization/first batch/first
  update before an expensive repeat. Do not attribute it to different library
  versions without evidence.
  New initial state, sampling order, batch, loss, gradients and first updated
  state now match the current formal functions under controlled conditions.
  Historical entry and protocol files have since been recovered with exact
  hashes from their run records and terminal artifacts. The original and
  published checkpoints' 42/50 model tensors, 3D normalization and 500-row
  histories also match exactly. The historical transitive dependency hashes
  and TF32/cuDNN/CPU-thread profile are still incomplete; finding these two
  sources does not itself resolve loss differences or establish full retraining.
  Separately, the current candidate FNO-3D has now completed all 500 epochs:
  1,000/1,000 historical losses, 50/50 model-state tensors and 4/4 normalization
  tensors match the reference exactly on the checked host. The older campaign's
  failures are retained; they must not be confused with this new verified run.
  The current candidate FNO-2D has likewise completed all 500 epochs:
  1,000/1,000 historical losses and 42/42 model-state tensors match the reference
  exactly on the checked host. These new results do not erase the older failed
  campaign. Subsequent independent inference with both fresh FNOs passed all
  M3 comparisons; M2 retained 29 U-NO failures and no failed FNO item. See the
  separately bound new-FNO M2/M3 evidence above; whole-project acceptance remains
  unestablished.
- **U-Net:** the prior corrected convolution-TF32 training profile produced all
  500 epoch records and all 60 state tensors exactly. Do not retrain it merely
  because other methods failed. Inference still needs its own profile check.
- **M1 PINN-SR-KC, 1% noise:** states at the NAdam boundary were identical, then
  diverged during L-BFGS. Two reported error values derive from the same gamma
  coefficient discrepancy. The upstream-to-formal adaptation is substantial,
  not a few path edits; see [M1 adapter constraints](adapters/M1.md).
  A bounded external-class prototype found exact initial prediction, library
  and derivatives, with the first difference in loss reduction/gradient
  accumulation. Matching that original float32 path recovered exact gradients,
  one NAdam update, slots and beta powers in both open/KC cases. This used tiny
  fixed batches and zero initial coefficients; it is not an ADO, L-BFGS,
  full-budget or resumed-training verification.
  The subsequent NAdam-only adapter passed exact two-step versus
  one-step/resume/one-step state recovery in both modes. KC also passed a
  nonzero-coefficient, sparse-mask, two-chunk full-gradient comparison. Open
  still differed in 160 of 26,133 gradient elements (maximum 2.33e-10), so the
  combined equivalence test failed (one pass, one failure). Its execution CLI
  is gated off; the observed difference has not been shown to affect report-level
  metrics. The external STRidge adapter now matches three synthetic NumPy cases
  exactly without importing TensorFlow or Torch. The first-party continuation
  driver around installed SciPy 1.7.3 L-BFGS-B matches two analytic quadratic
  traces and endpoints, including four fresh-process boundary resumptions.
  These are numerical-component tests, not a full PINN training acceptance.
  The KC phase controller then passed a real tiny integration check: two fixed
  logical runs, each with 17 NAdam updates, six STRidge rounds and a two-iteration
  L-BFGS budget. One ran continuously; the other exited at its first L-BFGS FG
  boundary and resumed in a new Python process. All 81 TF state variables,
  coefficient/mask/l0 state, phase events and L-BFGS traces matched exactly.
  This deliberately reduced test is not the full 31,000-NAdam-update protocol
  or report-level acceptance. The KC CLI and safe terminal-state export also
  passed one fixed tiny real CPU execution. The full-budget 1% CPU KC run was
  intentionally paused after its last valid `pre_nadam` commit at 2,000/5,000
  updates. Its frozen source, entire journal and launch evidence were preserved;
  a relocated new-process restore-only check passed with original and archived
  records unchanged. It did not train, forward or save another update. Continuing
  this old run requires its old frozen source and original CPU environment, not
  the current source selected with `--device cpu` and not a GPU continuation.
  GPU02 then passed the three bounded checks listed above. GPU01's four
  coefficient-gradient differences disappeared after the sole KC mask-shape
  correction; neither tolerance nor precision was changed. The public full-phase
  CLI now accepts `--device cpu|gpu` (CPU default, GPU only for known/KC), with
  the legacy environment's DLL directories supplied through child-process PATH.
  The NAdam-only primitive CLI remains CPU. The real GPU tiny CLI then passed
  a new-job terminal export and completed-run reuse; the latter left the entire
  job byte-exact and performed no extra updates. A separate noise001/known GPU
  run subsequently completed the original full budget from untrained
  initialization, not from the paused CPU run or tiny job. It passed source,
  input, terminal and budget checks but failed 2/12 scalar comparisons under
  the unchanged `0.0005 + 0.05 * abs(reference)` rule. Gamma absolute error is
  0.21602565050125122 versus 0.23254746198654175; relative error is
  21.602565050125122% versus 23.254746198654175%. All three parameter estimates
  themselves pass. Being closer to physical truth does not waive reproduction
  of the reported error columns. Execution evidence SHA is
  `28391A7892B5E8625ACE09A289F8A1F9CE8A6D5635AC5651929EC9159D4B9D13`.
  In this candidate run, the pre-NAdam 5,000 boundary was exact in all 60
  primary state tensors plus 22 diagnostic accumulators (82 total). Immediately
  after L-BFGS, all 18 network tensors and the coefficient tensor differed;
  the 38 NAdam slots, two beta powers and mask remained exact. The boundaries
  locate divergence within L-BFGS, not its first differing operation or cause.
  Parameter/gradient packing order was checked and ruled out: both paths use
  `W0,b0,...,W8,b8,coefficients`. No tolerance, source or parameter was adjusted.
  See the [actual scalar comparison](../validation/fresh_20260910/pinn_known_noise_001/comparison.json)
  and [saved boundary summaries](../validation/fresh_20260910/pinn_known_noise_001/boundary_comparisons.json).
  Open execution remains gated. One isolated CPU attempt to fix the open mask
  shape to `[90,1]` left the same 160/26,133 unequal gradient elements (maximum
  2.3283064365386963e-10); three updated parameter tensors also differed.
  This failed change was not integrated, and no full open training was run.
  The isolated diagnostic's result SHA is
  `AF3E6E51ECF001B2CE26B713871088CD461F102491553735C240AB351475D4B4`;
  the small known/KC evidence package does not contain or validate that open test.
- **S1:** per-state relative errors in very weak high-frequency bands are highly
  denominator-sensitive. The original metric is preserved; additional energy-
  stratified diagnostics must not silently replace it.
- **S2:** an uncorrected holdout trajectory (ID 1278, seed 20260821) genuinely
  diverged and dominated the t=7.5/8 mean and dispersion. Do not drop it or enable
  correction in the no-correction ablation. IDs 1200–1399 are held out from GIFT,
  but are part of baseline training: “holdout” is method-specific here. A new
  read-only comparison of saved per-trajectory metrics now also identifies
  ID1278 as the dominant contribution to the fresh-GIFT run's six failures:
  it accounts for over 99.9999997% of each affected mean difference versus the
  prior reference-weight quick run. Recorded source hashes agree 16/16 and
  runtime fields 20/20; both data records agree. Only model selection/input
  records differ; existing low state is 21/21 exact while each high state is
  only 5/27 exact. This narrows the observed difference but does not locate the
  first floating-point divergence or prove which high-weight difference caused
  it. No trajectory is excluded, metric replaced or tolerance changed. Details
  remain in the linked S2 trajectory and identity evidence above.
- **M2 output durability:** an optional keyframe plot failed when a prediction
  exceeded a fixed colour limit. The old runner wrote its final report only
  after plotting; the candidate must persist numeric evidence before plotting.
- **Original packaging:** full dataset-generation commands and complete training
  resume states were not all present in the authoritative folder. A source
  solver module alone is not proof that every archived dataset can be rebuilt.
  The reconstructed generator uses the same initial parameters and byte-identical
  solver source. A bounded CPU check on trajectory 0 nevertheless differs from
  archived fields already at t=0 (max 1.19e-6) and by 2.38e-6 after eight steps.
  There is no NumPy FFT in this path. Missing historical intermediate spectra
  and runtime prevent attributing the first floating-point difference to a
  specific Torch operation; no unsupported numerical patch was introduced.
  The formal M1 package contains final three-parameter PINN readouts, but not
  the complete 90/4-coefficient vectors or trained terminal network files needed
  for independent fast readout. The old code does save such files; its referenced
  work artifacts were simply not included in the checked formal package.
  Archived metrics must not be reverse-engineered into a purported checkpoint.
  A subsequent search of the historical work directory recovered all six
  method records with hashes exactly matching the formal report's native
  evidence references, plus terminal checkpoint files. All 384 checked hash
  declarations passed, and complete coefficient readouts exactly reproduced
  the 18 reported parameters. Six numeric-only terminal archives (82 tensors
  each) are now included under `artifacts/pinn_reference/`, without TF meta
  graphs or external algorithm code. Their checkpoint hashes were measured
  during recovery, not sealed in the historical run; they are reference states,
  not proven resume states for the new controller. All 90 open-library terms
  remain nonzero, so this must not be called verified sparse-structure recovery.

这些问题不会通过修改原实验文档、扩大验收容差或替换参照检查点来掩盖。
候选修复后的新输出必须单独验证，未解决项如实保留。

## Distribution and rights / 分发与权利

No third-party algorithm implementation should be committed, including copies
hidden in adapters, vendored utilities or optimizers. Some pinned repositories
have no explicit redistribution licence; see the source audit. Their absence
from this repository does not constitute a legal opinion about all possible uses.

The owner selected CC BY 4.0 for the generated dataset; auxiliary third-party
input exceptions are identified separately in the package. Creator/publication
metadata must still be supplied by the owner; no names or DOI were invented.
The Zenodo deposit is not created or uploaded by this task. U-NO's terminal
checkpoint is larger than GitHub's ordinary-file limit and is excluded from
ordinary Git tracking. It is now available as an authenticated
[private Release asset](https://github.com/ArmstrongInCN/GIFT/releases/tag/reference-checkpoints-20260913);
follow the [download and integrity-check instructions](../artifacts/README.md).
A plain clone still omits this asset. Earlier offline snapshot failures remain
historical facts and are not rewritten. Private distribution does not resolve
the numerical failures, data-creator metadata or initialization-rights questions.
