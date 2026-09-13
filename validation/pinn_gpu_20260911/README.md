# PINN known/KC GPU checks and CPU preservation

This is bounded numerical validation, not completed PINN training or M1 scientific acceptance.

- GPU01's original failed tiny gate is retained: four of 26,047 flat-gradient elements differed (maximum 4.377216100692749e-8). It is not reclassified as passed.
- The sole GPU01-to-GPU02 correction fixes the known/KC mask placeholder's static shape to 4x1. Precision and tolerance were not changed.
- GPU02 gate 1 passed both zero and nonzero/sparse-coefficient tiny cases exactly: prediction, library, derivatives, losses, full gradients, one NAdam update and slots. Each case uses 8 training/8 physics rows, chunk 4.
- GPU02 gate 2 passed continuous versus new-process own-GPU-run resume: 81 TF variables, mask/coefficients, phase/l0 state and complete L-BFGS cursor/trace. Each logical run uses only 17 NAdam updates and six STRidge calls; L-BFGS stops at its two-iteration diagnostic limit, not convergence.
- GPU02 gate 3 passed one update at all 24,000 training/84,000 physics rows, chunk 32,768. A full-row single update is not a complete training budget.
- The original CPU run was intentionally paused, with its actual exit code 4294967295 preserved. Its committed pre-NAdam 2,000/5,000 state was archived and successfully restored on CPU in a separate process: all 81 TF states plus mask exact, RNG/FSM intact, no forward/update/advance/save. The entire 31,000-update protocol was not completed. This is not CPU-to-GPU resume.

See [summary](summary.json), [scientific comparison hashes/results](gpu_array_comparisons.json), [GPU own-run resume](gpu_resume.json), [CPU preservation](cpu_preservation.json), and [historical-source mapping](source_mapping.json).

Historical CPU source SHAs are checked against the preserved source directory, not silently against newly changed candidate files. The old source-binding records remain untouched. Current TF32 is recorded in the actual GPU profiles; the historical training TF32 setting remains unrecorded.

No model weights, raw scientific arrays, training journals, third-party algorithm code or personal device UUID are distributed here. Packaging used the standard library only, with no TensorFlow/Torch import or GPU operation.
