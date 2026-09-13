# Candidate PINN GPU CLI: tiny diagnostic only

The actual candidate known/KC noise_001 CLI completed a fresh diagnostic run (17.756359 s), then a new process accepted its already-completed run (10.204045 s). Both exited 0. All recorded source/input checks and every completed job-file byte hash agree. The second invocation reported already_complete=true.

Budget: 8 train rows, 12 physics rows, chunk4; pre-NAdam2, L-BFGS-B maxiter2/maxfun4, 6 rounds of STRidge plus NAdam2, post-NAdam3. Total: 17 NAdam updates and 6 STRidge calls; 81 model-state arrays saved. L-BFGS stopped at the iteration budget, not convergence.

This verifies the candidate CLI's tiny fresh terminal export and completed-run resume. It does not establish full-budget scientific acceptance, interrupted-run resume (see the separate GPU02 evidence), open-mode support, or CPU-to-GPU continuation. Original diagnostic_test_only=true / eligible_for_formal_M1=false remain explicit. The full-budget GPU run is a separate run and has no completion claim here.

[Evidence](evidence.json) contains exact original record hashes, candidate source/data binding, terminal numeric summary and unchanged job-file inventory. No NPZ weights, third-party source, device UUID or absolute private path is copied. The seven parent-directory evidence files are unchanged.
