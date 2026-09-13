# S2: execution complete, numerical acceptance FAILED

The actual fresh-GIFT inference completed (child and harness exit 0), but 6/420 numeric values fail the unchanged rule `abs(delta) <= 0.0005 + 0.05*abs(reference)`. All 84 row keys and 168 counts agree. This is not a scientific PASS.

Failures are mean, sample SD and maximum at lead 2.5/3 for seed20260821, independent holdout, correction disabled. Original candidate CSV bytes and all six failures are retained; no reference values are substituted.

All models in this S2 experiment are freshly trained GIFT; this does not establish all-project retraining or complete-project acceptance. High-branch differences remain in the separate training evidence.

The saved-metric diagnostic compares this run with the prior published-weight quick run (whose table matches the archived reference bytes). It identifies the contribution of trajectory1278 without loading prediction/truth fields, changing metrics, excluding trajectories or rerunning a model. Attribution is descriptive, not proof of the first differing operation.

See [FAIL comparison](comparison.json), [execution/input/source provenance](provenance.json), and [per-trajectory diagnostic](trajectory_diagnostic.json). No weights, raw fields, training code or private paths are bundled.

The supplementary [execution identity comparison](identity_comparison.json) confirms identical 16/16 recorded source hashes and 20/20 runtime fields. Only model-selection configuration and model input records differ; both data records agree. Existing state checks show 21/21 low-model tensors exact and 5/27 high-model tensors exact for each seed. This narrows the observed differences but does not establish their first floating-point cause; numerical acceptance remains FAIL.
