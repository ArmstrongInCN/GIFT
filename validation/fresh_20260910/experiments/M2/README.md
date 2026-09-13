# M2 fresh-GIFT numerical evidence

Scope: `fresh_GIFT_plus_published_reference_baselines`. This does not establish all-project retraining or complete-project acceptance.

The CSV is copied byte-for-byte from the completed experiment. All original table rows pass `abs(delta) <= 0.0005 + 0.05 * abs(reference)`. No reference values were substituted.

The three newly trained high branches are not bitwise equal to historical weights; each has 5/27 exactly equal tensors. Their measured differences remain in linked training evidence. This experiment's numeric PASS does not erase them or certify other experiments.

M2 combines fresh GIFT with four published baselines: FNO2D, FNO3D, U-NO and U-Net. M3 combines fresh GIFT with only the two published FNO2D/FNO3D baselines. S1/S2/S3 use fresh GIFT only, so all models within those individual experiments are freshly trained; this is not a claim that every project model has been retrained. No weights, raw arrays or plots are bundled. M2 was executed with `--skip-plots`. Packaging performs no model forward, training or GPU operation.
