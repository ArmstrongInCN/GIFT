# Fresh U-NO full M2 evidence

Fresh models: this campaign's GIFT low, three GIFT high seeds, and the full 150-epoch U-NO terminal. Reference models: FNO2D, FNO3D and U-Net. This is not all-model or whole-project from-zero acceptance.

Actual M2 inference completed before packaging. The CSV is copied byte-for-byte, with all 49 rows, 245 scalar comparisons and 98 count comparisons. Original tolerance remains abs(delta) <= 0.0005 + 0.05 * abs(reference). Comparator result: **FAIL**; all recorded failures, if any, remain in comparison.json.

U-NO's measured historical training-loss differences are retained independently; this M2 comparison does not erase them. GIFT training provenance is linked in provenance.json. Inference used --skip-plots. No PT, dataset or raw prediction array was read by packaging; no GPU/forward/training was performed. Input hashes are the launcher/experiment's recorded bindings, not a new raw-input rehash or deep-verification claim.
