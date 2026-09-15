# Experiment figures / 实验图像

All five figures referenced by the experiment document are regenerated from numerical results using Python/matplotlib and exported as SVG. Plotting does not train a model or select a checkpoint. It can be rerun independently after numerical experiments finish.

实验文档的五幅图均由数值结果生成，只导出 SVG。当前预测图展示已完成的 GIFT-Lite 和相应基线；全数据 GIFT 的预测结果尚未纳入。保留既定字体、布局和报告时刻，示例轨迹预先固定在独立测试集内。只作容纳数据、标签和方法所需的布局、轴范围及色限调整，不以方法排名作为绘图通过条件。M1 图像不变。

| Figure | Evidence and layout | Statistics / selection |
| --- | --- | --- |
| M1 parameter identification | Three side-by-side parameter panels; five methods and three noise conditions; absolute percentage error | One estimate per method, condition and parameter; error recomputed from estimate and true parameter; no error bars or significance test |
| M2 mean error | Explicitly selected completed methods, 183 × 96 mm nominal canvas | 180 trajectories; each GIFT regime averages three seed-specific trajectory means; one model for each baseline; markers are measured times, PCHIP is only a visual guide |
| M2 keyframes | Scalar fields and signed residuals in aligned blocks, with a GIFT-Lite row | Prespecified test trajectory 1045, seed 20260820, t = 5, 6, 7, 8; no averaging of prediction fields |
| M3 mean error | Three temporal-error panels for N64, N96 and N128, 183 × 72 mm nominal canvas | Same 180 paired trajectories; each GIFT band's pointwise range spans three seed means, not a confidence interval; each baseline uses one model |
| M3 fields | Reference/prediction row and aligned signed residuals, with a GIFT-Lite column | Prespecified test trajectory 1150, seed 20260820, N128, t = 6; not selected by the newly measured errors |

## Data and image integrity

- Curves and document tables must use the same completed metrics CSV. Field panels use the corresponding raw predictions, with displayed relative-L2 errors checked against recorded metrics.
- Keep CSV/NPZ source data and SHA-256 provenance with each figure. Record the rendering script and experiment's model/data identities; do not mix runs when assembling a document.
- Selected field arrays use `source_fields.npz`. Figure folders contain SVG, numeric CSV/NPZ and JSON records; plotting code and instructions remain in the project's source and documentation directories. `figure_manifest.json` binds the final exported bytes after SVG display hints are applied.
- Reject missing or duplicate method/seed/time combinations and non-finite values. Do not drop a failed method or a difficult trajectory to obtain a figure.
- Keep existing axis/color limits when all values fit. If needed, expand a shared limit for all methods; never clip, normalize each method separately or change the underlying arrays to preserve appearance.
- Residuals are prediction minus reference. Fields cover the complete spatial grid, without cropping, smoothing or selective contrast changes. Use vector cells, not embedded raster images; keep text editable in SVG.
- SVG QuadMesh groups use `shape-rendering="crispEdges"` to prevent viewer-dependent white seams between adjacent cells. This display hint does not change cell geometry, colors or numerical arrays.
- M1 parameter labels use editable Unicode Greek letters, avoiding font-specific private-use glyphs that can display as unrelated symbols on another device.
- Zero error is a valid measurement. A logarithmic figure must either display it explicitly with a documented scale adjustment or stop with a clear message; never replace zero with a fabricated positive error.
- The t = 5 zero-error anchor is omitted only from the M3 logarithmic curves, as stated in the caption. Positive lead-time measurements are not omitted.
- Before document delivery, check every linked image is SVG, all files exist, text/labels are readable, no SVG embeds raster images, and figure/source hashes match the selected experiment outputs.

图像只负责呈现证据，不预设哪种模型获胜。若方法排序、稳定性或误差范围发生变化，正文与图注必须作相应的最小必要修改。

## Independent rendering commands

Run these from the project root after the respective numerical experiment is complete. Each output directory must be new; rendering never replaces an existing figure package. M2 and M3 also render automatically after their numerical outputs are committed, unless `--skip-plots` is used.

```shell
python -m experiments.formal.m1_equation_identification.aggregate --jobs runs/m1_jobs --output runs/M1 --skip-plots
python -m experiments.formal.m1_equation_identification.plot_parameters --result-dir runs/M1 --output-dir runs/figures_M1
python -m experiments.formal.m2_recursive_prediction.plot_mean_error --metrics runs/M2/summary/metrics.csv --output-dir runs/figures_M2_curve --gift-regimes GIFT-Lite
python -m experiments.formal.m2_recursive_prediction.plot_keyframes --input runs/M2/raw/predictions.h5 --output-dir runs/figures_M2_keyframes --gift-regimes GIFT-Lite
python -m experiments.formal.m3_cross_resolution.plot_results --result-dir runs/M3 --output-dir runs/figures_M3 --gift-regimes GIFT-Lite
```

For results containing both completed GIFT regimes, omit `--gift-regimes GIFT-Lite`;
the default requires both. An absent selected method is an error, not a reason
to omit it silently. The M1 input must contain all 15 completed method/condition
jobs (45 parameter rows), not a partial table. These commands require the numerical
completion record and verify the input hashes. `figure_manifest.json` records the
exported files, renderer, source identities, statistics and SVG checks. Rendering
a completed run does not establish that another training run reproduced it.
