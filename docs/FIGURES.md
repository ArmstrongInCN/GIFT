# Experiment figures / 实验图像

All eight figure placements referenced by the experiment document are generated from numerical results using Python/matplotlib and exported as SVG; they resolve to seven published files because the S4 section reuses the published M2 keyframe plate unchanged. Plotting does not train a model or select a checkpoint. It can be rerun independently after numerical experiments finish.

实验文档的八处图像引用对应七幅已发布 SVG，均由数值结果生成，只导出 SVG；S4 小节左幅改用同一作图程序按已发布的 M2 汇总数据重绘（M2 数值不变、不重算），以便与右幅共用同一纵轴。预测图同时展示 GIFT、GIFT-Lite 和相应基线。保留既定字体、布局和报告时刻，示例轨迹预先固定在独立测试集内。只作容纳数据、标签和方法所需的布局、轴范围及色限调整，不以方法排名作为绘图通过条件。色限是按初值分布预设的固定常数：四涡旋沿用正文图所用的 ±19 / ±21，高斯分布预设为 ±25；预设范围会裁切所选关键帧时渲染器直接拒绝出图，不伸缩色标，也不在正文中事后改口。M1 图像不变。

| Figure | Evidence and layout | Statistics / selection |
| --- | --- | --- |
| M1 parameter identification | Three side-by-side parameter panels; five methods and three noise conditions; absolute percentage error | One estimate per method, condition and parameter; error recomputed from estimate and true parameter; no error bars or significance test |
| M2 mean error | Six methods, 183 × 96 mm nominal canvas | 180 trajectories; each GIFT regime averages three seed-specific trajectory means; one model for each baseline; markers are measured times, PCHIP is only a visual guide |
| M2 keyframes | Scalar fields and signed prediction errors in aligned blocks, with separate GIFT and GIFT-Lite rows | Prespecified test trajectory 1045, seed 20260820, t = 5, 6, 7, 8; no averaging of prediction fields |
| M3 mean error | Three temporal-error panels for N64, N96 and N128, 183 × 72 mm nominal canvas | Same 180 paired trajectories; each GIFT band's pointwise range spans three seed means, not a confidence interval; each baseline uses one model |
| M3 fields | Reference/prediction row and aligned signed prediction errors, with separate GIFT and GIFT-Lite columns | Prespecified test trajectory 1150, seed 20260820, N128, t = 6; not selected by the newly measured errors |
| S4 initial-distribution comparison | Two panels drawn by the same renderer on one shared vertical range, then placed side by side | 180 trajectories per distribution; each GIFT regime averages three seed-specific trajectory means; one model per baseline; the left panel is redrawn from the published M2 summary values with M2 numbers unchanged and M2 not recomputed; the GIFT-Lite curve stops at t=6.5 because later report times retain fewer than 180 finite trajectories, and that is annotated on the panel |
| S4 keyframes | Same block layout as the M2 keyframe plate; prespecified Gaussian test trajectory 2265 | Prespecified trajectory 2265, seed 20260820, t = 5, 6, 7, 8; one shared colour scale per quantity at the prespecified ±25 for this population, wider than the paper's ±19 because trajectory 2265 reaches about 24.06, recorded in `figure_manifest.json` |

## Data and image integrity

- Curves and document tables must use the same completed metrics CSV. Field panels use the corresponding raw predictions, with displayed relative-L2 errors checked against recorded metrics.
- Keep CSV/NPZ source data and SHA-256 provenance with each figure. Record the rendering script and experiment's model/data identities; do not mix runs when assembling a document.
- Selected field arrays use `source_fields.npz`. Figure folders contain SVG, numeric CSV/NPZ and JSON records; plotting code and instructions remain in the project's source and documentation directories. `figure_manifest.json` binds the final exported bytes after SVG display hints are applied.
- Reject missing or duplicate method/seed/time combinations and non-finite values. Do not drop a failed method or a difficult trajectory to obtain a figure.
- Colour limits are prespecified constants, one pair per initial-condition population, and are never fitted to the plotted trajectory. Keep an existing shared limit when all values fit; a prespecified range the selected keyframe would exceed makes the renderer refuse the figure. Never clip, normalize each method separately or change the underlying arrays to preserve appearance.
- Prediction error is prediction minus reference. Fields cover the complete spatial grid, without cropping, smoothing or selective contrast changes. Use vector cells, not embedded raster images; keep text editable in SVG.
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
python -m experiments.formal.m2_recursive_prediction.plot_mean_error --metrics runs/M2/summary/metrics.csv --output-dir runs/figures_M2_curve
python -m experiments.formal.m2_recursive_prediction.plot_keyframes --input runs/M2/raw/predictions.h5 --output-dir runs/figures_M2_keyframes
python -m experiments.formal.m3_cross_resolution.plot_results --result-dir runs/M3 --output-dir runs/figures_M3
python -m experiments.formal.s4_initial_distribution.plot_results --result-dir runs/S4 --output-dir runs/figures_S4
```

The S4 command writes `curves/` and `keyframes/` under its output directory and
requires the published M2 package as the unchanged left-hand reference; it fails
rather than drawing if an input is missing. Its output directory must be new.

The default requires both GIFT regimes. Add `--gift-regimes GIFT-Lite` only for
an explicitly limited Lite-only figure in a separate directory. An absent
selected method is an error, not a reason
to omit it silently. The M1 input must contain all 15 completed method/condition
jobs (45 parameter rows), not a partial table. These commands require the numerical
completion record and verify the input hashes. `figure_manifest.json` records the
exported files, renderer, source identities, statistics and SVG checks. Rendering
a completed run does not establish that another training run reproduced it.
