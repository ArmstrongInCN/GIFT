# Full PINN-SR-KC noise_000: bounded numerical PASS

The independent clean/KC GPU run completed from canonical **untrained** initialization, without reusing a trained checkpoint: exit 0, 3,388.354533 seconds, 24,000 training / 84,000 physics rows, chunk 32,768. It completed 5,000 pre-NAdam updates, the original L-BFGS-B limits, six STRidge + 1,000 joint-NAdam rounds, final mask, and 20,000 post-NAdam updates: **31,000 NAdam updates and six STRidge rounds**, with no added post-L-BFGS.

The terminal-only audit passed all **12 scalar comparisons in the original three noise_000/KC rows**, using the unchanged rule `0.0005 + 0.05 * abs(original reference scalar)`. Estimates are nu=0.0301227830350399, beta=0.6776743531227112 and gamma=0.7861946225166321; they are not claimed bitwise equal to the original estimates. The audit also verified all **81 model/optimizer tensors plus mask** byte-for-byte against this run's own committed final journal—not against historical weights. L-BFGS reported relative-reduction convergence after 1,117 iterations / 1,140 evaluations; audit time was 1.3609105 seconds.

See [actual metrics](metrics.csv), [unchanged reference subset](reference_subset.csv), [12-value comparison](comparison.json), and [source/input/runtime/execution provenance](provenance.json). CSV bytes are preserved; comparison paths and provenance paths are portable, with raw receipt SHA-256 values retained. No weights, NPZ arrays, training logs or external algorithm source are bundled here.

**Scope:** this is one full-budget known-library condition, not complete M1, open-mode or project acceptance. [noise_001/KC still has two failed scalar comparisons](../pinn_known_noise_001/README.md); those failures and all original tolerances remain unchanged. Packaging performed no training, forward pass, checkpoint restore or scientific recomputation.

中文：无噪声 known/KC 独立完整训练已完成；原三行共12项标量均在既定容差内通过。本次终态与其自身最终训练日志中的81个状态张量及mask逐字节一致，不代表与历史权重逐字节一致。此结果仅覆盖 noise_000/KC，不代表完整M1、open模式或全项目通过；noise_001/KC原有2项失败继续保留。
