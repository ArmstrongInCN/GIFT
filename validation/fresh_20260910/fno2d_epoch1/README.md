# FNO-2D first full epoch / 首个完整训练轮次

The actual fresh run retained the formal **500-epoch** configuration and paused
after epoch 1: 1,000 trajectories, 150-step closed-loop objective, physical batch
10 with accumulation 2, and 50 optimizer updates. Epoch computation took
277.491 s; its successful training harness took 289.218 s (exit 0).

| Aggregate loss | Actual = archived epoch-1 CSV |
| --- | --- |
| Backward loss per trajectory | 134.83111840820314 |
| Full spacetime loss per trajectory | 0.9056980443000794 |

These **two scalar values are exactly equal**, without a tolerance. This does
not establish identical gradients, weight tensors or the entire trajectory:
reference epoch-1 weights are unavailable. The 500-epoch budget is not complete;
continuation is supported by the saved-state contract but has not been executed.
This is not full training or experimental acceptance.

此次从零训练只完成正式配置的第 1 轮，并按本次完整状态暂停；两个整轮聚合损失
与原 CSV 第 2 行精确相同。没有第 1 轮参照权重，不能推出梯度、权重或全轨迹相同，
更不能宣称 500 轮或正式实验已通过。这里仅发布校验信息，不附权重、journal 或数据。

The first read-only checker attempt (`audit_01`) failed on Windows source-path
separators. It remains recorded by its original SHA and failure in
[verification.json](verification.json). `audit_02` corrected only that checker:
canonical safe paths must still match the exact five frozen sources and hashes.
No training code or scientific result was changed, and no retraining occurred
for that fix. Absolute local prefixes are replaced by symbolic roots; original
audit, source, execution-receipt and artifact hashes are retained.

`runtime` preserves the original numerical flags. The 16-thread setting comes
from the harness receipt, not an independently sampled thread count in the
baseline journal. Publication performed only JSON/CSV and byte-hash checks;
it did not load tensors, rehash the full dataset, run forward or train.
