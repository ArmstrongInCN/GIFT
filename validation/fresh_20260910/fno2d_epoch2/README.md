# FNO-2D epoch 2 evidence / 第 2 轮实测证据

The same fresh run resumed its own epoch-1 checkpoint and completed epoch 2; the original epoch-1 checkpoint and full history row were preserved.

同一本次从零训练 run 从自身第 1 轮完整状态续算到第 2 轮；旧第 1 轮检查点与完整历史行均保留。

Formal configuration remains 500 epochs, 1,000 trajectories and 150-step training targets. This audited boundary had status `running`, not a full terminal model. 此审计边界是正式 500 轮预算内部暂停，不是完成全预算。

| Epoch | Backward loss / trajectory | Full spacetime loss / trajectory |
|---|---|---|
| 1 | 134.83111840820314 | 0.9056980443000794 |
| 2 | 118.62494018554688 | 0.8063657546043396 |

Every displayed aggregate equals its archived CSV value exactly; no tolerance is applied. 没有对应轮次的原参考权重，因此这些标量精确相同不能证明梯度、权重或整个训练轨迹相同，不能声称 500 轮或正式实验通过。

The actual CPU audit checked source/input/runtime binding, finite model/optimizer state, saved RNG, history and the paused boundary. The full released input SHA was checked by that audit. This publication step only rechecked JSON/CSV/file hashes; it did not load checkpoints or run forward/training.

此目录仅含可移植 JSON/Markdown 证据：无机器私有路径、运行 UUID、权重、journal 或科学输入。原始审计、源码、输入及自身检查点 SHA 保留以支持追溯。Own checkpoints were used only for same-run continuation, never as pretrained initialization. OMP/MKL 16 is established by the harness receipt, not independently bound in the old journal.

The immutable checkpoint and completed-audit hashes were rechecked. `historical_latest_pointer_at_audit` preserves the pointer SHA observed by that audit; it is not asserted to be the current pointer after subsequent legitimate continuation. 原 LATEST 历史 SHA 不回填、不要求指针倒退，源码和不可变检查点的严格核验不变。
