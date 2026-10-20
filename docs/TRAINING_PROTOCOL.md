# Training protocol / 训练协议

## W1/W2 固定训练协议

[N300/76点协议](W1_W2_PROTOCOL.md) 使用有序固定N300训练名单、20条验证轨迹、t5–8每0.04秒76个真实节点和单一真实t5初态。GIFT-Lite取同一名单前50条。各方法采用固定超参数与声明种子，每方法一个训练实例。W1包括GIFT、GIFT-Lite、FNO-2D、FNO-3D、U-NO、U-Net、ANIE；W2除ANIE外包括六方法。

| 方法 | 配方与模型选择 | 实际监督范围 |
| --- | --- | --- |
| FNO-2D | seed0，batch20，500epoch，Adam lr0.001/wd1e-4，StepLR100×0.5；epoch500终态 | 闭环课程[1,2,5,10,25,50,75]，前六阶段各20轮，从第121轮起完整75步。 |
| FNO-3D | seed0，batch10，500epoch，同FNO-2D optimizer/scheduler；epoch500终态 | 一次联合输出未来75点，时空relative L2 batch-sum。 |
| U-NO | seed0，batch16，保留尾12条，500epoch，Adam lr0.001/wd1e-5，StepLR100×0.5；epoch500终态 | t5闭环20步，训练目标至t5.80，batch/time-sum relative L2。 |
| U-Net | seed0，batch20，500epoch，Adam lr0.001/wd1e-4，StepLR100×0.5；epoch500终态 | t5闭环4步，训练目标至t5.16，batch/time-sum relative L2。 |
| ANIE | seed7，effective batch32/micro6，Adam lr1e-4/wd1e-4，延迟CosineT101，上限5000轮；第29轮validation-MSE-best，依patience1000规则在1029轮终止 | 官方NS训练c=0/y_init路径，原单位全76点MSE，含初态重建；仅W1。 |
| GIFT / GIFT-Lite | generator seed2026072301、branch seed20260820；generator250+250，branch400/50/25/25；原阶段AdamW和日程，500轮终态 | coarse FD及局部RK4；监督时距0.08/0.48秒。 |

GIFT内部RK4 dt = 0.02，每0.04秒保存；W1推进150个内部步。N300生成元/分支更新9500/13600，GIFT-Lite为2000/2600；相同阶段epoch不代表相同计算预算。W1/W2的GIFT-Lite来源键为`GIFT-N50`；辅助实验中的同名50样本模型属于独立N1000协议，使用单独日程和validation-best，不能互换权重。监督范围、loss及预算差异是实验条件，不能把表现差异单独归因于架构。网络与控制器来源见 [原生实现说明](W1_W2_PROTOCOL.md)。

确认IDs100000–100179与训练、验证独立，但两个评价人口的成绩已被查看，因此不是全部方法的盲测；探索IDs1040–1219另表报告。模型路径和SHA见 [CHECKPOINTS.md](CHECKPOINTS.md)，进程计时见 [PERFORMANCE.md](PERFORMANCE.md)。其余方法配置来源为 `prediction_extension/EXTENSION_LOCK.json`。

W3采用独立单轨迹参数识别协议；S1–S3采用下列N1000及GIFT-Lite辅助协议，其模型和统计范围分别声明。

ANIE坐标和训练配置来源为 `prediction_aligned/PROTOCOL_LOCK.json`，具体源码适配见 [ANIE实现说明](ANIE_ADAPTATION.md)。


## 独立辅助训练协议（N1000/46帧/0.02秒）

The four supervised prediction baselines use the same 1,000 training trajectories,
46-frame input history and **500 epochs**. Each epoch visits each trajectory once
and selects one reproducibly seeded training window per trajectory. Use the final
epoch, not test performance, to select the released baseline checkpoint.

四个同类预测网络使用相同的 1,000 条训练轨迹、46 帧输入历史和 500 epoch。
每个 epoch 将全部训练轨迹各访问一次，并以固定种子选择时间窗口。
各模型保持自己的预测方式和损失计算，不将相同 epoch 宣称为相同参数更新次数或计算量。

| Model | Epochs | Physical batch | Gradient accumulation | Expected parameter updates | Training prediction steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| FNO-2D | 500 | 10 | 2 | 25,000 | 150 recursive |
| FNO-3D | 500 | 5 | 2 | 50,000 | 150 jointly predicted |
| U-NO | 500 | 16 | 1 | 31,500 | 20 recursive |
| U-Net | 500 | 20 | 1 | 25,000 | 4 recursive |

U-NO retains its final eight-trajectory batch in every epoch; no trajectory is
dropped. All four schedulers advance once per epoch (StepLR: 100 epochs, factor
0.5). The initial learning rate is 0.001. Weight decay is 0.00001 for U-NO and
0.0001 for the other three. Test trajectories are never used for updates,
normalization, scheduling or checkpoint selection.

## N1000 GIFT and GIFT-Lite training / 辅助模型训练

Two training regimes share the same GIFT architecture. **GIFT** uses all 1,000
prediction-training trajectories; its generator and high-frequency branch each
receive 500 trajectory epochs. This is **500 + 500**, not a 500-epoch total and
not a compute-matched claim. **GIFT-Lite** uses independently trained
50-trajectory models and its own multi-stage schedule; Lite denotes reduced
training data, not a smaller network. W3 identification keeps its separate
protocol below: its generator is trained on, and its coefficients are read from,
the single local training trajectory 0 that the four baselines also use.

GIFT 的生成元与高频支路各训练 500 epoch，分项报告。每个 epoch 对
全部 1,000 条训练轨迹各访问一次，从每条轨迹选择一个固定种子控制的时间窗口。
GIFT-Lite 使用 50 条训练轨迹、20 条验证轨迹及验证选模方式；网络结构不缩小。
两者的训练预算也不同，因此不能把性能差异仅归因于训练数据数量。

### Data and model selection

Prediction training IDs are 0–999; GIFT-Lite uses 0–49. Validation IDs are
1000–1039; IDs 1000–1019 contain the complete 0–10 trajectory interval and are
used for checkpoint-validation measurements. Independent test IDs are
1040–1219 (180 trajectories), paired across N64/N96/N128. Identity mapping is
fixed before model evaluation and does not depend on errors or failures.
The identification dataset used by W3 retains its separate local identifiers; all
five W3 configurations identify their parameters from local training trajectory 0.

Full-data GIFT uses **terminal stage states and terminal epoch weights**.
Validation monitors the declared procedure and does not select a model or
alter learning rates. Test observations are not used for parameter updates,
normalization, stage decisions or checkpoint selection.

### Full-data stages and budget

| Component / stage | Trajectory epochs | Physical batch | Parameter updates |
| --- | ---: | ---: | ---: |
| Generator, phase 1 | 250 | 16 (final batch 8) | 15,750 |
| Generator, phase 2 | 250 | 16 (final batch 8) | 15,750 |
| Branch, derivative supervision | 400 | 16 (final batch 8) | 25,200 |
| Branch, short rollout | 50 | 5 | 10,000 |
| Branch, long rollout stage 1 | 25 | 5 | 5,000 |
| Branch, long rollout stage 2 | 25 | 5 | 5,000 |

The generator has 31,500 gradient updates; each branch has 45,200. Its affine
constant/linear fit is a separate analytic operation: one initial fit and one
fit on the same sampled observations at the end of each generator epoch,
501 fits total. These fits are disclosed, not relabelled as gradient updates.
The generator is trained once and reused by the three independently initialized
branches with seeds 20260820, 20260821 and 20260822.

Generator architecture, rank 8, P21 support, four-point centred derivative
numerator on five observed frames, normalized MSE, gradient clipping at 5,
rank-aware affine solver and AdamW are unchanged mathematical components.
The two phase learning rates are 0.01 and 0.003, with within-phase cosine decay
to 2% of the initial value and weight decay 1e-8. The affine fit and the
gradient updates use training observations only.

Branch architecture, Q21 derivative loss, dual-track RK4, local correction,
10-step short rollout, 50-step long rollout with 10-step truncated gradient
segments, and gradient clipping at 1 remain the GIFT mathematical components.
Stage learning rates are 0.0015, 0.0002, 0.00008 and 0.00008; AdamW weight decay
is 1e-6. The derivative stage uses cosine decay to 0.0001. Each new stage
initializes its own optimizer; continuation within a stage restores that
optimizer's complete state. No terminal state is replaced by a better test or
validation checkpoint.

Generator derivative centres are sampled from 2–498. Branch derivative
centres use the same available observation stencil. Rollout anchors are
sampled from 0–450 so every 50-step target is a saved observation within 0–10.
The full-data sampling unit is a trajectory epoch. The reduced-data derivative
stage instead passes over its fixed set of derivative examples.

### Reduced-data GIFT-Lite

GIFT-Lite uses a generator trained for 6,000 + 6,000 random-minibatch
updates on 50 trajectories, with 20 validation trajectories. These updates
are steps, not trajectory epochs. Its three branch runs use 18 derivative
passes, two short-rollout passes and two single-pass long-rollout stages,
5,834 updates per seed. The packaged branches are the validation-selected
weights from their respective fresh-initialization runs.
The shared generator cost is reported once, separately from the three branches.

### Independent commands and continuation

```shell
python -m training.train_gift_generator --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --output ../runs/gift_generator
python -m training.train_gift_predictor --run-training --dataset ../gift-data/fno/fno1000_n64_t0_t10_dt0p02.h5 --low-model ../runs/gift_generator/model.pt --seed 20260820 --output ../runs/gift_branch_20260820
```

Run the branch command separately for each of the three declared seeds. Replace
`--run-training` with `--resume` to restore that run's committed model, optimizer,
scheduler, epoch/stage counters and random state. The default boundary is every
10 epochs and each stage end; `--stop-after-epoch` commits an explicit boundary.
Uncommitted epochs may be repeated after interruption. A published unrelated
generator or a reduced-data checkpoint is not accepted as the full-data training
prerequisite.

All four GIFT training entry points share `--execution auto`: fresh GPU training
uses CUDA graphs, CPU uses eager, and resume retains the journal's backend.
Use `--execution eager` for the explicit reference implementation. This does not
change the full-data 500 + 500 epoch budgets, the reduced-data selection protocol,
or any batch/update counts. Resume checks the model, data, schedule and saved state;
source and environment versions remain descriptive metadata. See
[execution efficiency and measured scope](PERFORMANCE.md).

Full-data timers sum committed epoch time: sampling, forward/backward,
parameter updates, generator affine refits / branch frozen-RHS preparation,
and scheduled validation. They exclude initial input loading/calibration,
checkpoint writing, paused time and discarded uncommitted work. Report the
measured times from each completed run; do not substitute estimated timings.

## Independent runs, timing and continuation

### W3 无噪声50组参数训练协议

W3在固定的50组参数组合上逐组运行。每个case使用与其真实参数相对应的一条局部ID0轨迹、N64、t=0–10、501帧、保存间隔0.02、内部步长0.005；无测量噪声、无验证轨迹、不同case不复用轨迹。参数由锁定LHS生成，ν∈[0.0075,0.015]对数均匀，β和γ∈[0.75,1.25]线性，seed2026100901。完整CSV SHA及case逐项真值见[结果包](../results/formal/W3_single_trajectory_20261010/)。

GIFT每case独立训练，seed2026072301、batch16、6000+6000随机小批更新、lr0.01/0.003、P21/rank8。每阶段固定采用第6000步终态checkpoint，训练内监控不选模型。每case独立执行最终affine fit和参数读出；预算为每case12000次更新。

PDE-FIND和PDE-FIND-KC使用各case本身的ID0轨迹、原局部多项式导数、open90/known4项库、STRidge与seed0；无SVD去噪。PINN-SR和PINN-SR-KC分别使用open90/known4库、seed1234、`paper_ado_v1`、native FP32且TF32关闭。每case固定24,000训练标签、6,000同轨迹留出标签和84,000物理点；预算为5,000次联合NAdam、一次原生SciPy L-BFGS、六轮每轮1,000次网络NAdam及最后20,000次联合NAdam，固定终态。SciPy参数上限与逐例实测停止状态见结果包，达到上限不代表收敛。

五种方法合计250个唯一方法/case槽位；不增加种子、不选参、不用测试误差选权重，每组独立初始化，科学失败按原实例保留。完整的槽位成本、并行主机尝试和中断界限见run的FINAL_COST_CLOSURE。

### N1000 prediction baselines

Train each model with its own command, for example:

```shell
python -m scripts.run_training fno2d --run-training --output ../runs/fno2d
python -m scripts.run_training fno3d --run-training --output ../runs/fno3d
python -m scripts.run_training uno --run-training --output ../runs/uno
python -m scripts.run_training unet --run-training --output ../runs/unet
```

The environment, dataset, official external-source paths and user-prepared
`GIFT_EXTERNAL_ADAPTER_ROOT` outside GIFT must be configured first; follow
[EXTERNAL_MANUAL_ADAPTATION.md](EXTERNAL_MANUAL_ADAPTATION.md).
To continue an interrupted run, add `--resume` and keep the same output directory.
The default committed boundary is every 10 completed epochs; an explicit
`--stop-after-epoch` also commits that boundary. Uncommitted epochs may need to
be repeated. Model, optimizer, scheduler and random state are restored together;
changing source code, data or training settings does not qualify as same-run continuation.

Every committed checkpoint records actual update counts and per-epoch seconds.
The final weights also contain `training_cost`: cumulative committed-epoch wall
time, including batch loading, forward/backward, updates and GPU synchronization.
It excludes setup, checkpoint writing, paused time and discarded unfinished work;
this is not an end-to-end installation or recovery-time measurement. Include this
definition and hardware/runtime details with the experimental cost table. Do not
invent a measured runtime from the expected update counts above.

U-NO retains the upstream bicubic-interpolation backward operation. PyTorch can
report that this CUDA operation is nondeterministic; its warning is not suppressed
and `warn_only=True` does not turn it into a deterministic implementation. Fixed
seeds and complete continuation state preserve the declared procedure, not a
guarantee of bitwise-identical weights across devices or independent CUDA runs.
