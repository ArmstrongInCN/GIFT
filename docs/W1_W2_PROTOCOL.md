# W1/W2：单初态、N300、76点、固定配方

本协议描述W1/W2的最终数据、训练方法、评价指标和图像规范。机器可读模型与数据绑定位于`prediction_extension/EXTENSION_LOCK.json`，GIFT、FNO-2D、FNO-3D的训练配置位于`prediction_native/PROTOCOL_LOCK.json`。ANIE的训练及坐标配置位于`prediction_aligned/PROTOCOL_LOCK.json`。真实目录名称用于定位来源；模型文件与SHA见[模型登记](CHECKPOINTS.md)，完整数值和图见[实验记录](../EXPERIMENTS.md)。

## 1. 比较方法

| 方法 | 训练实例与选择 | W1 | W2 |
| --- | --- | --- | --- |
| GIFT | 生成元seed2026072301、分支seed20260820；终态 | 是 | N64/N96/N128 |
| FNO-2D | seed0；终态 | 是 | N64/N96/N128 |
| FNO-3D | seed0；终态 | 是 | N64/N96/N128 |
| ANIE | seed7；validation-MSE-best | 是 | 不参加 |
| U-NO | seed0；终态 | 是 | N64/N96/N128 |
| U-Net | seed0；终态 | 是 | N64/N96/N128 |
| GIFT-Lite | 生成元seed2026072301、分支seed20260820；终态 | 单列减量对照 | 单列减量对照 |

超参数预先固定，每方法训练一次。GIFT的生成元和分支是同一方法的两个组件。ANIE按声明的验证MSE选择同次训练的检查点，并执行原早停规则；其余方法使用终态。

## 2. 共同观测与评分

- 主训练人口：`default_rng(2026100304).permutation(1000)[:300]`，锁中保存真实有序 ID。GIFT-Lite 使用这份有序名单的前 50 条，不是原编号 0–49，也不按误差选样本。
- 验证：原 IDs1000–1019，共 20 条；不能把其余验证轨迹或测试轨迹加入训练、统计或选模。
- 观测：N64、原涡量单位、t=5,5.04,…,8，共 76 个真实点，从原 dense dt0.02 帧250,252,…,400精确抽取；不插值时间，不使用遗漏的 dense 帧监督。
- 真输入：t=5 的一个场，所有方法没有 46 帧历史或未来真值输入。训练 shuffle seed2026100301；GIFT 观测中心/anchor seed2026100303，初始化 seed 独立。
- W1：未来 75 个点，主指标先在每轨迹逐帧 relative L2 上取平均，再跨完整180人口平均；另报 t8 relative L2、future75原单位 MSE、曲线与失败人数。
- W2：t=5–6 每0.04秒，共26点，主指标对应未来25点；另报t6、future25 MSE、频带和失败指标。每个网格从自己的一个t5场出发。
- 已知初态不计入主 future 指标；ANIE 原训练 MSE 仍包含模型生成的初态重建。

“可用76点观测相同”不表示全部训练损失使用相同未来长度。算法目标、损失、总更新数和成本不同；本实验不是只改变架构的单因素消融，也不是相同 GPU 预算比较。

## 3. 固定神经网络配方与实际监督覆盖

| 方法 | seed / epoch上限 | 物理 batch、累积 | optimizer / weight decay | lr / scheduler | 选择及训练覆盖 |
| --- | --- | --- | --- | --- | --- |
| FNO-2D | 0 / 500 | 20，无累积 | 原 Adam / 1e-4 | .001；每epoch StepLR100×.5 | 终态；t5单初態完整闭环，课程[1,2,5,10,25,50,75]，前六段各20轮，121轮起75未来步 |
| FNO-3D | 0 / 500 | 10，无累积 | 原 Adam / 1e-4 | .001；每epoch StepLR100×.5 | 终态；一次联合输出75未来点 |
| ANIE | 7 / 5000 | effective32、micro6；按真实样本数累积 | 原 torch Adam / 1e-4 | 1e-4；CosineT101、eta_min1e-9；完成第17轮起推进 | 同次优化validation全76点MSE-best；每轮验证，原patience1000，等值不计counter；全76点MSE含初態 |
| U-NO | 0 / 500 | 16，无累积；尾12保留 | 公开 `UNO/Adam.py` / 1e-5 | .001；每epoch StepLR100×.5 | 终态；从t5闭环20步，监督t5.04–5.80 |
| U-Net | 0 / 500 | 20，无累积 | torch Adam / 1e-4 | .001；每epoch StepLR100×.5 | 终态；从t5闭环4步，监督t5.04–5.16 |

FNO-2D、U-NO、U-Net 使用每场逐轨迹 relative L2 对 batch 求和、再对时间求和；FNO-3D 使用展平的整个未来时空场 relative L2 对 batch 求和，不把它改成逐帧时间和。均不额外增加 `/batch` 或时间平均后改变 Adam 输入梯度。FNO-3D 使用仅由这300条声明训练输入/输出拟合的 UnitGaussianNormalizer；其他方法保持所声明原单位路径。ANIE 用物理 MSE，不改为 relative L2。状态与时空坐标统一space-major/time-minor排列，同一token严格对应同一空间位置和查询时间，训练及评价一致。U-NO 的非重入 activation checkpoint 只重算原 forward，完整闭环梯度不截断；U-Net 不做 activation checkpoint。没有 teacher forcing 或未来真值重启。

N300的实际更新数为FNO-2D 7500、FNO-3D 15000、U-NO9500、U-Net7500。ANIE上限5000轮、每轮10次更新；实际在第1029轮早停，共10290次更新，使用第29轮验证MSE最佳权重。U-NO/U-Net每10轮及终态在全部20条验证轨迹上进行完整75步诊断，模型选择和学习率日程保持预先声明的规则。有限终态即使验证预测失败也保留，并在完整测试集上评价。

FNO与ANIE的优化器和学习率配方来自各自固定的官方NS实现。U-NO保留公开架构、Adam与LpLoss，使用本项目的20步监督、每轮StepLR和终态；作者示例采用40步监督、不同调度器推进频率和validation-best。U-Net使用公开架构及本项目的batch、4步闭环、学习率和选择规则。本实验比较固定配方，不作为各原论文完整实验的复现。

公开模型的固定提交和必要接口调整见[外部适配](EXTERNAL_ADAPTATIONS.md)。本协议记录最终训练配置；各模型文件与源码身份见[模型登记](CHECKPOINTS.md)。

## 4. GIFT 与 GIFT-Lite

两者共享架构 cutoff21/rank8 及同一阶段/epoch配方，使用终态，验证只作监测：

| 组件 / 阶段 | epoch | batch | lr / AdamW weight decay | 日程 |
| --- | --- | --- | --- | --- |
| generator 1 / 2 | 250 / 250 | 16 | .01 / .003；1e-8 | 各phase独立optimizer，按update cosine到初始lr的2% |
| branch derivative | 400 | 16 | .0015；1e-6 | 按epoch cosine到.0001 |
| branch short | 50 | 5 | .0002；1e-6 | 常数 |
| branch long 1 / 2 | 25 / 25 | 5 | .00008 / .00008；1e-6 | 常数，各phase重置optimizer |

generator和branch的gradient clip分别为5与1；generator在初始化及每轮结束时进行仿射拟合，共501次。N300 generator/branch分别为9500/13600次更新，N50为2000/2600次。因此GIFT-Lite同时改变数据量与总优化预算。此处GIFT-Lite指N300协议的50样本配置，来源键为`GIFT-N50`。辅助实验中的同名配置属于独立N1000协议，采用6000+6000次生成元更新、独立分支日程与验证选模；两套权重不能互换。

GIFT FD仅使用76个真实观测节点和dt0.04。内部RK4采用dt0.02；名义0.1/0.5秒监督滞后按最近合法观测节点选择、等距时取较早节点，实际为0.08/0.48秒。训练short/long分别为10/50个内部步；完整W1为150个内部步，每2步保存一个节点，共76点；W2为50个内部步，保存26点。数值安全阈值固定，积分器拒绝计为失败。计算使用eager执行。

## 5. 评价人口与跨网格能力

W1/W2分别报告确认集IDs100000–100179和探索性测试集IDs1040–1219，各180条，与训练和验证集互不重叠。确认集的四种方法结果此前已被查看，因此不是全部方法的盲测。两个集合均不用于选择训练参数；全部轨迹保留。

W2的三个网格逐条使用相同ID和float64初始参数。N96/N128真值在各自网格上由ETDRK4积分，内部dt0.005，padding分别147/195；t5–6按dt0.02保存51个真实节点，再精取dt0.04的26点，不插值时间。

GIFT以冻结权重在目标网格构造谱算子；FNO-2D/U-NO/U-Net直接接受目标场并递归25次。FNO-3D保持训练时的75个联合时间token，完整预测后取前25点；冻结N64 normalizer统计仅作确定性的空间周期Fourier插值，不拟合目标网格数据。U-Net的固定像素核对应的物理长度随网格变化。ANIE不参加W2。具体接口与Nyquist/DFT约定见[W2实现](W1_W2_PROTOCOL.md)。

失败保留全人口；训练失败保留唯一slot，测试非有限或solver拒绝保留实际reason/前缀。全人口指标不可定义时JSON为null，finite-only值仅诊断，不按成功子集排名。W2 t5–6前缀完整有限时，不把只在t6以后出现的W1失败硬套为一秒失败。没有跨训练seed SD。

## 6. 耗时、图像与数值来源

七种方法的训练进程墙钟时间、计时范围和来源见[耗时记录](PERFORMANCE.md)。训练时间包括数据准备、训练、验证、检查点I/O和导出；GIFT的两个组件合计为同一方法的成本，W2不重复计入训练时间。各方法监督覆盖和更新预算不同，耗时须结合其配方解读。

确认与探索集分别保存raw、metric、report和figures。W1固定ID100005/1045及t5/6/7/8；W2固定ID100110/1150、N128及t6。正文采用PNG预览，原SVG曲线用于编辑和导出；图的色阶、指标单位及失败表示见[图像说明](FIGURES.md)。

汇总、曲线数值与选定场数组分别位于[W1结果包](../results/formal/W1_flow_prediction_20261007/published.json)和[W2结果包](../results/formal/W2_cross_resolution_20261007/published.json)。清单记录源数值、模型/数据身份和SHA-256；完整权重与预测HDF位于声明的外部实验目录。独立评价命令见[运行说明](EXPERIMENT_EXECUTION.md)。W3和S1–S3各有独立协议，结果分别解释。
