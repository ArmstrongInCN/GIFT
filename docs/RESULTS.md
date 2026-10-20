# Reading results / 阅读实验结果

## W1/W2 方法、人口与指标

W1比较GIFT、GIFT-Lite、FNO-2D、FNO-3D、U-NO、U-Net和ANIE，W2比较其中除ANIE外的六种方法。主方法使用300条N64训练轨迹、20条验证轨迹、t5–8每0.04秒的76个真实节点，以及每条轨迹自己的t5真实初态；GIFT-Lite使用有序N300名单前50条。每个方法采用固定超参数和声明种子，训练一个模型实例。配方、选择规则与权重绑定见 [训练协议](W1_W2_PROTOCOL.md)，完整结果表与耗时见 [EXPERIMENTS.md](../EXPERIMENTS.md)。

W1主指标是未来75点的逐帧relative L2均值；W2在N64/N96/N128上评价t5–6的未来25点。先对每条轨迹的未来节点取均值，再对完整180条人口取均值；另报终点误差、原涡量单位MSE和失败计数。各方法只有一个训练实例，不报告跨训练种子的均值或标准差。

两评价人口分别报告，不合并平均：

- 确认集IDs100000–100179，180条，与训练及验证不重叠。两个评价人口的结果已被查看，因此不能视为全部方法的盲测。
- 探索集IDs1040–1219，180条，其结果仅作探索性评价。

W2使用相同ID和逐元素相同的float64初值参数；N96/N128真值在各自网格真实积分，随后精取0.04秒节点。模型只在N64训练，推理不拟合目标网格权重或读取测试归一化统计。

## 方法差异与结果

共同输入与评价人口不意味着损失和预算相同。U-NO只监督20步至t5.80，U-Net只监督4步至t5.16；FNO、ANIE和GIFT使用各自声明的全窗口、课程或局部目标。GIFT-Lite沿用GIFT阶段epoch，但生成元/分支更新为2000/2600，N300为9500/13600，因此它同时改变样本量与计算预算。W1/W2的GIFT-Lite属于N300协议，来源键为`GIFT-N50`；S1–S3中的GIFT-Lite属于独立N1000协议，使用单独日程与验证选模，权重及预算不能互换。

GIFT采用0.08/0.48秒监督时距、内部RK4 dt = 0.02；W1推进150个内部步并每0.04秒保存，共76点。ANIE使用第29轮验证MSE最优模型，训练在第1029轮依原早停规则终止；ANIE仅参加W1。

| 确认集指标 | GIFT | GIFT-Lite |
| --- | ---: | ---: |
| W1 future75平均relative L2 | 0.024687 | 0.036608 |
| W1 t8 relative L2 | 0.038113 | 0.065787 |
| W2 N64 future25平均relative L2 | 0.021098 | 0.026436 |
| W2 N96 future25平均relative L2 | 0.020633 | 0.024404 |
| W2 N128 future25平均relative L2 | 0.020610 | 0.024372 |

表中relative L2为比例，0.01表示1%。W1七方法、W2六方法在两评价人口相应任务中均为0/180失败。结论反映固定配方的单模型表现，不隔离架构因素，不代表充分选参后的最优能力，也不估计训练随机性。非有限预测或求解器拒绝保留对应轨迹和原因；无法定义全人口指标时记录null，finite-only结果仅作诊断。

## 数值与图像文件

W3当前结果覆盖50组无噪声参数、每组一条ID0轨迹及五种方法的250个槽位、750行参数读出。完整结果与锁定协议见[W3协议](W3_PROTOCOL.md)及[结果清单](../results/formal/W3_single_trajectory_20261010/published.json)。

| 实验 | 结果目录与范围 |
| --- | --- |
| W1 | [W1结果包](../results/formal/W1_flow_prediction_20261007/)：两人口各180条、future75、t8、MSE和失败；确认与探索分别保存。 |
| W2 | [W2结果包](../results/formal/W2_cross_resolution_20261007/)：两人口、三个网格、future25、t6、MSE、频带指标和失败；不包含ANIE。 |
| W3 | [W3结果包](../results/formal/W3_single_trajectory_20261010/)：50组无噪声参数、250个方法/case槽位、750行有限参数读出；逐例数值、汇总表及[图4a–c逐参数绝对百分比误差](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)。 |
| S1–S3 | 各自独立辅助协议的数值，分别存于 `S1_high_frequency_branch/`、`S2_recursive_local_correction/` 和 `S3_initial_distribution/`。 |

W1/W2科学结果包包含CSV汇总、可编辑SVG、曲线CSV、选定场数组`source_fields.npz`与来源清单。表格保留真实模型键，图中使用论文算法名称；任务编号统一为W1/W2。PNG场板位于 `assets/figures/`，具体呈现和色阶见 [FIGURES.md](FIGURES.md)。清单记录源数值报告、模型/数据身份和文件SHA-256；选定场数组是图像证据，不是完整训练或预测数据集。

W1场板左侧为涡量场、右侧为有符号误差prediction−truth，固定t5/6/7/8与ID100005/1045。W2场板在N128、t6按上场下误差两行排列，固定ID100110/1150。全部场图采用原生矢量网格；场与误差分别使用全方法共同的对称色阶，有限值不裁切，轨迹不按误差挑选。曲线保留W1七报告点PCHIP及W2三网格对数轴；主表指标使用全部future75/future25节点。

ANIE的训练与两个W1人口的数值审计绑定于 `prediction_aligned`；其余模型和W2的来源见各自结果清单。

W1/W2权重、完整raw HDF及优化器/随机状态保存在声明的外部实验目录，未随Git打包。结果包适合读表与查看图像，不能用于恢复训练。重新推理需要对应权重与数据；重新训练需要完整训练配置。环境和命令见 [SETUP.md](SETUP.md) 与 [EXPERIMENT_EXECUTION.md](EXPERIMENT_EXECUTION.md)。

```shell
python -m scripts.verify_published_results --experiment W1 --result-dir results/formal/W1_flow_prediction_20261007
python -m scripts.verify_published_results --experiment W2 --result-dir results/formal/W2_cross_resolution_20261007
```

完整性命令核对文件字节和来源绑定，不运行训练或重算预测；不同硬件、数学库和上游随机算子可能改变浮点结果。

## 独立参数识别与辅助协议

W3使用独立参数case和局部ID0单轨迹，无观测噪声；50组真值各对应一条轨迹，五种方法各完成50次，共750行参数读出。均值/中位APE、完整系数与耗时见[实验记录](../EXPERIMENTS.md#43-w3-50组无噪声方程参数识别)和[结果包](../results/formal/W3_single_trajectory_20261010/README.md)。开放项库的完整支持单独报告；100次PINN L-BFGS均达到函数/梯度评估上限。GIFT的ν、β平均APE最低，γ最低为PDE-FIND；有限读出不等于参数均准确，GIFT的ν最大APE为59.32%。图4a–c分别展示ν、β、γ的绝对百分比误差分布，APE=100×|估计值−真值|/|真值|；每方法/参数保留全部50个真实读出。Tukey箱线图的实心箱体为逐项APE的Q1–Q3，贯穿箱体的加长线及每行Median数值表示50次APE的中位数。IQR=Q3−Q1，细须线及短端帽延伸至[Q1−1.5×IQR, Q3+1.5×IQR]内最远的实际观测（包含边界）；全部真实点包括须外点完整保留，不重复添加离群点符号，也不单独标记均值。这些是参数case的经验分布，不是置信区间。圆点在七条等间距轨道上确定性排布，不改变APE横坐标。圆点标称直径为2.1 pt，七轨道相邻间距为2.7 pt，使点层更紧凑；每个真实点用12层同中心、无描边的同色透明圆片叠加，按径向Gaussian透明度渐变形成柔边（相对半径sigma为0.42、标称中心alpha为0.95）。该渐变仅柔化单个真实点，不作KDE或数据平滑，也不添加虚构观测。算法名称为黑色粗体，方法圆点、实心箱体与须线使用鲜艳深色，中位线与箱体颜色完全一致，线宽1.6 pt、长度约为箱体高度的2.1倍，坐标轴为黑色实线；横轴刻度朝上指向图内，刻度数值仍在轴线下方，轴线在最右刻度后延伸当前横轴跨度的3%，不改变数据横坐标或刻度位置；图内仅保留算法名称、坐标刻度、带参数名称的横轴标题及每方法的Median数值。每幅图只有一个完整范围面板；五方法共用从零开始的线性轴，ν、β为[0,300]%，γ为[0,110]%，不截断有限值。[图4a：ν SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)（[PNG](../assets/figures/w3_parameter_nu_errors.png)） · [图4b：β SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_beta_errors.svg)（[PNG](../assets/figures/w3_parameter_beta_errors.png)） · [图4c：γ SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_gamma_errors.svg)（[PNG](../assets/figures/w3_parameter_gamma_errors.png)）；来源和绘图规则见[图像清单](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json)。

S1–S2使用N1000 GIFT与GIFT-Lite模型：GIFT生成元和分支各训练500个epoch并使用终态；GIFT-Lite使用50条训练轨迹、单独日程与验证选模。训练IDs0–999，测试IDs1040–1219；GIFT-Lite训练IDs0–49。这些模型定义和统计范围与W1/W2单模型协议分别陈述。

S1的分支开启配置降低全场与Q21状态预测误差，但瞬时Q21导数逐状态均值较差；小分母和长尾分布须与分位数共同解读。S2中N1000 GIFT未触发局部修正，开启/关闭均保持有限；GIFT-Lite关闭修正时轨迹1062在t6.74失稳，后续finite-only均值使用179条，不能据此获得总体精度优势。

S3采用独立N1000训练协议，比较四涡旋与平滑高斯随机场两种分布。基线输入t4.1–5.0的46帧历史，GIFT使用t5单状态，预测150步至t8；每种分布各自训练和测试，属于分布内比较。Gaussian训练IDs1220–2219、测试IDs2260–2439；四涡旋对照使用其对应N1000模型、IDs1040–1219和固定示例1045。该对照不是N300 W1测量。

S3中GIFT的t8均值为0.021035，最接近的U-NO为0.173805；GIFT、U-NO、FNO-2D、FNO-3D和U-Net均180/180条有限。GIFT-Lite有17/180条在t约6.9之后失稳，t7及以后仅为有限子集统计，曲线到t6.5为止。完整失败人数随结果报告，不能与其他方法的全人口值直接排名；该实验不支持任意初值分布的结论。
