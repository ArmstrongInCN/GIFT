# W3：50组无噪声方程参数识别

## 研究问题与数据

W3检验模型在真实物理参数变化下能否给出稳定、准确的方程参数读出。它不把GIFT定位为专用参数识别算法，也不设置噪声条件。固定50组参数由LHS seed 2026100901生成：ν对数均匀分布于[0.0075, 0.015]，β、γ分别线性分布于[0.75, 1.25]。真方程为ω_t=−β(uω_x+vω_y)+ν∇²ω+γq，q=−4cos(4y)。

每组参数对应一条独立的N64涡量轨迹，局部数据ID为0；t=0–10，共501帧，保存间隔0.02，内部积分步长0.005。每组仅使用对应的ID0轨迹，额外validation轨迹数为0；所有参数组使用同一初态参数，各组轨迹按其自身方程参数生成。设计CSV SHA-256为`43b8b7d0edb5d70f4b7c7519946c7e358e88011db582f09c5d7e6940f1893a14`。

## 方法和固定预算

五种方法为GIFT、PDE-FIND、PDE-FIND-KC、PINN-SR和PINN-SR-KC；每种方法对50个case各运行一次，共250个唯一方法/case槽位和750个参数读出。无外层超参数搜索、无额外训练seed、无test调参或失败槽位替换。

| 方法 | 固定的主要设置 |
|---|---|
| GIFT | 每case独立训练；seed2026072301；单训练轨迹、零validation轨迹；batch16、6000+6000更新、lr0.01/0.003、P21/rank8；阶段终态checkpoint；最终affine fit及P21参数读出。 |
| PDE-FIND | 每case独立导数与回归；open90项库；seed0；局部多项式和STRidge流程，无SVD去噪。 |
| PDE-FIND-KC | 每case独立导数与回归；known4项库；seed0；局部多项式和STRidge流程，无SVD去噪。 |
| PINN-SR | 每case fresh seed1234、open90项库、paper_ado_v1/native_fp32、TF32关闭；24,000训练观测、6,000同轨迹留出、84,000物理点；5,000联合NAdam、一次实际SciPy L-BFGS、6×1,000网络NAdam、20,000末段联合NAdam；终态checkpoint。 |
| PINN-SR-KC | 同PINN-SR完整预算，known4项库。 |

PINN L-BFGS `maxiter`和`maxfun`是上限，报告逐例真实终止原因；达到上限不表示收敛。PINN保留全部候选系数、末轮STRidge精确非零支持及各真项误差。GIFT训练内监控只作诊断，固定使用终态模型；不以验证集选择checkpoint。

各方法采用各自声明的观测处理、识别目标与预算，不能解释为等信息/等预算的架构对照。PDE-FIND系方法没有神经网络训练阶段；不同方法的wall-clock成本也不是统一硬件预算排名。

## 指标与结果

对真值θ与读出θ̂报告APE=100×|θ̂−θ|/|θ|。每个算法/参数各有50个case级读出；下表汇总50次APE的均值与中位数，不计算跨训练seed标准差。完整参数设计、750行读出、250个槽位结果、系数支持、输入/源码哈希和真实成本见[W3结果包](../results/formal/W3_single_trajectory_20261010/README.md)及其`published.json`清单。

| 方法 | ν APE均值（中位数） | β APE均值（中位数） | γ APE均值（中位数） |
|---|---:|---:|---:|
| GIFT | 12.86%（8.74%） | 2.57%（1.68%） | 9.70%（7.26%） |
| PDE-FIND | 97.41%（102.07%） | 10.94%（11.15%） | 7.57%（7.70%） |
| PDE-FIND-KC | 32.79%（33.30%） | 13.90%（13.99%） | 9.42%（9.26%） |
| PINN-SR | 153.57%（142.54%） | 42.63%（34.33%） | 39.18%（32.14%） |
| PINN-SR-KC | 89.71%（93.96%） | 40.76%（17.66%） | 15.58%（9.79%） |

250个实例全部完成，750个参数读出有限；执行失败数0不表示全部识别准确。GIFT的ν和β平均APE最低，γ平均APE最低的方法为PDE-FIND。GIFT的ν APE最大为59.32%，说明部分参数组合仍有明显偏差。100次PINN L-BFGS均达到函数/梯度评估上限，随后完成规定的后续预算；这些状态不能称为优化收敛。结果以逐例数值、汇总表和[图4](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)共同呈现。

## 参数误差的图示

[图4a：ν SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)（[PNG](../assets/figures/w3_parameter_nu_errors.png)） · [图4b：β SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_beta_errors.svg)（[PNG](../assets/figures/w3_parameter_beta_errors.png)） · [图4c：γ SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_gamma_errors.svg)（[PNG](../assets/figures/w3_parameter_gamma_errors.png)）

图4a、4b、4c分别展示ν、β、γ的绝对百分比误差，APE_p=100×|p̂−p|/|p|，由同一750行真值与估计值逐项计算。每幅图按固定顺序排列GIFT、PDE-FIND、PDE-FIND-KC、PINN-SR和PINN-SR-KC，每行保留对应方法全部50个真实读出，共250点。圆点的横坐标是该次读出的APE；圆点按APE排序，在七条等间距水平轨道上确定性排布，优先选择不与既有圆点相交的中央轨道，密集值仍可重叠。圆点标称直径为2.1 pt，七轨道相邻间距为2.7 pt，使点层更紧凑；每个真实点用12层同中心、无描边的同色透明圆片叠加，按径向Gaussian透明度渐变形成柔边（相对半径sigma为0.42、标称中心alpha为0.95）。该渐变仅柔化单个真实点，不作KDE或数据平滑，也不添加虚构观测。轨道仅用于整理圆点，不编码第二个变量；横坐标不作抖动、分箱或舍入。采用Tukey箱线图：实心箱体表示逐项APE的第25至75百分位区间（Q1–Q3），贯穿箱体的加长竖线表示50次APE的中位数，每行右侧的Median数值给出同一中位数。IQR=Q3−Q1，细须线及短端帽延伸至[Q1−1.5×IQR, Q3+1.5×IQR]内实际观测的最小值与最大值（包含边界），不将围栏值直接作为端点。须外真实观测仍按原坐标完整绘出，不额外重复绘制离群点符号；图中不单独标记均值，也不表示置信区间。全部统计量均在逐项取绝对值后计算，描述不同参数组合的经验分布，不是跨训练seed误差条。五种方法共用各图内从零开始的线性横轴，ν与β为[0,300]%、γ为[0,110]%，完整显示全部有限值。每幅图仅含一个完整范围面板。GIFT为鲜深蓝、PDE-FIND为深橙、PDE-FIND-KC为深绿、PINN-SR为深紫、PINN-SR-KC为深青，圆点和对应须线共用方法配色。算法名称使用黑色粗体，坐标轴使用黑色实线；圆点、实心箱体和须线使用对应方法的鲜艳深色，中位线使用与箱体完全相同的颜色，线宽为1.6 pt，贯穿并伸出箱体，线长约为箱体高度的2.1倍；横轴刻度朝上指向图内，刻度数值仍位于轴线下方，轴线在最右刻度后继续延伸当前横轴跨度的3%，数据横坐标与刻度位置不变；须线细、端帽短；图内仅保留算法名称、坐标刻度、带参数名称的横轴标题及每方法的Median数值，统计定义由正文说明。三幅图与正文主表使用同一APE定义，便于逐参数比较识别误差大小和离散范围，不表达参数间的相关关系。逐点误差与经验统计见[坐标CSV](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/source_points.csv)和[统计CSV](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/source_statistics.csv)，完整绘图规则与文件身份见[绘图元数据](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_metadata.json)及[图像清单](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json)。

## 来源与复现边界

50组参数CSV、协议锁、ID0数组等输入的字节身份与250槽位的source/input/claim绑定见W3结果包、run审计材料和`FINAL_SOURCE_AND_SCOPE_AUDIT.json`。完整双机执行控制、训练状态和外部实现代码均未纳入GIFT项目包；使用者须从[外部算法仓库记录](EXTERNAL_REPOSITORY_POLICY.md)自行下载对应上游代码，在GIFT仓库外实施本文与[手工适配说明](EXTERNAL_MANUAL_ADAPTATION.md)所述的适配。
