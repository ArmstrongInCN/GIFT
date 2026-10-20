# W3 单轨迹、50组方程参数实验

每个参数组合对应原始 ID 0 的无噪声涡量轨迹；共 50 条轨迹、5 种算法各 50 次独立拟合。网格 N64，t=0–10、501帧、保存间隔0.02、积分内部步长0.005。主结果是每个参数组合的参数相对绝对百分比误差（APE）；无跨种子误差条。

`summary/parameter_results.csv` 保存全部 750 个参数读出，`summary/parameter_cases.csv` 保存50组真值，`summary/method_metrics.csv` 是按参数及算法汇总的50次结果。`slots/` 保留250个槽位的原始数值、完整系数表和输入绑定元数据。

所有训练/回归槽位均完成，未报告失败槽位。每种算法原有的数据使用方式和训练预算保持锁定；不同算法的观测目标和预算并非完全等信息/等计算量，因此不作纯架构或效率排名。PINN的真实 L-BFGS 终止状态及完整稀疏支持见训练审计与各槽位记录。

GIFT 仅使用每组对应的 ID 0 轨迹，训练轨迹数1、外部验证轨迹数0，固定终态 checkpoint；GIFT 的其余模型、优化器、6000+6000 更新预算与读出步骤按锁定协议执行。每组轨迹按其自身方程参数生成，50组使用相同的初态参数。

科学源代码与数据按清单中的 SHA256 绑定；外部算法实现与权重未打包。结果以逐例数值、汇总表和图4呈现。

## 三参数误差图

[图4a：ν SVG](figures/parameter_distributions/w3_parameter_nu_errors.svg)（[PNG](../../../assets/figures/w3_parameter_nu_errors.png)） · [图4b：β SVG](figures/parameter_distributions/w3_parameter_beta_errors.svg)（[PNG](../../../assets/figures/w3_parameter_beta_errors.png)） · [图4c：γ SVG](figures/parameter_distributions/w3_parameter_gamma_errors.svg)（[PNG](../../../assets/figures/w3_parameter_gamma_errors.png)）

图4a、4b、4c分别展示ν、β、γ的绝对百分比误差，APE_p=100×|p̂−p|/|p|，由同一750行真值与估计值逐项计算。每幅图按固定顺序排列GIFT、PDE-FIND、PDE-FIND-KC、PINN-SR和PINN-SR-KC，每行保留对应方法全部50个真实读出，共250点。圆点的横坐标是该次读出的APE；圆点按APE排序，在七条等间距水平轨道上确定性排布，优先选择不与既有圆点相交的中央轨道，密集值仍可重叠。圆点标称直径为2.1 pt，七轨道相邻间距为2.7 pt，使点层更紧凑；每个真实点用12层同中心、无描边的同色透明圆片叠加，按径向Gaussian透明度渐变形成柔边（相对半径sigma为0.42、标称中心alpha为0.95）。该渐变仅柔化单个真实点，不作KDE或数据平滑，也不添加虚构观测。轨道仅用于整理圆点，不编码第二个变量；横坐标不作抖动、分箱或舍入。采用Tukey箱线图：实心箱体表示逐项APE的第25至75百分位区间（Q1–Q3），贯穿箱体的加长竖线表示50次APE的中位数，每行右侧的Median数值给出同一中位数。IQR=Q3−Q1，细须线及短端帽延伸至[Q1−1.5×IQR, Q3+1.5×IQR]内实际观测的最小值与最大值（包含边界），不将围栏值直接作为端点。须外真实观测仍按原坐标完整绘出，不额外重复绘制离群点符号；图中不单独标记均值，也不表示置信区间。全部统计量均在逐项取绝对值后计算，描述不同参数组合的经验分布，不是跨训练seed误差条。五种方法共用各图内从零开始的线性横轴，ν与β为[0,300]%、γ为[0,110]%，完整显示全部有限值。每幅图仅含一个完整范围面板。GIFT为鲜深蓝、PDE-FIND为深橙、PDE-FIND-KC为深绿、PINN-SR为深紫、PINN-SR-KC为深青，圆点和对应须线共用方法配色。算法名称使用黑色粗体，坐标轴使用黑色实线；圆点、实心箱体和须线使用对应方法的鲜艳深色，中位线使用与箱体完全相同的颜色，线宽为1.6 pt，贯穿并伸出箱体，线长约为箱体高度的2.1倍；横轴刻度朝上指向图内，刻度数值仍位于轴线下方，轴线在最右刻度后继续延伸当前横轴跨度的3%，数据横坐标与刻度位置不变；须线细、端帽短；图内仅保留算法名称、坐标刻度、带参数名称的横轴标题及每方法的Median数值，统计定义由正文说明。三幅图与正文主表使用同一APE定义，便于逐参数比较识别误差大小和离散范围，不表达参数间的相关关系。逐点误差与经验统计见[坐标CSV](figures/parameter_distributions/source_points.csv)和[统计CSV](figures/parameter_distributions/source_statistics.csv)，完整绘图规则与文件身份见[绘图元数据](figures/parameter_distributions/figure_metadata.json)及[图像清单](figures/parameter_distributions/figure_manifest.json)。
