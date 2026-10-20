# Experiment figures / 实验图像

## W1/W2图像规范

正文图号与论文一致：W1曲线与场板为图2A/2B，W2曲线与场板为图3A/3B，W3参数识别为图4。预定轨迹及布局按本项目声明的协议呈现。

图像从相应完整数值报告及真实预测数组生成。W1七方法、W2六方法，ANIE仅参加W1；每方法一个固定训练实例。主方法N300，GIFT-Lite使用有序N300名单前50条。共同观测为t5–8、dt=0.04、76点，推理输入为各轨迹自己的单个真实t5场。配方见[训练协议](W1_W2_PROTOCOL.md)。

| 图 | 数据与呈现 |
| --- | --- |
| W1 `mean_relative_l2_vs_time.svg` | 七个实际报告节点的180条轨迹均值，原PCHIP视觉引导、顶部三列图例、比例纵轴；六个原方法配色和标记保留，ANIE增加独立配色和标记。 |
| W1 `traj*_recursive_keyframes.svg` | t5/6/7/8；Reference与七方法按行，左四列涡量场、右四列有符号误差；两个横向色标，方法行严格对齐。 |
| W2 `cross_resolution_mean_relative_l2.svg` | N64/N96/N128横排三面板，对数比例纵轴及终点方法/数值标签。每网格显示十个实际未来报告节点，无跨种子色带。 |
| W2 `traj*_n128_t6_prediction_fields.svg` | N128、t6，上行Reference与六方法预测场，下行对应有符号误差；原生矢量网格，两个独立色标。 |

W1曲线按原七报告位置选择0.04秒网格的最近合法节点，等距时取较早者：t5/5.48/6/6.48/7/7.48/8。标记、PCHIP输入与CSV共同使用实际时刻；曲线中间位置仅显示趋势。W2按十个未来报告位置选择t5.08/5.20/5.28/5.40/5.48/5.60/5.68/5.80/5.88/6。主表future75/future25仍使用全部未知节点，不随图示抽样改变。

确认与探索集各180条，分别作图。两个评价人口均独立于训练和验证，成绩均已被查看，因此此次评价为非盲测的事后测量。W1固定确认ID100005和探索ID1045，均为各集合第6条；W2固定确认ID100110和探索ID1150，均为索引110。ID和时刻不按误差挑选。GIFT与GIFT-Lite场图固定seed20260820；ANIE的t5原始重建及非零误差完整展示。

U-NO监督20步至t5.80，U-Net监督4步至t5.16；其余方法各有声明的目标和原生损失。GIFT-Lite阶段epoch与GIFT相同，但数据与总更新减少。GIFT监督滞后为0.08/0.48秒，内部RK4 dt=0.02，每0.04秒保存。结论限定于这些固定配方，不能唯一归因于架构或当作等预算比较。

## 场板色阶与误差定义

W1/W2均使用prediction−truth的有符号误差；场色图为RdBu_r，误差色图为PuOr，零中心对称。W1两个评价集的选定场板共用场±19、误差±22；后者覆盖最大绝对残差21.354568。W2沿用统一规则max(默认值,ceil(实际最大绝对值))，场默认13、误差默认7：确认图实际±16/±18，探索图±14/±20。

每种量在一幅图内由所有方法共用一套色阶。完整空间范围采用原生vector pcolormesh，W1保留原y轴向上、W2保留原upper origin方向；不翻转源数组、不裁切有限值、不逐方法缩放、不裁剪或平滑场。Reference的误差格使用原浅灰零误差占位。非有限值或失败保留相应方法位置和文字，不能用有限子集取代全体结果。

## W3 and S3 figure status

图4采用Tukey箱线图：箱体为Q1–Q3，中线为中位数，须线端点是1.5×IQR围栏内最远的实际观测，与[Matplotlib官方文档](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)的默认1.5 IQR定义一致。[NIST手册](https://www.itl.nist.gov/div898/handbook/eda/section3/boxplot.htm)说明了四分位箱体与中位线。全部50个真实观测均叠加在箱线图上，包括须外观测；须外点不等于无效数据，也不被删除。

图4a、4b、4c分别展示ν、β、γ的绝对百分比误差，APE_p=100×|p̂−p|/|p|，由同一750行真值与估计值逐项计算。每幅图按固定顺序排列GIFT、PDE-FIND、PDE-FIND-KC、PINN-SR和PINN-SR-KC，每行保留对应方法全部50个真实读出，共250点。圆点的横坐标是该次读出的APE；圆点按APE排序，在七条等间距水平轨道上确定性排布，优先选择不与既有圆点相交的中央轨道，密集值仍可重叠。圆点标称直径为2.1 pt，七轨道相邻间距为2.7 pt，使点层更紧凑；每个真实点用12层同中心、无描边的同色透明圆片叠加，按径向Gaussian透明度渐变形成柔边（相对半径sigma为0.42、标称中心alpha为0.95）。该渐变仅柔化单个真实点，不作KDE或数据平滑，也不添加虚构观测。轨道仅用于整理圆点，不编码第二个变量；横坐标不作抖动、分箱或舍入。采用Tukey箱线图：实心箱体表示逐项APE的第25至75百分位区间（Q1–Q3），贯穿箱体的加长竖线表示50次APE的中位数，每行右侧的Median数值给出同一中位数。IQR=Q3−Q1，细须线及短端帽延伸至[Q1−1.5×IQR, Q3+1.5×IQR]内实际观测的最小值与最大值（包含边界），不将围栏值直接作为端点。须外真实观测仍按原坐标完整绘出，不额外重复绘制离群点符号；图中不单独标记均值，也不表示置信区间。全部统计量均在逐项取绝对值后计算，描述不同参数组合的经验分布，不是跨训练seed误差条。五种方法共用各图内从零开始的线性横轴，ν与β为[0,300]%、γ为[0,110]%，完整显示全部有限值。每幅图仅含一个完整范围面板。GIFT为鲜深蓝、PDE-FIND为深橙、PDE-FIND-KC为深绿、PINN-SR为深紫、PINN-SR-KC为深青，圆点和对应须线共用方法配色。算法名称使用黑色粗体，坐标轴使用黑色实线；圆点、实心箱体和须线使用对应方法的鲜艳深色，中位线使用与箱体完全相同的颜色，线宽为1.6 pt，贯穿并伸出箱体，线长约为箱体高度的2.1倍；横轴刻度朝上指向图内，刻度数值仍位于轴线下方，轴线在最右刻度后继续延伸当前横轴跨度的3%，数据横坐标与刻度位置不变；须线细、端帽短；图内仅保留算法名称、坐标刻度、带参数名称的横轴标题及每方法的Median数值，统计定义由正文说明。三幅图与正文主表使用同一APE定义，便于逐参数比较识别误差大小和离散范围，不表达参数间的相关关系。逐点误差与经验统计见[坐标CSV](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/source_points.csv)和[统计CSV](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/source_statistics.csv)，完整绘图规则与文件身份见[绘图元数据](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_metadata.json)及[图像清单](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json)。

[图4a：ν SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)（[PNG](../assets/figures/w3_parameter_nu_errors.png)） · [图4b：β SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_beta_errors.svg)（[PNG](../assets/figures/w3_parameter_beta_errors.png)） · [图4c：γ SVG](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_gamma_errors.svg)（[PNG](../assets/figures/w3_parameter_gamma_errors.png)）

S3 uses its independent N1000 training protocol, baseline 46-frame history, GIFT single-state input, dt=0.02 and 150-step rollout. Four-vortex and Gaussian populations are trained and evaluated separately; the four-vortex examples do not belong to N300 W1.

| 图 | 数据与统计 |
|---|---|
| W3图4a/b/c | ν、β、γ各一幅绝对百分比误差分布图；五方法各50个真实点，实心箱体为APE的Q1–Q3，与箱体同色的箱内中位线及Median数值表示中位数，细须线及短端帽为1.5×IQR围栏内最远的实际观测。全部真实点（包括须外点）保留，每图一个从零开始的完整范围线性轴；无跨seed误差条。 |
| S3分布对照曲线 | Two populations of 180; declared report times, axes and colors are shared. Existing finite-only scope is retained. |
| S3四涡旋/ Gaussian场板 | Existing selected examples and source bindings are unchanged. |



## 图像文件与数值来源

W1/W2曲线矢量源与选定数值证据分别位于：

- [W1图像与数值源](../results/formal/W1_flow_prediction_20261007/figures/)。
- [W2图像与数值源](../results/formal/W2_cross_resolution_20261007/figures/)。

各人口目录包含可编辑曲线及场板SVG、`source_data.csv`、有界`source_fields.npz`及`figure_manifest.json`；W1另含场误差表，W2场误差记录在清单中。PNG预览位于[assets/figures](../assets/figures/)，名称含`w1_flow_prediction`或`w2_cross_resolution`、集合名和原图名。PNG大小及SHA记录在科学结果包的`figure_manifest.json`；SVG与选定源数组均绑定其科学来源。绘图函数和数据读取入口的源码SHA记录在来源图清单中。

W1/W2的PNG预览与其科学图清单逐文件绑定。

S3曲线与Gaussian场板在 [初值分布图像目录](../results/formal/S3_initial_distribution/figures/)；四涡旋对照场板来源为 [ID1045矢量图](../results/formal/S3_initial_distribution/figures/four_vortex/traj1045_recursive_keyframes.svg)。

表格与曲线使用同一指标数据；场板来自对应真实预测，图示relative L2与记录指标一致。清单记录数据/模型绑定和SHA-256。正文以PNG预览兼顾Markdown显示，矢量文件用于编辑和导出。完整预测HDF和训练状态保存在声明的外部实验目录；选定图像数组不能替代完整预测人口。

## 绘图入口

W3使用[逐参数绝对百分比误差绘图入口](../experiments/formal/w3_equation_identification/plot_parameter_distributions.py)从summary/parameter_results.csv直接绘制图4a/b/c。只读取已保存的750行参数结果，输出三幅SVG及PNG、真实APE坐标、15组方法/参数经验统计和完整绘图元数据，不运行训练或重算参数读出。入口拒绝覆盖已有输出目录和预览文件，--help列出输入与输出参数。

W1/W2图由各自科学结果包直接保存；`scripts.verify_published_results`核验数值文件、图布局及来源绑定。任务编号、数据和绘图入口直接对应。

W1使用[数据读取入口](../scripts/plot_w1_original_style.py)连接[误差曲线函数](../experiments/formal/w1_recursive_prediction/plot_mean_error.py)及[关键帧函数](../experiments/formal/w1_recursive_prediction/plot_keyframes.py)。W2使用[数据读取入口](../experiments/additional/w2_original_style_figures.py)连接[跨网格绘图函数](../experiments/formal/w2_cross_resolution/plot_results.py)。输入为完成评价的numeric报告与保存的预测数组，输出为两个评价集各自的曲线、场板及来源清单。入口的`--help`列出输入与新输出目录参数。
