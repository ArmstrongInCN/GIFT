# 外部算法的下载、手工适配与接口

GIFT 仓库不包含外部算法实现或自动源码补丁。使用外部基线时，使用者需从下表的官方仓库下载固定提交，在 **GIFT 仓库文件夹之外** 完成以下文字所述的必要适配，并提供本文的接口。GIFT 内的 `adapters/__init__.py` 只连接外部接口，不定义网络、积分方程、稀疏回归或源码改写过程。

本机已用于实验的运行接口保存在 `D:/2_GIFT/gift-external-runtime/`，原官方 checkout 仍保存在各自的仓库外目录。这个目录是本机独立依赖，不属于 GIFT Git 仓库，也不随 GIFT 发布。新的使用者不能从 GIFT clone 中取得这些实现；应自行下载官方代码、依下述操作实施适配并提供相同接口。仅安装 GIFT 即可读取已发布的结果表；重新训练或推理外部方法还需要上述依赖。

## 目录与来源

| 路径或环境变量 | 用途 |
| --- | --- |
| `GIFT_EXTERNAL_ADAPTER_ROOT` | 仓库外运行接口目录，包含 `adapters/` 子目录。缺省为 GIFT 仓库相邻的 `gift-external-runtime/`。不得指向 GIFT 仓库或其子目录。 |
| `GIFT_EXTERNAL_ROOT` | FNO、U-NO、U-Net、PDE-FIND、PINN-SR、TensorFlow 固定 checkout 的父目录，子目录名见 `external_sources.json`。必须在 GIFT 仓库外。 |
| `GIFT_ANIE_SOURCE_ROOT` | ANIE 官方 checkout 的直接路径；也可以使用上述父目录下的 `anie/`。必须在 GIFT 仓库外。 |
| 仓库内 `external_sources.json` 与 `anie_sources.json` | 仓库地址、固定版本和选定文件等来源元数据；不是算法实现，也不含可执行适配补丁。 |

外部运行接口与官方 checkout 可以位于不同的外部目录。需要修改官方文件时，应在使用者自己的仓库外适配副本中操作，并记录原 commit 与修改文件；保留干净 checkout 便于来源核对。本机接口从未改写的官方 checkout 加载，实施下文明确声明的当前计算语义。已完成实验的原数值、权重及协议身份独立保存；更新训练接口不会将这些结果或恢复状态转换成另一协议。

| 方法 | 官方仓库 | 固定 commit | 主要文件 |
| --- | --- | --- | --- |
| FNO-2D / FNO-3D | [neuraloperator/neuraloperator](https://github.com/neuraloperator/neuraloperator) | `01d2aeca407f85ce1c1e78b313edb749973fc051` | `fourier_2d_time.py`、`fourier_3d.py`、`utilities3.py` |
| U-NO | [ashiq24/UNO](https://github.com/ashiq24/UNO) | `19462d82729ef64ef7b9e97056ddcaaf3044ad47` | `navier_stokes_uno2d.py`、`integral_operators.py`、`Adam.py`、`utilities3.py` |
| U-Net | [Rui1521/Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | `229da3e01acefa6f4dd6c77244da5015c9b965af` | `Baselines/U-net.py` |
| ANIE | [emazap7/ANIE](https://github.com/emazap7/ANIE) | `2071ad709b260a6c305450affcf7168fd825f692` | `IE_source/kernels.py`、`Galerkin_transformer.py`、`Attentional_IE_solver.py`；NS notebook 与 `experiments.py` 作设置参考 |
| PDE-FIND | [snagcliffs/PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | `86911349b9fac82996e35465a13d06205730cc05` | `PDE_FIND.py` |
| PINN-SR | [isds-neu/EQDiscovery](https://github.com/isds-neu/EQDiscovery) | `9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496` | `Examples/Discovery with Single Dataset/Vorticity/NS_Vorticity_UniformSetting_OneAdam_PreADOPt.py` |
| PINN 原生优化器 | [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow) | `590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b`，tag `v1.15.0` | `tensorflow/contrib/opt/python/training/nadam_optimizer.py` 与 `external_optimizer.py` |

下载后先核对 commit 和选定文件来源，再实施适配。公开 URL 和散列不授予额外使用或再分发权利；许可状态与原论文引用见 [README](../README.md)。不能把含外部算法的适配副本、压缩包、notebook 或依赖缓存放进 GIFT 文件夹。

## FNO-2D / FNO-3D

1. 从 `fourier_2d_time.py` 加载 `SpectralConv2d_fast` 和 `FNO2d`，从 `fourier_3d.py` 加载 `SpectralConv3d` 和 `FNO3d`。上游示例在模块顶层执行数据读取和训练，需在仓库外适配副本中把这些示例执行语句隔离在独立入口；加载网络时不能触发它们。保留类定义、谱层、forward 和初始化顺序。
2. 在两种模型构造器的唯一 `fc0` 输入投影处调整输入数。单初态 W1/W2：FNO-2D 从原 12 改为 **3**，由一场和两空间坐标组成；FNO-3D 从原 13 改为 **4**，由一场和三时空坐标组成。独立历史 46 帧协议分别为 **48 / 49**。必须在构造模型前确定维度，不能重复初态来冒充历史帧，也不能将 46 帧权重载入单初态模型。
3. 网络参数保持 FNO-2D modes=12/12、width=20；FNO-3D modes=8/8/8、width=20。从 `utilities3.py` 提供原 `UnitGaussianNormalizer` 和 `LpLoss`，隔离无关示例与数据读取。
4. FNO-2D 输入采用 `[batch,Y,X,history]`，单步输出同样保留末尾一通道。FNO-3D 将观察通道沿查询时间轴重复，输入为 `[batch,Y,X,query_time,history]`；由原模型追加坐标，输出完整查询网格。
5. 训练损失、课程、归一化和模型选择由 GIFT 实验协议决定，不属于输入投影修改。W2 的 FNO-3D 使用原 75 个 joint queries，再截取前 25 个未来点；空间 normalizer 只通过确定性 Fourier 网格插值使用冻结训练统计，不能拟合目标网格或测试人口。

## U-NO

1. 在仓库外保留原 `UNO`、`integral_operators`、`Adam` 和 `LpLoss`；不替换谱计算、插值梯度或优化器内核。加载工具文件时只需所用损失定义，不执行示例数据读取。
2. 将 `navier_stokes_uno2d.py` 的两个模块级随机种子设置移至用户的训练入口，在构造模型前设置协议种子。外部模块导入不能把训练入口已设置的种子重置为零；同时保留原网络初始化方法和参数创建顺序。
3. W1/W2 单初态构造设置：in_width=5、width=32、pad=0、factor=0.75。一场之外的四个周期坐标通道由原网络追加。历史 46 帧协议使用 in_width=50。输入和单步输出均为 `[batch,Y,X,channel]`。
4. 单初态训练时每次将上一步预测直接作为下一步输入，保留全部反馈梯度。既有 activation checkpoint 使用非重入计算，BatchNorm 状态及更新方式不得改写；它不允许借此改变物理 batch。当前监督 20 步，评价递归 75 步；这些预算与学习率调度由协议分别声明。

## U-Net

1. 从仓库外 `Baselines/U-net.py` 提供原 `U_net` 类。
2. W1/W2 通过原构造参数设置 input_channels=1、output_channels=1、kernel_size=3、dropout_rate=0；历史 46 帧协议仅将 input_channels 改为46。无需改卷积、BatchNorm、skip connection 或初始化。
3. 输入和输出均为 `[batch,channel,Y,X]`。单初态闭环每步将预测作为下一步输入，不截断训练梯度。当前四步监督、75步评价属于实验预算，不称为原作者 NS 训练脚本。N64/N96/N128 使用同一全卷积网络，像素核对应的物理尺度随分辨率变化。

## ANIE

1. 以固定 NS notebook 的构造顺序提供 `model_blocks` 积分网络、`SingleConvNeuralNet` 编码器、`Decoder_NN_2D` 解码器，以及原 `Integral_spatial_attention_solver_multbatch`。加载依赖时保留 Galerkin attention 定义，隔离上游示例训练、绘图、无关 ODE/Monte Carlo 入口和循环导入。类型注解可延后解析；不把原类或原 autograd 替换为本地复写实现。
2. 在 `Attentional_IE_solver.py` 中空间与时间拼接的位置（固定版本约1190行）修正坐标顺序：状态展平为 `[batch,space,time,channel]`，token 顺序先空间后时间；原重复空间坐标为 `[batch,time,space,coordinate]`，须交换 time 与 space 轴再展平。时间坐标为每个空间点连续保留整段查询时间，**不能将重复时间排序**。例如空间点0的全部时间必须排在空间点1的全部时间之前，坐标与状态逐 token 一致。保留状态排列、积分归一化、迭代公式和停止规则。
3. 在 `Galerkin_transformer.py` 的 `linear_attention` 功能性 attention dropout 位置（固定版本约758行）将是否启用 dropout 与调用方 Dropout 模块的 training 状态绑定。训练仍使用原功能性 dropout 默认概率 **0.5**，不替换为传入模块标注的0.1；eval 模式关闭该训练 dropout。其余 dropout、attention 与归一化不改。
4. 输入为有限 float32 `[batch,64,64]` 初始涡量场。编码器输出16×16×32潜空间；全局解码器保持8192→64→256→4096。attention 为 latent32、embedding64、4 heads、4 blocks、feedforward128，模块 dropout 参数0.1；原构造和参数创建顺序不改。空间坐标采用16点的0–1轴；局部物理查询时间0–3秒除以3后提供给原积分求解器。
5. 当前 `aligned_ns_single_v1` 使用原 NS 训练的 c=0 分支：只编码初态，沿查询时间重复编码状态作为初始迭代 y_init；保留 initialization=True、spatial_integration=True、use_support=False、accumulate_grads=True、最多3次迭代、smoothing_factor=0.5、global_error_tolerance=1e-6。不将编码状态改为每次右端求值的固定自由项，不加时间逐步反馈、残差锚定或局部解码器。
6. 求解器的 device 全局绑定输入设备，模型与输入保持同设备和 float32；不在优化器更新期间迁移模型参数。当前显式传入76个查询点，训练与评价同网格；省略时间参数的历史默认151点不能用于当前W1测量。原输出为 `[batch,time,64,64]`，包括预测的起始帧，不能用真实起始场覆盖该预测。首帧编码/解码重构另通过 `initial_decoder` 报告。
7. ANIE 仅参加 W1。训练 c=0/y_init 分支、联合 Adam、validation-MSE-best、延迟调度与 patience 等训练规则见固定协议；这些不因依赖移动而改变。旧 `native_ns_single_v1` 只实施评价 dropout 修正，旧 `aligned_legacy` 还允许历史 fixed_f 分支；加载旧权重时须使用其记录的 profile，不能静默将旧权重切换为新排列。

目标位置参考：[NS notebook](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/NIE_Attention_Navier-Stokes_patches.ipynb)、[求解器](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/IE_source/Attentional_IE_solver.py#L1183)、[attention](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/IE_source/Galerkin_transformer.py#L750)、[原 NS 初始化](https://github.com/emazap7/ANIE/blob/2071ad709b260a6c305450affcf7168fd825f692/IE_source/experiments.py#L5295)。两项运算修正会改变未修正版的函数，因此当前结果是已声明修正的官方架构实验，不能称为作者代码逐字原样复现。

## PDE-FIND

1. 从仓库外 `PDE_FIND.py` 提供原 `PolyDiffPoint`、`build_Theta`、`TrainSTRidge` 和 `STRidge`，不执行顶层示例。
2. 在所有 NumPy 版本下，将 `STRidge` 中唯一通过与空列表比较来判断 biginds 非空的位置改为按索引长度非零判断。NumPy 1.26 已会因数组与空列表形状不一致而报错，因此不能仅在 NumPy 2 下执行此操作。显式非空判断恢复原注释所述的最终最小二乘步骤；旧 NumPy 的单元素数组比较可能错误地跳过该步骤，故不能宣称所有退化输入都与旧行为逐位相同。不能顺带改阈值搜索、候选接受规则或最小二乘求解器。
3. 保持原 seed0 MT19937 抽样集合及其顺序，在局部 RNG 中执行，避免改变其他模型的全局随机状态。成员查找可使用布尔索引表加速，但不能改变被抽中的样本、排列或训练/测试集合。
4. GIFT 数据接口沿用已声明的周期速度重建、噪声下分别对涡量和速度做 SVD、五次局部多项式导数及混合导数计算；库行序为 y,x,time，已知项4列、开放项90列。数据处理和库几何是实验设置，不能视为等价格式转换。
5. 当前W3正式50组无噪声参数实验中，每个case的数据容器只含该组的1条training轨迹（局部ID0）和0条validation轨迹；不同参数case各自使用对应的轨迹，不复制或跨case复用。PDE-FIND读取本case的501×64×64 float32涡量值及t=0–10、间隔0.02的时间网格；不得复制ID0补足50条，不改变导数、候选库、抽样或STRidge设置。无噪声实例不进行SVD去噪。

## PINN-SR 与 TensorFlow 优化器

1. 在仓库外从固定 EQDiscovery 文件提供原 `PhysicsInformedNN`，隔离示例数据读取、训练和全局环境设置。采用 `tf.compat.v1` 图模式，并从固定 TensorFlow checkout 提供原 `NadamOptimizer` 与 `ScipyOptimizerInterface`；不复写优化器内核，不导入整个 TensorFlow 源码树执行构建。
2. 把唯一 lambda_w 系数向量的第一维从原60改为已知库4或开放库90。已知库依序提供涡量、平流、Laplacian 和已知强迫模式。开放库保留原60列，再按 q、q²、u q、v q、w q 顺序，各乘原六个导数因子追加30列；q为−4 cos(4y)。列顺序及名称必须与权重和系数读出一致。
3. 保留原 Xavier 初始化、seed1234、三坐标输入、8个宽60隐藏层和三场输出，float32 图计算关闭TF32。按协议使用单线程 Session，显式选择CPU/GPU及增长显存策略。默认 `native_fp32` 保留原坐标自动微分、幂运算和损失；每次原 NAdam 更新使用完整24,000观察/84,000配点人口。
4. 构图前启用TF1引用变量语义，并核对网络和原NAdam状态均为RefVariable。原TF1 `ScipyOptimizerInterface` 通过feed候选变量求损失与梯度；不能仅启用compat图模式后使用ResourceVariable，否则候选可能未进入实际计算。构图及编译后，分别以候选feed和显式赋值验证目标与全部梯度一致，再检查真实有限步L-BFGS确实改变参数。记录SciPy的success、status、message、nit、nfev、njev、初末损失和梯度；到达上限或接口返回不能当作收敛。
5. 独立稀疏回归调用原 `TrainSTRidge` 与 `STRidge`。保留原seed0 MT19937抽样集合及返回顺序，不排序训练行；保留从TF读取的float32继承系数快照，不额外转成float64；首轮l0基准在后续轮继承。最小二乘保留rcond=−1，唯一空索引判断按长度处理。使用局部RNG隔离抽样，不改变原选择顺序及数值。
6. 保持lambda_w为稳定变量，每轮STRidge完成后立即明确赋值并保存其向量及精确非零支持。普通系数、指标和日志读取不得隐含赋值。ADO按[论文Methods式(6b)](https://www.nature.com/articles/s41467-021-26434-1)固定回归系数，只更新网络权重/偏置；使用同一个原NAdam及已有网络动量和beta状态，不重新创建优化器。官方固定NS可执行脚本采用联合ADO且含延迟赋值，和论文子问题不同；当前训练声明 `paper_ado_v1`，不称逐事件复现该脚本。
7. 后训练从最后一次STRidge向量及支持开始，支持按原精确零/非零规则固定，不另设阈值。按论文式(8)联合更新网络与有效系数，残差及读出均使用原系数乘固定mask。保留原动量；被排除项的原始存储可能被旧动量推进，但有效系数必须严格为零，不能由演化后的原始向量重建支持。模型、optimizer slots、mask、最终回归向量/支持、RNG和阶段状态耦合恢复。L-BFGS进入前保存边界，内部中断时从该边界重做未提交阶段，不宣称求解器内部状态续算；不能将旧联合ADO结果当修正版恢复状态。
8. 可选 `optimized_xla_fp32` 只改变等式计算与编译：KC沿原tanh网络传播精确一阶导数及Laplacian，仍对网络参数自动微分；开放库合并坐标求导，并把原15种多项式/forcing因子与6种导数的90项收缩作等价因子计算，完整90列Phi仍供STRidge。关闭TF32、不使用半精度、不减少观测/配点/训练次数。显式GPU选择及XLA配置须在首次TF导入前完成，libdevice在仓库外按环境准备。编译后必须重验候选参数传播、全部梯度和优化器状态；完整同初态配对训练通过预测、物理残差、支持及活跃系数门槛后才能声明速度与精度收益。执行方式记录在来源metadata；默认仍为严格原生FP32。

## 外部接口约定

外部根目录下提供名为 `adapters/` 的 Python 模块集合。以下接口是 GIFT 训练和评价调用的连接约定；所有算法类和针对源码的适配操作均由使用者在外部目录实现。内部辅助函数可能被现有训练入口调用，名称也需保持。

| 外部模块 | 提供的名称及行为 |
| --- | --- |
| `models` | `build_model`、`load_checkpoint`、`source_record`；以及 `fno_utilities`、`uno_components`、`_fno_class`、`_import_file`。按声明method/history/device加载原网络和工具，strict加载模型state；模型带 `gift_external_source` 来源记录。 |
| `prediction` | `load_uno`、`load_unet`、`uno_rollout`、`unet_rollout`，提供历史46帧接口及仅用预测反馈的推演。 |
| `w1_extended` | `build_network` 与 `forward_model`；history=1，输入 `[B,Y,X]`，输出 `[B,steps+1,Y,X]`，首帧为提供的真实初态，未来为闭环预测；训练按协议保留完整反馈梯度。 |
| `anie_config` | `ANIEConfig`、`PINNED_COMMIT`、`source_identity` 与来源锁读取接口；返回固定架构和记录profile，拒绝不支持的架构或分支。 |
| `anie` | `ANIEPredictor`；提供标准PyTorch参数/state接口、`forward(initial,times)`、`initial_decoder(initial)`、`source_identity()` 与 `last_solver_iterations`，采用上文明确的原NS计算分支。 |
| `pde_find` | `load_upstream`、`source_record`、观察与周期数据处理、库生成、回归和参数读出接口；保留4/90列约定和来源记录。 |
| `pinn_runtime` | `PINNRuntime`、`read_inputs` 及原Session阶段操作、变量导出/恢复；按原算法执行，不用本地仿制网络。 |
| `pinn_stridge` | `fit_stridge`，接受有限float32矩阵/目标、4或90维继承系数、ADO轮序及固定l0 penalty；返回系数、接受历史、选定阈值、输入和来源记录。 |
| `pinn_reference` | `read_checkpoint`、`read_reference` 和已声明库/预算元数据，读取系数结果，不执行TensorFlow训练。 |
| `check` | 独立小规模模型构造检查入口；检查可加载与shape，不当作完整科学结果。 |

初始化随机数、参数名称及创建顺序、输入输出轴、dtype/device、梯度连通性、train/eval行为、checkpoint profile、归一化统计和库列序均属于接口的一部分。满足名称并不意味着计算等价。变更外部接口后，应先按相同seed构造模型比较全部state，按固定输入比较前向/梯度，再用已保存正式权重检查固定轨迹。已完成的科学数据、原始预测、模型权重及来源metadata须保持不变；来源目录移动不允许以改参数或重新训练代替一致性验证。
