# External algorithms / 外部算法适配

Download the fixed sources listed in `external_sources.json` into a directory outside this project. The loader verifies them before execution. No upstream source file is edited on disk or included in this repository.

这里分别说明对外部源码的少量调整和本项目的实验设置。固定源码、模型能够加载、训练能够恢复，分别是不同的检查，不代表实验结果已经得到验证。

| Used for | Upstream repository | Pinned commit |
| --- | --- | --- |
| FNO-2D / FNO-3D | [NeuralOperator](https://github.com/neuraloperator/neuraloperator) | `01d2aeca407f85ce1c1e78b313edb749973fc051` |
| U-NO | [UNO](https://github.com/ashiq24/UNO) | `19462d82729ef64ef7b9e97056ddcaaf3044ad47` |
| U-Net | [Turbulent-Flow-Nets](https://github.com/Rui1521/Turbulent-Flow-Nets) | `229da3e01acefa6f4dd6c77244da5015c9b965af` |
| PDE-FIND | [PDE-FIND](https://github.com/snagcliffs/PDE-FIND) | `86911349b9fac82996e35465a13d06205730cc05` |
| PINN-SR | [EQDiscovery](https://github.com/isds-neu/EQDiscovery) | `9a20ebe6783e00bc53e3cdd1fd7ff07d2170f496` |
| Native PINN optimizers | [TensorFlow](https://github.com/tensorflow/tensorflow) | `590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b` |

[SETUP.md](SETUP.md) gives installation and download commands. Adaptations below
are applied automatically in memory; readers do not need to edit the upstream
files. The file-by-file byte checks are in `external_sources.json`.

## FNO

1. Load only the network and required utility classes, without executing the example's data-loading or training statements.
2. Change FNO-2D's input projection from 12 to 48 inputs: 46 history frames and two coordinates.
3. Change FNO-3D's input projection from 13 to 49 inputs: 46 history frames and three coordinates.
4. Make these two dimension changes before initialization. Do not replace spectral layers, forward operations or gradients.

只加载模型和所需工具类。两种模型分别调整一处输入维度，其他网络计算保持上游定义。时间窗口、训练批量、损失汇总、归一化统计及跨分辨率统计映射属于实验设置，须另行明确报告。

## U-NO

1. Load the original network, integral operators, Adam and `LpLoss`.
2. Use `in_width=50`, comprising 46 observed history channels and four coordinates supplied by the model.
3. Do not execute the two import-time seed statements. Set the seed explicitly in the training command before constructing the network.
4. Load only the needed loss class from the utilities file. Do not import unused data readers or replace native interpolation gradients.

使用原模型、优化器和损失；只设置输入通道，并把随机种子设置放到训练入口。训练窗口、递归步数、学习率调度频率和最终模型选择由实验协议明确规定，不声称直接执行了上游示例训练脚本。

## U-Net

Import `U_net` directly and set `input_channels=46`, `output_channels=1`, `kernel_size=3`, `dropout_rate=0`. No network operation or initialization method is replaced.

直接通过上游已有参数设置输入、输出和 dropout，不改网络内部。本项目负责时间窗口、闭环相对 L2 损失、优化器调用和恢复。训练入口显式设置记录的数值环境；模型加载器本身不修改精度开关。

## PDE-FIND

1. Load the original library-construction and STRidge functions.
2. For NumPy 2, replace the single empty-index test `biginds != []` with `len(biginds) != 0`.
3. Keep the same sampled values and order in a local random-number generator. A membership lookup table only accelerates testing whether an index was sampled.

回归核心不替换。数据预处理单独包括周期速度重建、噪声条件下分别对涡量和速度进行 SVD，以及固定导数计算。导数使用五次局部多项式权重；混合导数对已算出的一阶导数再做中心差分。这些是实验处理步骤，不只是文件格式转换。

## PINN sparse regression / PINN 稀疏回归

The external STRidge methods use four explicit adjustments: sort the sampled row indices; supply the fixed inherited coefficients as float64; use `lstsq(rcond=-1)`; replace the single empty-index test with a length test. Do not change the sampled set, threshold-search objective or accepted-candidate rule.

四项调整分别是样本行排序、继承系数的计算精度、最小二乘设置和空数组判断。前三项属于需要报告的数值约定，不称为纯语法兼容。

## PINN network and training / PINN 网络与训练

1. Load `PhysicsInformedNN` without running the example's data-loading, training or environment statements.
2. Expose the TF1 API through `tf.compat.v1`, and load the pinned TensorFlow implementations of NAdam and `ScipyOptimizerInterface` from the external checkout. Configure the session's device, thread count and logging explicitly.
3. Change the coefficient vector's single dimension from 60 to 4 (known library) or 90 (open library).
4. For the known library, supply four columns: vorticity, advection, Laplacian and the known forcing pattern. For the open library, keep all 60 original columns and append the five forcing polynomials `q`, `q²`, `u q`, `v q`, `w q`, each multiplied by the six original derivative factors. Here `q = -4 cos(4y)`.
5. Initialize with the original Xavier method and seed 1234 in the declared TensorFlow environment. No initialization checkpoint is required.

Do not intercept `tf.gradients`, rewrite powers as multiplication, replace the loss, accumulate a separate gradient, or implement optimizer kernels. Each NAdam update directly runs the upstream `train_op_Adam` on the complete 24,000-observation / 84,000-collocation population. The project schedules the declared phases and saves native TensorFlow variables, including optimizer slots.

只加载原类；兼容 TF1 接口；调整一处系数数量；按物理问题提供候选项。网络、自动微分、幂运算、损失及 NAdam 更新均调用原实现。初始化由代码和种子生成，不读取初始化文件。图计算采用 float32，并显式关闭 TF32；这些运行环境设置与上述源码适配分开记录。

L-BFGS calls the original public optimizer interface. Its internal work arrays are not copied or serialized by this project. The preceding NAdam boundary is saved before entering L-BFGS. If interrupted inside L-BFGS, `--resume` restores that boundary and repeats only the uncommitted L-BFGS phase; it does not claim internal-solver-state continuation. NAdam, STRidge and completed phase boundaries have their own saved states.

L-BFGS 使用上游公开接口，按阶段恢复；阶段内部中断时重做当前 L-BFGS 阶段，不重做已完成的预训练。上游接口不返回完整的收敛状态，因此这里只报告回调计数及最终损失，不把达到迭代上限或正常返回擅自写成“收敛”。

See the [TensorFlow graph-mode API](https://www.tensorflow.org/api_docs/python/tf/compat/v1/disable_eager_execution) and [SciPy L-BFGS-B options](https://docs.scipy.org/doc/scipy-1.7.1/reference/reference/optimize.minimize-lbfgsb.html). A small native-operation or continuation test is not evidence that a full experimental budget has completed.

## Attribution / 致谢

We thank the authors of NeuralOperator, U-NO, Turbulent-Flow-Nets, EQDiscovery, PDE-FIND, TensorFlow and SciPy. Please cite the corresponding research and respect each upstream project's terms. This project's license does not grant additional rights over external code. The exact references, together with each upstream project's licence status at the pinned commit, are listed in [README.md](../README.md) under Citation.
