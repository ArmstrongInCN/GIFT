# 科学来源与完整性

实验按W1、W2、W3、S1、S2、S3直接组织。当前结果包保存数值CSV、选定场数组、矢量图及文件清单，不是另一编号结果的展示包装，也不能恢复训练。

完整训练目录、原始预测、模型和原始报告存于仓库外。`experiments.provenance.external_source(key)`从外部登记解析其实际位置；默认登记文件为相邻的`experiment_sources.json`，可用`GIFT_EXPERIMENT_SOURCE_REGISTRY`或`GIFT_SOURCE_<KEY>`指定。文档中的`prediction_native`、`prediction_extension`、`prediction_aligned`是来源键，需要先解析为实际目录。

| 来源键 | 内容 |
| --- | --- |
| `prediction_native` | GIFT、FNO-2D、FNO-3D可加载训练状态及共同观测 |
| `prediction_extension` | U-NO、U-Net、GIFT-Lite可加载状态及跨网格配置 |
| `prediction_aligned` | ANIE可加载状态和W1训练配置 |

```python
from experiments.provenance import external_source
prediction_root = external_source("prediction_extension")
```

可加载状态的规范化身份元数据与测量时版本分别保存，其模型张量、optimizer、RNG、normalizer和观测数组逐位一致；这些状态不是再次训练所得。原始报告、训练历史、raw数组和实际测量权重保持原件。

S3的可加载数据使用当前身份`gaussian_s3`，本机数据根为`D:/2_GIFT/canonical_data`，其`s3_gaussian/trajectories.h5`与`lite.h5`保存原观测数组的逐位副本。当前发布模型使用一致的S3身份和冻结生成元绑定。数组、模型张量、optimizer、RNG及训练预算均保持原值，身份元数据的文件SHA变化不表示再生数据、重新训练或新测量。实际可加载文件SHA由`artifacts/CHECKPOINTS.json`登记，已发表数值的测量时权重SHA与原报告保持原件。本机等价性证据为仓库外的`S3_CANONICAL_DATA_MIRROR.json`和`S3_PACKAGED_METADATA_NORMALIZATION.json`。

来源登记只定位证据，不转换观测或选择权重。结果清单中的`source_record_id`或`source-record:`引用对应仓库外的完整原始元数据；本机登记为`D:/2_GIFT/repository_audits/rename_w3_20261007/SOURCE_PROVENANCE.json`。每个当前引用绑定原文件SHA及所在JSON位置，原始报告、训练历史、raw数组和测量权重的SHA保留不变，当前公开文件的大小与SHA按其实际字节绑定。

实验身份与路径属于元数据，不能作为重新训练或重新计算的证据。W1/W2与S1–S3的数值和既有图保持原测量。当前W3正式无噪声结果来自run `w3_single_trajectory_20261010`：固定50组参数、每case单条ID0轨迹，250槽和750行均完成。所有本机锁定科学源码SHA均与SOURCE_LOCK匹配，远端24文件由通过的部署收据绑定；数值包清单逐文件校验通过。外部算法运行时和官方checkout仍在GIFT项目目录之外。W3结果包保存数值、系数支持、协议、来源清单以及[三参数绝对百分比误差分布图4a–c](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/w3_parameter_nu_errors.svg)，不含外部实现代码。图像以100×|估计值−真值|/|真值|逐项计算同一750行参数读出的APE，每参数仅含一个完整范围面板，七条等间距水平轨道仅整理圆点而不改变APE横坐标。圆点标称直径为2.1 pt，七轨道相邻间距为2.7 pt，使点层更紧凑；每个真实点用12层同中心、无描边的同色透明圆片叠加，按径向Gaussian透明度渐变形成柔边（相对半径sigma为0.42、标称中心alpha为0.95）。该渐变仅柔化单个真实点，不作KDE或数据平滑，也不添加虚构观测。Tukey箱线图的实心箱体为逐项APE的Q1–Q3，贯穿箱体的加长线与每行Median数值表示50次APE的中位数，细须线及短端帽为[Q1−1.5×IQR, Q3+1.5×IQR]内最远的实际观测，IQR=Q3−Q1，边界观测包含在内。全部真实点（包括须外点）保留，不重复添加离群点符号，不单独标记均值。图内仅保留黑色粗体算法名称、刻度、带参数名称的横轴标题及每方法的Median数值，坐标轴为实线，方法圆点、实心箱体及须线共用鲜艳深色，中位线与箱体颜色完全一致，线宽1.6 pt、长度约为箱体高度的2.1倍；横轴刻度朝上指向图内，刻度数值仍在轴线下方，轴线在最右刻度后延伸当前横轴跨度的3%，不改变数据横坐标或刻度位置；其逐点坐标、绘图入口与文件SHA单独绑定于[图像清单](../results/formal/W3_single_trajectory_20261010/figures/parameter_distributions/figure_manifest.json)；已闭合训练协议、科学源码锁与成本清单保持原字节。方法目标、观测处理和预算不同，不构成等信息/等预算架构比较。来源、实测成本及完整审计见[W3结果清单](../results/formal/W3_single_trajectory_20261010/published.json)。

```shell
python -m scripts.verify_published_results --experiment W1 --result-dir results/formal/W1_flow_prediction_20261007
python -m scripts.verify_published_results --experiment W2 --result-dir results/formal/W2_cross_resolution_20261007
python -m scripts.verify_published_results --experiment S3 --result-dir results/formal/S3_initial_distribution
```

文件完整性核验检查清单、哈希和图像绑定，不重新训练或重算预测。数值重算、同权重预测一致性与实际图像查看分别记录。
