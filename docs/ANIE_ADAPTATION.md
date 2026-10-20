# ANIE：时空坐标与状态token对应

ANIE使用作者公开仓库 `emazap7/ANIE`，固定提交 `2071ad709b260a6c305450affcf7168fd825f692`，source profile为 `aligned_ns_single_v1`、condition为 `native`。官方源码与适配接口均位于 GIFT 仓库外。本机既有外部接口从只读官方 checkout 加载并实施已声明操作；GIFT 只提供[文字操作说明](EXTERNAL_MANUAL_ADAPTATION.md)和来源元数据，不包含实现或可执行补丁。

状态张量按空间优先、时间次优顺序展平。空间索引s、时间索引j对应token `k=s*T+j`；token的状态特征、空间坐标和时间坐标均使用该索引。求解器内部各次更新保持同一排列，解码前按其逆排列还原。训练、validation和W1推理采用同一配置。

| 项目文件或仓库外接口 | 调整及必要性 |
| --- | --- |
| `仓库外接口 adapters/anie_config.py` | 允许并验证独立 `aligned_ns_single_v1` 配置；它采用native条件路径，训练控制来源沿用官方NS实现。 |
| 仓库外执行锁；GIFT内`anie_sources.json`为文字元数据 | 记录 `spatial_time_token_alignment` 与 `linear_attention_eval_dropout_mode` 两个操作标识；原配置及checkpoint读取行为保持绑定，执行文本只存于仓库外。 |
| `仓库外接口 adapters/anie.py` | 仓库外接口将空间和时间坐标与状态统一排列；操作见手工适配说明，原架构、注意力、积分迭代和输出解码保持。 |
| `training/w1_single_train.py` | 按声明profile构造原网络；全76点物理MSE、optimizer/scheduler、同次validation-MSE-best及原早停规则由固定训练配置确定。 |

已登记的functional attention dropout调整在训练时保留原概率，在evaluation时关闭，与模块training状态一致。未增加归一化、空间decoder修改、初态硬锚定、辅助重建或递归机制。

训练数据为共同N300名单和20条validation，t5–8每0.04秒共76点。输入仅为每条轨迹t5真场；NS训练和评价均使用c=0/y_init条件路径。seed7，effective batch32、micro6，Adam lr1e-4/wd1e-4，延迟CosineT101，eta_min1e-9，上限5000轮、patience1000。实际完成1029轮、10290次更新，选择第29轮；训练进程墙钟21176.742秒。

当前可加载权重：`prediction_aligned/training/anie_ns_seed7/best.pt`，SHA-256 `3adbe09b412c1a3a214ac06bebde51079298018799b80a685d8cfde4f93abb35`。当前权重的数值状态与原测量相同，原测量权重文件SHA保留在科学结果清单。训练历史及raw来源由仓库外来源登记保留。W1两人口各180条单列，主指标排除已知初态，只取future75。ANIE不参加W2。该实验描述固定配方在本任务的表现，不作为原论文数据与全部设置的完整复现。
