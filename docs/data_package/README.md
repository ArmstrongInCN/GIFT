# GIFT data / GIFT 数据

Observed trajectories for GIFT prediction and equation-identification experiments.
The package contains no trained model weights. Keep it outside the Git repository
and set `GIFT_DATA_ROOT` to this folder. All scientific input bytes are listed in
`manifest.json`; `DATA_DICTIONARY.md`, `schema.json` and `splits.json` define them.

## Prediction benchmark / 预测实验

| Split | Trajectory IDs | Count |
| --- | --- | ---: |
| Training | 0–999 | 1,000 |
| Validation | 1000–1039 | 40 |
| Independent test | 1040–1219 | 180 |

GIFT-Lite uses training IDs 0–49. Validation IDs 1000–1019 contain complete
0–10 trajectories; the other validation IDs provide the observation intervals
stored with the evaluation grids. The same ID across grids is the same physical
initial condition. Validation and test trajectories are never training examples.

训练轨迹统一编号在前。GIFT-Lite 使用其中前 50 条；其余方法的预测训练使用
全部 1,000 条。验证集与测试集互不重叠，不根据预测误差或是否失稳选择样本。
同一轨迹的不同分辨率观测使用同一个编号。

## Equation identification / 方程识别

M1 uses a separate local identification-data namespace. Its clean/noisy observed
fields and PINN sensor designs retain their exact recorded bytes and local IDs.
M1 is not an equal-epoch prediction comparison. See the data dictionary for paths.

## License and citation / 许可与引用

Data are licensed under CC BY 4.0; see `LICENSE.txt`. Cite the GIFT project and
the dataset record when reusing these fields. The DOI of the published record is
recorded in the repository README (`10.57760/sciencedb.013lo`), not inside this
package, whose bytes are hash-bound by `manifest.json`. The GitHub repository
contains code and checkpoints, not this data directory. The same fields can also
be regenerated from the recorded initial-condition parameters; see
`docs/DATA_GENERATION.md` in the repository.
