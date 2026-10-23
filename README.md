# JSBSim 仿真实验 v1：功能状态与可观测性分离

本工程生成用于研究“多源观测下效能结果评估”的**方法学样例数据**。JSBSim 只提供固定基线运动状态；另一个独立的合成变量表示功能状态。代码没有建模反制设备、作用机理、武器参数、交战过程或规避/攻击策略，也不应把输出当成真实装备性能数据。

## 1. 设计依据

结果与观测质量是两个不同维度：

1. **仿真真值** `ground_truth_outcome`：按合成的 `functional_capacity_truth` 阈值确定，只表示本实验的抽象功能结果。
2. **可观测性** `observability_status`：按传感器代理数据覆盖率和最长连续缺测时长确定。
3. **评估标签** `evaluation_label`：仅当可观测性达标时取真值类别；否则为 `indeterminate`。因此，`indeterminate` 不是第四种物理结果。

四个评估类别是 `no_effect`（未见明显功能影响）、`degraded`（功能下降，可对应“击伤”评估语义）、`disabled`（功能丧失，可对应“击毁”评估语义）、`indeterminate`（结果/管控不可判定）。其中前三类是抽象结果，最后一类是“证据不足”的状态。它们不构成真实命中、损伤或摧毁的物理判据。真值单独保存在 `oracle_labels.csv`，训练/评估输入应使用 `observations.csv`，并通过 `episode_id` 关联标签。严禁把 `functional_capacity_truth` 或 `ground_truth_outcome` 拼入模型输入。

## 2. 字段

### `observations.csv`：观测与评估集

| 字段 | 含义 |
|---|---|
| `episode_id`, `time_s` | 场景及时间索引 |
| `sim_altitude_ft`, `sim_speed_fps`, `sim_roll_rad`, `sim_pitch_rad`, `sim_heading_rad` | JSBSim 输出的基线运动状态；只用于仿真管线联调，不代表传感器真值或特定真实平台 |
| `health_proxy_observed` | 带噪声且可能缺测的合成功能代理观测 |
| `sensor_valid` | 当前时刻是否有代理观测，1/0 |
| `coverage_fraction`, `max_gap_s` | 本 episode 的有效观测比例和最长缺测时长 |
| `observability_status` | `observable` 或 `indeterminate` |
| `evaluation_label` | 四分类目标；不可判定时为 `indeterminate` |

### `oracle_labels.csv`：隔离的仿真真值

| 字段 | 含义 |
|---|---|
| `episode_id` | 与观测集关联键 |
| `ground_truth_outcome` | 三种真实合成结果之一：`no_effect`、`degraded`、`disabled` |
| `functional_capacity_truth` | 抽象功能完整度，取值 [0, 1] |
| `synthetic_effect_family` | 固定说明为抽象状态变量，不对应物理作用机制 |
| `truth_definition` | 阈值规则的说明 |

## 3. 生成逻辑

1. 根据随机种子轮换三种结果区间，降低类别失衡；区间阈值在 `config.json` 中。
2. 每个 episode 建立 JSBSim FDM，运行同一基线模型，不向模型注入干预动作或设备控制量。
3. 对隐藏的功能代理加入高斯噪声和随机缺测，生成观测序列。
4. 按覆盖率及最长缺测时长判断是否可观测；不达标即将评估标签置为 `indeterminate`，但保留单独文件中的真值结果。
5. 写出 `observations.csv`、`oracle_labels.csv` 和 `manifest.json`。固定 `seed` 可复现实验抽样。

## 4. 运行

先确认本机已有可用的 JSBSim Python 绑定。本工程不会自动下载或安装依赖；`requirements.txt` 仅记录依赖名，不要为了运行本实验重复安装已有的包。你提供的源码目录可以作为模型数据根目录传入，脚本会从中读取 `aircraft`、`engine`、`systems`：

```powershell
python generate_dataset.py --jsbsim-root "D:\Learn\雷达打击效果评估\JB\jsbsim-master"
```

当前只找到源码树，不代表 Python 绑定已经编译。Python 绑定就绪后，可用 `--jsbsim-root` 指向源码/模型数据目录；若只单独准备了 aircraft 目录，则可以传：

```powershell
python generate_dataset.py --aircraft-path "C:\path\to\jsbsim\aircraft"
```

可用 `--episodes 30` 临时缩小样本量，或用 `--model MODEL_NAME` 指定本机已有模型。脚本不会自行下载模型。首次接入某个 JSBSim 安装时，建议先用 1–3 个 episode 检查模型是否可加载、是否能完整推进，以及状态字段是否为有限数值。

## 5. 验证方案

当前环境未发现可调用的 JSBSim Python 绑定；你提供的目录包含源码和模型数据，但未发现已编译的 Python 扩展或独立可执行程序，因此尚未生成运行数据。绑定就绪后，按以下顺序验证：

1. **接口验证**：模型成功加载、初始条件成功初始化、每个 episode 都推进到末尾；位置/姿态/速度字段无 NaN/Inf。
2. **标签不变量**：真值只能属于前三类；若 `observability_status == indeterminate`，则 `evaluation_label == indeterminate`；若可观测，则评估标签等于对应真值。
3. **缺测边界**：用固定种子分别设置零缺测、高缺测，确认覆盖率/最长间断阈值会相应改变可判定比例。
4. **防泄漏**：训练特征只从 `observations.csv` 选择；禁止使用真值文件、隐藏功能代理、场景编号顺序或生成种子。
5. **统计验证**：检查 episode 级标签计数、混淆矩阵，以及可观测率和类别条件下的可观测率；按 episode 划分训练/验证，避免同一时间序列跨集合。
6. **敏感性分析**：改变传感器噪声、缺测概率、覆盖率阈值和类别阈值，报告四类比例及宏平均指标的变化。该合成实验只能验证流程与标签逻辑，不能验证现实效能。

## 6. 当前边界与下一版

这里的结果类别来自人为设定的功能完整度区间；基线飞行动力学与结果标签没有物理耦合。它适合检查数据结构、标签隔离、缺测处理和评估流程。下一版若要提高可信度，应先经领域人员定义可测功能指标及真值采集规程，再以校准数据建立状态转移，不应直接把本版阈值解释为现实损伤标准。
