# X4-950 PINN v2 宽域模型

2026-11-19 的六维宽域模型及复现记录。当前默认加载 [v3](x4-950-pinn-wind-randomized-v3.md)。v2、[X4 v1](x4-950-pinn-pilot-v1.md)、旧 `pinn` 和 0–3 m/s `analytic` 可显式选择。

## 接口和适用范围

- 世界系水平风满足 `hypot(wx,wy) ≤ 10 m/s` 且 `wz=0`。输入 [6,8,0] 合规，[10,10,0] 超域。根点水平飞速≤3 m/s、垂直速度±1 m/s、roll/pitch 各±20°、全 yaw、机体系角速度模长≤1.5 rad/s，用于训练和状态检查。相对空气速度可达约 13.04 m/s。状态超域时记录诊断。
- 输入为机体系根点相对空气速度与角速度，共六维。相同状态下做有风和无风推理，三轴差分转回世界系。实际施力先做 XY 投影和阻力面积换算，再经 EMA α=0.05、逐轴 ±6 m/s² 限幅并乘质量 8.0004 kg，根 body 外力的 Z 为零。
- 参考是同一固定初始气动几何的 29 部件逐轴二次阻力，密度 1.225 kg/m³，逐轴 Cd=1.0。旋翼诱导流、尾流、遮挡、壁面、活动臂、额外载荷和直接气动力矩不在本次模型内。下述误差针对该解析参考。

## 运行时等效阻力面积

| 参数 | 默认值 | 定义 |
| --- | --- | --- |
| 水平等效阻力面积 $C_DA$ | **0.25 m²** | 机体静止、水平、航向为零，世界系来风为 [4, 4, 0] m/s 时，以沿风力除以动压 |

随包宽域解析参考在此工况的沿风力约为 2.3424 N，动压为 19.6 Pa，等效面积约为 0.11951 m²。运行时取目标面积与参考面积之比（约 2.092）作为同一增益，缩放宽域 PINN 和使用随包宽域参考的解析后端的世界系 XY 加速度增量，再进入 EMA 和限幅。在该静态工况，解析沿风力目标为 4.90 N，PINN 结果受代理近似误差影响。其它来风角度及运动状态的等效面积随原模型响应变化。

$C_DA$ 是本次指定的仿真实验参数。原冻结参考、训练权重和三轴原始预测的诊断口径保留，下面的历史验收数据对应原施力设置。启动终端显示 $C_DA$ 和增益，使用运行日志时分别记录为 `wind_cda_m2` 和 `wind_drag_scale`。

## 解析后端阻力随机化与阵风

在仿真终端运行以下命令，然后点 **Apply Wind** 并开启风扰。参数只是可复现的实验示例。

```bash
python scripts/run_simulation.py --wind-backend analytic \
  --drag-variation 0.20 --gust-amplitude 1.5 --gust-period 4 --wind-seed 42 \
  --wind-x 4 --wind-y 4
```

- `--drag-variation 0.20` 在每次运行开始时，分别为机身、机臂及电机、机械臂抽取一个均匀分布于 [0.8, 1.2] 的阻力系数倍率。每组倍率在有风、无风两次力计算中共用，并在整个运行期间固定。0.25 m² 仍是标定工况的名义值，单次随机试验的实际等效面积可以偏离它。
- `--gust-amplitude 1.5 --gust-period 4` 叠加峰值为 1.5 m/s、历时 4 个仿真秒的水平脉冲。各脉冲方向按种子确定，起止都平滑回到基准风。总水平风速限于 10 m/s，关闭风扰时外力归零。
- 两个扰动可分别开启，任一启用时解析后端自动选择随包 0–10 m/s 参考。加上 `--run-log` 可记录各组倍率、种子、基准风和实际来流。

v2 权重对应原冻结参考。在匹配试验中可显式选择 v2 权重以重放同样的阵风，其阻力倍率恒为 1。采用分组倍率的代理参见 [v3 模型卡](x4-950-pinn-wind-randomized-v3.md)。

## 数据、预算与独立验收

开发集为 12,000 训练对和 3,000 验证对。按共享风条件的 10 对状态分组，风速分层覆盖弱风、常用区间、准确边界及不同风向，检查两分支六维输入无跨区重复。比较嵌套的 6,000/12,000 对、1,200/2,400 步、全局或低风加权目标及三个初始化种子。按开发验证最差门槛比值选择 **6,000 对、全局损失、seed 43、2,400 步**。使用原 6→64→64→3 Tanh 残差以及二次阻力先验，归一化仅来自该预算的训练对。

正式测试的 3,000 对样本（300 组）使用独立 seed 9253，依据已冻结的模型、旧权重及门槛生成。协议 SHA256 为 `fc49c14b350b67e9ce6a42cd34f7faa720775fd0e4d659e55d9506b03081472a`。候选 SHA256 为 `43c44e6b7ea509d0e54206aad8678cd8b5384aa69171ca77f9b89e278b8f89ba`，冻结参考内容 SHA256 为 `03702b592b8b936b44b7e3c166759feb12aba89c980f0be714794b1e35f36cd5`。

| 工况 | 对数 | 原始 XY 向量误差 P95 (m/s²) | 归一化向量 RMSE |
| --- | ---: | ---: | ---: |
| 全域 | 3,000 | 0.004272 | 0.401% |
| 0–3 m/s 非零风 | 1,420 | 0.002872 | 1.496% |
| 上一行同用例，v1 | 1,420 | 0.002947 | 2.036% |
| (0,0.25) m/s | 340 | 0.000415 | 4.382% |
| 10 m/s | 330 | 0.008689 | 0.357% |

全域 XY 每轴 RMSE 为 0.001595 / 0.001574 m/s²，向量 P99 为 0.008219，最大值 0.017491 m/s²。所有预先冻结的阈值通过，最接近门槛的是 0–3 m/s 向量 P95，门槛 0.003 m/s²。弱风绝对门槛为 P95≤0.001 m/s²，旧权重同组 P95 为 0.000341 m/s²。近匹配来流的 8 个挑战对参考 RMS 只有约 0.000684 m/s²，归一化误差 38.7%，P95 0.000383 m/s²。240 对零风增量和 3,000 个相同输入的原始差分精确为零。本次机器证据保存在 [独立测试报告](../../outputs/pinn/x4_950_wind_10_v2/formal_test/evaluation.json)。

实际编译模型和生产后端重新扫描全部 3,000 对。两条路径的实际 Fz 与手算 EMA/限幅/质量乘法差异均为零。离线、预热、CPU 单线程、含整个施力函数的 P50/P95/P99 为 PINN 0.588/0.663/0.805 ms、解析 0.651/0.699/0.881 ms。该计时范围包括校验、预测和施力写入，省略日志、MuJoCo 物理步和实时调度。 [生产施力离线检查](../../outputs/pinn/x4_950_wind10_runtime_2026-11-19/offline_check.json)

短时真实 SITL 中，无风和 +X 方向至 10 m/s 的配对门槛通过。+Y 10 m/s 首轮及补测合计 3 对，2 对水平 RMSE 超过替代门槛。后续方向和轨迹在该级停止。 [SITL 闭环结果](../X4_PINN_WIND_10_SITL_2026-11-19.md)

宽域接入时全量项目离线回归 239 项通过，SITL 驱动离线专项 4 项通过。190 个历史受保护文件与既有指纹一致，包括 v1 权重、旧模型与所列 PX4 文件。

## 显式选择 v2

PX4 SITL 启动后，在第二终端从项目根目录启动仿真。

```bash
conda activate env_uav
python scripts/run_simulation.py --pinn-checkpoint \
  src/wind_field_simulation/assets/checkpoints/x4_950_wind_10_v2/candidate.pt
```

权重和参考随包存放在 `src/wind_field_simulation/assets/checkpoints/x4_950_wind_10_v2/`，四份文件与通过验收的产物字节一致。纯解析宽域对照可用 `--wind-backend analytic --reference-config src/wind_field_simulation/assets/checkpoints/x4_950_wind_10_v2/reference_config.json`。切换到其他版本的 X4 权重时，使用 `--pinn-checkpoint` 和 `--reference-config` 指定配对文件。CLI 的 `--wind-x/y` 仍只预填草稿，需要单独 Apply 和开启风扰。

## 复现

使用新目录。以下命令重放 9253 批次的生成和评估。修改候选后的独立验收需要在配置中预先冻结尚未打开的测试 seed 和门槛。本轮冻结步骤在生成测试标签之前记录了候选哈希。重训时文件中的耗时或来源指纹可以变化，以张量和误差复现为准。

```bash
conda activate env_uav
WIDE_RUN=outputs/pinn/x4_950_wind_10_reproduce_01

python scripts/run_x4_wind_10_study.py export \
  --output-dir "$WIDE_RUN/reference"
python scripts/run_x4_wind_10_study.py prepare \
  --reference "$WIDE_RUN/reference/reference_config.json" \
  --output-dir "$WIDE_RUN/development"
python scripts/run_x4_wind_10_study.py train \
  --manifest "$WIDE_RUN/development/manifest.json" \
  --output-dir "$WIDE_RUN/training"
python scripts/evaluate_x4_wind_10_study.py freeze \
  --manifest "$WIDE_RUN/development/manifest.json" \
  --checkpoint "$WIDE_RUN/training/candidate.pt" \
  --protocol "$WIDE_RUN/evaluation_protocol.json"
python scripts/evaluate_x4_wind_10_study.py test \
  --protocol "$WIDE_RUN/evaluation_protocol.json"
```

开发集生成后补充了运行时元数据校验，正式测试冻结前用当时的源码重生成并逐项核对 21 个开发数组，结果完全一致。原正式测试的 11 份源文件按指纹归档。当前复现入口增加了门槛和样本组校验，冻结时用实际加载的权重重算开发验证。原权重、原正式协议与原测试产物保留。 [冻结时的源码](../../outputs/pinn/x4_950_wind_10_v2/frozen_evaluation_sources)

见 [收敛审计](../X4_PINN_CONVERGENCE_2026-11-19.md) 和 [闭环对照协议](../../outputs/closed_loop/x4_wind_10_2026-11-19/README.md)。
