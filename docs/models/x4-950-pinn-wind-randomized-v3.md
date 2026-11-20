# X4-950 PINN v3 分组阻力与阵风

2026-11-20。v3 是 X4-950 与 Airbot Play 固定几何下的九输入气动力代理，由默认 `pinn-x4` 加载。无参数启动采用三个单位倍率和零阵风幅值。[v2](x4-950-pinn-wind-10-v2.md) 与 [v1](x4-950-pinn-pilot-v1.md) 可显式加载。

## 输入、输出与施力

| 项目 | 定义 |
| --- | --- |
| 世界系环境风 | 水平合风速 0–10 m/s，竖直分量为零 |
| 单分支九维输入 | 机体系根点相对气速 XYZ、机体系角速度 XYZ、机身、机臂及电机、机械臂三个倍率 |
| 单分支输出 | 机体系完整气动加速度 XYZ，单位 m/s² |
| 生产施力 | 有风减无风，转世界系并投影到 XY，乘一次名义增益，再过 EMA 与逐轴限幅，乘整机质量后施加给根 body |

两分支共用根点速度、姿态、角速度及三个倍率，仅改变环境风。无风时同态差分精确为零。状态域沿用 v2，包含根点水平速度 ≤3 m/s、垂直速度 ±1 m/s、roll/pitch 各 ±20°、全 yaw 和角速度模长 ≤1.5 rad/s。运行时记录超域状态。v3 支持每组倍率在 [0.8, 1.2] 内按独立均匀抽样生成，每次运行固定。

名义参考为机体静止、水平、零航向、无阵风、世界系来风 [4, 4, 0] m/s。在这个参考点，\(C_DA=0.25\,\mathrm{m^2}\) 对应沿风力 4.90 N，世界系 XY 增益为 2.0918753。倍率改变单次试验的等效阻力面积，名义增益以同一冻结参考计算。标签保留增益前的力。生产路径使用 EMA α=0.05、逐轴 ±6 m/s² 限幅及 8.0004 kg 的整机质量。

每个组的单位倍率响应为二次阻力先验加神经网络残差。网络对六个运动变量使用 6→64→64→9 的 Tanh MLP，输出三个三轴残差，随后按其物理倍率加权求和。这个结构保留解析模型中阻力随代理面积线性变化的性质，同时学习各组响应随相对气速和角速度变化的部分。六个运动输入和各组目标尺度仅用训练数据拟合。三个倍率按物理单位直接参与加权。训练损失联合监督三个组在两个分支的完整加速度和按倍率合成的世界系水平同态增量。

## 合成数据与训练

训练、验证、测试分别为 30,000／6,000／6,000 对，抽样种子为 17653／18659／19663。每组 10 对共享一套固定倍率、一条基准风和一个阵风种子。三套数据使用独立种子空间和组号。阵风组的峰值在 [0.5, 2.0] m/s 均匀抽样，脉冲周期在 [2, 6] s 均匀抽样。每组覆盖三个脉冲内的十个固定相位，每个相位配一个合成状态。样本用于同态查询，状态来自 v2 分层抽样，并非模拟飞行时间序列。阵风调用生产后端的 `HorizontalGust`，用平滑抬升余弦窗口和种子方向，合风速上限为 10 m/s。部分组采用恒风，包括零风与名义 10 m/s 组。

训练和验证排除三个倍率都满足 |倍率−1| ≥0.12 的联合角落。在测试中预留 150 个完整组覆盖这些角落，并核对精确倍率组合、种子、组号无跨区重复。测试还有 389 组启用阵风、540 对实际零风、298 对低于 0.25 m/s 的非零风、686 对在 10 m/s 边界。标签来自宽域 29 代理冻结解析几何，采用运行时相同的代理分组。有风和无风均保存机体系相对气速、逐组加速度、机体系及世界系完整力和增量。基准风、实际风、脉冲参数、已过时间及种子均可回放。归档中的 `speed_stratum`、`direction_sector` 和 `challenge_id` 描述基准风抽样，评估按实际风速分组。

CPU PyTorch 2.11.0+cu128 单线程、float32，NumPy 2.4.4，训练种子 47，AdamW 6,000 步，每步 256 对，余弦学习率由 0.003 降到 0.00015。按验证集增益前 XY 向量误差 P95 选模型。选中第 6,000 步，P95 为 0.0034133 m/s²。测试生成前写入权重、参考、开发清单和门槛的 SHA-256。预设门槛为全域水平力 P95 ≤0.15 N、NRMSE ≤2%、逐组反事实 NRMSE ≤8%，并要求零风精确、生产施力垂直分量精确为零，误差低于 v2 与固定倍率消融。

## 独立测试

力误差是同一状态及瞬时风下，相对于随机化解析标签的水平力增量误差，已经乘名义增益，处于 EMA 和限幅之前。NRMSE 用该子集的解析力向量 RMS 归一化。

| 工况 | 对数 | 解析 RMSE (N) | v2 RMSE (N) | v3 RMSE (N) | v3 P95 (N) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全域 | 6,000 | 0 | 0.70227 | 0.02926 | 0.05933 |
| 未见倍率联合角落 | 1,500 | 0 | 0.91396 | 0.02955 | 0.05901 |
| 带阵风的过程 | 3,890 | 0 | 0.59121 | 0.02859 | 0.05702 |
| 0–0.25 m/s 非零风 | 298 | 0 | 0.00703 | 0.00391 | 0.00862 |
| 10 m/s 边界 | 686 | 0 | 1.45475 | 0.04690 | 0.09226 |
| 零风 | 540 | 0 | 0 | 0 | 0 |

全域 NRMSE 为 v2 的 7.56% 和 v3 的 0.315%。v3 最大单对力误差为 0.234 N。v3 强制使用 [1, 1, 1] 时，全域 RMSE 回到 0.70084 N，联合角落升至 0.91359 N。保持每个测试状态、瞬时来流与其余倍率固定，只将其中一组从 0.8 改为 1.2，得到以下力差。

| 倍率组 | 解析力差 RMS (N) | v3 力差误差 RMSE (N) | NRMSE |
| --- | ---: | ---: | ---: |
| 机身 | 1.81661 | 0.00634 | 0.349% |
| 机臂及电机 | 0.60580 | 0.00462 | 0.763% |
| 机械臂 | 1.29415 | 0.00693 | 0.536% |

三组倍率均对输出产生与解析参考一致的可测响应。消融结果说明，本轮从 v2 到 v3 的主要降误差来源是输入真实倍率。各组误差表征教师模型近似，不能用来判断实机上三个倍率的可辨识性。

### 完整生产施力路径计时

在 Intel Core i7-14650HX 上，CPU PyTorch 单线程。使用生产 `_apply_wind_force` 回放八条留出阵风过程，每后端预热后计时 3,200 次。每次包含取 MuJoCo 状态、阵风求值、两分支预测、投影和增益、EMA、限幅、乘质量及写入外力，且从同一回放状态和风开始。计时范围止于施力函数返回。十个归档相位循环回放，脉冲方向缓存更新频率高于连续5 ms 仿真步。连续闭环计时另列。所有应用的 Fz 均为零。独立生产解析回放与归档原始全轴标签的最大数值差为 3.33×10⁻¹⁵ N。

| 后端 | P50 (ms) | P95 (ms) | P99 (ms) | 滤波后 XY 施力 RMSE (N) |
| --- | ---: | ---: | ---: | ---: |
| 随机化解析 | 0.707 | 0.791 | 0.842 | 0 |
| 六维 v2 | 0.624 | 0.734 | 0.826 | 0.33107 |
| 九维 v3 | 0.668 | 0.781 | 0.891 | 0.01160 |

v3 的中位耗时略低，P99 更高。这组时延不足以证明 v3 在同一生产路径上有稳定的提速收益。 独立测试与计时 JSON `outputs/pinn/x4_950_wind_randomized_v3/test/evaluation.json`

## PX4 SITL 匹配闭环

每个后端分别以种子 42、43、44 重启 PX4 和 MuJoCo，基准风为 [4, 4, 0]、[-6, 4, 0]、[0, 8.5, 0] m/s。每组共用 ±20% 抽样、1.5 m/s 阵风峰值及 4 s 周期。进入 FLYING 后 8 个仿真秒开启风，开启后 10 s 启动直径 1 m、周期 10 s 的单圈轨迹。三条路径使用相同的冻结参考、目标轨迹时序、名义增益、EMA 和限幅。v2 按原模型使用名义阻力，实际来流与同组解析和 v3 一致。PX4 使用 MuJoCo 提供的真值状态反馈。

9 次均完成全圈、落地、上锁和受管理退出。每次全圈包含 500 个连续有效样本。记录中未出现域外状态或限幅，全部物理步实际 Fz 为零。全部带风样本可由各自的已过时间和种子逐样本回放。每种风况各后端一次独立重启。

| 种子 | 解析 XY 跟踪 RMSE (m) | v2 (m) | v3 (m) | v3 − 解析 (mm) | 施加水平力 RMS，解析／v2／v3 (N) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.376 | 0.361 | 0.367 | −8.13 | 4.725／4.295／4.727 |
| 43 | 0.323 | 0.333 | 0.322 | −0.95 | 7.633／8.125／7.630 |
| 44 | 0.312 | 0.794 | 0.331 | +19.11 | 15.235／17.310／15.273 |

在种子 44 的工况，v2 轨迹水平峰值误差为 1.916 m，解析和 v3 分别为 0.454 m、0.537 m。该组 v2 的施力 RMS 比解析高 13.6%。三组中 v3 与解析水平施力 RMS 的绝对差均小于 0.039 N。轨迹误差衡量 PX4 对各后端施加风扰的闭环响应，同态力精度由独立测试衡量。单次重启对照未估计闭环重复性，也未做统计等效检验。

连续仿真中，各后端三次全圈采样的完整施力 P95 范围是解析 0.998–1.563 ms、v2 1.046–1.205 ms、v3 1.021–1.227 ms。种子 43 下 v3 的 P95 高于解析。实测同时受到运行时调度影响。 闭环汇总与逐次收据 `outputs/closed_loop/x4_wind_randomized_v3/summary.json`

## 运行与复现

常规双终端启动。PX4 终端先运行 `bash scripts/run_px4_sitl.sh`。仿真终端从项目根目录启动以下命令，填写或保留预填来流后点击 **Apply Wind**，然后开启 **Wind ON**。

```bash
conda activate env_uav
V3=src/wind_field_simulation/assets/checkpoints/x4_950_wind_randomized_v3/candidate.pt
python scripts/run_simulation.py \
  --drag-variation 0.20 --gust-amplitude 1.5 --gust-period 4 --wind-seed 42 \
  --wind-x 4 --wind-y 4
```

解析对照使用 `--wind-backend analytic` 和同一组风参数。六维对照显式传入 v2 的 `candidate.pt`，以同一瞬时来流运行，内部倍率固定为 [1, 1, 1]。运行日志分别记录试验抽样值 `experiment_drag_group_scales` 和模型实际倍率 `drag_group_scales`，v2 的后一字段为空。

复现匹配闭环需释放所选本机 SITL 端口。下例逐次运行九个组合，使用项目随包 PX4 二进制和同一份仿真参数种子。每次运行保存生产 50 Hz 日志、PX4 控制台、ULog 及闭环指标。

```bash
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SITL_RUN=outputs/closed_loop/x4_wind_randomized_v3_reproduce_01
for seed in 42 43 44; do
  for method in analytic v2 v3; do
    python scripts/run_x4_wind_randomized_sitl.py --method "$method" --wind-seed "$seed" \
      --checkpoint "$V3" --instance "$((seed - 35))" --output-dir "$SITL_RUN/${seed}_${method}"
  done
done
```

PX4 二进制 SHA-256 为 `db1c803bb28077856f8989b059c12421e0c8d3815be69aa3bb81b3262ae735ca`，随项目提供的参数种子 SHA-256 为 `d26c2bbaf50b6ae5fbd9f0e81e8e1d5ce08e30901706e83a77809b1c6d0ce9da`。 [参数种子](../../configs/pinn/x4_950_sitl_seed_parameters.bson)

下列数据归档和运行日志由复现命令在本地生成。

在新目录中复现数据、训练和冻结评估。评估阶段生成测试数据并写入测试数据哈希。路径和计时可随运行环境变化，模型张量和误差按记录的库版本比对。

```bash
conda activate env_uav
export MPLBACKEND=Agg
RUN=outputs/pinn/x4_950_wind_randomized_v3_reproduce_01
python scripts/run_x4_wind_randomized_study.py prepare --output-dir "$RUN/development"
python scripts/run_x4_wind_randomized_study.py train --steps 6000 \
  --manifest "$RUN/development/manifest.json" --output-dir "$RUN/training"
python scripts/run_x4_wind_randomized_study.py evaluate \
  --manifest "$RUN/development/manifest.json" --checkpoint "$RUN/training/candidate.pt" \
  --output-dir "$RUN/test"
```

| 产物 | SHA-256 |
| --- | --- |
| 训练数据 `outputs/pinn/x4_950_wind_randomized_v3/development/train.npz` | `8b193675a883ab03fcb917fbdad3e04044f88c574eb34000d26d75ccab28cc9b` |
| 验证数据 `outputs/pinn/x4_950_wind_randomized_v3/development/validation.npz` | `76061cd55fb71347f8954f8aef389ca4ca2e8abe6ca4c76a97852606e4d75b24` |
| 开发清单 `outputs/pinn/x4_950_wind_randomized_v3/development/manifest.json` | `4f09151edbef8dfc6528cd7292aa1133c51cf08aa035da0fda4a50d68de404f3` |
| [随包 v3 权重](../../src/wind_field_simulation/assets/checkpoints/x4_950_wind_randomized_v3/candidate.pt) | `37124dd930a990cef6bd8baa3137d72f2df28a771a113847d844e8574b32891c` |
| 冻结测试协议 `outputs/pinn/x4_950_wind_randomized_v3/test/protocol.json` | `52182cc937afa53ce552490d49a304c8fe3bf8f4a145ee3405b9ca04096ebb12` |
| 独立测试数据 `outputs/pinn/x4_950_wind_randomized_v3/test/test.npz` | `d67b8a166ce75d4767698dd1436a0ad1036a93d38aec457405c6d0b02630a544` |

独立重跑生成与训练后，训练和验证归档、开发清单及候选权重均与原产物逐字节一致。重新生成的独立测试归档也逐字节一致。项目离线回归为 **260 项通过**。

冻结参考内容指纹为 `03702b592b8b936b44b7e3c166759feb12aba89c980f0be714794b1e35f36cd5`，使用随包 v2 参考、车辆参数和几何清单。来源及验证轨迹记录见 开发清单 `outputs/pinn/x4_950_wind_randomized_v3/development/manifest.json` 和 训练报告 `outputs/pinn/x4_950_wind_randomized_v3/training/training.json`。

## 何时有实际价值

当前任务的解析模型直接求出生成标签的力，且单步力只依赖当前状态、瞬时来流和已知倍率。只需要复现这 29 个代理的合成阻力时，解析后端给出零近似误差，生产链路耗时相近。v3 的直接收益是以现有 PINN 接口代替固定倍率的 v2，并在同一个代理上重放多种倍率。

更有实际价值的情形是高保真气动求解器超出在线周期，代理在目标硬件、目标批量和误差预算下实测更快，或加入了独立测得的残差，在留出风况中比校准后的准定常解析基线更准确。若要验证非定常预测，应在带桨整机、已知机械臂构型下同步记录三维来流、旋翼转速、姿态和六分量力／力矩，扫过方向、频率和幅值，留出完整阵风试次与构型。对瞬时输入相同、来流前史不同的样本比较力的幅值、相位及滞后。若能复现历史效应，需要引入可测的历史或动力学状态，并在独立试验上与无记忆模型对照。高保真非定常 CFD 可补充覆盖，需由测力数据验证。三个倍率若要用于实机，也需要有独立标定或估计证据。 [NASA 整机测力试验](https://rotorcraft.arc.nasa.gov/Research/Programs/UAS_Performance_Test.html) · [主动格栅风洞的阵风与来流测量](https://amt.copernicus.org/articles/17/4941/2024/index.html)
