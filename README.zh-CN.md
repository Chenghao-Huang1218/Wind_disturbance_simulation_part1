# Wind_disturbance_simulation_part1

[English](README.md) | **简体中文**

基于 MuJoCo 与 PX4 SITL 的单无人机风扰动仿真，使用搭载 Airbot Play 机械臂的 X4-950，支持 PINN 气动力预测与交互式轨迹控制。

## 演示

| X4-950 + Airbot Play 模型 | 控制面板 |
| --- | --- |
| ![Quadrotor and arm in MuJoCo](docs/media/images/MJCF%20model.png) | ![Flight, trajectory and horizontal wind controls](docs/media/images/Control%20UI.png) |

| 演示视频（动态预览） | 圆形轨迹跟踪 |
| --- | --- |
| [<img src="docs/media/videos/wind-field-simulation-preview.gif" width="420" alt="Animated wind simulation preview; click for the full MP4">](docs/media/videos/Wind%20field%20simulation%20Demo.mp4) | <img src="docs/media/images/Motion%20tracking.png" width="420" alt="Example target and actual XY trajectories"> |
| [播放 / 下载完整 MP4（约 66 秒，20.6 MB）](docs/media/videos/Wind%20field%20simulation%20Demo.mp4) | 灰色虚线：目标；红色：实际轨迹 |


## 功能

- PX4 闭环飞行，支持 XYZ 目标与八字、椭圆、圆形、方形轨迹。
- 可调的均匀水平风，默认合风速范围 0–10 m/s。
- 使用预训练 X4-950 PINN，将风输入转换为附加气动力。
- MuJoCo 场景、控制面板和轨迹对比图。

## 目录结构

~~~text
Wind_disturbance_simulation_part1/
├── src/
│   ├── wind_field_simulation/
│   │   ├── simulation.py       # MuJoCo、飞行控制与面板
│   │   ├── wind_backends.py    # PINN 与解析受力接口
│   │   ├── training/           # 参考数据、训练与评估
│   │   └── assets/             # MuJoCo 模型与资源
│   │       ├── scenes/default/
│   │       │   └── robot_quadrotor_950_with_arm.xml
│   │       ├── robot_models/
│   │       │   ├── quadrotor_950/
│   │       │   │   ├── quadrotor_950.xml
│   │       │   │   └── meshes/
│   │       │   ├── airbot_play/
│   │       │   │   ├── airbot_play.xml
│   │       │   │   └── meshes/
│   │       │   └── scenes/
│   │       │       └── ground.xml
│   │       ├── checkpoints/     # 预训练权重
│   │       └── configs/         # PX4 参数
│   └── pymavlink/              # 内置 MAVLink 库
├── configs/pinn/               # 训练配置
├── runtime/px4/                # 预编译 PX4 SITL
├── scripts/                    # 启动与实验入口
├── tests/                      # 离线检查
├── docs/                       # 安装说明、报告与演示素材
├── licenses/                   # 第三方许可证
├── outputs/                    # 生成的日志与结果
├── pyproject.toml
└── requirements.txt
~~~

MuJoCo 主场景入口：[robot_quadrotor_950_with_arm.xml](src/wind_field_simulation/assets/scenes/default/robot_quadrotor_950_with_arm.xml)。

## 安装

验证环境：**Ubuntu 24.04 x86-64 / Python 3.12**，MuJoCo 3.6.0、NumPy 2.4.4、PyTorch 2.11.0、Matplotlib 3.10.9；图形界面需要 Tk 与 OpenGL。

PX4 运行包需要 Gazebo Transport 13、Messages 10、Protobuf 等系统库。按[安装说明](docs/INSTALL.zh-CN.md)补齐依赖，启动前确认 **ldd** 无 “not found”。

~~~bash
git clone https://github.com/Chenghao-Huang1218/Wind_disturbance_simulation_part1.git
cd Wind_disturbance_simulation_part1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
ldd runtime/px4/bin/px4
~~~

## 启动

在仓库根目录打开两个终端。

**终端 1：PX4 SITL**

~~~bash
source .venv/bin/activate
bash scripts/run_px4_sitl.sh
~~~

**终端 2：MuJoCo 与控制面板**

~~~bash
source .venv/bin/activate
python scripts/run_simulation.py
~~~

模型权重与 PX4 程序已随仓库提供，正常运行无需重新训练或编译。

## 控制面板

1. 等待 **PX4: connected**，点击 **Unlock**，确认 **armed** 后点击 **Takeoff**。
2. 使用 **X± / Y± / Z±**，按 0.5 m 步长调整世界坐标目标，或选择预设轨迹。圆轨迹默认直径 1 m、周期 10 s。
3. 输入 X/Y 风速（m/s），点击 **Apply Wind**；点击 **Wind OFF** 开启风，**Wind ON** 关闭风。
4. 点击 **Land**，等待落地后点击 **Exit**，关闭仿真与启动器拥有的 PX4 进程。

| 控件 | 功能 |
| --- | --- |
| Figure-8 / Ellipse / Circle / Square | 八字 / 椭圆 / 圆形 / 方形轨迹 |
| Apply Wind | 应用 X/Y 新值，不自动开启风 |
| Set / Speed | 已应用分量 / 合风速 |
| Emergency | 立即停止仿真转子 |

初始风速输入为空。水平合风速满足 $\sqrt{w_x^2+w_y^2}\le10$ m/s，正值表示空气沿世界坐标正方向流动。竖直风速与实际竖直扰动力均为零，蓝色标记显示水平风向。

## 实现方法

默认 **pinn-x4** 后端（X4 PINN v3）在每步以同一飞机状态计算有风、无风的加速度差：

$$
\mathbf a_{\mathrm{wind}}^W
=P_{xy}R_{WB}\bigl[
g_\theta(R_{WB}^{T}(\mathbf v_O^W-\mathbf w^W),\boldsymbol\omega^B,\mathbf s)
-g_\theta(R_{WB}^{T}\mathbf v_O^W,\boldsymbol\omega^B,\mathbf s)
\bigr].
$$

$R_{WB}$ 将机体系映射到世界系，$\mathbf v_O^W$ 为机体原点速度，$\mathbf w^W$ 为风速，$P_{xy}=\mathrm{diag}(1,1,0)$ 保留水平加速度。

默认水平等效阻力面积为 **0.25 m²**。按参考工况计算的增益作用于世界坐标 XY 加速度，随后依次经过 EMA（$\alpha=0.05$）、各轴 ±6 m/s² 限幅，再乘整机质量 8.0004 kg，施加到根 body。

v3 使用机体系相对气速、机体系角速度和三组阻力倍率 $\mathbf s$，共九维。每组采用二次阻力先验与学习残差，再按倍率合成总响应。默认倍率均为 1、阵风幅值为 0。使用 `--drag-variation 0.20 --gust-amplitude 1.5 --gust-period 4 --wind-seed 42` 可启用分组阻力随机化与阵风。

### PINN v3 数据怎么生成

训练标签来自 29 个由 X4-950 与 Airbot Play 的 XML 提取的固定几何代理。对每个合成状态，用相同的根点速度、姿态、角速度、几何和阻力倍率计算两次，一次使用瞬时来流 $\mathbf w^W(t)$，一次将环境风设为零。对代理 $j$，先计算点速度在代理局部坐标系中的分量，再按轴计算二次阻力：

$$
\begin{aligned}
\mathbf u_j(\mathbf w)
&=Q_j^\mathsf{T}\!\left[R_{WB}^\mathsf{T}(\mathbf v_O^W-\mathbf w^W)
                          +\boldsymbol\omega^B\times\mathbf r_j^B\right],\\
\mathbf f_j(\mathbf w)
&=-\tfrac12\rho\,s_{g(j)}\,(\mathbf C_D\odot\mathbf A_j)
                   \odot\mathbf u_j(\mathbf w)\odot|\mathbf u_j(\mathbf w)|,\\
\mathbf F^W(\mathbf w)&=R_{WB}\sum_j Q_j\mathbf f_j(\mathbf w),\\
\mathbf y_{xy}^W&=P_{xy}\,[\mathbf F^W(\mathbf w(t))-\mathbf F^W(\mathbf 0)]/m .
\end{aligned}
$$

$Q_j$ 将代理局部坐标映射到机体系，$\mathbf r_j^B$ 是根点到代理中心的偏移，$\mathbf A_j$ 是局部三轴参考面积，$\odot$ 表示逐元素相乘。数据保存两分支的完整三轴力、各组单位倍率响应和水平加速度增量。名义 $C_DA=0.25\,\mathrm{m^2}$ 的换算在运行时施加一次，标签生成在该换算、滤波及限幅之前。

在原来的固定参数公式上加入两类随机量：

| 随机量 | 抽样及进入公式的方式 |
| --- | --- |
| 三组阻力倍率 $s_{g(j)}$ | 机身、机臂及电机、机械臂各抽一个倍率，基础分布为 $\mathcal U(0.8,1.2)$，对组内全部代理的 $C_D A$ 同比例缩放。每个过程和有风／无风配对共用同一组倍率。 |
| 水平阵风 | 有阵风的过程抽取峰值 $G\sim\mathcal U(0.5,2.0)$ m/s、脉冲周期 $T\sim\mathcal U(2,6)$ s，以及每个脉冲的方向 $\phi_k\sim\mathcal U(0,2\pi)$。脉冲内已过时间为 $\tau=t\bmod T$，在基准风上叠加 $G\sin^2(\pi\tau/T)(\cos\phi_k,\sin\phi_k,0)$，再将水平合风速约束在 10 m/s 内。 |

$\rho=1.225\,\mathrm{kg/m^3}$、三轴基准 $C_D=1$、代理几何和质量固定。基准风按风速和风向分层，保留零风、弱风及 10 m/s 边界。状态抽样包含逆风和随风运动。每 10 对共用一个风过程及倍率抽样，按整组分成 **30,000 训练／6,000 验证／6,000 测试对**，种子分别为 **17653／18659／19663**。训练和验证排除三个倍率同时距 1 至少 0.12 的联合角落，测试单独保留 1,500 个这样的样本对。数据是参数化解析模型生成的合成样本。[v3 复现命令和验证结果](docs/models/x4-950-pinn-wind-randomized-v3.md)

<details>
<summary>PINN 训练图与技术文档</summary>

![PINN 试验训练与留出集误差](docs/figures/x4-950-pinn-pilot-v1.png)

- [参考模型](docs/x4-950-reference-v1.md)
- [v2 宽域基线与复现](docs/models/x4-950-pinn-wind-10-v2.md)
- [v1 模型卡](docs/models/x4-950-pinn-pilot-v1.md)
- [训练报告](docs/X4_PINN_PILOT_2026-11-10.md)
- [SITL 集成报告](docs/X4_PINN_INTEGRATION_2026-11-10.md)
- [PX4 运行包与源码来源](docs/px4.md)

</details>
