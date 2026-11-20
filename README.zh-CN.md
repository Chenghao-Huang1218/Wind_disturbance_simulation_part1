# Wind_disturbance_simulation_part1

[English](README.md) | **简体中文**

基于 MuJoCo 与 PX4 SITL 的单无人机风扰动仿真，使用搭载 Airbot Play 机械臂的 X4-950，支持 PINN 气动力预测与交互式轨迹控制。

## 仿真演示

https://github.com/user-attachments/assets/5404e7aa-410a-489f-a724-ee3c1227da1d

[下载原始 MP4（约 66 秒）](docs/media/videos/Wind%20field%20simulation%20Demo.mp4)

| X4-950 + Airbot Play 模型 | 控制面板 |
| --- | --- |
| ![Quadrotor and arm in MuJoCo](docs/media/images/MJCF%20model.png) | ![Flight, trajectory and horizontal wind controls](docs/media/images/Control%20UI.png) |

**圆形轨迹跟踪**

<img src="docs/media/images/Motion%20tracking.png" width="420" alt="圆轨迹目标与实际 XY 轨迹示例">

灰色虚线为目标轨迹，红线为实际轨迹。


## 仿真功能

- PX4 闭环飞行，支持 XYZ 目标与八字、椭圆、圆形、方形轨迹。
- 可调的均匀水平风，默认合风速范围 0–10 m/s。
- 使用预训练 X4-950 PINN，将风输入转换为附加气动力。
- MuJoCo 场景、控制面板和轨迹对比图。

## 项目结构

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

## 仿真启动

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

## 气动模型

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

v3 使用机体系相对气速、机体系角速度和三组阻力倍率 $\mathbf s$，共九维。每组采用二次阻力先验与学习残差，再按倍率合成总响应。随包 v3 默认开启三组阻力倍率随机化（±20%，各组范围 [0.8, 1.2]）与水平阵风（峰值 1.5 m/s、周期 4 s、种子 42）。倍率每次启动抽取一次，同种子可复现。在面板点击 **Apply Wind** 并开启风扰后，阵风叠加到设定的基准风上。需要名义阻力和恒定来流时，可同时传入 `--drag-variation 0 --gust-amplitude 0`。

### v3 合成数据生成

#### 原始解析模型

二次阻力的标量形式为 $D=\tfrac12\rho V^2 C_D A$，其中 $V$ 是相对气速。本项目在固定几何的 29 个部件代理上逐轴应用该模型。代理 $j$ 的局部相对气速 $\mathbf u_j$ 与名义局部阻力 $\mathbf f_{j,0}$ 为：

$$
\begin{aligned}
\mathbf u_j(\mathbf w^W)
 &=Q_j^{\mathsf T}\left[R_{WB}^{\mathsf T}(\mathbf v_O^W-\mathbf w^W)
   +\boldsymbol\omega^B\times\mathbf r_j^B\right],\\
\mathbf f_{j,0}(\mathbf w^W)
 &=-\tfrac12\rho\,(\mathbf C_D\odot\mathbf A_j)
     \odot\mathbf u_j\odot|\mathbf u_j|.
\end{aligned}
$$

$Q_j$ 将代理局部坐标映射到机体系，$\mathbf r_j^B$ 是根点到代理中心的偏移，$\mathbf A_j$ 是局部三轴参考面积。$\odot$ 和绝对值均按分量计算。将各代理阻力旋转到世界系并求和，对同一状态取有风和无风之差，得到水平扰动加速度。v3 引入组倍率 $\mathbf s$ 后，计算式为：

$$
\begin{aligned}
\mathbf F^W(\mathbf w^W;\mathbf s)
 &= R_{WB}\sum_{j=1}^{29}Q_j\,s_{g(j)}\,\mathbf f_{j,0}(\mathbf w^W),\\
\mathbf y_{xy}^W
 &=\frac{P_{xy}}{m}
   \left[\mathbf F^W(\mathbf w^W(t);\mathbf s)-\mathbf F^W(\mathbf 0;\mathbf s)\right].
\end{aligned}
$$

$g(j)$ 是代理所属的阻力组。取 $\mathbf s=(1,1,1)$ 即得原始固定参数模型。两分支共享根点速度、姿态、角速度、几何和倍率，分别保存完整三轴力。数据还保存各组单位倍率响应与配对加速度增量。[二次阻力公式](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/) · [项目解析参考](docs/x4-950-reference-v1.md)

#### 关键参数

| 参数 | 取值与定义 |
| --- | --- |
| 空气密度 $\rho$ | 1.225 kg/m³ |
| 代理阻力系数 $\mathbf C_D$ | $(1,1,1)$，三个局部轴的基准值 |
| 代理参考面积 $\mathbf A_j$ | 由 X4-950 与 Airbot Play 的 XML 几何导出，各代理采用各自的三轴面积 |
| 整机质量 $m$ | 8.0004 kg |
| 水平合风速 | 0–10 m/s，竖直分量为零 |
| 解析参考等效阻力面积 $(C_DA)_{\mathrm{ref}}$ | 约 0.11951 m²，由冻结几何求得 |
| 仿真名义等效阻力面积 $(C_DA)_{\mathrm{nom}}$ | 0.25 m²，用于运行时水平施力 |

等效阻力面积以机体静止、水平、零航向，来风为 $(4,4,0)$ m/s、倍率均为 1 且无阵风的工况定义。该工况下，以沿风力除以动压 $\tfrac12\rho V^2$ 得到 $(C_DA)_{\mathrm{ref}}$。运行时将水平增量乘以 $0.25/0.11951\approx2.0919$，名义沿风力为 4.90 N。训练标签保留增益、滤波和限幅前的解析结果。

#### 随机化与样本划分

| 随机量 | 抽样与作用 |
| --- | --- |
| 三组阻力倍率 $s_g$ | 机身、机臂及电机、机械臂各从基础分布 $\mathcal U(0.8,1.2)$ 抽取一个倍率，对组内代理的阻力等比例缩放。倍率在同一风过程及有风／无风配对内固定。 |
| 水平阵风 | 有阵风的过程抽取峰值 $G\sim\mathcal U(0.5,2.0)$ m/s 和周期 $T\sim\mathcal U(2,6)$ s。每个脉冲抽取方向 $\phi_k\sim\mathcal U(0,2\pi)$，向基准风叠加 $G\sin^2(\pi\tau/T)(\cos\phi_k,\sin\phi_k,0)$，其中 $\tau=t\bmod T$。叠加后限制水平合风速至 10 m/s。 |

基准风按风速和风向分层，覆盖零风、弱风和上限工况。每 10 对样本共享一个风过程和一组倍率，按整组划分为 **30,000 训练／6,000 验证／6,000 测试对**，种子为 **17653／18659／19663**。训练和验证排除三组倍率均偏离 1 至少 0.12 的组合，测试集中有 1,500 对样本覆盖这些未见组合。标签由参数化解析模型合成。[v3 复现与验证](docs/models/x4-950-pinn-wind-randomized-v3.md)

<details>
<summary>模型验证与复现</summary>

![PINN 试验训练与留出集误差](docs/figures/x4-950-pinn-pilot-v1.png)

- [参考模型](docs/x4-950-reference-v1.md)
- [v2 宽域基线与复现](docs/models/x4-950-pinn-wind-10-v2.md)
- [v1 模型卡](docs/models/x4-950-pinn-pilot-v1.md)
- [训练报告](docs/X4_PINN_PILOT_2026-11-10.md)
- [SITL 集成报告](docs/X4_PINN_INTEGRATION_2026-11-10.md)
- [PX4 运行包与源码来源](docs/px4.md)

</details>
