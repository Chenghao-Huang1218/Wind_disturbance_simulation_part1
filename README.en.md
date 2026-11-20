# Wind_disturbance_simulation_part1

**English** | [简体中文](README.zh-CN.md)

Single-UAV wind disturbance simulation in MuJoCo with PX4 SITL, a PINN aerodynamic surrogate, and interactive trajectory control. The aircraft is an X4-950 with an Airbot Play arm.

## Demo

| X4-950 + Airbot Play model | Interactive control panel |
| --- | --- |
| ![Quadrotor and arm in MuJoCo](docs/media/images/MJCF%20model.png) | ![Flight, trajectory and horizontal wind controls](docs/media/images/Control%20UI.png) |

| Demo video (animated preview) | Circular trajectory tracking example |
| --- | --- |
| [<img src="docs/media/videos/wind-field-simulation-preview.gif" width="420" alt="Animated wind simulation preview; click for the full MP4">](docs/media/videos/Wind%20field%20simulation%20Demo.mp4) | <img src="docs/media/images/Motion%20tracking.png" width="420" alt="Example target and actual XY trajectories"> |
| [Watch / download the full MP4 (approximately 66 seconds, 20.6 MB)](docs/media/videos/Wind%20field%20simulation%20Demo.mp4) | Dashed gray: target; red: actual trajectory |


## Features

- PX4 closed-loop flight with XYZ targets and Figure-8, Ellipse, Circle and Square trajectories.
- Adjustable, uniform horizontal wind from 0 to 10 m/s with the default backend.
- A pretrained X4-950 PINN converts wind into additional aerodynamic forces.
- MuJoCo visualization, a control panel and trajectory comparison plots.

## Project structure

~~~text
Wind_disturbance_simulation_part1/
├── src/
│   ├── wind_field_simulation/
│   │   ├── simulation.py       # MuJoCo, flight control and panel
│   │   ├── wind_backends.py    # PINN and analytical force adapters
│   │   ├── training/           # Reference data, training and evaluation
│   │   └── assets/             # MuJoCo models and assets
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
│   │       ├── checkpoints/     # Pretrained weights
│   │       └── configs/         # PX4 parameters
│   └── pymavlink/              # Bundled MAVLink library
├── configs/pinn/               # Training configurations
├── runtime/px4/                # Prebuilt PX4 SITL
├── scripts/                    # Launch and experiment entry points
├── tests/                      # Offline checks
├── docs/                       # Guides, reports and demo media
├── licenses/                   # Third-party licenses
├── outputs/                    # Generated logs and results
├── pyproject.toml
└── requirements.txt
~~~

MuJoCo scene entry: [robot_quadrotor_950_with_arm.xml](src/wind_field_simulation/assets/scenes/default/robot_quadrotor_950_with_arm.xml).

## Installation

Tested on **Ubuntu 24.04 x86-64 / Python 3.12**, with MuJoCo 3.6.0, NumPy 2.4.4, PyTorch 2.11.0 and Matplotlib 3.10.9. The GUI requires Tk and OpenGL.

The bundled PX4 runtime needs Gazebo Transport 13, Messages 10 and Protobuf libraries. Follow the [installation guide](docs/INSTALL.en.md); resolve every “not found” entry from **ldd** before launch.

~~~bash
git clone https://github.com/Chenghao-Huang1218/Wind_disturbance_simulation_part1.git
cd Wind_disturbance_simulation_part1
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
ldd runtime/px4/bin/px4
~~~

## Run

From the repository root, open two terminals.

**Terminal 1 — PX4 SITL**

~~~bash
source .venv/bin/activate
bash scripts/run_px4_sitl.sh
~~~

**Terminal 2 — MuJoCo and control panel**

~~~bash
source .venv/bin/activate
python scripts/run_simulation.py
~~~

The supplied checkpoint and PX4 binary are ready to use; normal operation requires no training or PX4 build.

## Control panel

1. Wait for **PX4: connected**. Click **Unlock**, wait for **armed**, then click **Takeoff**.
2. Use **X± / Y± / Z±** for 0.5 m world-frame target increments, or select a trajectory. Circle defaults to a 1 m diameter and a 10 s period.
3. Enter X/Y wind in m/s and click **Apply Wind**. Click **Wind OFF** to enable wind; **Wind ON** disables it.
4. Click **Land**, wait for touchdown, then **Exit**. Exit closes the simulation and the launcher's PX4 process.

| Control | Behavior |
| --- | --- |
| Figure-8 / Ellipse / Circle / Square | Predefined flight trajectories |
| Apply Wind | Accept new X/Y values; does not enable wind |
| Set / Speed | Accepted components / resultant wind speed |
| Emergency | Immediately stop simulated rotors |

Wind inputs start empty. The resultant speed must satisfy $\sqrt{w_x^2+w_y^2}\le10$ m/s; positive values mean air moving along the positive world axis. Wind Z and applied force Z are zero. Blue markers show horizontal wind direction.

## Model

Each step, the default **pinn-x4** backend (X4 PINN v3) evaluates wind and no-wind inputs at the same aircraft state:

$$
\mathbf a_{\mathrm{wind}}^W
=P_{xy}R_{WB}\bigl[
g_\theta(R_{WB}^{T}(\mathbf v_O^W-\mathbf w^W),\boldsymbol\omega^B,\mathbf s)
-g_\theta(R_{WB}^{T}\mathbf v_O^W,\boldsymbol\omega^B,\mathbf s)
\bigr].
$$

$R_{WB}$ maps body to world, $\mathbf v_O^W$ is root-point velocity, $\mathbf w^W$ is wind velocity, and $P_{xy}=\mathrm{diag}(1,1,0)$ retains horizontal acceleration.

The default horizontal equivalent drag area is **0.25 m²**. The gain derived at the reference condition scales the world-frame XY acceleration before it passes through an EMA ($\alpha=0.05$), per-axis clipping at ±6 m/s², and multiplication by the 8.0004 kg aircraft mass before application to the root body.

v3 takes nine inputs: body-frame relative air velocity, body angular rate, and three drag multipliers $\mathbf s$. It combines a quadratic prior and learned residuals per group, then sums the responses with the three multipliers. The bundled v3 defaults to three randomized drag multipliers (±20%, range [0.8, 1.2] each) and horizontal gusts with a 1.5 m/s peak, a 4 s period, and seed 42. The multipliers are drawn once at startup and replay with the same seed. After you click **Apply Wind** and switch wind on, gusts are added to the configured baseline wind. For nominal drag and steady inflow, pass `--drag-variation 0 --gust-amplitude 0`.

### How the PINN v3 dataset is generated

The synthetic labels come from 29 fixed geometric proxies extracted from the X4-950 and Airbot Play XML files. Each sampled state is evaluated twice with the same root velocity, attitude, angular rate, geometry, and drag multipliers. The wind branch uses the instantaneous inflow $\mathbf w^W(t)$ and the other branch sets environmental wind to zero. For proxy $j$, the model first resolves the local relative air velocity and then computes axis-wise quadratic drag:

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

$Q_j$ maps proxy axes to body axes, $\mathbf r_j^B$ is the offset from the root to the proxy center, and $\mathbf A_j$ contains its three reference areas. The operator $\odot$ denotes element-wise multiplication. The dataset stores complete force vectors for both branches, each group's unit-multiplier response, and the paired horizontal acceleration increment. Labels precede the nominal $C_DA=0.25\,\mathrm{m^2}$ runtime scaling, filtering, and clipping.

The fixed-parameter formula gains two sources of randomization:

| Random quantity | Sampling and role in the formula |
| --- | --- |
| Three drag multipliers $s_{g(j)}$ | One draw per group (frame, booms and motors, manipulator) from a base $\mathcal U(0.8,1.2)$ distribution. The multiplier scales $C_D A$ for every proxy in its group and stays fixed across a wind process and both branches of each pair. |
| Horizontal gusts | Gust-enabled processes draw peak magnitude $G\sim\mathcal U(0.5,2.0)$ m/s and pulse period $T\sim\mathcal U(2,6)$ s. Each pulse draws its own direction $\phi_k\sim\mathcal U(0,2\pi)$. At time $\tau=t\bmod T$ within the pulse, add $G\sin^2(\pi\tau/T)(\cos\phi_k,\sin\phi_k,0)$ to the baseline wind and cap the horizontal resultant at 10 m/s. |

Density $\rho=1.225\,\mathrm{kg/m^3}$, nominal axis coefficients $C_D=1$, proxy geometry, and mass stay fixed. States and baseline wind are sampled with coverage for zero wind, weak wind, and the 10 m/s boundary. Each group of ten pairs shares a wind process and a multiplier draw. Entire groups go into **30,000 training, 6,000 validation, and 6,000 test pairs**, with seeds **17653, 18659, and 19663**. Training and validation exclude corners where all three multipliers differ from one by at least 0.12. The test reserves 1,500 pairs in those corners. The labels represent the parameterized analytical model. [v3 reproduction commands and validation results](docs/models/x4-950-pinn-wind-randomized-v3.md)

<details>
<summary>PINN training plot and technical reports</summary>

![PINN pilot training and holdout errors](docs/figures/x4-950-pinn-pilot-v1.png)

- [Reference model](docs/x4-950-reference-v1.md)
- [v2 wide-range baseline and reproduction](docs/models/x4-950-pinn-wind-10-v2.md)
- [v1 model card](docs/models/x4-950-pinn-pilot-v1.md)
- [Pilot training report](docs/X4_PINN_PILOT_2026-11-10.md)
- [SITL integration report](docs/X4_PINN_INTEGRATION_2026-11-10.md)
- [PX4 runtime and source provenance](docs/px4.md)

</details>
