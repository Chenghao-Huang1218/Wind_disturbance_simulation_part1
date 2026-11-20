# Wind_disturbance_simulation_part1

**English** | [简体中文](README.zh-CN.md)

Single-UAV wind disturbance simulation in MuJoCo with PX4 SITL, a PINN aerodynamic surrogate, and interactive trajectory control. The aircraft is an X4-950 with an Airbot Play arm.

## Simulation demonstration

https://github.com/user-attachments/assets/5404e7aa-410a-489f-a724-ee3c1227da1d

[Download the original MP4 (about 66 seconds)](docs/media/videos/Wind%20field%20simulation%20Demo.mp4)

| X4-950 + Airbot Play model | Interactive control panel |
| --- | --- |
| ![Quadrotor and arm in MuJoCo](docs/media/images/MJCF%20model.png) | ![Flight, trajectory and horizontal wind controls](docs/media/images/Control%20UI.png) |

**Circular trajectory tracking**

<img src="docs/media/images/Motion%20tracking.png" width="420" alt="Target and actual XY trajectories">

The dashed gray line is the target. The red line is the actual trajectory.


## Simulation capabilities

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

## Simulation startup

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

## Aerodynamic model

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

### v3 synthetic data generation

#### Baseline drag model

The scalar quadratic drag law is $D=\tfrac12\rho V^2 C_D A$, where $V$ is relative airspeed. The reference model applies this law along each local axis of 29 fixed-geometry proxies. For proxy $j$, the local relative velocity $\mathbf u_j$ and nominal local force $\mathbf f_{j,0}$ are:

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

$Q_j$ maps the proxy axes to the body frame, $\mathbf r_j^B$ is the offset from the root to the proxy center, and $\mathbf A_j$ contains the three reference areas. Multiplication $\odot$ and absolute value are element-wise. The model sums the proxy forces in world coordinates and subtracts the zero-wind result at the same vehicle state. With the v3 group multipliers $\mathbf s$, the horizontal acceleration label is:

$$
\begin{aligned}
\mathbf F^W(\mathbf w^W;\mathbf s)
 &= R_{WB}\sum_{j=1}^{29}Q_j\,s_{g(j)}\,\mathbf f_{j,0}(\mathbf w^W),\\
\mathbf y_{xy}^W
 &=\frac{P_{xy}}{m}
   \left[\mathbf F^W(\mathbf w^W(t);\mathbf s)-\mathbf F^W(\mathbf 0;\mathbf s)\right].
\end{aligned}
$$

$g(j)$ identifies the proxy's drag group. Setting $\mathbf s=(1,1,1)$ recovers the fixed-parameter model. Both branches share root velocity, attitude, angular rate, geometry, and multipliers. The dataset stores both full force vectors, the unit-multiplier group responses, and the paired acceleration increment. [Quadratic drag law](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/) · [Project reference model](docs/x4-950-reference-v1.md)

#### Model parameters

| Parameter | Value and definition |
| --- | --- |
| Air density $\rho$ | 1.225 kg/m³ |
| Proxy drag coefficients $\mathbf C_D$ | $(1,1,1)$ on the three local axes |
| Proxy reference areas $\mathbf A_j$ | Three axis-wise areas per proxy, derived from the X4-950 and Airbot Play XML geometry |
| Aircraft mass $m$ | 8.0004 kg |
| Horizontal wind magnitude | 0–10 m/s with zero vertical wind |
| Analytical reference drag area $(C_DA)_{\mathrm{ref}}$ | Approximately 0.11951 m² from the frozen geometry |
| Nominal simulation drag area $(C_DA)_{\mathrm{nom}}$ | 0.25 m² for runtime horizontal force scaling |

The reference drag area is defined for a stationary, level aircraft at zero yaw with a world-frame wind of $(4,4,0)$ m/s, unit drag multipliers, and zero gust amplitude. At this condition, divide the along-wind force by the dynamic pressure $\tfrac12\rho V^2$ to obtain $(C_DA)_{\mathrm{ref}}$. Runtime horizontal increments are scaled by $0.25/0.11951\approx2.0919$, giving a nominal along-wind force of 4.90 N. The synthetic labels precede this scaling, filtering, and clipping.

#### Randomization and data splits

| Random quantity | Sampling and role |
| --- | --- |
| Three drag multipliers $s_g$ | One draw per group (frame, booms and motors, manipulator) from a base $\mathcal U(0.8,1.2)$ distribution. Each draw scales the proxy forces in its group and stays fixed for the wind process and both branches of each pair. |
| Horizontal gusts | Gust-enabled processes draw peak magnitude $G\sim\mathcal U(0.5,2.0)$ m/s and period $T\sim\mathcal U(2,6)$ s. Each pulse draws a direction $\phi_k\sim\mathcal U(0,2\pi)$ and adds $G\sin^2(\pi\tau/T)(\cos\phi_k,\sin\phi_k,0)$ to the baseline wind, where $\tau=t\bmod T$. The horizontal resultant is capped at 10 m/s. |

Baseline winds are stratified by speed and direction, with coverage of zero wind, weak wind, and the upper limit. Each group of ten pairs shares a wind process and multiplier draw. Entire groups go into **30,000 training, 6,000 validation, and 6,000 test pairs**, with seeds **17653, 18659, and 19663**. Training and validation exclude combinations in which all three multipliers differ from 1 by at least 0.12. The test set contains 1,500 pairs in these held-out combinations. All labels are synthetic outputs of the parameterized analytical model. [v3 reproduction and validation](docs/models/x4-950-pinn-wind-randomized-v3.md)

<details>
<summary>Validation and reproducibility</summary>

![PINN pilot training and holdout errors](docs/figures/x4-950-pinn-pilot-v1.png)

- [Reference model](docs/x4-950-reference-v1.md)
- [v2 wide-range baseline and reproduction](docs/models/x4-950-pinn-wind-10-v2.md)
- [v1 model card](docs/models/x4-950-pinn-pilot-v1.md)
- [Pilot training report](docs/X4_PINN_PILOT_2026-11-10.md)
- [SITL integration report](docs/X4_PINN_INTEGRATION_2026-11-10.md)
- [PX4 runtime and source provenance](docs/px4.md)

</details>
