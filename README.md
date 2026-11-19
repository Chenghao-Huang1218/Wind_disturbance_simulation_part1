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

Each step, the default **pinn-x4** backend evaluates wind and no-wind inputs at the same aircraft state:

$$
\mathbf a_{\mathrm{wind}}^W
=P_{xy}R_{WB}\bigl[
g_\theta(R_{WB}^{T}(\mathbf v_O^W-\mathbf w^W),\boldsymbol\omega^B)
-g_\theta(R_{WB}^{T}\mathbf v_O^W,\boldsymbol\omega^B)
\bigr].
$$

$R_{WB}$ maps body to world, $\mathbf v_O^W$ is root-point velocity, $\mathbf w^W$ is wind velocity, and $P_{xy}=\mathrm{diag}(1,1,0)$ retains horizontal acceleration.

The default horizontal equivalent drag area is **0.25 m²**. The gain derived at the reference condition scales the world-frame XY acceleration before it passes through an EMA ($\alpha=0.05$), per-axis clipping at ±6 m/s², and multiplication by the 8.0004 kg aircraft mass before application to the root body.

The surrogate combines a quadratic-drag prior with a **6→64→64→3 Tanh residual network**. The earlier v1 pilot uses **3,000 paired synthetic samples** from 29 component drag proxies extracted from model XML. Sampling/training seeds are 7/42. This simplified reference model has no real-aircraft aerodynamic calibration.

<details>
<summary>PINN training plot and technical reports</summary>

![PINN pilot training and holdout errors](docs/figures/x4-950-pinn-pilot-v1.png)

- [Reference model](docs/x4-950-reference-v1.md)
- [Default wide-range model and reproduction](docs/models/x4-950-pinn-wind-10-v2.md)
- [v1 model card](docs/models/x4-950-pinn-pilot-v1.md)
- [Pilot training report](docs/X4_PINN_PILOT_2026-11-10.md)
- [SITL integration report](docs/X4_PINN_INTEGRATION_2026-11-10.md)
- [PX4 runtime and source provenance](docs/px4.md)

</details>
