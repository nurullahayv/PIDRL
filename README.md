# Vision-Based Pursuit-Evasion Control: 2.5D Dogfight HUD Error Vector Nullification

A comprehensive research project comparing classical control (PID, Kalman Filter + PID) with modern Deep Reinforcement Learning (SAC) for **3D error vector nullification** in an egocentric, dogfight HUD-inspired tracking environment with **depth perception and multi-target capability**.

## 📋 Project Overview

### 🆕 NEW: 2.5D (Pseudo-3D) Upgrade

We've upgraded from 2D to **2.5D with depth perception**:

- **3D State Space**: Targets now have position (x, y, **z**) where z represents depth/range
- **Visual Depth Encoding**: Depth is perceptually encoded as **target size** (closer = bigger, farther = smaller)
- **3D Action Space**: Agent controls (ax, ay, **az**) - lateral steering + forward/backward thrust
- **Multi-Target Support**: Track multiple color-coded targets simultaneously
- **Range Management**: Must control closure rate and maintain optimal engagement range

### The Core Concept: 3D Error Vector Nullification

Inspired by fighter jet dogfight HUDs, this project implements a **purely egocentric (first-person) control environment** where:

- **🎯 The Agent (You)**: Always at the center of your universe, represented by a fixed **green crosshair**
- **🔴 The Targets (Enemies)**: Color-coded targets executing evasive maneuvers in 3D space
- **📐 The 3D Error Vector (Mission)**: The cyan arrow from your crosshair to the target - **nullify in ALL three dimensions**
- **⚡ The Control Challenge**: Generate 3D acceleration commands (steering + thrust) to drive the 3D error vector to zero
- **🎚️ Depth Perception**: Target size scales with depth - larger means closer, smaller means farther away

This is **not** a realistic 3D flight simulator. Instead, we deliberately abstract away complex graphics and visual detection to focus on the **pure 3D control problem**: Given a simplified "radar" view with depth cues, can different control strategies effectively minimize 3D tracking error while managing range?

### The Egocentric Framework

The simulation operates from a purely first-person perspective:

1. **Agent's Viewpoint**: You are always at the center. The world rotates around you.
2. **Error Vector**: The line from your center (green crosshair) to the target (red circle) represents the error that must be nullified
3. **Control Loop**: As the target moves, the error vector changes → Controller computes acceleration → Agent moves to reduce error → Repeat

### Research Question

**How do different control paradigms (classical PID, state estimation with Kalman filtering, and end-to-end Deep RL) compare when solving this error vector nullification problem using only vision-based, egocentric observations?**

### Key Features

- **Dogfight HUD-Style Environment**: Radar rings, crosshair, error vector visualization, status indicators
- **Pure Egocentric Control**: Agent always centered, learns to nullify error vectors
- **Three Control Approaches**:
  1. **PID Controller**: Classical feedback control with OpenCV visual detection
  2. **Kalman Filter + PID**: State estimation for smooth, robust tracking
  3. **SAC Deep RL**: End-to-end learning directly from pixels (64×64 stacked frames)
- **Comprehensive Evaluation**: Error magnitude, tracking success rate, statistical comparison
- **Publication-Ready**: Automated figure generation and LaTeX tables

## 🏗️ Project Structure

```
PIDRL/
├── environments/           # Custom Gymnasium environment
│   ├── pursuit_evasion_env.py
│   └── __init__.py
├── controllers/            # Classical controllers
│   ├── pid_controller.py
│   ├── kalman_filter.py
│   ├── kalman_pid_controller.py
│   └── __init__.py
├── agents/                 # Deep RL agents
│   ├── networks.py
│   └── __init__.py
├── utils/                  # Utilities
│   ├── visual_detection.py
│   ├── visualization.py
│   └── __init__.py
├── experiments/            # Training and evaluation scripts
│   ├── train_sac.py
│   ├── evaluate.py
│   └── compare_methods.py
├── configs/                # Configuration files
│   └── config.yaml
├── demo.py                 # Interactive demo
├── test_environment.py     # Quick test script
└── requirements.txt        # Dependencies
```

## 🎯 Visual Guide: Understanding the HUD

When you run the demo, you'll see a **dogfight-inspired HUD** with these elements:

### HUD Elements

| Element | Color | Description |
|---------|-------|-------------|
| **Green Crosshair** | 🟢 Green | Your agent - always at the center. This is your fixed reference point. |
| **Red Circle** | 🔴 Red | The target executing evasive maneuvers (Brownian motion) |
| **Cyan Arrow** | 🔵 Cyan | **THE ERROR VECTOR** - Points from you (center) to target. Your goal is to nullify this! |
| **Radar Rings** | ⚪ Gray | Range indicators (33%, 66%, 100% of view radius) |
| **Light Green Line** | 🟢 Green | Your velocity vector (where you're moving) |
| **Light Red Line** | 🔴 Red | Target's velocity vector (where it's moving) |

### HUD Information Display

**Top-Left:**
- `ERROR: X.XX` - **Error magnitude** (distance to target) - Lower is better!
- `[LOCKED]` or `[TRACKING]` - Status indicator (LOCKED when error < threshold)

**Top-Right:**
- `STEP: X/500` - Current timestep in episode

**Bottom-Left:**
- `AGENT VEL: X.XX` - Your current speed
- `TARGET VEL: X.XX` - Target's current speed

**Bottom-Right:**
- `Goal: Nullify Error Vector` - Reminder of your objective

**Center (when tracking):**
- `XXX°` - Angle to target in degrees

### What You're Watching

When you run `python demo.py pid`:

1. **Green crosshair** stays at center (that's you, the agent)
2. **Red target** moves around executing random evasive maneuvers
3. **Cyan error vector** points from center to target
4. **Controller tries to "follow" the vector** by applying accelerations
5. **Goal**: Make the cyan arrow as short as possible (ideally zero)

The **error vector is the key concept**: Unlike traditional tracking where you move toward the target, here your perspective is egocentric - the target appears to move relative to you, and you must apply forces to keep it centered in your crosshair.

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
cd PIDRL

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python test_environment.py
```

## 🎮 Quick Start

### 1. Demo Visualization

**🆕 NEW: 2.5D Demo with Depth Perception**

```bash
# Single target with depth (recommended first try)
python demo_3d.py --n-episodes 3

# Multi-target scenario (3 targets)
python demo_3d.py --num-targets 3 --n-episodes 2

# Try 5 simultaneous targets (challenge mode!)
python demo_3d.py --num-targets 5 --n-episodes 2
```

**Classic 2D Demo (original version)**

Test each controller interactively:

```bash
# PID Controller (2D)
python demo.py pid --n-episodes 3

# Kalman Filter + PID (2D)
python demo.py kalman-pid --n-episodes 3

# SAC Agent (requires trained model - 2D)
python demo.py sac --sac-model models/sac/final_model --n-episodes 3
```

**Note:** The 2.5D environment uses random actions in the demo. For actual control, you'll need to adapt the controllers to handle 3D actions (see below).

### 2. Train SAC Agent

```bash
python experiments/train_sac.py \
    --config configs/config.yaml \
    --save-dir models/sac \
    --tensorboard-log logs/sac
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs/sac
```

### 3. Evaluate All Methods

```bash
python experiments/evaluate.py \
    --config configs/config.yaml \
    --sac-model models/sac/final_model \
    --n-episodes 100 \
    --output-dir results
```

### 4. Generate Comparison Plots

```bash
python experiments/compare_methods.py \
    --config configs/config.yaml \
    --sac-model models/sac/final_model \
    --n-episodes 100 \
    --output-dir results
```

This will generate:
- Performance comparison plots
- Statistical analysis
- LaTeX tables for the paper
- CSV files with detailed results

## 📊 Evaluation Metrics

The framework computes the following metrics:

1. **Mean Episode Reward**: Cumulative reward per episode
2. **Tracking Error**: Mean squared distance to target
3. **Success Rate**: Percentage of time target is within threshold
4. **Episode Length**: Steps per episode
5. **Detection Rate**: Vision system reliability (PID/Kalman-PID only)

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

- Environment parameters (physics, observation space)
- PID gains (Kp, Ki, Kd)
- Kalman Filter noise parameters
- SAC hyperparameters (learning rate, network architecture)
- Evaluation settings

## 📈 Experimental Workflow

### Full Research Pipeline

```bash
# 1. Train SAC agent
python experiments/train_sac.py --config configs/config.yaml

# 2. Evaluate all methods and generate plots
python experiments/compare_methods.py \
    --config configs/config.yaml \
    --sac-model models/sac/final_model \
    --n-episodes 100

# 3. Results are saved to results/
#    - results/evaluation_summary.csv
#    - results/figures/*.png
#    - results/performance_table.tex
```

## 🧪 Testing

Run quick tests to verify components:

```bash
# Test environment
python test_environment.py

# Test PID controller
python -c "from controllers import PIDAgent; print('PID OK')"

# Test Kalman Filter
python -c "from controllers import KalmanFilter; print('Kalman OK')"

# Test SAC networks
python -c "from agents.networks import CustomCNN; print('Networks OK')"
```

## 📝 Research Paper Integration

### Generated Assets

After running experiments, you'll have:

1. **Figures** (`results/figures/`):
   - `reward_comparison.png`
   - `tracking_error_comparison.png`
   - `success_rate_comparison.png`
   - `distance_over_time_ep*.png`

2. **LaTeX Table** (`results/performance_table.tex`):
   - Ready-to-include performance comparison table

3. **Data** (`results/*/`):
   - NumPy arrays for custom analysis
   - CSV files for spreadsheet analysis

### Citing

If you use this code in your research, please cite:

```bibtex
@article{yourname2024pidrl,
  title={Vision-Based Pursuit-Evasion Control: Comparing Classical and Deep Reinforcement Learning Approaches},
  author={Your Name},
  journal={Your Journal/Conference},
  year={2024}
}
```

## 🛠️ Customization

### Adding New Controllers

1. Implement controller in `controllers/`
2. Add agent wrapper with `predict()` method
3. Update `experiments/evaluate.py` to include new method

### Modifying Environment

Edit `environments/pursuit_evasion_env.py`:
- Change observation space
- Adjust physics parameters
- Modify reward function
- Add new features

### Tuning Hyperparameters

Use `configs/config.yaml`:
```yaml
pid:
  kp: 0.5    # Proportional gain
  ki: 0.01   # Integral gain
  kd: 0.2    # Derivative gain

sac:
  learning_rate: 3.0e-4
  buffer_size: 100000
  batch_size: 256
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `pygame.error: No available video device`
- **Solution**: Set `render_mode=None` or use virtual display

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in SAC config

**Issue**: Slow training
- **Solution**: Reduce `frame_size` or `buffer_size`

## 📚 Dependencies

Key libraries:
- **Gymnasium**: Environment API
- **Stable-Baselines3**: SAC implementation
- **PyTorch**: Deep learning
- **OpenCV**: Computer vision
- **Pygame**: Rendering
- **Matplotlib/Seaborn**: Visualization

See `requirements.txt` for complete list.

## 🎯 Research Questions

This project helps answer:

1. How does end-to-end RL compare to classical control?
2. Does Kalman filtering improve PID performance?
3. What is the sample efficiency trade-off?
4. How do methods generalize to different motion patterns?

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional controllers (MPC, LQR)
- More RL algorithms (PPO, TD3, DQN)
- 3D environments
- Multi-agent scenarios
- Real-world deployment

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

## 🙏 Acknowledgments

- OpenAI Gym/Gymnasium for the RL framework
- Stable-Baselines3 for SAC implementation
- OpenCV community for computer vision tools

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Happy Researching! 🚀**
