# 3D HUD Pursuit-Evasion: Modern Deep RL

**A modern deep reinforcement learning framework for 3D pursuit-evasion with dogfight-inspired HUD visualization.**

## 🎯 Project Overview

### 3D HUD System
Fighter jet inspired, egocentric (first-person) HUD where:
- **Agent (pursuer)** is always centered with green crosshair
- **Target (evader)** moves in 3D space with depth-based sizing (closer = larger)
- **Cyan error vector** points from agent to target - the goal is to nullify this vector
- **Real-time telemetry**: velocity vectors, radar boundaries, distance indicators, focus status

### Two Training Scenarios

#### Scenario 1: RL Agent vs Random Target
- Agent learns to track randomly moving target
- Target uses Brownian motion + evasive maneuvers
- Best for: Initial RL algorithm testing and baseline performance

#### Scenario 2: Competitive MARL (RL vs RL)
- Both pursuer and evader are RL agents
- Pursuer learns to catch, evader learns to escape
- Zero-sum competitive game with alternating training
- Best for: Advanced multi-agent research

### Supported RL Algorithms

| Algorithm | Type | Action Space | Best For |
|-----------|------|--------------|----------|
| **PPO** | Policy Gradient (On-Policy) | Continuous | General purpose, stable training |
| **SAC** | Actor-Critic (Off-Policy) | Continuous | Sample efficient, good exploration |
| **TD3** | Actor-Critic (Off-Policy) | Continuous | Robust, reduced overestimation |
| **DQN** | Value-Based (Off-Policy) | Discrete | Discrete action spaces |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
cd PIDRL

# Install dependencies
pip install -r requirements.txt
```

### 1. Test Random Actions (Baseline)

```bash
# See the 3D HUD with random agent
python demo_3d.py --scenario 1 --random --n-episodes 3
```

### 2. Train Your First Agent (Scenario 1)

**On Kaggle (GPU, No Rendering):**
```bash
# Upload to Kaggle and run:
python training/train_rl.py --algo ppo --scenario 1 --timesteps 500000

# Or use SAC for better exploration:
python training/train_rl.py --algo sac --scenario 1 --timesteps 500000
```

**On Local Machine:**
```bash
python training/train_rl.py --algo ppo --scenario 1 --timesteps 500000 \
                            --save-dir models --tensorboard-log logs
```

Monitor training:
```bash
tensorboard --logdir logs
```

### 3. Test Trained Model (Local with Rendering)

```bash
# Test PPO model
python testing/test_rl.py --scenario 1 --algo ppo \
                          --model models/ppo_pursuer_final.zip \
                          --episodes 10 --stats

# Test SAC model
python testing/test_rl.py --scenario 1 --algo sac \
                          --model models/sac_pursuer_final.zip \
                          --episodes 10 --stats
```

### 4. Train Competitive MARL (Scenario 2)

```bash
# Alternating training: Pursuer → Evader → Pursuer → ...
python training/train_rl.py --algo sac --scenario 2 --timesteps 500000 \
                            --competitive --alternating-rounds 10
```

This creates:
- `models/sac_pursuer_latest.zip`
- `models/sac_evader_latest.zip`

### 5. Demo Competitive MARL

```bash
python demo_3d.py --scenario 2 \
                  --pursuer-algo sac --pursuer-model models/sac_pursuer_latest.zip \
                  --evader-algo sac --evader-model models/sac_evader_latest.zip \
                  --n-episodes 5
```

## 📁 Project Structure

```
PIDRL/
├── configs/
│   └── config.yaml              # Configuration for all algorithms
├── environments/
│   ├── pursuit_evasion_env_3d.py  # 3D HUD environment
│   └── __init__.py
├── agents/
│   ├── networks.py              # CNN architectures
│   └── __init__.py
├── training/
│   └── train_rl.py              # Training script (Kaggle optimized, no render)
├── testing/
│   └── test_rl.py               # Testing script (local, with render)
├── models/                      # Saved models
├── logs/                        # TensorBoard logs
├── demo_3d.py                   # Interactive demo
├── competitive_marl/            # Competitive MARL system
│   ├── config.py
│   ├── environment/
│   ├── agents/
│   ├── training/
│   └── testing/
└── README.md
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

### Environment Parameters
```yaml
environment:
  max_velocity: 25.0        # Agent max speed (INCREASED)
  max_acceleration: 3.0     # Agent acceleration (INCREASED)
  target_size: 7.0          # Target size (LARGER for easier lock-on)
  target_max_speed_ratio: 1.0  # Target speed = agent speed
  success_threshold: 9.0    # Focus area (30% of FOV)
```

### Algorithm-Specific Configs

Each algorithm (DQN, PPO, SAC, TD3) has its own section in `config.yaml`:
- Network architecture (CNN layers, MLP units)
- Hyperparameters (learning rate, batch size, etc.)
- Training settings (buffer size, exploration params, etc.)

## 🎓 Training Recommendations

### Algorithm Selection

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Scenario 1 (RL vs Random) | **PPO** or **SAC** | PPO: stable, SAC: sample efficient |
| Scenario 2 (RL vs RL) | **SAC** or **TD3** | Better for competitive settings |
| Quick Testing | **PPO** | Faster convergence |
| Research | **SAC** | Best overall performance |

### Training Times (Kaggle GPU)

| Algorithm | Steps | Expected Time | Recommended |
|-----------|-------|---------------|-------------|
| PPO | 500K | ~1.5 hours | ✅ Good balance |
| SAC | 500K | ~2 hours | ✅ Best performance |
| TD3 | 500K | ~2 hours | ✅ Robust alternative |
| DQN | 500K | ~1.5 hours | ⚠️ Discrete actions only |

### Hyperparameter Tuning Tips

**Too slow convergence?**
- Increase `learning_rate` (e.g., 3e-4 → 5e-4)
- Decrease `batch_size` for faster updates

**Unstable training?**
- Decrease `learning_rate`
- Increase `batch_size`
- For PPO: reduce `clip_range`
- For SAC/TD3: increase `tau` (slower target updates)

**Poor exploration?**
- For SAC: set `ent_coef: "auto"`
- For PPO: increase `ent_coef`
- For DQN: increase `exploration_initial_eps`

## 📊 Evaluation Metrics

The testing script computes:
- **Episode Reward**: Cumulative reward per episode
- **Focus Time**: Percentage of time target is in focus area (<9.0 units)
- **Final Distance**: Distance to target at episode end
- **Episode Length**: Steps to completion

Example output:
```
OVERALL STATISTICS
Episodes: 10

Rewards:
  Mean: 125.30 ± 15.20
  Min/Max: 95.30 / 145.70

Focus Time (%):
  Mean: 68.3% ± 8.5%
  Min/Max: 55.2% / 78.9%
```

## 🧪 Kaggle GPU Training

### Setup

1. **Upload `PIDRL/` to Kaggle**
2. **Enable GPU** (Settings → Accelerator → GPU T4 x2)
3. **Run training:**

```python
# In Kaggle notebook:
!cd PIDRL && python training/train_rl.py --algo sac --scenario 1 --timesteps 500000

# Monitor with TensorBoard:
%load_ext tensorboard
%tensorboard --logdir logs
```

4. **Download models** from Output panel

### Kaggle Best Practices

✅ **Always use GPU** (5-10x faster than CPU)
✅ **No rendering** during training (automatic in `train_rl.py`)
✅ **Save checkpoints** every 10K steps
✅ **Monitor TensorBoard** for training curves
✅ **Download models** before session ends

## 🔬 Research Use Cases

### 1. Algorithm Comparison
Train all algorithms on Scenario 1 and compare:
```bash
for algo in ppo sac td3; do
    python training/train_rl.py --algo $algo --scenario 1 --timesteps 500000
done
```

### 2. Competitive MARL Research
Study co-evolution of pursuer and evader strategies:
```bash
python training/train_rl.py --algo sac --scenario 2 --competitive \
                            --alternating-rounds 50 --timesteps 1000000
```

### 3. Transfer Learning
Train on Scenario 1, fine-tune on Scenario 2:
```bash
# Step 1: Train on random target
python training/train_rl.py --algo sac --scenario 1 --timesteps 500000

# Step 2: Fine-tune on competitive
python training/train_rl.py --algo sac --scenario 2 --competitive \
                            --resume models/sac_pursuer_final.zip \
                            --timesteps 500000
```

## 🎮 3D HUD Controls

During demo/testing:
- **Observe**: Green crosshair (you) at center
- **Track**: Red/colored targets moving in 3D
- **Goal**: Keep target in focus area (inner circle, <9.0 units)
- **Depth**: Target size indicates distance (large = close, small = far)

HUD Elements:
- **Green crosshair**: Agent (always centered)
- **Colored circles**: Targets with depth-based sizing
- **Green arrow**: Agent velocity vector
- **Red arrow**: Target velocity vector
- **Cyan arrow**: Error vector (agent → target)
- **White circle**: Focus area boundary

## 🐛 Troubleshooting

**Training too slow?**
- Use Kaggle GPU (not CPU)
- Reduce `batch_size` or `buffer_size`

**Agent not learning?**
- Check TensorBoard for reward curves
- Increase `total_timesteps` (try 1M)
- Tune `learning_rate`

**"No module" errors?**
```bash
pip install gymnasium stable-baselines3[extra] torch numpy pygame pyyaml
```

**Rendering errors (Kaggle)?**
- Normal! Training uses `render_mode=None`
- Only test locally with rendering

## 📄 Citation

```bibtex
@software{pidrl_3d_hud,
  title={3D HUD Pursuit-Evasion: Modern Deep RL},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/PIDRL}
}
```

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: your-email@example.com

---

**Good Luck Training! 🚀**

_Note: This project focuses on 3D HUD pursuit-evasion with modern deep RL. Previous 2D and classical control methods have been removed to streamline the codebase._
