# Vision-Based Pursuit-Evasion Control: PID vs Deep RL

A comprehensive research project comparing classical control (PID, Kalman Filter + PID) with modern Deep Reinforcement Learning (SAC) for vision-based pursuit-evasion tasks.

## 📋 Project Overview

This project implements a 2D pursuit-evasion simulation to benchmark different control strategies. The challenge: an agent must learn to track a target moving with Brownian motion using only egocentric, vision-based observations (64×64 stacked frames).

### Key Features

- **Custom Gymnasium Environment**: Dynamic physics with acceleration-based control
- **Three Control Approaches**:
  1. **PID Controller**: Classical control with OpenCV-based visual detection
  2. **Kalman Filter + PID**: State estimation for robust tracking
  3. **SAC Deep RL**: End-to-end learning from pixels
- **Comprehensive Evaluation**: Metrics, visualizations, and statistical comparison
- **Publication-Ready**: Automated figure generation and LaTeX table export

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

Test each controller interactively:

```bash
# PID Controller
python demo.py pid --n-episodes 3

# Kalman Filter + PID
python demo.py kalman-pid --n-episodes 3

# SAC Agent (requires trained model)
python demo.py sac --sac-model models/sac/final_model --n-episodes 3
```

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
