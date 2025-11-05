# Quick Start Guide

This guide will help you get started with the PIDRL research project in 5 minutes.

## 1. Installation (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_environment.py
```

Expected output: All tests should pass ✅

## 2. Run Demo (1 minute)

Try each controller:

```bash
# PID Controller (Classical)
python demo.py pid --n-episodes 2

# Kalman Filter + PID (Enhanced Classical)
python demo.py kalman-pid --n-episodes 2
```

You should see a window with:
- Green circle = Agent (chaser)
- Red circle = Target (moving with Brownian motion)
- Agent tries to keep target centered

## 3. Train Deep RL Agent (30+ minutes)

```bash
# Start training (will take 30-60 minutes on CPU)
python experiments/train_sac.py --config configs/config.yaml

# Monitor training in another terminal
tensorboard --logdir logs/sac
```

For quick testing, reduce timesteps in `configs/config.yaml`:
```yaml
sac:
  total_timesteps: 50000  # Instead of 500000
```

## 4. Evaluate & Compare (5 minutes)

```bash
# Evaluate all three methods
python experiments/compare_methods.py \
    --config configs/config.yaml \
    --sac-model models/sac/final_model \
    --n-episodes 20
```

Results saved to:
- `results/evaluation_summary.csv` - Performance metrics
- `results/figures/*.png` - Comparison plots
- `results/performance_table.tex` - LaTeX table

## 5. Understanding the Results

### Key Metrics

1. **Mean Reward**: Higher is better
   - Negative squared distance to target
   - Closer tracking → higher reward

2. **Tracking Error**: Lower is better
   - Mean distance to target
   - Measures precision

3. **Success Rate**: Higher is better
   - Percentage of time target is within threshold
   - Measures robustness

### Expected Performance

Typical results on default configuration:

| Method | Mean Reward | Tracking Error | Success Rate |
|--------|-------------|----------------|--------------|
| PID | -50 to -100 | 3-5 units | 60-70% |
| Kalman-PID | -40 to -80 | 2.5-4 units | 70-80% |
| SAC (trained) | -30 to -60 | 2-3.5 units | 75-85% |

## Tips & Tricks

### Speed Up Training

1. **Reduce resolution**: In `configs/config.yaml`
```yaml
environment:
  frame_size: 48  # Instead of 64
```

2. **Use GPU**: Install CUDA and PyTorch with GPU support

3. **Reduce buffer size**: In `configs/config.yaml`
```yaml
sac:
  buffer_size: 50000  # Instead of 100000
```

### Tune PID Controller

Edit `configs/config.yaml`:
```yaml
pid:
  kp: 0.5   # Increase for faster response
  ki: 0.01  # Increase to reduce steady-state error
  kd: 0.2   # Increase to reduce oscillations
```

### Improve Tracking

Make task easier:
```yaml
environment:
  target_brownian_std: 1.0  # Reduce from 2.0 (slower target)
  view_radius: 40.0         # Increase from 30.0 (larger view)
```

## Troubleshooting

### "No available video device"

Run without rendering:
```bash
python experiments/evaluate.py --config configs/config.yaml
# (Don't use demo.py which requires rendering)
```

### Training is slow

Normal on CPU. Expected times:
- CPU (4 cores): 60-90 minutes for 500k steps
- GPU (RTX 3080): 15-20 minutes for 500k steps

### Low success rates

This is a challenging task! Try:
1. Increase PID gains
2. Reduce target motion (lower `target_brownian_std`)
3. Train SAC longer
4. Tune Kalman Filter noise parameters

## Next Steps

### For Research

1. **Run Multiple Seeds**: Get statistical significance
```bash
for seed in 42 43 44 45 46; do
    python experiments/compare_methods.py --seed $seed --output-dir results/run_$seed
done
```

2. **Ablation Studies**: Test different configurations
   - Vary PID gains
   - Change observation resolution
   - Modify reward shaping

3. **Generate Paper Figures**: Use visualization utilities
```python
from utils.visualization import generate_all_plots
# See utils/visualization.py for custom plots
```

### For Extension

1. **Add new controllers**: LQR, MPC, etc.
2. **Try different RL algorithms**: PPO, TD3, DQN
3. **3D environment**: Extend to 3D pursuit
4. **Multiple targets**: Track multiple objects
5. **Adversarial targets**: Evasive instead of random

## Getting Help

- Check `README.md` for detailed documentation
- Review code comments in source files
- Open GitHub issue for bugs
- Email: [your-email@example.com]

## Summary Commands

```bash
# Quick test
python test_environment.py

# Demo controllers
python demo.py pid
python demo.py kalman-pid

# Train SAC
python experiments/train_sac.py

# Full comparison
python experiments/compare_methods.py --sac-model models/sac/final_model

# View results
ls results/figures/
cat results/evaluation_summary.csv
```

Happy researching! 🚀
