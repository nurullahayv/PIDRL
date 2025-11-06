# Troubleshooting Guide

This guide helps you resolve common issues when setting up and running the PIDRL project.

## ⚠️ Critical: Kaggle Model Loading Issues (UPDATED 2024-11-06)

### Issue: Numpy Version Compatibility Error

**Error Message:**
```
ModuleNotFoundError: No module named 'numpy._core.numeric'
```
or
```
UserWarning: Could not deserialize object policy_kwargs
```

**Cause:**
Models trained on Kaggle (numpy 2.0+) have compatibility issues when loaded on local machines with numpy 1.x, or vice versa.

**Solution (RECOMMENDED):**

**✅ The latest code includes automatic fixes!** Simply update your repository:

```bash
git pull origin claude/pid-nn-rl-research-011CUpVJyyPR2RaPkVsoCSU3
```

The updated `test_trained_model.py` includes:
- Automatic numpy version detection
- Compatibility shims for both numpy 1.x and 2.x
- Graceful handling of deserialization warnings
- Custom objects loading for network architectures

**Manual Fix (if needed):**

If you still encounter issues, match numpy versions:

```bash
# Option 1: Upgrade to numpy 2.x (matches Kaggle)
pip install "numpy>=2.0.0"

# Option 2: Downgrade to numpy 1.x
pip install "numpy>=1.24.0,<2.0.0"
```

**Verify the fix:**
```bash
python test_trained_model.py --model models/sac/best_model/best_model.zip --episodes 1
```

You should see:
```
✓ Model loaded successfully!
```

**Technical Details:**
- Kaggle uses numpy 2.0+ by default
- Numpy 2.0 changed internal structure: `numpy.core` → `numpy._core`
- Models are serialized with cloudpickle which includes numpy references
- The fix adds compatibility module aliases

---

## Installation Issues

### Issue 1: FrameStack Import Error

**Error Message:**
```
ImportError: cannot import name 'FrameStack' from 'gymnasium.wrappers'
```

**Cause:** Different versions of gymnasium have FrameStack in different locations.

**Solution:** The code now includes automatic fallbacks. If you still encounter issues:

```bash
# Option 1: Update gymnasium
pip install --upgrade "gymnasium>=0.29.0,<1.0.0"

# Option 2: The code will use the built-in FrameStack implementation
# (this is automatic, no action needed)
```

**Verification:**
```bash
python -c "from environments import make_env; print('Import successful!')"
```

### Issue 2: Pygame Display Error (Windows)

**Error Message:**
```
pygame.error: No available video device
```

**Cause:** Pygame cannot find a display on Windows or headless systems.

**Solutions:**

1. **For Demo (requires display):**
   - Make sure you're not running in WSL without X server
   - Install VcXsrv or similar X server for WSL
   - Or run directly on Windows (not WSL)

2. **For Training (no display needed):**
   - Don't use `demo.py` (it requires rendering)
   - Use evaluation without rendering:
   ```bash
   python experiments/evaluate.py --config configs/config.yaml
   # (no --render flag)
   ```

### Issue 3: OpenCV Import Error

**Error Message:**
```
ImportError: No module named 'cv2'
```

**Solution:**
```bash
pip install opencv-python
# or
pip install opencv-python-headless  # for servers without display
```

### Issue 4: NumPy Version Conflicts

**Error Message:**
```
numpy.ndarray size changed, may indicate binary incompatibility
```

**Solution:**
```bash
# Reinstall all packages with compatible numpy
pip uninstall numpy
pip install "numpy>=1.24.0,<2.0.0"
pip install --force-reinstall --no-cache-dir -r requirements.txt
```

### Issue 5: PyTorch Installation (GPU Support)

**Issue:** PyTorch installs CPU version by default

**Solution for CUDA GPU:**

1. Check your CUDA version:
```bash
nvidia-smi
```

2. Install PyTorch with CUDA support (replace cu118 with your CUDA version):
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slower training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Runtime Issues

### Issue 6: Slow Training

**Symptoms:** Training takes hours on CPU

**Solutions:**

1. **Use GPU if available:**
   - Install PyTorch with CUDA (see Issue 5)
   - Verify GPU usage:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
   ```

2. **Reduce computational load:**
   Edit `configs/config.yaml`:
   ```yaml
   environment:
     frame_size: 48  # Reduce from 64

   sac:
     total_timesteps: 100000  # Reduce from 500000 for testing
     buffer_size: 50000  # Reduce from 100000
     batch_size: 128  # Reduce from 256
   ```

3. **Use fewer evaluation episodes:**
   ```bash
   python experiments/evaluate.py --n-episodes 20  # Instead of 100
   ```

### Issue 7: Memory Errors

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size in config:**
   ```yaml
   sac:
     batch_size: 128  # or even 64
   ```

2. **Reduce buffer size:**
   ```yaml
   sac:
     buffer_size: 50000  # Instead of 100000
   ```

3. **Use CPU instead:**
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   python experiments/train_sac.py
   ```

### Issue 8: Detection Rate is 0%

**Symptoms:** PID or Kalman-PID controller has 0% detection rate

**Cause:** Target is not visible in the frame or detection threshold is too high

**Solutions:**

1. **Check if target is in view:**
   ```python
   python demo.py pid --n-episodes 1
   # You should see a red circle (target) in the window
   ```

2. **Adjust detection parameters in** `utils/visual_detection.py`:
   ```python
   detector = VisualDetector(
       min_contour_area=5,  # Reduce from 10
       blur_kernel=3,  # Reduce from 5
   )
   ```

3. **Increase view radius:**
   Edit `configs/config.yaml`:
   ```yaml
   environment:
     view_radius: 40.0  # Increase from 30.0
   ```

### Issue 9: Agent Performance is Poor

**Symptoms:** All methods have low success rates or high tracking errors

**Diagnosis:**

1. **Check if environment works:**
   ```bash
   python test_environment.py
   ```

2. **Visualize agent behavior:**
   ```bash
   python demo.py pid --n-episodes 3
   ```

**Solutions:**

1. **Tune PID gains** (in `configs/config.yaml`):
   ```yaml
   pid:
     kp: 0.8  # Increase for faster response
     ki: 0.02  # Increase to reduce steady-state error
     kd: 0.3  # Increase to reduce oscillations
   ```

2. **Make task easier:**
   ```yaml
   environment:
     target_brownian_std: 1.0  # Reduce from 2.0
     max_steps: 1000  # Increase from 500
   ```

3. **Train SAC longer:**
   ```yaml
   sac:
     total_timesteps: 1000000  # Increase from 500000
   ```

### Issue 10: TensorBoard Not Showing Data

**Symptoms:** TensorBoard loads but shows no data

**Solutions:**

1. **Check if logs exist:**
   ```bash
   ls logs/sac/
   ```

2. **Make sure you started training:**
   ```bash
   python experiments/train_sac.py
   ```

3. **Point TensorBoard to correct directory:**
   ```bash
   tensorboard --logdir logs/sac
   # Not logdir logs/ or logdir logs/sac/SAC_1/
   ```

## Platform-Specific Issues

### Windows

**Issue:** Path separators causing problems

**Solution:** Use forward slashes or raw strings:
```python
config_path = "configs/config.yaml"  # Works on all platforms
# Not: config_path = "configs\config.yaml"
```

**Issue:** Long path names

**Solution:** Enable long path support in Windows or install closer to root:
```bash
# Install in C:\PIDRL instead of C:\Users\...\Desktop\PIDRL-main\PIDRL-main
```

### Linux/Mac

**Issue:** Permission errors

**Solution:**
```bash
chmod +x demo.py
chmod +x experiments/*.py
```

**Issue:** tkinter not found (for matplotlib)

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Mac
brew install python-tk
```

### WSL (Windows Subsystem for Linux)

**Issue:** No display for pygame

**Solutions:**

1. **Install X server:** Install VcXsrv on Windows

2. **Set DISPLAY variable:**
   ```bash
   export DISPLAY=:0
   ```

3. **Or use headless mode:**
   ```bash
   # Don't use demo.py
   # Use evaluation without rendering
   python experiments/evaluate.py
   ```

## Verification Commands

After fixing issues, verify your setup:

```bash
# 1. Test imports
python -c "import gymnasium; import torch; import cv2; print('All imports OK')"

# 2. Test environment
python test_environment.py

# 3. Quick training test (1 minute)
python experiments/train_sac.py --config configs/config.yaml &
# Wait 1 minute, then Ctrl+C

# 4. Check if model was created
ls models/sac/
```

## Getting More Help

If issues persist:

1. **Check Python version:**
   ```bash
   python --version  # Should be 3.8+
   ```

2. **Create fresh virtual environment:**
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Enable verbose error messages:**
   ```python
   # Add at top of script
   import traceback
   import sys
   sys.excepthook = lambda *args: traceback.print_exception(*args)
   ```

4. **Check GitHub issues:** Look for similar problems in the repository

5. **Create minimal reproduction:**
   ```bash
   python -c "from environments import make_env; import yaml; \
              config = yaml.safe_load(open('configs/config.yaml')); \
              env = make_env(config); print('Success!')"
   ```

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Import errors | `pip install --upgrade -r requirements.txt` |
| Display errors | Use scripts without `--render` flag |
| Slow training | Reduce `frame_size` and `total_timesteps` in config |
| Memory errors | Reduce `batch_size` and `buffer_size` in config |
| Poor performance | Tune PID gains or train longer |
| No GPU | Install PyTorch with CUDA support |

## Still Having Issues?

1. Copy the full error message
2. Note your Python version and OS
3. Check if the issue is in the GitHub issues
4. Include output of: `pip list | grep -E "(gymnasium|torch|numpy)"`
