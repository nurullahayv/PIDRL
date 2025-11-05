# Error Vector Nullification: Technical Concept

## Overview

This document provides a technical explanation of the **error vector nullification** approach used in our dogfight HUD-inspired pursuit-evasion control research.

## Conceptual Framework

### 1. Egocentric Reference Frame

Unlike traditional pursuit-evasion formulations where both agent and target positions are defined in a global coordinate system, our approach is **purely egocentric**:

- **Agent Reference Frame**: The agent is always at the origin of its own coordinate system
- **World Transforms**: As the agent moves in the global frame, its local frame moves with it
- **Observation**: The target's position is observed relative to the agent's current position and orientation

Mathematically:
```
r_target^local = r_target^global - r_agent^global
```

Where:
- `r_target^local` is the target position in agent's frame (what the agent "sees")
- `r_target^global` is the target position in world coordinates
- `r_agent^global` is the agent position in world coordinates

### 2. Error Vector Definition

The **error vector** e(t) is defined as:

```
e(t) = r_target^local(t)
```

In our egocentric frame, the error vector **is** the target's relative position. The agent's goal is to nullify this vector:

```
Goal: ||e(t)|| → 0
```

### 3. Control Objective

The controller must generate acceleration commands a(t) that drive the error vector to zero:

```
a(t) = f(e(t), ė(t), ∫e(τ)dτ, ...)
```

Where:
- `f()` is the control law (PID, RL policy, etc.)
- `e(t)` is the current error
- `ė(t)` is the error rate of change
- `∫e(τ)dτ` is the accumulated error (for integral control)

### 4. Why "Dogfight HUD"?

The visualization is inspired by fighter jet Heads-Up Displays (HUDs):

1. **Fixed Crosshair**: Like a flight HUD, the agent's crosshair is fixed at center
2. **Target Indicator**: The target appears as a tracked object on the HUD
3. **Vector Display**: The error vector is explicitly shown (like a lead indicator)
4. **Status Information**: Error magnitude, velocity, and lock status are displayed

This abstraction deliberately simplifies the problem to focus on the **control challenge** rather than perception complexity.

## Problem Formulation

### State Space

**Agent State (not directly observed):**
```
x_agent = [p_x, p_y, v_x, v_y]^T
```

**Target State (not directly observed):**
```
x_target = [p_x, p_y, v_x, v_y]^T
```

**Observation Space (what the agent sees):**
```
o(t) = stack(I_t-3, I_t-2, I_t-1, I_t)
```

Where I_t is a 64×64 grayscale image with:
- Background: Black (0)
- Target: White circle (255) at position e(t)
- Rendered from egocentric perspective

### Action Space

Continuous 2D acceleration:
```
a ∈ [-a_max, a_max]²
```

Where `a_max` is the maximum acceleration magnitude.

### Dynamics

**Agent Dynamics** (with friction):
```
v_agent(t+1) = γ·v_agent(t) + a(t)·Δt
p_agent(t+1) = p_agent(t) + v_agent(t)·Δt
```

Where:
- γ ∈ (0,1) is a friction coefficient
- Δt is the timestep duration

**Target Dynamics** (Brownian motion):
```
a_target(t) ~ N(0, σ²I)
v_target(t+1) = γ·v_target(t) + a_target(t)·Δt
p_target(t+1) = p_target(t) + v_target(t)·Δt
```

Where σ is the Brownian motion standard deviation.

### Reward Function

```
r(t) = -α·||e(t)||²
```

With penalty for losing track:
```
r(t) = -α·||e(t)||² - 100   if ||e(t)|| > r_max
```

Where:
- α is a reward scaling factor
- r_max is the maximum view radius

This reward structure incentivizes:
1. **Minimizing error magnitude** (staying close)
2. **Keeping target in view** (avoiding large errors)

## Control Approaches

### Approach 1: PID Controller

The PID controller operates on the error vector directly:

```
a(t) = K_p·e(t) + K_i·∫e(τ)dτ + K_d·ė(t)
```

**Error extraction**:
1. Use OpenCV to detect white blob in image
2. Convert pixel position to error vector in world coordinates
3. Apply PID control law

**Advantages**:
- Interpretable, tunable parameters
- No training required
- Fast inference

**Disadvantages**:
- Requires manual visual detection
- No memory of past observations
- Assumes perfect detection

### Approach 2: Kalman Filter + PID

Enhances PID with state estimation:

```
State: x̂ = [e_x, e_y, ė_x, ė_y]^T

Prediction: x̂(t|t-1) = F·x̂(t-1|t-1)
Update: x̂(t|t) = x̂(t|t-1) + K·(z(t) - H·x̂(t|t-1))

Control: a(t) = K_p·e(t|t) + K_i·∫e(τ)dτ + K_d·ė(t|t)
```

Where:
- F is the state transition matrix (constant velocity model)
- K is the Kalman gain
- z(t) is the noisy measurement from vision

**Advantages**:
- Smooths noisy measurements
- Estimates velocity for better derivative term
- More robust to detection failures

**Disadvantages**:
- Still requires vision system
- Assumes linear dynamics
- Manual tuning of noise parameters

### Approach 3: Deep Reinforcement Learning (SAC)

End-to-end learning from pixels:

```
π(a|o) = Neural Network Policy
a(t) ~ π(·|o(t))
```

**Architecture**:
```
Input: (4, 64, 64) stacked frames
  ↓
CNN Feature Extractor (3 conv layers)
  ↓
MLP Actor-Critic (2 layers each)
  ↓
Output: Action distribution
```

**Advantages**:
- No manual feature engineering
- Learns optimal policy from experience
- Can capture temporal patterns via frame stacking

**Disadvantages**:
- Requires extensive training
- Black-box (less interpretable)
- Sample inefficiency

## Research Contributions

This framework enables fair comparison of:

1. **Classical vs. Modern Control**: PID vs. Deep RL
2. **Role of State Estimation**: Raw measurements vs. Kalman filtering
3. **Vision-Based Control**: Learning from pixels vs. engineered features
4. **Error Vector Paradigm**: Egocentric formulation simplifies the problem

## Key Insights

1. **Abstraction Level**: By simplifying to a "radar view," we isolate the control problem from perception complexity

2. **Egocentric Perspective**: The error vector formulation naturally arises from first-person viewpoint

3. **Nullification Goal**: Unlike "reach target" objectives, our goal is "maintain zero error," which is subtly different and more challenging with dynamic targets

4. **Comparable Baselines**: All three approaches use the same observation space and reward function, enabling direct comparison

## Applications

This research applies to:

- **UAV Target Tracking**: Drones tracking moving objects
- **Missile Guidance**: Proportional navigation problems
- **Robot Pursuit**: Mobile robots following targets
- **Game AI**: NPC behavior in pursuit scenarios
- **Human-Robot Interaction**: Following or escorting behaviors

## References

This approach draws inspiration from:

- **Proportional Navigation**: Classical missile guidance
- **Visual Servoing**: Robot control from visual feedback
- **Pursuit-Evasion Games**: Differential game theory
- **Deep RL for Control**: Learning-based approaches to continuous control

---

For implementation details, see the main README and source code documentation.
