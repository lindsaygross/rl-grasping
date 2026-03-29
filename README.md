# Robotic Grasping with Reinforcement Learning

Train a Kuka IIWA robot arm to grasp objects using PPO in PyBullet simulation.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Runs on CPU (no GPU needed).

## Train

```bash
python train.py
```

Trains a PPO agent for 300k timesteps (~30-60 min on a laptop). Progress is printed to the terminal. Best model is saved to `best_model/best_model.zip`.

To monitor training live with TensorBoard:

```bash
tensorboard --logdir ./logs/tensorboard/
```

## Evaluate

```bash
python evaluate.py
```

Runs 20 episodes with the best model and prints success rate and average reward. Results are saved to `results.txt`.

Options:
- `--model PATH` — use a specific model file (default: `best_model/best_model.zip`)
- `--episodes N` — number of eval episodes (default: 20)
- `--render` — show PyBullet GUI visualization

Example with GUI rendering:

```bash
python evaluate.py --render --episodes 5
```

## Plot Training Curves

```bash
python plot_results.py
```

Plots episode reward and length over training from Monitor CSV logs. Saves `training_curve.png`.

## Project Structure

```
rl-grasping/
├── env.py                  # Custom PyBullet Gymnasium environment
├── train.py                # PPO training with SB3
├── evaluate.py             # Evaluation and demo
├── plot_results.py         # Plot training curves
├── sim2real_discussion.md  # Sim-to-real transfer discussion
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## Design Decisions

- **Action space**: 3D delta end-effector position (dx, dy, dz) using inverse kinematics. Simpler than joint-space control and faster to learn.
- **Gripper**: Heuristic auto-grasp that closes when the end-effector is within 5cm of the object. A fixed constraint holds the object during the grasp to compensate for PyBullet's limited contact model.
- **Observations**: Joint positions (7) + EE position (3) + gripper state (1) + object position (3) = 14D.
- **Reward**: Dense (negative distance) + proximity bonus + lift bonus (+10) + time penalty (-0.01).

## Results

*Run `python train.py` then `python evaluate.py` to fill in.*

| Metric | Value |
|--------|-------|
| Training timesteps | 300k |
| Avg eval reward | — |
| Grasp success rate | — |
