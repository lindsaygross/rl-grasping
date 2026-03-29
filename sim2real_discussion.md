# Sim-to-Real Transfer Challenges for Robotic Grasping

## Introduction

Training robotic grasping policies in simulation is attractive because it is fast, safe, and infinitely parallelizable. However, deploying a sim-trained policy on real hardware — the **sim-to-real transfer** problem — remains one of the hardest open challenges in robot learning. This document discusses the key obstacles and current solutions, with specific attention to what would happen if we tried to deploy our PyBullet-trained Kuka policy on a physical robot.

---

## 1. The Reality Gap

Simulation physics engines like PyBullet use simplified models of contact, friction, and rigid-body dynamics. The gap between these models and real-world physics is called the **reality gap**.

**Contact dynamics.** PyBullet resolves contacts using impulse-based solvers with a fixed time step. Real contacts involve surface deformation, micro-slip, and stick-slip transitions that these solvers do not capture. For grasping, this matters enormously: whether a cube slips from a gripper depends on the exact distribution of contact forces across the finger pads, which simulation approximates crudely.

**Friction.** Coulomb friction in simulation is parameterized by a single coefficient. Real friction depends on surface texture, temperature, moisture, wear, and contact area. A policy that learns to rely on a specific friction value in sim will fail when the real object has a different surface finish.

**Object physics.** Simulation assumes rigid bodies. Real objects — especially food, cloth, or rubber — deform under contact. Even rigid objects like a ceramic mug have mass distributions, center-of-gravity offsets, and surface imperfections that differ from their simulated models.

**Actuator dynamics.** In simulation, joint torques are applied instantaneously and perfectly. Real motors have bandwidth limits, backlash, friction in the gearbox, and communication delays. A policy that issues rapid, precise joint corrections in sim may produce jerky or unstable behavior on real hardware.

---

## 2. Sensor Discrepancies

Our simulated environment uses **perfect state information**: the exact 3D position of the object and the exact joint angles and end-effector pose. A real system has none of this.

**Vision.** A real Kuka arm would estimate the object pose from RGB or depth cameras, which introduce noise, occlusion, lighting variation, and calibration error. Object detection and pose estimation models have their own failure modes (e.g., reflective or transparent objects).

**Proprioception.** Real joint encoders are accurate but still have quantization noise and may drift. The forward kinematics model used to compute the end-effector pose assumes perfect link dimensions, which may be off by millimeters.

**Force/tactile sensing.** Our simulation does not model force feedback at all. Real grasping benefits greatly from tactile sensors that detect slip and contact — their absence in sim means the policy cannot learn reactive grasp adjustments.

---

## 3. Domain Randomization

**Domain randomization** addresses the reality gap by training the policy across a wide distribution of simulation parameters, so that the real world appears as just another sample from that distribution.

Common parameters to randomize for grasping:

- Object mass, friction coefficient, and size
- Table height and surface friction
- Lighting conditions and camera pose (for vision-based policies)
- Actuator gains, delays, and noise
- Object textures and colors

**Example: OpenAI's Rubik's Cube (2019).** OpenAI trained a dexterous manipulation policy entirely in simulation using massive domain randomization across hundreds of physical parameters (gravity, friction, object dimensions, actuator noise). The resulting policy could solve a Rubik's cube with a real Shadow Dexterous Hand — despite never seeing the real world during training. This demonstrated that sufficient randomization can substitute for accurate simulation.

The key insight is that the policy learns to be robust rather than optimal: it develops strategies that work across many parameter settings, including (hopefully) reality.

---

## 4. System Identification

An alternative to randomization is **system identification**: measuring the real system's physical parameters and calibrating the simulator to match.

For a Kuka arm, this would involve:

- Measuring actual link masses and inertias (or fitting them from trajectory data)
- Identifying motor gains, joint friction, and backlash
- Measuring the gripper's friction coefficients with target objects
- Calibrating camera intrinsics and extrinsics

System identification can produce very accurate simulators for a specific setup, but it is brittle: any change to the hardware, gripper, or object set requires re-identification. In practice, many teams combine system identification (to get the simulator in the right ballpark) with domain randomization (to handle residual inaccuracies).

---

## 5. Sim-to-Real Solutions

Beyond domain randomization, several techniques have been proposed to close the reality gap:

**Progressive networks (Rusu et al., 2017).** Train a policy in sim, then use a progressive neural network to adapt to real data. The sim-trained columns are frozen and lateral connections allow the real-world column to leverage sim-learned features without forgetting them.

**Transfer learning / fine-tuning.** Pre-train in simulation, then fine-tune on a small number of real-world episodes. This works when the sim policy provides a reasonable initialization, but can be sample-inefficient if the gap is large.

**Residual reinforcement learning.** Train a base policy in sim that handles the coarse behavior (reaching, approach trajectory). Then train a small residual policy on real hardware that outputs corrections on top of the base policy's actions. This limits the real-world search space, requiring fewer real samples. Johannink et al. (2019) demonstrated this approach for block pushing and peg insertion tasks.

**Sim-to-real via GAN-based adaptation.** Use a generative adversarial network to translate simulated images to look like real images (or vice versa), so a vision-based policy trained on "fake real" images transfers better. GraspGAN (Bousmalis et al., 2018) applied this to robotic grasping with RGB input.

---

## 6. Our Project: What Would Break?

If we took our PPO policy trained in PyBullet and deployed it on a physical Kuka IIWA with a gripper, several things would immediately fail:

### Observation mismatch
Our policy observes the object's exact 3D position from the simulator state. On a real robot, we would need a perception pipeline (e.g., an RGB-D camera with object detection) to estimate this, introducing latency and noise. The policy has never seen noisy observations and would likely behave erratically.

### Gripper auto-grasp heuristic
We use a distance-based heuristic to trigger grasping and a fixed constraint to hold the object. This is not physically meaningful. A real gripper would need to actually close fingers with appropriate force, and the grasp success would depend on the approach angle, finger placement, and friction — none of which our policy has learned to control.

### No force feedback
Our policy has no concept of contact forces. On a real arm, it might push the object off the table before attempting to grasp, or apply too much force and damage the gripper or object. Real grasping policies typically use force/torque sensing to modulate grip strength.

### Action scale and timing
Our policy issues actions at simulation speed (10 sim steps per action, ~24 Hz effective control). The real Kuka IIWA has specific control loop rates and joint velocity/torque limits that differ from our simulation. The action scaling (0.05m per step) might be too aggressive or too timid.

### What we would need to change
1. **Replace state-based observations with camera input** — or add a perception module that estimates object pose from cameras and feeds it to the policy.
2. **Replace the auto-grasp heuristic with learned gripper control** — add gripper open/close to the action space and train the policy to decide when and how hard to grasp.
3. **Add domain randomization** — randomize object mass, friction, position noise, action delays, and camera noise during training.
4. **Add force feedback to observations** — include wrist force/torque sensor readings so the policy can react to contact.
5. **Fine-tune on real hardware** — use residual RL or a small number of real-world rollouts to adapt to the physical system.

---

## Conclusion

Our project demonstrates the core RL loop for robotic grasping: defining an environment, training a policy with PPO, and evaluating grasp success. This is a valuable foundation, but sim-to-real transfer would require substantial additional engineering — primarily in perception, domain randomization, and hardware-aware training. The field is making rapid progress on these challenges, but robust sim-to-real grasping remains an active research area.

---

## References

- Tobin, J. et al. (2017). "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World."
- OpenAI et al. (2019). "Solving Rubik's Cube with a Robot Hand." arXiv:1910.07113.
- Rusu, A. et al. (2017). "Sim-to-Real Robot Learning from Pixels with Progressive Nets." CoRL.
- Johannink, T. et al. (2019). "Residual Reinforcement Learning for Robot Control." ICRA.
- Bousmalis, K. et al. (2018). "Using Simulation and Domain Adaptation to Improve Efficiency of Deep Robotic Grasping." ICRA.
