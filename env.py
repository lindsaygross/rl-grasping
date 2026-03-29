"""
Custom Gymnasium environment for Kuka IIWA robotic grasping in PyBullet.

Action space: 3D delta end-effector position (dx, dy, dz).
Gripper uses heuristic auto-grasp: closes when EE is near the target object.
Observation: joint positions (7) + EE position (3) + gripper state (1) + object position (3) = 14D.
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class KukaGraspEnv(gym.Env):
    metadata = {"render_modes": ["human", "headless"], "render_fps": 60}

    def __init__(self, render_mode="headless", max_steps=200):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_count = 0

        # Action: delta EE position (dx, dy, dz), scaled by 0.05 per step
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: joint_pos(7) + ee_pos(3) + gripper_state(1) + obj_pos(3) = 14
        obs_low = -np.ones(14, dtype=np.float32) * 5.0
        obs_high = np.ones(14, dtype=np.float32) * 5.0
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Grasp parameters
        self.grasp_distance = 0.05  # distance threshold to trigger auto-grasp
        self.lift_height = 0.2  # object must be above this z to count as grasped
        self.ee_action_scale = 0.05  # max EE movement per step

        # PyBullet state
        self.physics_client = None
        self.kuka_id = None
        self.object_id = None
        self.table_id = None
        self.gripper_closed = False
        self.grasp_constraint = None

        # Kuka joint indices for the arm (not gripper)
        self.arm_joint_indices = list(range(7))
        # End-effector link index (tip of the arm before gripper)
        self.ee_link_index = 6
        # Gripper finger joint indices in kuka_with_gripper2.sdf
        # Joints 7-13 are gripper-related; 8 and 11 are the main finger joints
        self.finger_joint_indices = [8, 11]

        self._connect_physics()

    def _connect_physics(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except p.error:
                pass

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.physics_client)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pc = self.physics_client

        p.resetSimulation(physicsClientId=pc)
        p.setGravity(0, 0, -9.81, physicsClientId=pc)
        p.setTimeStep(1.0 / 240.0, physicsClientId=pc)

        # Load ground plane and table
        p.loadURDF("plane.urdf", physicsClientId=pc)
        self.table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, 0.0, -0.63],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            physicsClientId=pc,
        )

        # Load Kuka arm with gripper
        self.kuka_id = p.loadSDF(
            "kuka_iiwa/kuka_with_gripper2.sdf", physicsClientId=pc
        )[0]
        p.resetBasePositionAndOrientation(
            self.kuka_id, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=pc,
        )

        # Set arm to a reasonable starting pose (slightly raised)
        start_joint_positions = [0.0, 0.4, 0.0, -1.2, 0.0, 0.8, 0.0]
        for i, pos in zip(self.arm_joint_indices, start_joint_positions):
            p.resetJointState(self.kuka_id, i, pos, physicsClientId=pc)

        # Open gripper
        for j in self.finger_joint_indices:
            p.resetJointState(self.kuka_id, j, 0.0, physicsClientId=pc)

        # Spawn target object at random position on the table
        # Table surface is roughly at z=0.0, object region ~20x20cm in front of robot
        obj_x = self.np_random.uniform(0.4, 0.6)
        obj_y = self.np_random.uniform(-0.1, 0.1)
        obj_z = 0.05  # slightly above table surface

        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02],
                                        physicsClientId=pc)
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02],
                                     rgbaColor=[1, 0, 0, 1], physicsClientId=pc)
        self.object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[obj_x, obj_y, obj_z],
            physicsClientId=pc,
        )
        # Increase friction so the object doesn't slide too easily
        p.changeDynamics(self.object_id, -1, lateralFriction=1.5,
                         spinningFriction=0.5, physicsClientId=pc)

        # Reset state
        self.gripper_closed = False
        self.grasp_constraint = None
        self.step_count = 0

        # Let the scene settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=pc)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        pc = self.physics_client
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.ee_action_scale

        # Get current EE position and compute target
        ee_state = p.getLinkState(self.kuka_id, self.ee_link_index, physicsClientId=pc)
        current_ee_pos = np.array(ee_state[0])
        target_ee_pos = current_ee_pos + delta

        # Use IK to get joint positions for the target EE position
        # Keep the gripper pointing downward
        target_orn = p.getQuaternionFromEuler([0, -np.pi, 0])
        joint_positions = p.calculateInverseKinematics(
            self.kuka_id, self.ee_link_index, target_ee_pos.tolist(),
            targetOrientation=target_orn,
            maxNumIterations=100,
            residualThreshold=1e-4,
            physicsClientId=pc,
        )

        # Apply joint position control for the arm joints
        for i in range(7):
            p.setJointMotorControl2(
                self.kuka_id, i, p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=200,
                physicsClientId=pc,
            )

        # Heuristic auto-grasp: close gripper when EE is close to object
        obj_pos = np.array(p.getBasePositionAndOrientation(
            self.object_id, physicsClientId=pc)[0])
        ee_to_obj = np.linalg.norm(current_ee_pos - obj_pos)

        if ee_to_obj < self.grasp_distance and not self.gripper_closed:
            self._close_gripper()
        elif ee_to_obj > self.grasp_distance * 2 and self.gripper_closed:
            # Release if moved far away (shouldn't normally happen during a good grasp)
            self._open_gripper()

        # Step simulation multiple times for stability
        for _ in range(10):
            p.stepSimulation(physicsClientId=pc)

        self.step_count += 1

        # Compute reward
        obs = self._get_obs()
        reward, grasped = self._compute_reward(current_ee_pos, obj_pos)

        terminated = grasped
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {"is_success": grasped}

    def _compute_reward(self, ee_pos, obj_pos_before):
        """
        Reward components:
        1. Dense: negative distance between EE and object (encourages reaching)
        2. Proximity bonus: small reward when gripper is very close to object
        3. Lift bonus: +10 when object is above lift threshold
        4. Time penalty: -0.01 per step
        """
        pc = self.physics_client
        obj_pos = np.array(p.getBasePositionAndOrientation(
            self.object_id, physicsClientId=pc)[0])

        ee_state = p.getLinkState(self.kuka_id, self.ee_link_index, physicsClientId=pc)
        ee_pos_now = np.array(ee_state[0])

        distance = np.linalg.norm(ee_pos_now - obj_pos)

        # Dense distance reward (negative, closer = less penalty)
        reward = -distance

        # Proximity bonus: reward for being very close (within grasp range)
        if distance < self.grasp_distance:
            reward += 0.5

        # Check if object is lifted
        grasped = False
        if obj_pos[2] > self.lift_height and self.gripper_closed:
            reward += 10.0
            grasped = True

        # Time penalty
        reward -= 0.01

        return reward, grasped

    def _close_gripper(self):
        pc = self.physics_client
        # Close finger joints
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.kuka_id, j, p.POSITION_CONTROL,
                targetPosition=0.5,  # closed position
                force=50,
                physicsClientId=pc,
            )

        # Step a few times to let fingers close
        for _ in range(20):
            p.stepSimulation(physicsClientId=pc)

        # Create a fixed constraint to "grasp" the object if close enough
        # This compensates for PyBullet's imperfect friction/contact model
        ee_state = p.getLinkState(self.kuka_id, self.ee_link_index, physicsClientId=pc)
        obj_pos = p.getBasePositionAndOrientation(self.object_id, physicsClientId=pc)[0]
        dist = np.linalg.norm(np.array(ee_state[0]) - np.array(obj_pos))

        if dist < self.grasp_distance and self.grasp_constraint is None:
            self.grasp_constraint = p.createConstraint(
                self.kuka_id, self.ee_link_index,
                self.object_id, -1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.05],
                childFramePosition=[0, 0, 0],
                physicsClientId=pc,
            )
            self.gripper_closed = True

    def _open_gripper(self):
        pc = self.physics_client
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.kuka_id, j, p.POSITION_CONTROL,
                targetPosition=0.0,
                force=50,
                physicsClientId=pc,
            )
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint, physicsClientId=pc)
            self.grasp_constraint = None
        self.gripper_closed = False

    def _get_obs(self):
        pc = self.physics_client

        # Joint positions (7 arm joints)
        joint_positions = []
        for i in self.arm_joint_indices:
            joint_positions.append(p.getJointState(self.kuka_id, i, physicsClientId=pc)[0])

        # End-effector position
        ee_state = p.getLinkState(self.kuka_id, self.ee_link_index, physicsClientId=pc)
        ee_pos = list(ee_state[0])

        # Gripper state (0=open, 1=closed)
        gripper_state = [1.0 if self.gripper_closed else 0.0]

        # Object position
        obj_pos = list(p.getBasePositionAndOrientation(
            self.object_id, physicsClientId=pc)[0])

        obs = np.array(joint_positions + ee_pos + gripper_state + obj_pos, dtype=np.float32)
        return obs

    def render(self):
        # Rendering is handled by PyBullet GUI when render_mode="human"
        pass

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect(self.physics_client)
            except p.error:
                pass
            self.physics_client = None
