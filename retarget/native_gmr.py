"""GMR frame-task solve path for X2 without robot-specific post-solvers."""

import mink
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting import GeneralMotionRetargeting


class NativeGMR(GeneralMotionRetargeting):
    """Use GMR/Mink tasks directly, without analytic or projected IK passes."""

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
        self.tasks1 = []
        self.tasks2 = []

        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []

        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

        # X2 has narrow waist-pitch limits and an Euler null space in its
        # shoulder chain.  A native Mink posture task selects the neutral IK
        # branch while the much stronger frame tasks still own the motion.
        posture_cost = np.full(self.model.nv, 0.05)
        posture_cost[:6] = 0.0
        for joint_id in range(self.model.njnt):
            name = self.model.joint(joint_id).name
            dof_address = self.model.jnt_dofadr[joint_id]
            if name and ("wrist" in name or "head" in name):
                posture_cost[dof_address] = 0.5
            if name and "shoulder_pitch_joint" in name:
                posture_cost[dof_address] = 2.0
            if name and "shoulder_yaw_joint" in name:
                posture_cost[dof_address] = 8.0
            if name and "shoulder_roll_joint" in name:
                # The X2 Euler shoulder has an equivalent solution near
                # +/-pi roll.  Without a roll null-space cost, otherwise
                # ordinary arm poses can jump onto that inverted branch.
                posture_cost[dof_address] = 5.0
            if name == "waist_pitch_joint":
                posture_cost[dof_address] = 10.0
        posture_task = mink.PostureTask(self.model, cost=posture_cost)
        posture_task.set_target(self.model.qpos0.copy())
        self.tasks1.append(posture_task)
        self.tasks2.append(posture_task)

    def update_targets(self, human_data, offset_to_ground=False):
        human_data = self.to_numpy(human_data)
        scaled_human_data = self.scale_human_data(
            human_data, self.human_root_name, self.human_scale_table
        )
        human_data1 = self.offset_human_data(
            scaled_human_data, self.pos_offsets1, self.rot_offsets1
        )
        human_data2 = self.offset_human_data(
            scaled_human_data, self.pos_offsets2, self.rot_offsets2
        )
        human_data1 = self.apply_ground_offset(human_data1)
        human_data2 = self.apply_ground_offset(human_data2)
        if offset_to_ground:
            human_data1 = self.offset_human_data_to_ground(human_data1)
            human_data2 = self.offset_human_data_to_ground(human_data2)
        self.scaled_human_data = human_data1

        if self.use_ik_match_table1:
            for body_name, task in self.human_body_to_task1.items():
                position, rotation = human_data1[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3(rotation), position
                    )
                )

        if self.use_ik_match_table2:
            for body_name, task in self.human_body_to_task2.items():
                position, rotation = human_data2[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(
                        mink.SO3(rotation), position
                    )
                )

    def retarget(self, human_data, offset_to_ground=False):
        self.update_targets(human_data, offset_to_ground)
        self._run_ik_stages()
        return self.configuration.data.qpos.copy()
