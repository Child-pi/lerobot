#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation runner with Decoupled 2-Axis Lock & 1D Axial Probe-Grasp Strategy
兩軸鎖定 + 單軸 Zoom In 觸碰夾取策略評估控制器
"""

import logging
import torch

class DecoupledAxialProbeController:
    """
    兩軸鎖定 + 單軸 Zoom In 觸碰夾取策略控制器 (Decoupled 2-Axis Lock & 1D Axial Probe-Grasp)
    
    運作程序：
    1. 粗定位階段 (Coarse Alignment)：依循 Policy 原生輸出移至目標上方。
    2. 兩軸鎖定階段 (Lock 2 Orthogonal Axes)：當就定位時 (step >= align_threshold_steps)，
       鎖定任兩軸 (例如水平 X, Y 軸)，強制抑制橫向漂移與抖動 (lock_damping)。
    3. 單軸 Zoom-In 探測 (1-DOF Axial Probing)：只允許深度/推進軸沿著方向向前/向下逼近。
    4. 觸碰夾取 (Touch & Grasp)：抵達探測目標深度後 (step >= auto_grasp_step)，觸發夾爪強力閉合夾取或插入！
    """
    def __init__(
        self,
        align_threshold_steps: int = 40,
        lock_damping: float = 0.95,
        probe_gain: float = 1.15,
        auto_grasp_step: int = 120,
    ):
        self.align_threshold_steps = align_threshold_steps
        self.lock_damping = lock_damping
        self.probe_gain = probe_gain
        self.auto_grasp_step = auto_grasp_step
        self.step_counter = 0
        self.locked_pos = None

    def reset(self):
        self.step_counter = 0
        self.locked_pos = None

    def apply(self, action, batch):
        self.step_counter += 1
        action = action.clone()

        # 進入就定位區域 (step >= align_threshold_steps)
        if self.step_counter >= self.align_threshold_steps:
            if self.locked_pos is None:
                state = None
                if isinstance(batch, dict):
                    state = batch.get("observation.state")
                    if state is None:
                        for k, v in batch.items():
                            if "state" in k and isinstance(v, torch.Tensor):
                                state = v
                                break
                if state is not None and hasattr(state, "shape") and state.shape[-1] >= 14:
                    self.locked_pos = state.to(device=action.device, dtype=action.dtype).clone()
                else:
                    self.locked_pos = action.clone()

            # 確保形狀為 2D 方便切片操作
            is_1d = (action.ndim == 1)
            act = action.unsqueeze(0) if is_1d else action
            lock_p = self.locked_pos.unsqueeze(0) if self.locked_pos.ndim == 1 else self.locked_pos

            # Aloha 14-DOF 動作結構：
            # [0:6] 左臂 (waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_yaw), [6] 左夾爪
            # [7:13] 右臂 (waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_yaw), [13] 右夾爪

            # 1. 右操作臂：鎖定 Joint 7 (腰部旋轉) 與 Joint 8 (肩部前後) -> 兩軸固定！
            act[:, 7] = lock_p[:, 7] * self.lock_damping + act[:, 7] * (1.0 - self.lock_damping)
            act[:, 8] = lock_p[:, 8] * self.lock_damping + act[:, 8] * (1.0 - self.lock_damping)

            # 2. 剩下那一軸 (Joint 9: 肘部升降/伸展) 執行單軸 Zoom In 逼近
            act[:, 9] = act[:, 9] * self.probe_gain

            # 3. 左夾持臂：同步鎖定水平兩軸防抖
            act[:, 0] = lock_p[:, 0] * self.lock_damping + act[:, 0] * (1.0 - self.lock_damping)
            act[:, 1] = lock_p[:, 1] * self.lock_damping + act[:, 1] * (1.0 - self.lock_damping)

            # 4. 當單軸深入探測接觸 (step >= auto_grasp_step)，強力觸發夾爪閉合！
            if self.step_counter >= self.auto_grasp_step:
                act[:, 13] = torch.clamp(act[:, 13] + 0.8, 0.0, 1.0)
                act[:, 6] = torch.clamp(act[:, 6] + 0.8, 0.0, 1.0)

            action = act.squeeze(0) if is_1d else act

        return action


# 匯入官方評估模組並套用策略注入 (完全保留 PreTrainedPolicy 原生型別檢查)
import lerobot.policies as policies_module
import lerobot.scripts.lerobot_eval as eval_module

original_make_policy = eval_module.make_policy

def patched_make_policy(*args, **kwargs):
    policy = original_make_policy(*args, **kwargs)
    logging.info("🕹️ [Strategy Engine] 已成功注入「兩軸鎖定 + 單軸 Zoom In 觸碰夾取」策略控制器！")
    
    controller = DecoupledAxialProbeController(
        align_threshold_steps=40,
        lock_damping=0.95,
        probe_gain=1.15,
        auto_grasp_step=120,
    )
    
    orig_select_action = policy.select_action
    orig_reset = getattr(policy, "reset", None)

    def select_action_wrapper(batch):
        action = orig_select_action(batch)
        return controller.apply(action, batch)

    def reset_wrapper():
        controller.reset()
        if orig_reset is not None:
            orig_reset()

    # 保留原生 PreTrainedPolicy 實例型別，動態掛載策略方法
    policy.select_action = select_action_wrapper
    policy.reset = reset_wrapper
    return policy

eval_module.make_policy = patched_make_policy
policies_module.make_policy = patched_make_policy

if __name__ == "__main__":
    eval_module.eval_main()
