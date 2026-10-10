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
        align_threshold_steps: int = 200,
        lock_damping: float = 0.70,
        probe_gain: float = 1.25,
        auto_grasp_step: int = 200,
        locked_axes: list = None,
        probe_axes: list = None,
        adaptive_convergence: bool = True,
        convergence_delta: float = 0.015,
    ):
        self.align_threshold_steps = align_threshold_steps
        self.lock_damping = lock_damping
        self.probe_gain = probe_gain
        self.auto_grasp_step = auto_grasp_step
        
        # 多軸配置：
        # Aloha 14-DOF:
        # 左臂 [0:6], 左夾爪 [6] | 右臂 [7:13], 右夾爪 [13]
        # 鎖定 X, Y 平面軸與姿態穩定軸 (預設 0,1 左臂腰肩, 3,4,5 左臂手腕固定插座姿態, 10,11,12 右臂手腕三軸姿態鎖定，特別固定 11 號俯仰角防止插頭上翹)
        self.locked_axes = locked_axes if locked_axes is not None else [0, 1, 3, 4, 5, 10, 11, 12]
        
        # 沿水平插接軸深入探索 (預設 7,8,9: 右腰Yaw向內、右肩Pitch向前、右肘Pitch向前，三軸協同水平直線插入母座)
        self.probe_axes = probe_axes if probe_axes is not None else [7, 8, 9]
        
        self.adaptive_convergence = adaptive_convergence
        self.convergence_delta = convergence_delta
        
        self.step_counter = 0
        self.is_aligned = False
        self.locked_pos = None
        self.prev_state = None
        self.stable_counter = 0

    def reset(self):
        self.step_counter = 0
        self.is_aligned = False
        self.locked_pos = None
        self.prev_state = None
        self.stable_counter = 0

    def apply(self, action, batch):
        self.step_counter += 1
        action = action.clone()

        state = None
        if isinstance(batch, dict):
            state = batch.get("observation.state")
            if state is None:
                for k, v in batch.items():
                    if "state" in k and isinstance(v, torch.Tensor):
                        state = v
                        break

        # 檢測 (X,Y) 是否已就定位
        if not self.is_aligned:
            if self.step_counter >= self.align_threshold_steps:
                self.is_aligned = True
            elif self.adaptive_convergence and state is not None and self.prev_state is not None:
                # 只有在經過充分交會逼近期後才允許自適應就定位判定
                xy_diff = torch.abs(state[..., [7, 8]] - self.prev_state[..., [7, 8]]).sum().item()
                if xy_diff < self.convergence_delta:
                    self.stable_counter += 1
                    min_step = int(self.align_threshold_steps * 0.85)
                    if self.stable_counter >= 3 and self.step_counter >= min_step:
                        self.is_aligned = True
                        logging.info(f"🎯 [Strategy] (X,Y) 自適應判定就定位 (Step {self.step_counter}, Diff: {xy_diff:.4f})")
                else:
                    self.stable_counter = 0

            if state is not None:
                self.prev_state = state.clone()

        # (X,Y) 就定位後：鎖定水平多軸，開始往 Z 座標深入探索！
        is_1d = (action.ndim == 1)
        act = action.unsqueeze(0) if is_1d else action

        if self.is_aligned:
            if self.locked_pos is None:
                if state is not None and hasattr(state, "shape") and state.shape[-1] >= 14:
                    self.locked_pos = state.to(device=action.device, dtype=action.dtype).clone()
                else:
                    self.locked_pos = action.clone()

            lock_p = self.locked_pos.unsqueeze(0) if self.locked_pos.ndim == 1 else self.locked_pos

            # 1. 多軸鎖定：固定 (X,Y) 平面與姿態穩定軸 (以就定位點 lock_p 為基準阻尼抑制漂移)
            for axis in self.locked_axes:
                if axis < act.shape[-1]:
                    delta = act[:, axis] - lock_p[:, axis]
                    # 手腕俯仰軸 (11) 特別施加高阻尼 (0.90) 確保水平水平推進、嚴防插頭上翹
                    damping = 0.90 if axis == 11 else self.lock_damping
                    act[:, axis] = lock_p[:, axis] + delta * (1.0 - damping)

            # 2. 往水平插接軸探索：多軸協同推進 (腰7, 肩8, 肘9 聯動水平直線向前插入)
            for axis in self.probe_axes:
                if axis < act.shape[-1]:
                    delta = act[:, axis] - lock_p[:, axis]
                    act[:, axis] = lock_p[:, axis] + delta * self.probe_gain

        # 3. 全程防脫落夾爪鎖緊保護 (Anti-Slip Clamp Guard):
        # Aloha 規格 0.0 為緊閉 (Closed)，1.0 為完全張開 (Open)。
        # 在 AlohaInsertion-v0 對接任務中，雙手預設即持有物件；夾爪張開會直接導致插頭/插座滑落。
        # 因此全程強制將夾爪指令限縮在牢固緊閉區間 [0.0, 0.03]，杜絕滑脫。
        if act.shape[-1] >= 14:
            act[:, 13] = torch.clamp(act[:, 13], 0.0, 0.03)  # 右夾爪鎖死 (緊持插頭)
            act[:, 6] = torch.clamp(act[:, 6], 0.0, 0.03)    # 左夾爪鎖死 (緊持插座)

        action = act.squeeze(0) if is_1d else act
        return action


# 匯入官方評估模組並套用策略注入 (完全保留 PreTrainedPolicy 原生型別檢查)
import os
import lerobot.policies as policies_module
import lerobot.scripts.lerobot_eval as eval_module

original_make_policy = eval_module.make_policy

def parse_int_list(val, default):
    if not val:
        return default
    try:
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    except Exception:
        return default

def patched_make_policy(*args, **kwargs):
    policy = original_make_policy(*args, **kwargs)
    
    # 從環境變數動態讀取超參數配置
    align_thresh = int(os.environ.get("STRATEGY_ALIGN_THRESHOLD_STEPS", "200"))
    lock_damping = float(os.environ.get("STRATEGY_LOCK_DAMPING", "0.75"))
    probe_gain = float(os.environ.get("STRATEGY_PROBE_GAIN", "1.25"))
    auto_grasp = int(os.environ.get("STRATEGY_AUTO_GRASP_STEP", "200"))
    locked_axes = parse_int_list(os.environ.get("STRATEGY_LOCKED_AXES"), [0, 1, 3, 4, 5, 10, 11, 12])
    probe_axes = parse_int_list(os.environ.get("STRATEGY_PROBE_AXES"), [7, 8, 9])

    logging.info("=" * 60)
    logging.info("🕹️ [Strategy Engine] 啟用「(X,Y) 定位後 -> 往 Z 軸探索」多軸策略控制器！")
    logging.info(f"   - (X,Y) 鎖定軸組 (Locked Axes): {locked_axes} (鎖定阻尼: {lock_damping})")
    logging.info(f"   - Z 探索軸組 (Probe Axes): {probe_axes} (Z 探索增益: {probe_gain})")
    logging.info(f"   - 就定位門檻: {align_thresh} 步 | 觸碰閉合步數: {auto_grasp} 步")
    logging.info("=" * 60)
    
    controller = DecoupledAxialProbeController(
        align_threshold_steps=align_thresh,
        lock_damping=lock_damping,
        probe_gain=probe_gain,
        auto_grasp_step=auto_grasp,
        locked_axes=locked_axes,
        probe_axes=probe_axes,
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
