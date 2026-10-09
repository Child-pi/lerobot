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
Evaluation script with Decoupled 2-Axis Lock & 1D Axial Probe-Grasp Strategy
兩軸鎖定 + 單軸 Zoom In 觸碰夾取策略評估控制器
"""

import logging
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
import numpy as np
import torch
import draccus

from lerobot.common.constants import ACTION
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.environments.factory import make_env
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import get_safe_torch_device
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.processor import (
    make_env_pre_post_processors,
    make_pre_post_processors,
)
from lerobot.scripts.lerobot_eval import eval_policy_all


class DecoupledAxialProbeWrapper(torch.nn.Module):
    """
    兩軸鎖定 + 單軸 Zoom In 觸碰夾取策略控制器 (Decoupled 2-Axis Lock & 1D Axial Probe-Grasp)
    
    運作程序：
    1. 粗定位階段 (Coarse Alignment)：依循 Policy 原生輸出移至目標上方。
    2. 兩軸鎖定階段 (Lock 2 Orthogonal Axes)：當就定位時，鎖定任兩軸 (例如水平 X, Y 軸)，
       強制抑制橫向漂移與抖動 (lock_damping)。
    3. 單軸 Zoom-In 探測 (1-DOF Axial Probing)：只允許深度/推進軸沿著方向向前/向下逼近。
    4. 觸碰夾取 (Touch & Grasp)：抵達探測目標深度後，觸發夾爪強力閉合夾取或插入！
    """
    def __init__(
        self,
        policy,
        align_threshold_steps: int = 40,
        lock_damping: float = 0.95,
        probe_gain: float = 1.15,
        auto_grasp_step: int = 120,
        enabled: bool = True,
    ):
        super().__init__()
        self.policy = policy
        self.align_threshold_steps = align_threshold_steps
        self.lock_damping = lock_damping
        self.probe_gain = probe_gain
        self.auto_grasp_step = auto_grasp_step
        self.enabled = enabled
        self.step_counter = 0
        self.locked_state = None

    @property
    def config(self):
        return self.policy.config

    def reset(self):
        self.step_counter = 0
        self.locked_state = None
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def select_action(self, batch):
        action = self.policy.select_action(batch)
        if not self.enabled:
            return action

        self.step_counter += 1
        state = batch.get("observation.state")
        if state is None:
            return action

        # 進入就定位區域 (step >= align_threshold_steps)
        if self.step_counter >= self.align_threshold_steps:
            if self.locked_state is None:
                self.locked_state = state.clone()

            # Aloha 14-DOF 動作結構：
            # [0:6] 左臂 (waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_yaw), [6] 左夾爪
            # [7:13] 右臂 (waist, shoulder, elbow, forearm_roll, wrist_pitch, wrist_yaw), [13] 右夾爪

            # 1. 右操作臂：鎖定 Joint 7 (腰部旋轉) 與 Joint 8 (肩部前後) -> 兩軸鎖定！
            action[:, 7] = self.locked_state[:, 7] * self.lock_damping + action[:, 7] * (1.0 - self.lock_damping)
            action[:, 8] = self.locked_state[:, 8] * self.lock_damping + action[:, 8] * (1.0 - self.lock_damping)

            # 2. 剩下那一軸 (Joint 9: 肘部升降/伸展) 執行單軸 Zoom In 逼近
            action[:, 9] = action[:, 9] * self.probe_gain

            # 3. 左夾持臂：同步鎖定水平兩軸防抖
            action[:, 0] = self.locked_state[:, 0] * self.lock_damping + action[:, 0] * (1.0 - self.lock_damping)
            action[:, 1] = self.locked_state[:, 1] * self.lock_damping + action[:, 1] * (1.0 - self.lock_damping)

            # 4. 當單軸深入探測接觸 (step >= auto_grasp_step)，強力觸發夾爪閉合！
            if self.step_counter >= self.auto_grasp_step:
                action[:, 13] = torch.clamp(action[:, 13] + 0.8, 0.0, 1.0)
                action[:, 6] = torch.clamp(action[:, 6] + 0.8, 0.0, 1.0)

        return action


@parser.wrap()
def main(cfg: EvalPipelineConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)

    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    base_policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )
    base_policy.eval()

    # 包裹「兩軸鎖定 + 單軸 Zoom In 觸碰夾取」策略控制器
    policy = DecoupledAxialProbeWrapper(
        base_policy,
        align_threshold_steps=40,
        lock_damping=0.95,
        probe_gain=1.15,
        auto_grasp_step=120,
        enabled=True,
    )

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=cfg.env, policy_cfg=cfg.policy
    )

    recording_dir = Path(cfg.output_dir) / "recordings" if cfg.eval.recording else None
    max_episodes_rendered = 0 if cfg.eval.recording else 10
    videos_dir = None if cfg.eval.recording else Path(cfg.output_dir) / "videos"

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_postprocessor=env_postprocessor,
            n_episodes=cfg.eval.n_episodes,
            videos_dir=videos_dir,
            max_episodes_rendered=max_episodes_rendered,
            start_episode_index=0,
            recordings_dir=recording_dir,
            return_observations=False,
            device=device,
        )

    logging.info(f"評估結果: {info}")


if __name__ == "__main__":
    main()
