"""
Probe数据收集器实现。

这个文件补全了 run_collection.py 所需的 ProbeDataCollector，并提供最小的收集/保存逻辑，
使 dummy 策略下的快速测试能够运行。
"""
import numpy as np
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. Episode data will be saved as .npz files instead of .hdf5. Install h5py for HDF5 support.")
from pathlib import Path
from typing import Any, Dict, Optional

from configs.probe_actions import get_probe_config
from utils.contact_detection import ContactDetector, ProbeResponseRecorder


class ProbeDataCollector:
    def __init__(
        self,
        env,
        policy,
        task_name: str,
        save_dir: Path,
        max_episode_steps: int = 500,
        save_images: bool = True,
    ):
        self.env = env
        self.policy = policy
        self.task_name = task_name
        self.save_dir = Path(save_dir)
        self.max_episode_steps = max_episode_steps
        self.save_images = save_images

        self.contact_detector = ContactDetector(force_threshold=0.5, min_contact_steps=2)
        self.response_recorder = ProbeResponseRecorder()
        self.probe_config = get_probe_config(task_name)

        self.total_episodes = 0
        self.successful_episodes = 0

    def collect_episode(self, episode_index: int) -> Optional[Dict[str, Any]]:
        obs, info = self._reset_env()
        self.policy.reset()
        self.contact_detector.reset()
        self.response_recorder.reset()

        episode_images = []
        outcome = 'failure'
        last_reward = 0.0
        step = 0

        while step < self.max_episode_steps:
            image = self._get_rgb_image(obs)
            instruction = f"Execute task {self.task_name}"
            action = self.policy.predict(image, instruction, proprio=self._get_proprio(obs))
            action = np.asarray(action, dtype=np.float32)

            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False

            if self.save_images and image is not None:
                episode_images.append(image)

            self._record_step(obs, info, action)
            last_reward = float(reward) if np.isscalar(reward) else float(np.asarray(reward).sum())
            step += 1

            if terminated or truncated:
                break

        if self._infer_success(info, last_reward, step):
            outcome = 'success'

        episode_data = {
            'episode_id': int(episode_index),
            'task_name': self.task_name,
            'outcome': outcome,
            'steps': step,
            'reward': last_reward,
            'probe_response': self.response_recorder.compute_response(),
            'planned_action': action.tolist(),
        }

        self._save_episode(episode_data, episode_images, episode_index)

        self.total_episodes += 1
        if outcome == 'success':
            self.successful_episodes += 1

        return episode_data

    def get_statistics(self) -> Dict[str, Any]:
        success_rate = 0.0
        if self.total_episodes > 0:
            success_rate = self.successful_episodes / self.total_episodes

        return {
            'total_episodes': self.total_episodes,
            'successful_episodes': self.successful_episodes,
            'success_rate': success_rate,
        }

    def _reset_env(self):
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}
        return obs, info

    def _get_rgb_image(self, obs: Any) -> Optional[np.ndarray]:
        if obs is None:
            return None

        if isinstance(obs, dict):
            for key in ['rgb', 'image', 'rgb_image', 'color', 'img']:
                if key in obs:
                    image = obs[key]
                    return self._normalize_image(image)

            if 'rgbd' in obs and isinstance(obs['rgbd'], dict):
                return self._normalize_image(obs['rgbd'].get('rgb'))

        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            return obs

        return None

    def _normalize_image(self, image: Any) -> Optional[np.ndarray]:
        if image is None:
            return None
        image = np.asarray(image)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image

    def _get_proprio(self, obs: Any) -> Optional[np.ndarray]:
        if isinstance(obs, dict):
            if 'agent' in obs and 'state' in obs['agent']:
                return np.asarray(obs['agent']['state'], dtype=np.float32)
            if 'robot_state' in obs:
                return np.asarray(obs['robot_state'], dtype=np.float32)
        return None

    def _record_step(self, obs: Any, info: Dict[str, Any], action: np.ndarray):
        force_mag = 0.0
        is_contact = False
        if isinstance(obs, dict):
            tcp_wrench = obs.get('extra', {}).get('tcp_wrench') if 'extra' in obs else None
            if tcp_wrench is not None:
                force_mag = float(np.linalg.norm(np.asarray(tcp_wrench)[:3]))

        contacts = info.get('contacts', []) if isinstance(info, dict) else []
        if isinstance(contacts, dict) and 'contacts' in contacts:
            contacts = contacts['contacts']

        is_contact, contact_info = self.contact_detector.detect_contact(obs, contacts)

        ee_pos = self._extract_position(obs, 'ee_pos')
        obj_pos = self._extract_position(obs, 'obj_pos')
        if ee_pos is None:
            ee_pos = np.zeros(3)
        if obj_pos is None:
            obj_pos = np.zeros(3)

        self.response_recorder.record_step(force_mag, is_contact, ee_pos, obj_pos)

    def _extract_position(self, obs: Any, key: str) -> Optional[np.ndarray]:
        if not isinstance(obs, dict):
            return None
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32)
        if 'agent' in obs and key in obs['agent']:
            return np.asarray(obs['agent'][key], dtype=np.float32)
        return None

    def _infer_success(self, info: Dict[str, Any], reward: float, steps: int) -> bool:
        if isinstance(info, dict):
            if info.get('is_success') or info.get('success') or info.get('task_success'):
                return True

        return reward > 0.5

    def _save_episode(
        self,
        episode_data: Dict[str, Any],
        images: Any,
        episode_index: int,
    ):
        if HAS_H5PY:
            file_path = self.save_dir / f"episode_{episode_index:06d}.hdf5"
            with h5py.File(file_path, 'w') as f:
                for key, value in episode_data.items():
                    if key == 'probe_response':
                        grp = f.create_group('probe_response')
                        for subkey, subval in value.items():
                            grp.create_dataset(subkey, data=subval)
                    elif isinstance(value, str):
                        f.attrs[key] = value
                    elif np.isscalar(value):
                        f.attrs[key] = value
                    elif isinstance(value, list):
                        f.create_dataset(key, data=np.asarray(value))
                    else:
                        try:
                            f.create_dataset(key, data=np.asarray(value))
                        except Exception:
                            f.attrs[key] = str(value)

                if self.save_images and images:
                    img_group = f.create_group('images')
                    for idx, image in enumerate(images):
                        img_group.create_dataset(f'step_{idx:03d}', data=image, compression='gzip')
            print(f"Saved to {file_path}")
        else:
            npz_path = self.save_dir / f"episode_{episode_index:06d}.npz"
            serializable_data = {}
            for key, value in episode_data.items():
                if key == 'probe_response' and isinstance(value, dict):
                    for subkey, subval in value.items():
                        serializable_data[f'probe_response_{subkey}'] = np.asarray(subval)
                elif isinstance(value, str):
                    serializable_data[key] = np.asarray(value, dtype=object)
                else:
                    serializable_data[key] = np.asarray(value)
            if images:
                serializable_data['images'] = np.asarray(images)
            np.savez_compressed(npz_path, **serializable_data)
            file_path = npz_path
            print(f"Saved to {file_path}")


__all__ = ['ProbeDataCollector']
__all__ = ['ProbeDataCollector']
