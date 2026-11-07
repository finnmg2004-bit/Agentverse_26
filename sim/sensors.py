#!/usr/bin/env python3
"""
Milestone 4: Simulated sensors (`sim/sensors.py`) and WAV audio asset generator.
Provides:
  1. 36-ray 2D LiDAR rangefinders (`[-90 deg, +90 deg]`, `z=0.32m`, `max_range=4.0m`)
  2. Head RGB-D camera (`320x240`, `fovy=60 deg`, JPEG RGB + float16 depth in meters)
  3. Stereo ear audio synthesizer (`16 kHz`, `0.5s` window = `8000x2` int16 PCM)
     with distance attenuation, wall occlusion (`0.45x` per wall), ILD, and ITD.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mujoco
import numpy as np
from scipy.io import wavfile

from sim.house_layout import SOUND_SOURCES, count_walls_crossed

# mj_ray geom groups: house (0, 1, 4, 5) + owner/chair (0); skip robot visual/collision (2, 3)
LIDAR_GEOMGROUP = np.array([1, 1, 0, 0, 1, 1], dtype=np.uint8)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_ROOT / "assets" / "audio"

SAMPLE_RATE: int = 16000
AUDIO_WINDOW_SEC: float = 0.5
AUDIO_WINDOW_SAMPLES: int = int(SAMPLE_RATE * AUDIO_WINDOW_SEC)  # 8000 samples

LIDAR_NUM_RAYS: int = 36
LIDAR_FOV_RAD: float = math.pi  # [-90 deg, +90 deg]
LIDAR_MAX_RANGE: float = 4.0
LIDAR_HEIGHT_Z: float = 0.32

CAM_WIDTH: int = 320
CAM_HEIGHT: int = 240
CAM_FOVY_DEG: float = 60.0


def ensure_audio_assets() -> Dict[str, np.ndarray]:
    """
    Ensure `assets/audio/{doorbell,knock,smoke_alarm}.wav` exist on disk as
    16 kHz mono int16 WAV files and return their normalized float32 waveforms.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    duration = 2.0
    n_samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0.0, duration, n_samples, endpoint=False, dtype=np.float64)

    # 1. Doorbell: Two-tone Ding-Dong chime (659.25 Hz E5 -> 523.25 Hz C5) repeating every 0.5s
    doorbell_path = AUDIO_DIR / "doorbell.wav"
    if not doorbell_path.exists():
        sig = np.zeros_like(t)
        t_mod = np.mod(t, 0.5)
        ding_mask = t_mod < 0.24
        dong_mask = ~ding_mask
        t_ding = t_mod[ding_mask]
        t_dong = t_mod[dong_mask] - 0.24

        sig[ding_mask] = (
            0.70 * np.sin(2.0 * np.pi * 659.25 * t_ding)
            + 0.22 * np.sin(2.0 * np.pi * 1318.5 * t_ding)
            + 0.08 * np.sin(2.0 * np.pi * 1977.8 * t_ding)
        ) * np.exp(-6.5 * t_ding)

        sig[dong_mask] = (
            0.75 * np.sin(2.0 * np.pi * 523.25 * t_dong)
            + 0.20 * np.sin(2.0 * np.pi * 1046.5 * t_dong)
            + 0.05 * np.sin(2.0 * np.pi * 1569.8 * t_dong)
        ) * np.exp(-5.5 * t_dong)

        sig = np.clip(sig * 0.9, -0.98, 0.98)
        wavfile.write(str(doorbell_path), SAMPLE_RATE, (sig * 32767).astype(np.int16))

    # 2. Knock: 3 crisp wooden door knocks every 0.5s (160 Hz + 290 Hz wood resonance + impact click)
    knock_path = AUDIO_DIR / "knock.wav"
    if not knock_path.exists():
        rng = np.random.default_rng(42)
        sig = np.zeros_like(t)
        t_mod = np.mod(t, 0.5)
        for pulse_t in (0.04, 0.19, 0.34):
            dt_k = t_mod - pulse_t
            active = (dt_k >= 0.0) & (dt_k < 0.09)
            tau = dt_k[active]
            wood = (
                0.65 * np.sin(2.0 * np.pi * 165.0 * tau)
                + 0.35 * np.sin(2.0 * np.pi * 295.0 * tau)
                + 0.20 * np.sin(2.0 * np.pi * 520.0 * tau)
            ) * np.exp(-42.0 * tau)
            click = 0.35 * rng.standard_normal(tau.shape) * np.exp(-140.0 * tau)
            sig[active] += wood + click
        sig = np.clip(sig / (np.max(np.abs(sig)) + 1e-6) * 0.92, -0.98, 0.98)
        wavfile.write(str(knock_path), SAMPLE_RATE, (sig * 32767).astype(np.int16))

    # 3. Smoke Alarm: High-frequency 3150 Hz piezo temporal-three fast pulsing alarm
    smoke_path = AUDIO_DIR / "smoke_alarm.wav"
    if not smoke_path.exists():
        carrier = (
            0.78 * np.sin(2.0 * np.pi * 3150.0 * t)
            + 0.16 * np.sin(2.0 * np.pi * 3220.0 * t)
            + 0.06 * np.sin(2.0 * np.pi * 6300.0 * t)
        )
        # Rapid 8 Hz beep envelope so every 0.5s window has 4 sharp high-pitched beeps
        env = np.where(np.mod(t, 0.125) < 0.085, 1.0, 0.05)
        sig = np.clip(carrier * env * 0.92, -0.98, 0.98)
        wavfile.write(str(smoke_path), SAMPLE_RATE, (sig * 32767).astype(np.int16))

    clips: Dict[str, np.ndarray] = {}
    for name in ("doorbell", "knock", "smoke_alarm"):
        sr, data = wavfile.read(str(AUDIO_DIR / f"{name}.wav"))
        if data.ndim > 1:
            data = data[:, 0]
        clips[name] = data.astype(np.float32) / 32768.0
    return clips


class SimSensors:
    """Computes LiDAR, Head RGB-D Camera, and Stereo Ear Audio in the simulator."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, enable_camera: bool = True) -> None:
        self.model = model
        self.data = data
        self.audio_clips = ensure_audio_assets()
        self.audio_cursor: int = 0
        self.rng = np.random.default_rng(123)

        # LiDAR ray angles in body frame: [-90 deg, +90 deg]
        self.lidar_angles = np.linspace(
            -0.5 * LIDAR_FOV_RAD, +0.5 * LIDAR_FOV_RAD, LIDAR_NUM_RAYS, dtype=np.float64
        )
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        # Sound beacons are a viewer aid for humans, not part of the world: the robot's camera
        # must not see them (the cyan doorbell beacon looks like the owner's teal jacket).
        self.beacon_geom_ids = [i for i in range(model.ngeom) if model.geom(i).name.startswith("beacon_")]

        # Initialize offscreen renderer for head_cam (320x240 RGB + Depth)
        self.renderer: Optional[mujoco.Renderer] = None
        if enable_camera:
            try:
                self.renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)
            except Exception as exc:
                print(f"[sensors] Note: Offscreen GL renderer unavailable ({exc}); using synthetic fallback.")
                self.renderer = None

    def compute_lidar(self, robot_pose: Tuple[float, float, float]) -> List[float]:
        """
        Cast 36 horizontal rays across [-90 deg, +90 deg] in robot body frame at z=0.32m.
        Returns a list of 36 distances in [0.0, 4.0] meters.
        """
        rx, ry, ryaw = robot_pose
        # Start rays at the nose/perimeter of the robot (0.28m forward from center along ray)
        # or from center with bodyexclude; starting at radius 0.27m outside the Go2 base
        # guarantees we never hit the Go2's own calf/thigh geoms.
        start_offset = 0.27
        pnt = np.array([rx, ry, LIDAR_HEIGHT_Z], dtype=np.float64)

        vec = np.zeros((LIDAR_NUM_RAYS, 3), dtype=np.float64)
        world_angles = ryaw + self.lidar_angles
        vec[:, 0] = np.cos(world_angles)
        vec[:, 1] = np.sin(world_angles)
        vec[:, 2] = 0.0

        geomid = np.full(LIDAR_NUM_RAYS, -1, dtype=np.int32)
        dist = np.full(LIDAR_NUM_RAYS, -1.0, dtype=np.float64)

        # Exclude robot's own geoms by starting each ray 0.27m out along its direction
        ranges: List[float] = []
        single_geom = np.array([-1], dtype=np.int32)
        for i in range(LIDAR_NUM_RAYS):
            ray_start = pnt + start_offset * vec[i]
            d = mujoco.mj_ray(
                self.model,
                self.data,
                ray_start,
                vec[i],
                LIDAR_GEOMGROUP,
                1,
                self.base_body_id,
                single_geom,
            )
            if d < 0.0 or (d + start_offset) > LIDAR_MAX_RANGE:
                ranges.append(float(LIDAR_MAX_RANGE))
            else:
                ranges.append(round(float(d + start_offset), 4))
        return ranges

    def render_head_camera(
        self, robot_pose: Tuple[float, float, float], owner_pos: Tuple[float, float]
    ) -> Tuple[bytes, bytes, np.ndarray, np.ndarray]:
        """
        Render `head_cam` at 320x240 -> returns (rgb_jpeg_bytes, depth_float16_bytes, rgb_array, depth_array).
        Includes a fast geometric pinhole fallback if offscreen OpenGL is unavailable.
        """
        if self.renderer is not None:
            try:
                saved_alpha = self.model.geom_rgba[self.beacon_geom_ids, 3].copy()
                self.model.geom_rgba[self.beacon_geom_ids, 3] = 0.0
                try:
                    self.renderer.update_scene(self.data, camera="head_cam")
                finally:
                    self.model.geom_rgba[self.beacon_geom_ids, 3] = saved_alpha
                rgb = self.renderer.render()  # uint8 (240, 320, 3) RGB
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render()  # float32 (240, 320) meters
                self.renderer.disable_depth_rendering()
                depth = np.clip(depth, 0.1, 10.0).astype(np.float16)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, jpeg_buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                jpeg_bytes = jpeg_buf.tobytes() if ok else b""
                return jpeg_bytes, depth.tobytes(), rgb, depth
            except Exception:
                pass

        # Fallback pinhole synthesis (used in headless environments without EGL/GLX)
        rgb = np.full((CAM_HEIGHT, CAM_WIDTH, 3), (45, 52, 64), dtype=np.uint8)
        rgb[CAM_HEIGHT // 2 :, :] = (38, 44, 52)
        depth = np.full((CAM_HEIGHT, CAM_WIDTH), 4.0, dtype=np.float32)

        rx, ry, ryaw = robot_pose
        # Camera world position (mounted 0.30m forward on robot base)
        cam_x = rx + 0.30 * math.cos(ryaw)
        cam_y = ry + 0.30 * math.sin(ryaw)
        ox, oy = owner_pos
        dx = ox - cam_x
        dy = oy - cam_y
        # Transform owner into camera frame: forward = +cos/sin(ryaw), left = -sin/+cos(ryaw)
        fwd = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        left = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        walls_between = count_walls_crossed((cam_x, cam_y), (ox, oy))

        if fwd > 0.25 and walls_between == 0:
            fy = 0.5 * CAM_HEIGHT / math.tan(math.radians(0.5 * CAM_FOVY_DEG))
            fx = fy
            u_center = int(0.5 * CAM_WIDTH - (left / fwd) * fx)
            v_center = int(0.5 * CAM_HEIGHT - ((0.75 - 0.40) / fwd) * fy)
            box_w = max(14, int((0.46 / fwd) * fx))
            box_h = max(28, int((1.35 / fwd) * fy))
            u1 = max(0, u_center - box_w // 2)
            u2 = min(CAM_WIDTH, u_center + box_w // 2)
            v1 = max(0, v_center - box_h // 2)
            v2 = min(CAM_HEIGHT, v_center + box_h // 2)
            if u2 > u1 and v2 > v1:
                # Draw cyan/teal jacket + head matching the MuJoCo Owner body
                rgb[v1:v2, u1:u2] = (13, 184, 224)  # RGB cyan/teal
                head_v2 = v1 + max(6, (v2 - v1) // 5)
                rgb[v1:head_v2, u1:u2] = (235, 189, 153)  # RGB skin tone
                depth[v1:v2, u1:u2] = float(fwd)

        depth_f16 = depth.astype(np.float16)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, jpeg_buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return (jpeg_buf.tobytes() if ok else b""), depth_f16.tobytes(), rgb, depth_f16

    def synthesize_stereo_audio(
        self,
        robot_pose: Tuple[float, float, float],
        active_sounds: Dict[str, bool],
    ) -> Tuple[bytes, Dict[str, object]]:
        """
        Synthesize 0.5s (8000 samples x 2 channels) of 16 kHz stereo int16 PCM audio
        as heard by the robot's Left `(0.22, +0.08, 0.15)` and Right `(0.22, -0.08, 0.15)` ears.
        Includes distance attenuation, wall occlusion (0.45x/wall), ILD, and ITD.
        """
        rx, ry, ryaw = robot_pose
        cos_y = math.cos(ryaw)
        sin_y = math.sin(ryaw)

        # Left and right ear positions in world frame
        ear_l = (rx + 0.22 * cos_y - 0.08 * sin_y, ry + 0.22 * sin_y + 0.08 * cos_y)
        ear_r = (rx + 0.22 * cos_y + 0.08 * sin_y, ry + 0.22 * sin_y - 0.08 * cos_y)

        # Subtle ambient room noise floor
        left_ch = self.rng.normal(0.0, 0.002, size=AUDIO_WINDOW_SAMPLES).astype(np.float32)
        right_ch = self.rng.normal(0.0, 0.002, size=AUDIO_WINDOW_SAMPLES).astype(np.float32)

        active_list: List[str] = []
        # Priority order: if smoke_alarm is active, it dominates or mixes with active sounds
        for sound_name, is_on in active_sounds.items():
            if not is_on or sound_name not in self.audio_clips:
                continue
            active_list.append(sound_name)
            clip = self.audio_clips[sound_name]
            idx = (np.arange(AUDIO_WINDOW_SAMPLES) + self.audio_cursor) % len(clip)
            mono = clip[idx]

            sx, sy, _sz = SOUND_SOURCES[sound_name]["pos"]
            dx = sx - rx
            dy = sy - ry
            dist = max(0.25, math.hypot(dx, dy))

            # Azimuth of sound source in robot body frame: +left is positive, -right is negative
            world_angle = math.atan2(dy, dx)
            rel_azimuth = math.atan2(math.sin(world_angle - ryaw), math.cos(world_angle - ryaw))

            # Distance + wall occlusion attenuation. Occlusion is capped at 3 walls: in the
            # open-plan house sound also reaches the robot around walls through the archways.
            walls = count_walls_crossed((rx, ry), (sx, sy))
            dist_atten = 1.0 / (1.0 + 0.22 * dist)
            wall_atten = (0.45 ** min(walls, 3))
            # Source loudness: a smoke alarm (~85 dB @ 3 m) is far louder than a doorbell chime.
            level = float(SOUND_SOURCES[sound_name].get("level", 1.0))
            base_gain = level * max(0.12, dist_atten * wall_atten)

            # Interaural Level Difference (ILD): sound on left (+rel_azimuth) is louder in Left ear
            ild = 0.38 * math.sin(rel_azimuth)
            gain_l = base_gain * (1.0 + ild)
            gain_r = base_gain * (1.0 - ild)

            # Interaural Time Difference (ITD): sample shift between ears
            # Physical + horn-baffle effective ear baseline gives up to +-6 samples at 16 kHz
            delay_samples = int(round(6.0 * math.sin(rel_azimuth)))
            if delay_samples > 0:
                # Closer to left ear -> left leads, right lags
                mono_l = mono
                mono_r = np.roll(mono, delay_samples)
            elif delay_samples < 0:
                mono_l = np.roll(mono, -delay_samples)
                mono_r = mono
            else:
                mono_l = mono
                mono_r = mono

            left_ch += (gain_l * mono_l).astype(np.float32)
            right_ch += (gain_r * mono_r).astype(np.float32)

        self.audio_cursor = (self.audio_cursor + AUDIO_WINDOW_SAMPLES) % (SAMPLE_RATE * 2)

        stereo = np.stack([left_ch, right_ch], axis=-1)
        stereo = np.clip(stereo, -0.99, 0.99)
        pcm_int16 = (stereo * 32767.0).astype(np.int16)

        meta = {
            "sample_rate": SAMPLE_RATE,
            "channels": 2,
            "samples": AUDIO_WINDOW_SAMPLES,
            "active_gt": active_list,
        }
        return pcm_int16.tobytes(), meta
