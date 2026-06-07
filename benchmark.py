"""
Benchmark robocasa environment step throughput.

Usage:
    MUJOCO_GL=egl python benchmark.py
    MUJOCO_GL=egl python benchmark.py --env TurnSinkSpout --steps 200
"""

import argparse
import time
import statistics

import gymnasium as gym
import numpy as np

import robocasa  # noqa: F401 — registers envs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="TurnSinkSpout")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--rollouts", type=int, default=1)
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Creating env: robocasa/{args.env}")
    t0 = time.perf_counter()
    env = gym.make(f"robocasa/{args.env}", split="pretrain", seed=0)
    print(f"Env created in {time.perf_counter() - t0:.2f}s\n")

    action = {k: np.zeros_like(v) for k, v in env.action_space.sample().items()}

    all_step_times = []
    rollout_fps_list = []

    for rollout in range(args.rollouts):
        obs, _ = env.reset()

        # warmup
        for _ in range(args.warmup):
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

        # timed steps
        step_times = []
        t_start = time.perf_counter()
        for _ in range(args.steps):
            t0 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.perf_counter() - t0)
            if terminated or truncated:
                obs, _ = env.reset()
        t_total = time.perf_counter() - t_start

        fps = args.steps / t_total
        rollout_fps_list.append(fps)
        all_step_times.extend(step_times)
        print(
            f"  rollout {rollout + 1}/{args.rollouts}  {args.steps} steps  "
            f"wall={t_total:.2f}s  fps={fps:.1f}"
        )

    env.close()

    ms = [v * 1000 for v in all_step_times]
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  step mean   : {statistics.mean(ms):.2f} ms")
    print(f"  step median : {statistics.median(ms):.2f} ms")
    print(f"  step p99    : {np.percentile(ms, 99):.2f} ms")
    print(
        f"  throughput  : {statistics.mean(rollout_fps_list):.1f} steps/s"
        + (
            f" ± {statistics.stdev(rollout_fps_list):.1f}"
            if len(rollout_fps_list) > 1
            else ""
        )
    )


if __name__ == "__main__":
    main()
