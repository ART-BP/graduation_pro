#!/usr/bin/env python3
"""
实时解析 RSL-RL / Isaac Lab 的 Docker 训练日志，并绘制主要指标曲线。

示例：
    python3 monitor_go2w_training.py go2w-v2-train
    python3 monitor_go2w_training.py go2w-v2-train --history-lines 30000 --window 600
    python3 monitor_go2w_training.py go2w-v2-train --csv ~/go2w_v2_metrics.csv

退出：
    在终端按 Ctrl+C，或关闭全部绘图窗口。
"""

from __future__ import annotations

import argparse
import csv
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt

# 优先使用常见中文字体；系统没有这些字体时，英文和数值仍可正常显示。
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Micro Hei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
NUMBER_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"

# 日志标签 -> CSV/绘图字段名
SCALAR_PATTERNS = {
    "steps_per_second": re.compile(rf"Computation:\s*{NUMBER_RE}\s*steps/s"),
    "action_noise_std": re.compile(rf"Mean action noise std:\s*{NUMBER_RE}"),
    "value_function_loss": re.compile(rf"Mean value_function loss:\s*{NUMBER_RE}"),
    "surrogate_loss": re.compile(rf"Mean surrogate loss:\s*{NUMBER_RE}"),
    "entropy_loss": re.compile(rf"Mean entropy loss:\s*{NUMBER_RE}"),
    "learning_rate": re.compile(rf"Mean diagnostic_learning_rate loss:\s*{NUMBER_RE}"),
    "action_mean_abs": re.compile(rf"Mean diagnostic_action_mean_abs loss:\s*{NUMBER_RE}"),
    "diagnostic_action_std": re.compile(rf"Mean diagnostic_action_std loss:\s*{NUMBER_RE}"),
    "success_ema": re.compile(rf"Mean diagnostic_success_ema loss:\s*{NUMBER_RE}"),
    "mean_reward": re.compile(rf"Mean reward:\s*{NUMBER_RE}"),
    "mean_episode_length": re.compile(rf"Mean episode length:\s*{NUMBER_RE}"),
    "success_rate": re.compile(rf"Episode/success_rate:\s*{NUMBER_RE}"),
    "collision_rate": re.compile(rf"Episode/collision_rate:\s*{NUMBER_RE}"),
    "unstable_rate": re.compile(rf"Episode/unstable_rate:\s*{NUMBER_RE}"),
    "stuck_rate": re.compile(rf"Episode/stuck_rate:\s*{NUMBER_RE}"),
    "timeout_rate": re.compile(rf"Episode/timeout_rate:\s*{NUMBER_RE}"),
    "out_of_bounds_rate": re.compile(rf"Episode/out_of_bounds_rate:\s*{NUMBER_RE}"),
    "observation_failure_rate": re.compile(rf"Episode/observation_failure_rate:\s*{NUMBER_RE}"),
    "curriculum_mean_level": re.compile(rf"Curriculum/mean_level:\s*{NUMBER_RE}"),
    "curriculum_goal_maximum_m": re.compile(rf"Curriculum/mean_goal_maximum_m:\s*{NUMBER_RE}"),
    "reward_progress": re.compile(rf"Reward/progress:\s*{NUMBER_RE}"),
    "reward_regression": re.compile(rf"Reward/regression:\s*{NUMBER_RE}"),
    "reward_heading": re.compile(rf"Reward/heading:\s*{NUMBER_RE}"),
    "reward_forward_to_goal": re.compile(rf"Reward/forward_to_goal:\s*{NUMBER_RE}"),
    "reward_goal_reached": re.compile(rf"Reward/goal_reached:\s*{NUMBER_RE}"),
    "reward_collision": re.compile(rf"Reward/collision:\s*{NUMBER_RE}"),
    "reward_unstable": re.compile(rf"Reward/unstable:\s*{NUMBER_RE}"),
    "reward_timeout": re.compile(rf"Reward/timeout:\s*{NUMBER_RE}"),
    "reward_linear_action_rate": re.compile(rf"Reward/linear_action_rate:\s*{NUMBER_RE}"),
    "reward_angular_action_rate": re.compile(rf"Reward/angular_action_rate:\s*{NUMBER_RE}"),
    "reward_angular_speed": re.compile(rf"Reward/angular_speed:\s*{NUMBER_RE}"),
    "reward_time": re.compile(rf"Reward/time:\s*{NUMBER_RE}"),
    "total_timesteps": re.compile(r"Total timesteps:\s*(\d+)"),
    "iteration_time_s": re.compile(rf"Iteration time:\s*{NUMBER_RE}s"),
}

TERRAIN_NAMES = [
    "flat",
    "ramp",
    "step",
    "stairs",
    "rough",
    "pit",
    "wall",
    "pillar",
    "mixed",
    "multi_route",
]

TERRAIN_CN = {
    "flat": "平地",
    "ramp": "坡道",
    "step": "台阶",
    "stairs": "楼梯",
    "rough": "崎岖",
    "pit": "坑洼",
    "wall": "墙体",
    "pillar": "柱状障碍",
    "mixed": "混合",
    "multi_route": "多路径",
}

TERRAIN_RATE_PATTERNS = {
    name: re.compile(rf"Terrain/{name}_success_rate:\s*{NUMBER_RE}")
    for name in TERRAIN_NAMES
}

CSV_FIELDS = [
    "timestamp",
    "iteration",
    "max_iterations",
    "total_timesteps",
    "steps_per_second",
    "iteration_time_s",
    "mean_reward",
    "mean_episode_length",
    "success_rate",
    "collision_rate",
    "unstable_rate",
    "stuck_rate",
    "timeout_rate",
    "out_of_bounds_rate",
    "observation_failure_rate",
    "curriculum_mean_level",
    "curriculum_goal_maximum_m",
    "action_noise_std",
    "value_function_loss",
    "surrogate_loss",
    "entropy_loss",
    "learning_rate",
    "action_mean_abs",
    "diagnostic_action_std",
    "success_ema",
    "reward_progress",
    "reward_regression",
    "reward_heading",
    "reward_forward_to_goal",
    "reward_goal_reached",
    "reward_collision",
    "reward_unstable",
    "reward_timeout",
    "reward_linear_action_rate",
    "reward_angular_action_rate",
    "reward_angular_speed",
    "reward_time",
] + [f"terrain_{name}_success_rate" for name in TERRAIN_NAMES]


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line.rstrip("\n"))


def parse_block(lines: Iterable[str]) -> Optional[Dict[str, float]]:
    text_lines = list(lines)
    result: Dict[str, float] = {}

    for line in text_lines:
        match = ITER_RE.search(line)
        if match:
            result["iteration"] = int(match.group(1))
            result["max_iterations"] = int(match.group(2))
            break

    if "iteration" not in result:
        return None

    for field, pattern in SCALAR_PATTERNS.items():
        for line in text_lines:
            match = pattern.search(line)
            if match:
                value = match.group(1)
                result[field] = int(value) if field == "total_timesteps" else float(value)
                break

    for terrain, pattern in TERRAIN_RATE_PATTERNS.items():
        for line in text_lines:
            match = pattern.search(line)
            if match:
                result[f"terrain_{terrain}_success_rate"] = float(match.group(1))
                break

    result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


class LogReader(threading.Thread):
    def __init__(self, process: subprocess.Popen[str], output_queue: queue.Queue[str]):
        super().__init__(daemon=True)
        self.process = process
        self.output_queue = output_queue

    def run(self) -> None:
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            self.output_queue.put(clean_line(raw_line))


class MetricHistory:
    def __init__(self, maxlen: int):
        self.rows: Deque[Dict[str, float]] = deque(maxlen=maxlen)

    def append(self, row: Dict[str, float]) -> None:
        # docker logs 的历史部分和实时部分偶尔可能重复最后一个 block。
        if self.rows and self.rows[-1].get("iteration") == row.get("iteration"):
            self.rows[-1] = row
        else:
            self.rows.append(row)

    def x(self) -> List[float]:
        return [row["iteration"] for row in self.rows]

    def values(self, field: str) -> List[float]:
        return [row.get(field, float("nan")) for row in self.rows]


class Dashboard:
    def __init__(self, history: MetricHistory):
        self.history = history
        plt.ion()

        self.fig_rates, self.ax_rates = plt.subplots(num="Go2W：任务结果")
        self.fig_reward, self.ax_reward = plt.subplots(num="Go2W：奖励趋势")
        self.fig_curriculum, self.ax_curriculum = plt.subplots(num="Go2W：课程学习")
        self.fig_critic, self.ax_critic = plt.subplots(num="Go2W：Critic 状态")
        self.fig_policy, self.ax_policy = plt.subplots(num="Go2W：策略诊断")
        self.fig_terrain, self.ax_terrain = plt.subplots(num="Go2W：分地形成功率")

        self.figures = [
            self.fig_rates,
            self.fig_reward,
            self.fig_curriculum,
            self.fig_critic,
            self.fig_policy,
            self.fig_terrain,
        ]

    @staticmethod
    def _finish_axis(ax, title: str, ylabel: str, percentage: bool = False) -> None:
        ax.set_title(title)
        ax.set_xlabel("训练迭代")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if percentage:
            ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="best")
        ax.figure.tight_layout()

    def update(self) -> None:
        if not self.history.rows:
            return

        x = self.history.x()
        latest = self.history.rows[-1]

        self.ax_rates.clear()
        self.ax_rates.plot(x, self.history.values("success_rate"), label="成功率")
        self.ax_rates.plot(x, self.history.values("collision_rate"), label="碰撞率")
        self.ax_rates.plot(x, self.history.values("unstable_rate"), label="失稳率")
        self.ax_rates.plot(x, self.history.values("timeout_rate"), label="超时率")
        self._finish_axis(
            self.ax_rates,
            (
                f"任务结果｜当前成功 {latest.get('success_rate', float('nan')):.3f}，"
                f"碰撞 {latest.get('collision_rate', float('nan')):.3f}，"
                f"失稳 {latest.get('unstable_rate', float('nan')):.3f}"
            ),
            "比例",
            percentage=True,
        )

        self.ax_reward.clear()
        self.ax_reward.plot(x, self.history.values("mean_reward"), label="平均回报")
        self.ax_reward.plot(x, self.history.values("reward_progress"), label="进度奖励")
        self.ax_reward.plot(x, self.history.values("reward_goal_reached"), label="到达奖励")
        self.ax_reward.plot(x, self.history.values("reward_timeout"), label="超时惩罚")
        self._finish_axis(
            self.ax_reward,
            f"奖励趋势｜当前平均回报 {latest.get('mean_reward', float('nan')):.2f}",
            "每回合奖励",
        )

        self.ax_curriculum.clear()
        self.ax_curriculum.plot(
            x, self.history.values("curriculum_mean_level"), label="平均课程等级"
        )
        self.ax_curriculum.plot(
            x, self.history.values("curriculum_goal_maximum_m"), label="最大目标距离（m）"
        )
        self._finish_axis(
            self.ax_curriculum,
            (
                f"课程学习｜当前等级 {latest.get('curriculum_mean_level', float('nan')):.2f}，"
                f"目标上限 {latest.get('curriculum_goal_maximum_m', float('nan')):.2f} m"
            ),
            "等级 / 距离",
        )

        self.ax_critic.clear()
        self.ax_critic.plot(
            x, self.history.values("value_function_loss"), label="Value function loss"
        )
        self._finish_axis(
            self.ax_critic,
            f"Critic 状态｜当前 Value loss {latest.get('value_function_loss', float('nan')):.2f}",
            "损失",
        )

        self.ax_policy.clear()
        self.ax_policy.plot(
            x, self.history.values("action_noise_std"), label="动作噪声 std"
        )
        self.ax_policy.plot(
            x, self.history.values("diagnostic_action_std"), label="诊断动作 std"
        )
        self.ax_policy.plot(
            x, self.history.values("action_mean_abs"), label="动作均值绝对值"
        )
        self.ax_policy.plot(
            x, self.history.values("success_ema"), label="成功率 EMA"
        )
        self._finish_axis(
            self.ax_policy,
            (
                f"策略诊断｜噪声 {latest.get('action_noise_std', float('nan')):.3f}，"
                f"成功率 EMA {latest.get('success_ema', float('nan')):.3f}"
            ),
            "数值",
        )

        self.ax_terrain.clear()
        for terrain in TERRAIN_NAMES:
            field = f"terrain_{terrain}_success_rate"
            values = self.history.values(field)
            if any(value == value for value in values):  # 排除全 NaN
                self.ax_terrain.plot(x, values, label=TERRAIN_CN[terrain])
        self._finish_axis(
            self.ax_terrain,
            "分地形成功率（日志中的平滑成功率）",
            "成功率",
            percentage=True,
        )

        for figure in self.figures:
            figure.canvas.draw_idle()
            figure.canvas.flush_events()

    def any_open(self) -> bool:
        return any(plt.fignum_exists(fig.number) for fig in self.figures)


class CsvWriter:
    def __init__(self, path: Optional[Path]):
        self.path = path
        self.file = None
        self.writer = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not path.exists() or path.stat().st_size == 0
            self.file = path.open("a", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=CSV_FIELDS)
            if is_new:
                self.writer.writeheader()
                self.file.flush()

    def append(self, row: Dict[str, float]) -> None:
        if self.writer is None or self.file is None:
            return
        self.writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def print_summary(row: Dict[str, float]) -> None:
    print(
        "iter={iteration:.0f}/{max_iterations:.0f} "
        "reward={mean_reward:.2f} "
        "success={success_rate:.3f} "
        "collision={collision_rate:.3f} "
        "unstable={unstable_rate:.3f} "
        "timeout={timeout_rate:.3f} "
        "level={curriculum_mean_level:.2f} "
        "goal_max={curriculum_goal_maximum_m:.2f}m "
        "value_loss={value_function_loss:.2f}".format(
            iteration=row.get("iteration", float("nan")),
            max_iterations=row.get("max_iterations", float("nan")),
            mean_reward=row.get("mean_reward", float("nan")),
            success_rate=row.get("success_rate", float("nan")),
            collision_rate=row.get("collision_rate", float("nan")),
            unstable_rate=row.get("unstable_rate", float("nan")),
            timeout_rate=row.get("timeout_rate", float("nan")),
            curriculum_mean_level=row.get("curriculum_mean_level", float("nan")),
            curriculum_goal_maximum_m=row.get(
                "curriculum_goal_maximum_m", float("nan")
            ),
            value_function_loss=row.get("value_function_loss", float("nan")),
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="实时可视化 Docker 中 Go2W RSL-RL 训练指标。"
    )
    parser.add_argument(
        "container",
        nargs="?",
        default="go2w-v2-train",
        help="训练容器名称，默认：go2w-v2-train",
    )
    parser.add_argument(
        "--history-lines",
        type=int,
        default=200000,
        help="启动时读取最近多少行 Docker 日志；0 表示只看新日志。默认：20000",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10000,
        help="图中最多保留多少个 iteration。默认：500",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.5,
        help="绘图刷新周期（秒）。默认：0.5",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="可选：持续保存解析后的 CSV，例如 ~/go2w_v2_metrics.csv",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    command = [
        "docker",
        "logs",
        "--tail",
        str(args.history_lines),
        "-f",
        args.container,
    ]

    print("执行：", " ".join(command))
    print("按 Ctrl+C 退出；关闭全部绘图窗口也会退出。")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print("错误：找不到 docker 命令。", file=sys.stderr)
        return 1

    line_queue: queue.Queue[str] = queue.Queue()
    reader = LogReader(process, line_queue)
    reader.start()

    history = MetricHistory(maxlen=args.window)
    dashboard = Dashboard(history)
    csv_writer = CsvWriter(args.csv)

    current_block: List[str] = []
    inside_block = False
    last_refresh = 0.0

    try:
        while True:
            received = False

            while True:
                try:
                    line = line_queue.get_nowait()
                except queue.Empty:
                    break

                received = True

                if "Learning iteration" in line:
                    current_block = [line]
                    inside_block = True
                    continue

                if inside_block:
                    current_block.append(line)

                    if re.search(r"^\s*ETA:", line):
                        row = parse_block(current_block)
                        inside_block = False
                        current_block = []

                        if row is not None:
                            history.append(row)
                            csv_writer.append(row)
                            print_summary(row)

            now = time.monotonic()
            if received or now - last_refresh >= args.refresh:
                dashboard.update()
                plt.pause(0.01)
                last_refresh = now

            if process.poll() is not None and line_queue.empty():
                print(f"docker logs 已退出，返回码：{process.returncode}")
                print("日志读取结束，保留图形窗口。关闭窗口后程序退出。")

                dashboard.update()

                # 由交互模式切换为阻塞显示，防止窗口立即关闭
                plt.ioff()
                plt.show()
                return 0

            if not dashboard.any_open():
                print("全部绘图窗口已关闭。")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在退出……")
    finally:
        csv_writer.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        # plt.close("all")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
