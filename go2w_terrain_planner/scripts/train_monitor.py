#!/usr/bin/env python3
"""Monitor the small set of metrics needed to judge a Go2W training run.

Examples:
    python3 scripts/train_monitor.py go2w-v2-train
    python3 scripts/train_monitor.py go2w-v2-train --csv ~/go2w_metrics.csv
    python3 scripts/train_monitor.py go2w-v2-train --no-gui

The dashboard deliberately omits individual reward terms and duplicated action
diagnostics.  It focuses on task outcome, curriculum mastery, PPO stability,
sensor health, and the latest cumulative success rate of observed terrains.
"""

from __future__ import annotations

import argparse
import csv
import math
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


plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Micro Hei",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


MetricRow = Dict[str, object]
MONITOR_VERSION = "essential-v2"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
NUMBER_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def metric_pattern(label: str) -> re.Pattern[str]:
    return re.compile(rf"{re.escape(label)}:\s*{NUMBER_RE}")


# Only retain metrics that can change a training decision.
SCALAR_PATTERNS = {
    "steps_per_second": re.compile(rf"Computation:\s*{NUMBER_RE}\s*steps/s"),
    "action_noise_std": metric_pattern("Mean action noise std"),
    "value_function_loss": metric_pattern("Mean value_function loss"),
    "surrogate_loss": metric_pattern("Mean surrogate loss"),
    "learning_rate": metric_pattern("Mean diagnostic_learning_rate loss"),
    "mean_reward": metric_pattern("Mean reward"),
    "mean_episode_length": metric_pattern("Mean episode length"),
    "success_rate": metric_pattern("Episode/success_rate"),
    "collision_rate": metric_pattern("Episode/collision_rate"),
    "unstable_rate": metric_pattern("Episode/unstable_rate"),
    "stuck_rate": metric_pattern("Episode/stuck_rate"),
    "timeout_rate": metric_pattern("Episode/timeout_rate"),
    "out_of_bounds_rate": metric_pattern("Episode/out_of_bounds_rate"),
    "observation_failure_rate": metric_pattern(
        "Episode/observation_failure_rate"
    ),
    "completed_count": metric_pattern("Episode/completed_count"),
    "curriculum_mean_level": metric_pattern("Curriculum/mean_level"),
    "curriculum_mean_sampled_stage": metric_pattern(
        "Curriculum/mean_sampled_stage"
    ),
    "curriculum_frontier_success_ema": metric_pattern(
        "Curriculum/frontier_success_ema"
    ),
    "curriculum_full_success_ema": metric_pattern(
        "Curriculum/full_difficulty_success_ema"
    ),
    "curriculum_full_episode_count": metric_pattern(
        "Curriculum/mean_full_difficulty_episode_count"
    ),
    "curriculum_frontier_difficulty": metric_pattern(
        "Curriculum/mean_frontier_difficulty"
    ),
    "curriculum_sampled_difficulty": metric_pattern(
        "Curriculum/mean_sampled_difficulty"
    ),
    "curriculum_goal_maximum_m": metric_pattern(
        "Curriculum/mean_goal_maximum_m"
    ),
    "sensor_observed_ratio": metric_pattern("Sensor/observed_ratio"),
    "sensor_height_valid_ratio": metric_pattern("Sensor/height_valid_ratio"),
    "sensor_point_slots": metric_pattern("Sensor/point_slots_per_frame"),
    "sensor_mean_point_count": metric_pattern("Sensor/mean_point_count"),
    "safety_stop_steps": metric_pattern("Safety/stop_step_count"),
    "total_timesteps": re.compile(r"Total timesteps:\s*(\d+)"),
    "iteration_time_s": re.compile(rf"Iteration time:\s*{NUMBER_RE}s"),
}

TERRAIN_NAMES = (
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
    "low_obstacle",
)

TERRAIN_CN = {
    "flat": "平地",
    "ramp": "坡道",
    "step": "台阶",
    "stairs": "楼梯",
    "rough": "崎岖",
    "pit": "深坑",
    "wall": "墙体",
    "pillar": "立柱",
    "mixed": "混合",
    "multi_route": "多路径",
    "low_obstacle": "矮障碍",
}

TERRAIN_RATE_PATTERNS = {
    name: metric_pattern(f"Terrain/{name}_success_rate")
    for name in TERRAIN_NAMES
}
TERRAIN_COUNT_PATTERNS = {
    name: metric_pattern(f"Terrain/{name}_episode_count")
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
    "other_failure_rate",
    "completed_count",
    "curriculum_mean_level",
    "curriculum_mean_sampled_stage",
    "curriculum_frontier_difficulty",
    "curriculum_progress",
    "curriculum_sampled_difficulty",
    "curriculum_frontier_success_ema",
    "curriculum_full_success_ema",
    "curriculum_full_episode_count",
    "curriculum_goal_maximum_m",
    "action_noise_std",
    "value_function_loss",
    "surrogate_loss",
    "learning_rate",
    "sensor_observed_ratio",
    "sensor_height_valid_ratio",
    "sensor_hit_ratio",
    "sensor_mean_point_count",
    "safety_stop_steps",
] + [
    field
    for name in TERRAIN_NAMES
    for field in (
        f"terrain_{name}_success_rate",
        f"terrain_{name}_episode_count",
    )
]


def clean_line(line: str) -> str:
    return ANSI_RE.sub("", line.rstrip("\n"))


def _first_match(lines: List[str], pattern: re.Pattern[str]) -> Optional[str]:
    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def parse_block(lines: Iterable[str]) -> Optional[MetricRow]:
    text_lines = list(lines)
    result: MetricRow = {}

    for line in text_lines:
        match = ITER_RE.search(line)
        if match:
            result["iteration"] = int(match.group(1))
            result["max_iterations"] = int(match.group(2))
            break
    if "iteration" not in result:
        return None

    for field, pattern in SCALAR_PATTERNS.items():
        value = _first_match(text_lines, pattern)
        if value is not None:
            result[field] = int(value) if field == "total_timesteps" else float(value)

    for terrain in TERRAIN_NAMES:
        rate = _first_match(text_lines, TERRAIN_RATE_PATTERNS[terrain])
        count = _first_match(text_lines, TERRAIN_COUNT_PATTERNS[terrain])
        if rate is not None:
            result[f"terrain_{terrain}_success_rate"] = float(rate)
        if count is not None:
            result[f"terrain_{terrain}_episode_count"] = float(count)

    other_failures = (
        float(result.get("stuck_rate", 0.0))
        + float(result.get("out_of_bounds_rate", 0.0))
        + float(result.get("observation_failure_rate", 0.0))
    )
    result["other_failure_rate"] = other_failures

    level = result.get("curriculum_mean_level")
    frontier_difficulty = result.get("curriculum_frontier_difficulty")
    if level is not None and frontier_difficulty is not None:
        result["curriculum_progress"] = float(level) + float(frontier_difficulty)

    point_count = result.get("sensor_mean_point_count")
    point_slots = result.get("sensor_point_slots")
    if point_count is not None and point_slots is not None and float(point_slots) > 0.0:
        result["sensor_hit_ratio"] = float(point_count) / float(point_slots)

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
        self.rows: Deque[MetricRow] = deque(maxlen=maxlen)
        self.observed_terrains = set()

    def append(self, row: MetricRow) -> MetricRow:
        # Episode/curriculum logs appear only when at least one environment
        # resets. Carry their last value across iterations so the dashboard and
        # CSV do not alternate between a useful value and NaN.
        same_iteration = bool(
            self.rows and self.rows[-1].get("iteration") == row.get("iteration")
        )
        merged = dict(self.rows[-1]) if self.rows else {}
        if not same_iteration:
            # These are counts from the current log block, not running totals.
            merged.pop("completed_count", None)
            for terrain in TERRAIN_NAMES:
                merged.pop(f"terrain_{terrain}_episode_count", None)
        merged.update(row)

        merged["other_failure_rate"] = sum(
            float(merged.get(field, 0.0))
            for field in (
                "stuck_rate",
                "out_of_bounds_rate",
                "observation_failure_rate",
            )
        )
        if (
            "curriculum_mean_level" in merged
            and "curriculum_frontier_difficulty" in merged
        ):
            merged["curriculum_progress"] = float(
                merged["curriculum_mean_level"]
            ) + float(merged["curriculum_frontier_difficulty"])
        if (
            float(merged.get("sensor_point_slots", 0.0)) > 0.0
            and "sensor_mean_point_count" in merged
        ):
            merged["sensor_hit_ratio"] = float(
                merged["sensor_mean_point_count"]
            ) / float(merged["sensor_point_slots"])

        for terrain in TERRAIN_NAMES:
            if float(row.get(f"terrain_{terrain}_episode_count", 0.0)) > 0.0:
                self.observed_terrains.add(terrain)

        if same_iteration:
            self.rows[-1] = merged
        else:
            self.rows.append(merged)
        return merged

    def x(self) -> List[float]:
        return [float(row["iteration"]) for row in self.rows]

    def values(self, field: str) -> List[float]:
        return [float(row.get(field, float("nan"))) for row in self.rows]

    def terrain_was_observed(self, terrain: str) -> bool:
        return terrain in self.observed_terrains


class Dashboard:
    def __init__(self, history: MetricHistory):
        self.history = history
        plt.ion()
        self.figure, axes = plt.subplots(
            2,
            3,
            num=f"Go2W训练必要指标（{MONITOR_VERSION}）",
            figsize=(16, 9),
        )
        (
            self.ax_outcomes,
            self.ax_curriculum,
            self.ax_mastery,
            self.ax_training,
            self.ax_sensor,
            self.ax_terrain,
        ) = axes.reshape(-1)
        self.ax_value = self.ax_training.twinx()

    @staticmethod
    def _finish_axis(ax, title: str, ylabel: str, *, percentage: bool = False) -> None:
        ax.set_title(title)
        ax.set_xlabel("训练迭代")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if percentage:
            ax.set_ylim(-0.03, 1.03)
        ax.legend(loc="best", fontsize=8)

    def update(self) -> None:
        if not self.history.rows:
            return
        x = self.history.x()
        latest = self.history.rows[-1]

        self.ax_outcomes.clear()
        for field, label in (
            ("success_rate", "成功"),
            ("collision_rate", "碰撞"),
            ("unstable_rate", "失稳"),
            ("timeout_rate", "超时"),
            ("other_failure_rate", "其他失败"),
        ):
            self.ax_outcomes.plot(x, self.history.values(field), label=label)
        self._finish_axis(
            self.ax_outcomes,
            f"回合结果｜长度 {value(latest, 'mean_episode_length', 1)} step",
            "比例",
            percentage=True,
        )

        self.ax_curriculum.clear()
        self.ax_curriculum.plot(
            x,
            self.history.values("curriculum_progress"),
            label="前沿等级+难度",
        )
        self.ax_curriculum.plot(
            x,
            self.history.values("curriculum_mean_sampled_stage"),
            label="实际采样阶段",
            alpha=0.8,
        )
        self.ax_curriculum.set_ylim(0.8, 10.1)
        self._finish_axis(
            self.ax_curriculum,
            f"课程进度｜目标上限 {value(latest, 'curriculum_goal_maximum_m', 2)} m",
            "阶段",
        )

        self.ax_mastery.clear()
        self.ax_mastery.plot(
            x,
            self.history.values("curriculum_frontier_success_ema"),
            label="前沿成功率EMA",
        )
        self.ax_mastery.plot(
            x,
            self.history.values("curriculum_full_success_ema"),
            label="完整难度成功率EMA",
        )
        self.ax_mastery.axhline(0.85, color="black", linestyle="--", alpha=0.5, label="升级门槛")
        self._finish_axis(
            self.ax_mastery,
            f"课程掌握度｜完整难度样本 {value(latest, 'curriculum_full_episode_count', 0)}",
            "EMA",
            percentage=True,
        )

        self.ax_training.clear()
        self.ax_value.clear()
        reward_line = self.ax_training.plot(
            x,
            self.history.values("mean_reward"),
            color="tab:blue",
            label="平均回报",
        )
        value_line = self.ax_value.plot(
            x,
            self.history.values("value_function_loss"),
            color="tab:orange",
            label="Value loss",
        )
        self.ax_training.set_title(
            "训练稳定性｜"
            f"surrogate {value(latest, 'surrogate_loss', 4)}，"
            f"lr {scientific(latest, 'learning_rate')}"
        )
        self.ax_training.set_xlabel("训练迭代")
        self.ax_training.set_ylabel("平均回报", color="tab:blue")
        self.ax_value.set_ylabel("Value loss", color="tab:orange")
        self.ax_training.grid(True, alpha=0.25)
        lines = reward_line + value_line
        self.ax_training.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

        self.ax_sensor.clear()
        for field, label in (
            ("sensor_observed_ratio", "观测覆盖率"),
            ("sensor_height_valid_ratio", "高度有效率"),
            ("sensor_hit_ratio", "射线命中率"),
        ):
            self.ax_sensor.plot(x, self.history.values(field), label=label)
        self._finish_axis(
            self.ax_sensor,
            f"传感器健康｜安全停止 {value(latest, 'safety_stop_steps', 1)} step/回合",
            "比例",
            percentage=True,
        )

        self.ax_terrain.clear()
        observed_terrains = [
            name for name in TERRAIN_NAMES if self.history.terrain_was_observed(name)
        ]
        if observed_terrains:
            rates = [
                float(latest.get(f"terrain_{name}_success_rate", float("nan")))
                for name in observed_terrains
            ]
            self.ax_terrain.bar(
                [TERRAIN_CN[name] for name in observed_terrains],
                rates,
                color="tab:green",
                alpha=0.8,
            )
            self.ax_terrain.axhline(0.85, color="black", linestyle="--", alpha=0.5)
            self.ax_terrain.set_ylim(0.0, 1.0)
            self.ax_terrain.tick_params(axis="x", rotation=35)
        else:
            self.ax_terrain.text(
                0.5,
                0.5,
                "等待完成回合",
                ha="center",
                va="center",
                transform=self.ax_terrain.transAxes,
            )
        self.ax_terrain.set_title("已观测地形累计成功率")
        self.ax_terrain.set_ylabel("成功率")
        self.ax_terrain.grid(True, axis="y", alpha=0.25)

        self.figure.suptitle(
            f"Go2W训练监控 {MONITOR_VERSION}｜"
            f"iteration {int(float(latest.get('iteration', 0)))}/"
            f"{int(float(latest.get('max_iterations', 0)))}｜"
            f"{value(latest, 'steps_per_second', 0)} steps/s｜"
            f"noise {value(latest, 'action_noise_std', 3)}"
        )
        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        self.figure.canvas.draw_idle()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.figure.number)


class CsvWriter:
    def __init__(self, path: Optional[Path]):
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

    def append(self, row: MetricRow) -> None:
        if self.writer is None or self.file is None:
            return
        self.writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        self.file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def value(row: MetricRow, field: str, digits: int) -> str:
    try:
        number = float(row.get(field, float("nan")))
    except (TypeError, ValueError):
        return "--"
    return f"{number:.{digits}f}" if math.isfinite(number) else "--"


def scientific(row: MetricRow, field: str) -> str:
    try:
        number = float(row.get(field, float("nan")))
    except (TypeError, ValueError):
        return "--"
    return f"{number:.2e}" if math.isfinite(number) else "--"


def print_summary(row: MetricRow) -> None:
    print(
        " ".join(
            (
                f"iter={value(row, 'iteration', 0)}/{value(row, 'max_iterations', 0)}",
                f"sps={value(row, 'steps_per_second', 0)}",
                f"reward={value(row, 'mean_reward', 2)}",
                f"len={value(row, 'mean_episode_length', 1)}",
                f"success={value(row, 'success_rate', 3)}",
                f"collision={value(row, 'collision_rate', 3)}",
                f"unstable={value(row, 'unstable_rate', 3)}",
                f"timeout={value(row, 'timeout_rate', 3)}",
                f"other={value(row, 'other_failure_rate', 3)}",
            )
        ),
        flush=True,
    )
    print(
        "  "
        + " ".join(
            (
                f"curriculum={value(row, 'curriculum_mean_level', 0)}+{value(row, 'curriculum_frontier_difficulty', 3)}",
                f"sampled={value(row, 'curriculum_mean_sampled_stage', 2)}",
                f"ema={value(row, 'curriculum_frontier_success_ema', 3)}/{value(row, 'curriculum_full_success_ema', 3)}",
                f"full_n={value(row, 'curriculum_full_episode_count', 0)}",
                f"value={value(row, 'value_function_loss', 3)}",
                f"surrogate={value(row, 'surrogate_loss', 5)}",
                f"lr={scientific(row, 'learning_rate')}",
                f"noise={value(row, 'action_noise_std', 3)}",
                f"map={value(row, 'sensor_observed_ratio', 3)}/{value(row, 'sensor_height_valid_ratio', 3)}",
                f"hits={value(row, 'sensor_hit_ratio', 3)}",
                f"stop={value(row, 'safety_stop_steps', 1)}",
            )
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="监控Go2W Docker训练的必要指标。"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {MONITOR_VERSION}",
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
        default=30000,
        help="启动时读取的Docker历史日志行数；0表示只看新日志。默认：30000",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=2000,
        help="曲线最多保留的iteration数量。默认：2000",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.5,
        help="绘图刷新周期，单位秒。默认：0.5",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="可选的指标CSV输出路径。",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="仅输出终端摘要/CSV，不创建Matplotlib窗口。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.history_lines < 0 or args.window <= 0 or args.refresh <= 0.0:
        raise ValueError("history-lines必须非负，window和refresh必须大于0")

    command = [
        "docker",
        "logs",
        "--tail",
        str(args.history_lines),
        "-f",
        args.container,
    ]
    print(f"Go2W训练监控版本：{MONITOR_VERSION}")
    print("执行：", " ".join(command))
    print("按 Ctrl+C 退出。")

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError:
        print("错误：找不到docker命令。", file=sys.stderr)
        return 1

    line_queue: queue.Queue[str] = queue.Queue()
    LogReader(process, line_queue).start()
    history = MetricHistory(maxlen=args.window)
    dashboard = None if args.no_gui else Dashboard(history)
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
                            merged = history.append(row)
                            csv_writer.append(merged)
                            print_summary(merged)

            now = time.monotonic()
            if dashboard is not None and (
                received or now - last_refresh >= args.refresh
            ):
                dashboard.update()
                plt.pause(0.01)
                last_refresh = now

            if process.poll() is not None and line_queue.empty():
                print(f"docker logs已退出，返回码：{process.returncode}")
                if dashboard is not None:
                    dashboard.update()
                    plt.ioff()
                    plt.show()
                return 0 if process.returncode == 0 else int(process.returncode)

            if dashboard is not None and not dashboard.is_open():
                print("监控窗口已关闭。")
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n收到Ctrl+C，正在退出……")
    finally:
        csv_writer.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
