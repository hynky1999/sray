#!/usr/bin/env python3
"""Command-line utility to launch Ray clusters on Slurm via srun."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional


DEFAULT_NODES = 2
DEFAULT_CPUS_PER_NODE = 4
DEFAULT_GPUS_PER_NODE = 0
DEFAULT_MEM_PER_NODE = "16G"


@dataclass
class WorkerProcess:
    """Container tracking a worker Slurm task."""

    index: int
    process: subprocess.Popen


class GracefulKiller:
    """Helper for handling shutdown signals."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        signal.signal(signal.SIGINT, self._handler)  # type: ignore[arg-type]
        signal.signal(signal.SIGTERM, self._handler)  # type: ignore[arg-type]

    def _handler(self, signum: int, frame: Optional[object]) -> None:  # pragma: no cover - signature required by signal
        print(f"Received signal {signum}, shutting down...", file=sys.stderr)
        self._stop.set()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop

    def wait(self) -> None:
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._stop.set()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Ray workers on a Slurm cluster.")
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES, help="Total number of nodes to allocate, including the head node.")
    parser.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE, help="GPUs to request per node.")
    parser.add_argument("--cpus-per-node", type=int, default=DEFAULT_CPUS_PER_NODE, help="CPUs to request per node.")
    parser.add_argument("--mem-per-node", default=DEFAULT_MEM_PER_NODE, help="Memory to request per node (Slurm format, e.g. 32G).")
    parser.add_argument("--partition", default=None, help="Optional Slurm partition name.")
    parser.add_argument("--dashboard-port", type=int, default=8265, help="Port for the Ray dashboard.")
    parser.add_argument("--port", type=int, default=6379, help="Port for the Ray GCS service (head node).")
    parser.add_argument("--ray-client-port", type=int, default=10001, help="Port for the Ray client server.")
    return parser.parse_args(argv)


def get_head_ip() -> str:
    env_ip = os.environ.get("SRAY_HEAD_IP")
    if env_ip:
        return env_ip
    try:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
    except OSError:
        ip_addr = "127.0.0.1"
    return ip_addr


def build_head_command(args: argparse.Namespace, ip_address: str) -> List[str]:
    cmd = [
        "ray",
        "start",
        "--head",
        f"--node-ip-address={ip_address}",
        f"--port={args.port}",
        f"--dashboard-port={args.dashboard_port}",
        f"--ray-client-server-port={args.ray_client_port}",
        f"--num-cpus={args.cpus_per_node}",
    ]
    if args.gpus_per_node:
        cmd.append(f"--num-gpus={args.gpus_per_node}")
    cmd.append("--block")
    return cmd


def build_worker_command(args: argparse.Namespace, head_address: str) -> str:
    ray_cmd = [
        "ray",
        "start",
        f"--address={head_address}",
        f"--num-cpus={args.cpus_per_node}",
    ]
    if args.gpus_per_node:
        ray_cmd.append(f"--num-gpus={args.gpus_per_node}")
    ray_cmd.append("--block")
    return " ".join(ray_cmd)


def build_srun_command(args: argparse.Namespace, worker_script: str) -> List[str]:
    cmd = [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={args.cpus_per_node}",
        f"--mem={args.mem_per_node}",
    ]
    if args.gpus_per_node:
        cmd.append(f"--gpus-per-task={args.gpus_per_node}")
    if args.partition:
        cmd.append(f"--partition={args.partition}")
    cmd.extend([
        "bash",
        "-lc",
        worker_script,
    ])
    return cmd


def launch_head(args: argparse.Namespace, ip_address: str) -> subprocess.Popen:
    cmd = build_head_command(args, ip_address)
    print("Starting Ray head node:", " ".join(cmd))
    return subprocess.Popen(cmd, preexec_fn=os.setsid)


def launch_worker(args: argparse.Namespace, head_address: str, index: int) -> subprocess.Popen:
    worker_script = build_worker_command(args, head_address)
    cmd = build_srun_command(args, worker_script)
    print(f"Launching worker {index}:", " ".join(cmd))
    return subprocess.Popen(cmd, preexec_fn=os.setsid)


def terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print(f"Force killing {name} (pid={proc.pid}).", file=sys.stderr)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def monitor_workers(args: argparse.Namespace, head_address: str, workers: List[WorkerProcess], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        for position, worker in enumerate(list(workers)):
            ret = worker.process.poll()
            if ret is None:
                continue
            if stop_event.is_set():
                break
            print(f"Worker {worker.index} exited with code {ret}. Restarting...")
            new_process = launch_worker(args, head_address, worker.index)
            workers[position] = WorkerProcess(worker.index, new_process)
        time.sleep(2)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if args.nodes < 1:
        print("--nodes must be at least 1", file=sys.stderr)
        return 1

    head_ip = get_head_ip()
    head_address = f"{head_ip}:{args.port}"

    print("Ray head will listen on:", head_address)

    killer = GracefulKiller()

    head_process = launch_head(args, head_ip)
    time.sleep(3)

    if head_process.poll() is not None:
        print("Ray head failed to start. See output above for details.", file=sys.stderr)
        return head_process.returncode or 1

    print("Ray head started. Connect using: ray.init(address=\"ray://%s:%s\")" % (head_ip, args.ray_client_port))
    print(f"Dashboard available at http://{head_ip}:{args.dashboard_port}")
    print("Press Ctrl+C to shut down the cluster.")

    workers: List[WorkerProcess] = []
    worker_count = max(args.nodes - 1, 0)

    for index in range(worker_count):
        proc = launch_worker(args, head_address, index + 1)
        workers.append(WorkerProcess(index + 1, proc))
        time.sleep(1)

    monitor_thread: Optional[threading.Thread] = None
    if workers:
        monitor_thread = threading.Thread(
            target=monitor_workers,
            args=(args, head_address, workers, killer.stop_event),
            daemon=True,
        )
        monitor_thread.start()

    try:
        killer.wait()
    finally:
        killer.stop_event.set()
        print("Stopping workers...")
        for worker in workers:
            terminate_process(worker.process, f"worker-{worker.index}")

        print("Stopping head node...")
        terminate_process(head_process, "ray-head")
        subprocess.run(["ray", "stop"], check=False)
        if monitor_thread is not None:
            monitor_thread.join(timeout=5)
        print("Cluster shut down.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
