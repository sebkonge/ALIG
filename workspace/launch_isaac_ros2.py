#!/usr/bin/env python3
"""
Usage:
  ./launch_isaac_ros2.py                       # full setup + interactive shell
  ./launch_isaac_ros2.py --fast                # skip colcon build too
  ./launch_isaac_ros2.py --setup               # redo the one-time dependency install
  ./launch_isaac_ros2.py --echo /clock         # run `ros2 topic echo /clock` instead of a shell
  ./launch_isaac_ros2.py --usd ~/other/scene.usd   # override the default USD file
  ./launch_isaac_ros2.py --no-usd              # start Isaac Sim without opening a file
  ./launch_isaac_ros2.py --no-isaac            # only start the ROS 2 side
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------- CONFIG ---------------------------------------
HOME = Path.home()

HOST_WS = HOME / "IsaacSim-ros_workspaces" / "jazzy_ws"

ISAAC_DIR = HOME / "isaacsim"

USD_FILE = HOME / "ALIG" / "workspace" / "kandidat_setup_template.usd"

DOCKER_IMAGE = "osrf/ros:jazzy-desktop"
READY_IMAGE = "jazzy-isaac-ready:latest"
SETUP_CONTAINER_NAME = "ros_ws_docker_setup"
CONTAINER_NAME = "ros_ws_docker"
ROS_DOMAIN_ID = "0"
FASTDDS_PROFILE = "/jazzy_ws/fastdds.xml"
# -----------------------------------------------------------------------------


def log(msg):
    print(f"\033[1;36m[launch]\033[0m {msg}", flush=True)


def fail(msg):
    print(f"\033[1;31m[error]\033[0m {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def check_prereqs():
    if shutil.which("docker") is None:
        fail("docker not found in PATH.")
    if not HOST_WS.is_dir():
        fail(f"Workspace folder not found: {HOST_WS}\nEdit HOST_WS at the top of this script.")

    res = subprocess.run(["docker", "ps"], capture_output=True, text=True)
    if res.returncode != 0:
        user = os.environ.get("USER", "$USER")
        fail(
            "Cannot talk to Docker as your user.\n"
            f"  1) Add yourself:   sudo usermod -a -G docker {user}\n"
            "  2) Then either log out/in, or run this script via:  newgrp docker\n"
            f"Docker said: {res.stderr.strip()}"
        )

    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def allow_x11():
    if shutil.which("xhost"):
        subprocess.run(["xhost", "+"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        log("xhost not found - GUI tools (rviz etc.) in the container may not open.")


def start_isaac(usd_file):
    script = ISAAC_DIR / "isaac-sim.sh"
    if not script.is_file():
        fail(f"Could not find {script}\nEdit ISAAC_DIR at the top of this script.")

    cmd = [str(script)]
    if usd_file:
        usd = Path(usd_file).expanduser().resolve()
        if not usd.is_file():
            fail(f"USD file not found: {usd}")
        open_stage = ISAAC_DIR / "open_stage.py"
        script_ref = str(open_stage) if open_stage.is_file() else "open_stage.py"
        cmd += ["--exec", f"{script_ref} {usd}"]
    return subprocess.Popen(cmd, cwd=ISAAC_DIR, start_new_session=True)


def stop_isaac(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def ensure_ready_image(force):
    """
    Use --setup to redo this (e.g. after adding new packages to src/).
    """
    exists = subprocess.run(["docker", "image", "inspect", READY_IMAGE],
                            capture_output=True).returncode == 0
    if exists and not force:
        log(f"Using existing image {READY_IMAGE} (dependencies already installed).")
        return

    log("One-time setup: installing dependencies (this is only needed once) ...")
    subprocess.run(["docker", "rm", "-f", SETUP_CONTAINER_NAME],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    setup_script = "\n".join([
        "export DEBIAN_FRONTEND=noninteractive",
        "export PIP_BREAK_SYSTEM_PACKAGES=1",
        "cd /jazzy_ws",
        "apt-get update",
        "rosdep update",
        "rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y"
        " || echo '[launch] rosdep reported errors'",
    ])
    rc = subprocess.call([
        "docker", "run",
        "--net=host",
        "-v", f"{HOST_WS}:/jazzy_ws",
        "--name", SETUP_CONTAINER_NAME,
        DOCKER_IMAGE,
        "/bin/bash", "-c", setup_script,
    ])
    if rc != 0:
        subprocess.run(["docker", "rm", "-f", SETUP_CONTAINER_NAME],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fail("Dependency setup failed - see the output above.")

    subprocess.run(["docker", "commit", SETUP_CONTAINER_NAME, READY_IMAGE],
                   stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["docker", "rm", "-f", SETUP_CONTAINER_NAME],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"Saved image {READY_IMAGE}.")


def build_inner_script(fast, echo_topic):
    """Bash that runs INSIDE the container (replaces all the manual typing)."""
    lines = [
        "cd /jazzy_ws",
        "export PIP_BREAK_SYSTEM_PACKAGES=1",
        "source /opt/ros/jazzy/setup.bash",
        f"export FASTRTPS_DEFAULT_PROFILES_FILE={FASTDDS_PROFILE}",
        f"export ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
        "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
    ]

    if not fast:
        lines += [
            "colcon build --symlink-install || echo '[launch] colcon build reported errors'",
        ]

    lines += [
        "source /jazzy_ws/install/setup.bash",
        "echo",
        "echo '[launch] ROS 2 environment ready.'",
        "echo '[launch] Reminder: in Isaac Sim, open the Action Graph (Edit Graph) and press Play.'",
        "echo",
    ]

    if echo_topic:
        lines += [
            "ros2 topic list",
            f"ros2 topic echo {echo_topic}",
        ]
    else:
        lines += ["exec bash"]

    return "\n".join(lines)


def run_container(fast, echo_topic):
    cmd = ["docker", "run"]
    cmd.append("-it" if sys.stdin.isatty() else "-i")
    cmd += [
        "--rm",
        "--net=host",
        "--env=DISPLAY",
        f"--env=ROS_DOMAIN_ID={ROS_DOMAIN_ID}",
        "-v", f"{HOST_WS}:/jazzy_ws",
        "--name", CONTAINER_NAME,
        READY_IMAGE,
        "/bin/bash", "-c", build_inner_script(fast, echo_topic),
    ]
    log("Starting ROS 2 Jazzy container ...")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description="Launch Isaac Sim + ROS 2 Jazzy bridge")
    parser.add_argument("--usd", default=str(USD_FILE),
                        help=f"USD file to open in Isaac Sim (default: {USD_FILE})")
    parser.add_argument("--no-usd", action="store_true",
                        help="Start Isaac Sim with an empty stage instead of opening the USD file")
    parser.add_argument("--no-isaac", action="store_true", help="Don't start Isaac Sim")
    parser.add_argument("--fast", action="store_true",
                        help="Skip colcon build (use the existing install/ folder as is)")
    parser.add_argument("--setup", action="store_true",
                        help="Redo the one-time apt/rosdep dependency install (after adding new packages)")
    parser.add_argument("--echo", metavar="TOPIC",
                        help="Run `ros2 topic echo TOPIC` instead of opening a shell (e.g. /clock)")
    args = parser.parse_args()

    check_prereqs()
    ensure_ready_image(args.setup)
    allow_x11()

    isaac = None
    try:
        if not args.no_isaac:
            isaac = start_isaac(None if args.no_usd else args.usd)
            time.sleep(2)  # let it begin loading while the container builds
        run_container(args.fast, args.echo)
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        stop_isaac(isaac)
        log("Done.")


if __name__ == "__main__":
    main()
