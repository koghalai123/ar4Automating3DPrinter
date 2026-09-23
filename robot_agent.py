#!/usr/bin/env python3
"""Robot-side command server, run on the computer wired to the arm.

The website computer POSTs commands here; this process looks each one up in a
whitelist and calls the matching printerAutomation method. Standard library
only on the HTTP side, so there is nothing to pip install into the ROS
environment. Otherwise it is another runner script like runScrapePlate.py --
same node, same offset configs, same save file, with commands arriving over
HTTP instead of from a menu.

A move takes seconds to minutes, so the POST doesn't wait for it:

    202  {"received": true, "id": 4, "name": "go_home", ...}

is the delivery confirmation, and the caller then polls

    GET /command/4  ->  {"state": "running"}          ... later ...
                        {"state": "done", "result": "home", "duration_s": 12.4}

No request is held open for the length of a trajectory that way.

Commands run one at a time. A second one arriving mid-move is refused (409)
rather than queued, so the operator hears about it immediately instead of
finding a stale move ran minutes later; handlers can assume nothing else is
driving the arm, like the run* scripts do.

Run it:  python3 robot_agent.py     (after sourcing the ROS workspace and
                                     launching the arm stack)
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Address, port and token live in robot_link.py, the file shared byte-for-byte
# with the website computer, so the two ends can't drift.
from tools.robot_link import (AUTH_TOKEN, BIND_HOST, PROTOCOL_VERSION, ROBOT_HOST,
                             ROBOT_PORT)

# ---- config (edit these; no CLI args) ------------------------------------
ROBOT = 'xarm6'      # 'ar4' | 'lite6' | 'xarm6' (see ar4_automation/robot_config.py)
SPEED_SCALE = 0.5    # Default MoveIt vel/accel scaling; editable per motion command

# 1 = the Gazebo stack (launchVirtualXArm6.sh), images off the bridged camera.
# 0 = the real arm, images off the USB webcam. Getting this wrong hangs the
# first command: the webcam path prompts for a camera index when it can't find
# one, and nothing here can answer a prompt.
SIM = 1

# 1 = testing mode: handlers print the call they would have made and return it
# as the result, so the website's whole path can be checked with no arm and no
# ROS. Set to 0 on the robot computer to drive the arm.
DRY_RUN = 0

# 1 = plan against the collision scene: a ground plane under the base plus a box
# model of every printer added to it. 0 = plan in an EMPTY world, where only
# self-collisions and joint limits constrain the path, so a move is free to sweep
# the EEF through the floor or a printer. Diagnostic only — this agent takes
# motion commands from a website with nobody's hand on a stop button.
COLLISIONS = 1

HISTORY_LIMIT = 50   # finished commands kept around for polling


# ==========================================================================
# The ROS node, created lazily
# ==========================================================================
# Started on first use rather than at import, so /ping answers while the arm
# stack is still coming up. Reused after that: MoveIt/TF state has to persist.

_node = None
_node_lock = threading.Lock()


def dry(description: str) -> str:
    """Print the call that would have run and return it as the command's
    result, so it reaches the website's result panel too."""
    line = f"[DRY RUN] {description}"
    print(line, flush=True)
    return line


def get_node():
    """The shared printerAutomation node, starting it on first use."""
    global _node
    with _node_lock:
        if _node is not None:
            return _node
        import rclpy
        from ar4_automation.runner_common import start_node

        # Nothing can type an answer here, and select_camera falls back to
        # input() when no camera matches its keyword. Dropping stdin turns that
        # into an immediate error instead of a command stuck at "running".
        sys.stdin = None
        rclpy.init()
        node = start_node(sim=bool(SIM), robot=ROBOT, collisions=COLLISIONS)
        node.moveit2.max_velocity = SPEED_SCALE
        node.moveit2.max_acceleration = SPEED_SCALE
        if not node.load_state():
            node.get_logger().warn(
                "No save file found -- marker poses are unknown. Run "
                "scanFor2Markers.py before sending motion commands.")
        else:
            from ar4_automation.runner_common import restore_saved_printers
            restore_saved_printers(node)
        _node = node
        return _node


# ==========================================================================
# Command registry
# ==========================================================================

HANDLERS: dict[str, callable] = {}


def command(fn):
    """Expose `fn` to the website under its own name. This dict is the
    whitelist: nothing undecorated can be invoked."""
    HANDLERS[fn.__name__] = fn
    return fn


def command_specs() -> list[dict]:
    """Name, docstring and parameters of every command. The website builds its
    input forms from this, so a @command added here shows up in the UI with no
    frontend change."""
    specs = []
    for name, fn in sorted(HANDLERS.items()):
        params = []
        for pname, p in inspect.signature(fn).parameters.items():
            default = None if p.default is inspect.Parameter.empty else p.default
            params.append({
                "name": pname,
                "default": default,
                # input type in the browser, inferred from the default
                "type": ("number" if isinstance(default, (int, float))
                         and not isinstance(default, bool) else "text"),
            })
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        specs.append({"name": name, "doc": doc, "params": params,
                      "moves": name in MOTION_COMMANDS})
    return specs


# Commands that physically move the arm. The website warns before running one
# (and skips the warning in DRY_RUN, where nothing moves).
MOTION_COMMANDS = {"go_home", "scan_marker", "pickup_plate", "scrape_plate"}


# ---- link tests (no hardware) ----

@command
def echo(message: str = "hello"):
    """Round-trip test: proves the link works without moving anything."""
    return f"robot received: {message}"


@command
def slow_task(seconds: float = 5.0):
    """Fake long-running move, for exercising the poll-for-result flow."""
    time.sleep(float(seconds))
    return f"slept {seconds}s"


# ---- read-only state ----

@command
def list_markers():
    """Markers the node currently knows about, and whether each pose is a real
    scan or still just an estimate."""
    if DRY_RUN:
        dry("read node.marker_poses")
        # same shape as the real return, so the website's rendering of a
        # non-string result gets exercised too
        return [{"id": 1, "position": [0.30, -0.10, 0.05], "estimated": True},
                {"id": 2, "position": [0.40, 0.10, 0.05], "estimated": False}]
    return [
        {"id": e["id"],
         "position": [round(float(v), 4) for v in e["positionInBase"]],
         "estimated": bool(e.get("estimated", False))}
        for e in get_node().marker_poses if "positionInBase" in e
    ]


# ---- motion ----
# Under DRY_RUN each prints the call it would have made and returns that line
# as its result.

def motion_node(velocity_scaling):
    """Validate and apply this command's velocity and acceleration scaling."""
    if isinstance(velocity_scaling, bool):
        raise ValueError("velocity_scaling must be a number greater than 0 and at most 1")
    try:
        scale = float(velocity_scaling)
    except (TypeError, ValueError):
        raise ValueError("velocity_scaling must be a number greater than 0 and at most 1") from None
    if not 0 < scale <= 1:
        raise ValueError("velocity_scaling must be greater than 0 and at most 1")
    if DRY_RUN:
        dry(f"velocity and acceleration scaling={scale}")
        return None
    node = get_node()
    node.moveit2.max_velocity = scale
    node.moveit2.max_acceleration = scale
    return node


@command
def go_home(velocity_scaling: float = SPEED_SCALE):
    node = motion_node(velocity_scaling)
    if DRY_RUN:
        return dry(f"node.go_home(velocity_scaling={velocity_scaling})")
    node.go_home(velocity_scaling=float(velocity_scaling))
    return "home"


@command
def scan_marker(marker_id: int = 1, viewing_distance: float = 0.15,
                velocity_scaling: float = SPEED_SCALE):
    node = motion_node(velocity_scaling)
    if DRY_RUN:
        return dry(f"node.scanMarkerApproach(marker_id={int(marker_id)}, "
                   f"viewing_distance={float(viewing_distance)})")
    node.scanMarkerApproach(marker_id=int(marker_id),
                                  viewing_distance=float(viewing_distance))
    return f"scanned marker {marker_id}"


@command
def pickup_plate(source_id: int = 2, velocity_scaling: float = SPEED_SCALE):
    node = motion_node(velocity_scaling)
    if DRY_RUN:
        return dry(f"node.pickupOnly(source_id={int(source_id)}, "
                   f"wait_after_pickup=False)")
    ok = node.pickupOnly(source_id=int(source_id),
                               wait_after_pickup=False)
    return "picked up" if ok else "pickup failed"


@command
def scrape_plate(source_id: int = 2, scrape_id: int = 1,
                 velocity_scaling: float = SPEED_SCALE):
    node = motion_node(velocity_scaling)
    if DRY_RUN:
        return dry(f"node.scrapePlate(source_id={int(source_id)}, "
                   f"scrape_id={int(scrape_id)}, wait_after_pickup=False)")
    ok = node.scrapePlate(source_id=int(source_id),
                                scrape_id=int(scrape_id),
                                wait_after_pickup=False)
    return "scraped" if ok else "scrape failed"


# ==========================================================================
# Execution state
# ==========================================================================

class CommandRunner:
    """Tracks the running command and recently finished ones. Locked because
    the HTTP server is threaded: polls arrive while the worker thread is still
    writing these records."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id = 1
        self._commands: dict[int, dict] = {}
        self._order: list[int] = []
        self._busy_id: int | None = None

    def submit(self, name: str, params: dict) -> dict:
        """Start a command and return the receipt. Raises RuntimeError if
        another one is still running."""
        with self._lock:
            if self._busy_id is not None:
                busy = self._commands[self._busy_id]
                raise RuntimeError(
                    f"busy running '{busy['name']}' (id {busy['id']}), "
                    f"started {time.time() - busy['started_at']:.0f}s ago")
            cmd_id = self._next_id
            self._next_id += 1
            now = time.time()
            self._commands[cmd_id] = {
                "id": cmd_id, "name": name, "params": params,
                "state": "running", "result": None, "error": "",
                "received_at": now, "started_at": now,
                "finished_at": None, "duration_s": None,
            }
            self._order.append(cmd_id)
            self._trim()
            self._busy_id = cmd_id

        threading.Thread(target=self._run, args=(cmd_id, name, params),
                         daemon=True, name=f"cmd-{cmd_id}-{name}").start()
        return {"received": True, "id": cmd_id, "name": name, "params": params,
                "received_at": now}

    def _run(self, cmd_id: int, name: str, params: dict) -> None:
        started = time.time()
        try:
            value = HANDLERS[name](**params)
            state, result, error = "done", value, ""
        except Exception as e:
            # a failed move is reported, not fatal to the server
            traceback.print_exc()
            state, result, error = "failed", None, f"{type(e).__name__}: {e}"
        with self._lock:
            self._commands[cmd_id].update(
                state=state, result=result, error=error,
                finished_at=time.time(),
                duration_s=round(time.time() - started, 3))
            self._busy_id = None

    def get(self, cmd_id: int) -> dict | None:
        with self._lock:
            record = self._commands.get(cmd_id)
            return dict(record) if record else None

    def status(self) -> dict:
        with self._lock:
            busy = self._commands.get(self._busy_id) if self._busy_id else None
            return {"ok": True, "protocol": PROTOCOL_VERSION, "robot": ROBOT,
                    "dry_run": bool(DRY_RUN),
                    "commands": command_specs(), "busy": busy is not None,
                    "current": dict(busy) if busy else None,
                    "history": [dict(self._commands[i]) for i in
                                reversed(self._order[-10:])]}

    def _trim(self) -> None:
        """Drop the oldest records past the limit (lock held)."""
        while len(self._order) > HISTORY_LIMIT:
            self._commands.pop(self._order.pop(0), None)


RUNNER = CommandRunner()


# ==========================================================================
# HTTP layer
# ==========================================================================

class Handler(BaseHTTPRequestHandler):
    server_version = "robot_agent/1.0"

    def _send(self, code: int, payload: dict) -> None:
        # default=str so a handler returning a numpy array or ROS message can't
        # turn a finished move into a 500.
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authed(self) -> bool:
        if self.headers.get("X-Auth-Token") == AUTH_TOKEN:
            return True
        self._send(401, {"error": "bad or missing X-Auth-Token"})
        return False

    def log_message(self, fmt, *args):
        """One line per request, naming the command rather than its id so the
        log can be matched against the button pressed. The handlers below set
        _log_path once they know the name."""
        line = fmt % args
        shown = getattr(self, "_log_path", None)
        if shown:
            line = line.replace(self.path, shown, 1)
        print(f"[agent] {self.address_string()} {line}", flush=True)

    def do_GET(self):
        if not self._authed():
            return
        if self.path == "/ping":
            self._send(200, RUNNER.status())
        elif self.path.startswith("/command/"):
            cmd_id = int(self.path.rsplit("/", 1)[-1])
            record = RUNNER.get(cmd_id)
            if record is None:
                self._send(404, {"error": f"unknown command id {cmd_id}"})
            else:
                self._log_path = f"/command/{record['name']}"
                self._send(200, record)
        else:
            self._send(404, {"error": f"no route {self.path}"})

    def do_POST(self):
        if not self._authed():
            return
        if self.path != "/command":
            self._send(404, {"error": f"no route {self.path}"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            name = body["name"]
        except (ValueError, KeyError):
            self._send(400, {"error": 'body must be {"name": ..., "params": {...}}'})
            return
        # set before the checks below so a rejected command still names itself
        self._log_path = f"/command/{str(name)[:60]}"
        params = body.get("params") or {}
        if name not in HANDLERS:
            self._send(404, {"error": f"unknown command '{name}'",
                             "commands": sorted(HANDLERS)})
            return

        try:
            receipt = RUNNER.submit(name, params)
        except RuntimeError as e:      # already running something
            self._send(409, {"error": str(e)})
            return
        self._send(202, receipt)


def main():
    server = ThreadingHTTPServer((BIND_HOST, ROBOT_PORT), Handler)
    print(f"robot agent listening on {BIND_HOST}:{ROBOT_PORT}")
    print(f"  clients connect to http://{ROBOT_HOST}:{ROBOT_PORT}")
    print(f"  robot={ROBOT}")
    print(f"  commands: {sorted(HANDLERS)}")
    if DRY_RUN:
        print("  *** TESTING MODE (DRY_RUN = 1): nothing will move. Each "
              "command prints the call it would have made. ***")
    else:
        print("  LIVE: commands will move the arm.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
