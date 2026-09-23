#!/usr/bin/env python3
"""
Pick the plate off one printer, scrape it against another, put it back.
Set RUN_SIM = 1 for Gazebo (start scripts/launchVirtualRobot.sh first).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy

from ar4_automation.marker_sources import SCAN
from ar4_automation.runner_common import (
    start_node,
    restore_saved_printers,
    spawn_printers_from_markers,
    spawn_sim_printers,
    sim_printer_specs,
    require_scanned_markers,
)

# ---- Configuration ----
RUN_SIM         = 1          # 1 = Gazebo (sim camera + spawned printers), 0 = hardware
ROBOT           = 'xarm6'     # 'ar4' | 'lite6' | 'xarm6' (sim launch: launchVirtualRobot.sh / launchVirtualXArmLite6.sh / launchVirtualXArm6.sh)
SOURCE_ID       = 2           # marker to pick the plate from (and return it to)
SCRAPE_ID       = 1           # marker whose surface gets scraped against
# 1 = sim printers stand where the last scan MEASURED their markers
# (data/printer_state.json, written by scanFor2Markers.py), so Gazebo shows the
# same layout the arm is working from. 0 = spawn them at the runner_common
# layout and take the marker estimates from there instead — no scan needed, but
# the save file is then ignored and the scene won't match a scanned run.
SPAWN_FROM_SCAN = 1
# 1 = collisions on, in BOTH places they exist: the MoveIt planning scene (a
# ground plane under the base plus a box model of every printer) and Gazebo
# physics (printers spawn with real <collision> geometry). 0 = turn off both —
# the planning scene is emptied, including anything an earlier run left in
# move_group, and printers spawn visual-only so the arm passes through them
# instead of stalling against them. Self-collisions and joint limits are ALWAYS
# enforced. Use 0 to tell "the plan is in collision" apart from "the goal is
# unreachable", not for a real run.
COLLISIONS      = 1
# all motion and tilt angles come from the waypoint lists in
# printerAutomation.__init__'s offset_configs


def main():
    rclpy.init()
    node = start_node(sim=RUN_SIM, robot=ROBOT, collisions=COLLISIONS)
    speed_scale = 1.0
    node.moveit2.max_velocity = speed_scale
    node.moveit2.max_acceleration = speed_scale
    if RUN_SIM and not SPAWN_FROM_SCAN:
        # the spawned printers ARE the ground truth here: marker estimates are
        # derived from where they were placed, and the save file is ignored
        spawn_sim_printers(node, sim_printer_specs(ROBOT, 2))
    else:
        # save file restores marker poses, offset config, printer configs.
        # Markers are pinned by default (updates only during their own scan
        # waypoints), so the scrape marker keeps its file pose automatically.
        if not node.load_state():
            node.get_logger().error("No save file found — run scanFor2Markers.py first to create one.")
            return

        if RUN_SIM:
            # stand each printer where the scan measured its marker. Nothing is
            # re-registered, so the scanned poses stay exactly as saved;
            # fallback=False keeps an unscanned marker from getting a printer at
            # a made-up pose (require_scanned_markers below names it).
            spawn_printers_from_markers(node, sim_printer_specs(ROBOT, 2),
                                        source=SCAN, fallback=False)
        else:
            restore_saved_printers(node)
        # only officially scanned poses (scanFor2Markers.py) are accepted here
        # — hand-taught/geometric estimates must go through a scan first
        if not require_scanned_markers(node, [SOURCE_ID, SCRAPE_ID]):
            return

    # the post-scrape wrist roll is a {'rotate': deg} entry in the scrape
    # waypoint list (offset_configs), not an argument here
    ok = node.scrapePlate(
        source_id=SOURCE_ID,
        scrape_id=SCRAPE_ID,
        wait_after_pickup=False,
        wait_duration=10.0,
    )
    node.get_logger().info("scrapePlate succeeded." if ok else "scrapePlate failed.")


if __name__ == '__main__':
    main()
