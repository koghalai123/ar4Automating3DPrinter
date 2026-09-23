#!/usr/bin/env python3
"""
Print-and-scrape loop. For each print in the YAML queue: print it on the
Bambu at SOURCE_ID, wait, move the head clear, scrape the plate against
SCRAPE_ID and return it. Edit the config section and queue file first.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
import yaml

from ar4_automation.runner_common import (
    start_webcam_node,
    restore_saved_printers,
    register_manual_estimates,
)
from ar4_automation.printerclass import BambuPrinter, load_printer_config, strip_startup_gcode


# ---- Configuration ----
ROBOT           = 'xarm6'       # 'ar4' | 'lite6' | 'xarm6' (see ar4_automation/robot_config.py).
                              # Marker poses and printer configs come from the
                              # per-robot save file (run scanFor2Markers.py for
                              # this ROBOT first) — no hardcoded frame values here.
SOURCE_ID       = 2           # marker to pick the plate from (and return it to)
SCRAPE_ID       = 1           # marker whose surface gets scraped against
SCAN_DISTANCE   = 0.15        # marker scan distance (m)
# 1 = seed the initial scans from data/manual_marker_estimates.json (written
# by teachMarkersByHand.py), overriding any saved poses; with estimates for
# both SOURCE_ID and SCRAPE_ID a save file is optional
USE_MANUAL_ESTIMATES = 1
# scrape motion (standoff/depth/retract) comes from the 'scrape' waypoint
# list of the scrape marker's offset config in printerAutomation.__init__

# credentials come from printer_config.yaml (copy the example file)
PRINTER_NAME = "a1"

# lists print folder, file names and repeat counts (see print_queue.yaml)
PRINT_QUEUE_FILE = "print_queue.yaml"

# strip the printer's startup gcode (purge, bed level, nozzle wipe, re-home)
# before uploading, keeping only heat-and-home plus the blob squirt. the
# stock startup shoves the plate around and breaks the scrape.
STRIP_STARTUP = False
# 1 = plan against the collision scene: a ground plane under the base plus a box
# model of every printer added to it. 0 = plan in an EMPTY world, where only
# self-collisions and joint limits constrain the path, so a move is free to sweep
# the EEF through the floor or a printer. Use 0 to tell "the plan is in
# collision" apart from "the goal is unreachable" — never for an unattended job
# like this one, which drives real hardware for the whole print queue.
COLLISIONS = 1
# ---- End Configuration ----


def load_print_queue(queue_file):
    """
    Read the YAML queue, return the ordered list of local print-file paths
    with each file repeated 'count' times.

    Format:
        print_folder: gcode
        prints:
          - name: bed_scraper_a1mini.gcode.3mf
            count: 20
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    queue_path = os.path.join(base_dir, queue_file)
    with open(queue_path, "r") as f:
        cfg = yaml.safe_load(f)

    folder = cfg.get("print_folder", "")
    entries = cfg.get("prints") or []
    if not entries:
        raise ValueError(f"No 'prints' entries found in {queue_path}")

    queue = []
    for entry in entries:
        name = entry["name"]
        count = int(entry.get("count", 1))
        if count < 1:
            continue
        local_file = os.path.join(folder, name)
        if not os.path.exists(os.path.join(base_dir, local_file)):
            raise FileNotFoundError(f"Print file not found: {local_file} "
                                    f"(from {queue_path})")
        queue.extend([local_file] * count)
    return queue


def main():
    # Load and validate the queue first so a bad YAML fails before any
    # hardware is touched.
    print_queue = load_print_queue(PRINT_QUEUE_FILE)

    rclpy.init()
    node = start_webcam_node(robot=ROBOT, collisions=COLLISIONS)

    # Load save file — restores marker poses, offset config, and printer configs
    have_save = node.load_state()
    if have_save:
        restore_saved_printers(node)

    # hand-taught estimates (after load_state so stale saved poses can't
    # shadow them) — the official scanMarkerApproach calls below re-measure
    # both markers before any motion uses them
    manual_ids = []
    if USE_MANUAL_ESTIMATES:
        manual_ids = register_manual_estimates(
            node, marker_ids=[SOURCE_ID, SCRAPE_ID]
        )
    if not have_save and not {SOURCE_ID, SCRAPE_ID}.issubset(manual_ids):
        node.get_logger().error(
            f"No save file and no hand-taught estimates for markers "
            f"{sorted({SOURCE_ID, SCRAPE_ID})} — run scanFor2Markers.py or "
            f"teachMarkersByHand.py first."
        )
        return

    # Connect the Bambu printer and register it with the node.
    printer_cfg = load_printer_config(PRINTER_NAME)
    bambu = BambuPrinter(
        ip=printer_cfg["ip"],
        access_code=printer_cfg["access_code"],
        serial=printer_cfg["serial"],
    )
    bambu.connect()
    bambu.enable_debug_listener()
    node.register_bambu_printer(SOURCE_ID, bambu)

    # Prepare and upload each unique file in the queue once, and remember the
    # name to print it under. upload_file_timeout stores files at the SD card
    # root under their basename, so prints are started by basename.
    remote_names = {}
    for local_file in dict.fromkeys(print_queue):  # unique, in queue order
        if STRIP_STARTUP:
            try:
                local_file_prepped = strip_startup_gcode(local_file)
                node.get_logger().info(
                    f"Stripped startup gcode: {local_file} -> {local_file_prepped}"
                )
            except ValueError as e:
                # e.g. a file sliced with a custom start-gcode profile that
                # doesn't have the stock Bambu markers — print it as-is.
                node.get_logger().warning(
                    f"Could not strip startup from {local_file}: {e} "
                    f"Uploading it UNMODIFIED (full startup will run)."
                )
                local_file_prepped = local_file
        else:
            local_file_prepped = local_file
        if not bambu.upload_file_timeout(local_file_prepped):
            raise RuntimeError(f"Upload failed for {local_file_prepped}")
        remote_names[local_file] = os.path.basename(local_file_prepped)

    # Marker setup. Marker poses, offset config, and printer configs all come
    # from the per-robot save file (load_state + restore_saved_printers above),
    # created by running scanFor2Markers.py for this ROBOT first — so this runs
    # on any arm without hardcoded, AR4-frame seed poses. Markers are pinned by
    # default: marker 1 (scrape) is scanned once here and keeps that pose for
    # every cycle (the scrape waypoints have no scan entries); marker 2 (pickup
    # source) is re-detected each cycle by its pickup-scan windows.
    viewing_distance = SCAN_DISTANCE
    # re-assert the scrape/source offset configs in case the save file predates
    # them (load_state restores whatever was saved).
    node.marker_offset_config[SCRAPE_ID] = 'box_offset'
    node.marker_offset_config[SOURCE_ID] = 'printer_offset'

    node.get_logger().info("Scanning for scrape marker 1...")
    node.scanMarkerApproach(marker_id=1, viewing_distance=viewing_distance)
    # marker 1 is now pinned at this freshly scanned pose: updates only happen
    # inside a scan window for its own ID, and nothing below scans marker 1

    node.get_logger().info("Scanning for source marker 2...")
    node.scanMarkerApproach(marker_id=2, viewing_distance=viewing_distance)
    node.get_logger().info("Initial scan complete.")

    # Main print-and-scrape loop over the queue
    total = len(print_queue)
    for cycle, local_file in enumerate(print_queue, start=1):
        remote_file = remote_names[local_file]
        node.get_logger().info(
            f"=== Cycle {cycle}/{total}: starting print {remote_file} ==="
        )

        bambu.start_print(remote_file)
        bambu.waitUntilPrintFinished()

        node.get_logger().info(f"=== Cycle {cycle}/{total}: print done, preparing for pickup ===")
        bambu.prepare_for_pickup()

        node.get_logger().info(f"=== Cycle {cycle}/{total}: scraping plate ===")
        # the post-scrape wrist roll is a {'rotate': deg} entry in the scrape
        # waypoint list (offset_configs), not an argument here
        ok = node.scrapePlate(
            source_id=SOURCE_ID,
            scrape_id=SCRAPE_ID,
            wait_after_pickup=True,
            wait_duration=90.0,
        )
        node.get_logger().info(
            f"=== Cycle {cycle}/{total}: scrapePlate {'succeeded' if ok else 'FAILED'} ==="
        )

        #node.save_state()

    node.get_logger().info("All cycles complete.")
    bambu.disconnect()


if __name__ == '__main__':
    main()
