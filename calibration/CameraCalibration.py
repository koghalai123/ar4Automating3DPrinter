import cv2
import numpy as np
import glob
import os
import sys
import argparse

# Make the repo root importable so ar4_automation resolves when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ar4_automation.web_video_server import select_camera
from calibration.charuco_utils import (
    calibrate_camera, create_board, detect, detector_parameters)

# --- ChArUco board parameters (adjust to match your physical board) ---
SQUARES_X = 11         # number of chessboard squares in X direction
SQUARES_Y = 8          # number of chessboard squares in Y direction
SQUARE_LENGTH = 0.015   # meters (length of one chessboard square)
MARKER_LENGTH = 0.011   # meters (length of one ArUco marker)
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# --- Image source ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_GLOB = os.path.join(_SCRIPT_DIR, "calibration_images", "*.jpg")
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "camera_matrix.npz")


def _detect_charuco(gray, board, aruco_dict, detector_params):
    """Detect ChArUco corners in a grayscale image (OpenCV 4.6 compatible)."""
    marker_corners, marker_ids, _, charuco_corners, charuco_ids = detect(
        gray, board, aruco_dict, detector_params)
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def collect_calibration_images(num_images: int = 30, camera_index: int = None):
    """Capture calibration frames from a live camera and save them to disk."""
    if camera_index is None:
        camera_index = select_camera(preset_keyword="GENERAL WEBCAM")
    os.makedirs(os.path.join(_SCRIPT_DIR, "calibration_images"), exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = create_board(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detector_params = detector_parameters()

    saved = 0
    print(f"Press SPACE to capture a frame ({num_images} needed), ESC to quit early.")
    while saved < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = _detect_charuco(
            gray, board, aruco_dict, detector_params
        )

        display = frame.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        if charuco_corners is not None and len(charuco_corners) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

        cv2.putText(display, f"Captured: {saved}/{num_images}  SPACE=save  ESC=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect calibration images", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        if key == 32 and charuco_corners is not None and len(charuco_corners) >= 4:
            path = os.path.join(_SCRIPT_DIR, "calibration_images", f"frame_{saved:03d}.jpg")
            cv2.imwrite(path, frame)
            print(f"  Saved {path}")
            saved += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {saved} images.")


def calibrate_from_images(image_glob: str = IMAGE_GLOB) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run ChArUco-based camera calibration on a set of images.

    Returns
    -------
    camera_matrix : np.ndarray (3x3)
    dist_coeffs   : np.ndarray (1x5)
    rms_error     : float  (reprojection error in pixels)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = create_board(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detector_params = detector_parameters()

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    image_paths = sorted(glob.glob(image_glob))
    if not image_paths:
        raise FileNotFoundError(f"No images found at '{image_glob}'")

    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]   # (width, height)

        charuco_corners, charuco_ids, _, marker_ids = _detect_charuco(
            gray, board, aruco_dict, detector_params
        )
        if marker_ids is None or len(marker_ids) < 4:
            print(f"  Skipping {path}: too few markers detected")
            continue
        if charuco_corners is None or len(charuco_corners) < 20:
            print(f"  Skipping {path}: too few ChArUco corners ({len(charuco_corners) if charuco_corners is not None else 0})")
            continue

        all_charuco_corners.append(charuco_corners)
        all_charuco_ids.append(charuco_ids)
        print(f"  Accepted {path}: {len(charuco_corners)} corners")

    if len(all_charuco_corners) < 5:
        raise RuntimeError(f"Need at least 5 valid frames, got {len(all_charuco_corners)}")

    print(f"\nCalibrating with {len(all_charuco_corners)} frames...")
    # Fix k3/k4/k5 to prevent the solver overfitting with too few diverse poses.
    # Only k1, k2, p1, p2 are estimated. Loosen this once you have a diverse
    # image set and the RMS drops below ~1 px.
    calib_flags = (cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5)
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = calibrate_camera(
        all_charuco_corners, all_charuco_ids, board, image_size,
        flags=calib_flags)

    print(f"\nReprojection RMS error: {rms:.4f} px")

    # Compare against previously saved calibration if available
    import pathlib
    prev_path = pathlib.Path(OUTPUT_FILE)
    if prev_path.exists():
        prev = np.load(prev_path)
        prev_cm = prev["camera_matrix"]
        prev_dc = prev["dist_coeffs"]
        print("\n--- Comparison with previous calibration ---")
        print(f"{'':30s}  {'Previous':>12}  {'New':>12}  {'Delta':>12}")
        labels = ["fx", "fy", "cx", "cy"]
        values = [
            (prev_cm[0,0], camera_matrix[0,0]),
            (prev_cm[1,1], camera_matrix[1,1]),
            (prev_cm[0,2], camera_matrix[0,2]),
            (prev_cm[1,2], camera_matrix[1,2]),
        ]
        for label, (old_v, new_v) in zip(labels, values):
            print(f"  {label:28s}  {old_v:>12.4f}  {new_v:>12.4f}  {new_v-old_v:>+12.4f}")
        print(f"\n  Previous dist coeffs: {prev_dc.ravel()}")
        print(f"  New      dist coeffs: {dist_coeffs.ravel()}")
        print(f"  Delta    dist coeffs: {(dist_coeffs.ravel() - prev_dc.ravel())}")
        print("--------------------------------------------")
    else:
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients:\n{dist_coeffs}")

    return camera_matrix, dist_coeffs, rms


def save_calibration(camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                     output_file: str = OUTPUT_FILE):
    np.savez(output_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\nCalibration saved to '{output_file}'")


def diagnose_detection(image_glob: str = IMAGE_GLOB, num_images: int = 3):
    """
    Open the first `num_images` accepted calibration images in a window and draw:
      - detected ArUco markers (green boxes)
      - detected ChArUco corners with their IDs labelled
    This lets you verify that the board parameters match the physical board.
    Press any key to advance, ESC to quit.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = create_board(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detector_params = detector_parameters()

    shown = 0
    for path in sorted(glob.glob(image_glob)):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = _detect_charuco(
            gray, board, aruco_dict, detector_params
        )
        if charuco_corners is None or len(charuco_corners) < 20:
            continue

        vis = img.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)

        # Label every corner with its ID so mismatch is obvious
        for corner, cid in zip(charuco_corners, charuco_ids.ravel()):
            pt = tuple(corner.ravel().astype(int))
            cv2.putText(vis, str(cid), (pt[0]+4, pt[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        h, w = vis.shape[:2]
        cv2.putText(vis,
            f"{path}  corners={len(charuco_corners)}  board={SQUARES_X}x{SQUARES_Y}",
            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        print(f"[diagnose] {path}: showing {len(charuco_corners)} corners")
        print(f"  Corner IDs detected: {sorted(charuco_ids.ravel().tolist())}")
        print(f"  Expected IDs 0..{(SQUARES_X-1)*(SQUARES_Y-1)-1}")
        cv2.imshow("diagnose_detection — any key=next, ESC=quit", vis)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == 27:
            break
        shown += 1
        if shown >= num_images:
            break


def generate_board_pdf(output_pdf: str = "charuco_board.pdf", dpi: int = 300,
                       paper: str = "A4"):
    """
    Render the ChArUco board centred on a standard sheet of paper so it can be
    sent directly to a printer.

    IMPORTANT: print at 100% / actual size — disable any 'fit to page' or
    'shrink to printable area' option so the square dimensions are preserved.

    Parameters
    ----------
    output_pdf : str   Output PDF path.
    dpi        : int   Rasterisation resolution (300 is sufficient).
    paper      : str   Paper size — 'A4' or 'Letter'.
    """
    from PIL import Image, ImageDraw, ImageFont

    paper_sizes_mm = {"A4": (210.0, 297.0), "Letter": (215.9, 279.4)}
    if paper not in paper_sizes_mm:
        raise ValueError(f"Unknown paper '{paper}'. Choose from: {list(paper_sizes_mm)}")
    page_w_mm, page_h_mm = paper_sizes_mm[paper]

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = create_board(
        SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict
    )

    board_w_mm = SQUARES_X * SQUARE_LENGTH * 1000.0
    board_h_mm = SQUARES_Y * SQUARE_LENGTH * 1000.0
    if board_w_mm > page_w_mm or board_h_mm > page_h_mm:
        raise RuntimeError(
            f"Board ({board_w_mm:.1f} x {board_h_mm:.1f} mm) does not fit on "
            f"{paper} ({page_w_mm:.1f} x {page_h_mm:.1f} mm)"
        )

    def mm_to_px(mm):
        return int(round(mm / 25.4 * dpi))

    page_px_w = mm_to_px(page_w_mm)
    page_px_h = mm_to_px(page_h_mm)
    board_px_w = mm_to_px(board_w_mm)
    board_px_h = mm_to_px(board_h_mm)

    board_img = board.draw((board_px_w, board_px_h))
    if board_img is None or board_img.size == 0:
        raise RuntimeError("board.draw() returned an empty image — check board parameters")

    # White page; board centred horizontally, centred vertically
    page = Image.new("L", (page_px_w, page_px_h), color=255)
    paste_x = (page_px_w - board_px_w) // 2
    paste_y = (page_px_h - board_px_h) // 2
    page.paste(Image.fromarray(board_img), (paste_x, paste_y))

    # Label below the board with board parameters and print reminder
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=28)
    label = (
        f"ChArUco {SQUARES_X}x{SQUARES_Y}  "
        f"square={SQUARE_LENGTH*1000:.1f} mm  marker={MARKER_LENGTH*1000:.1f} mm  |  "
        f"Print at ACTUAL SIZE / 100% — do NOT scale or fit-to-page"
    )
    text_y = paste_y + board_px_h + mm_to_px(4)   # 4 mm gap below board
    draw.text((paste_x, text_y), label, fill=0, font=font)

    page.save(output_pdf, "PDF", resolution=dpi)
    print(
        f"Board PDF saved to '{output_pdf}'  "
        f"({board_w_mm:.1f} x {board_h_mm:.1f} mm centred on {paper}, {dpi} dpi)\n"
        f"  -> Print at 100% actual size — disable 'fit to page'"
    )


def load_calibration(file: str = OUTPUT_FILE) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(file)
    return data["camera_matrix"], data["dist_coeffs"]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true",
                        help="capture fresh images from the USB webcam first")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args(argv)
    # Diagnostic: visually verify corner IDs match the physical board layout.
    # Check that ID=0 is top-left and IDs increase left→right, top→bottom.
    # If they look scrambled, swap SQUARES_X and SQUARES_Y.
    if args.diagnose:
        diagnose_detection(IMAGE_GLOB, num_images=3)

    # Step 0: generate (or regenerate) the printable board PDF.
    generate_board_pdf(os.path.join(_SCRIPT_DIR, "charuco_board.pdf"),
                       dpi=300, paper="A4")

    # Step 1 (optional): capture fresh calibration images from a live camera.
    if args.capture:
        collect_calibration_images(num_images=args.num_images,
                                   camera_index=args.camera_index)

    # Step 2: compute the camera matrix from the saved images.
    camera_matrix, dist_coeffs, rms = calibrate_from_images(IMAGE_GLOB)

    # Step 3: persist the results.
    save_calibration(camera_matrix, dist_coeffs)


if __name__ == "__main__":
    main()
