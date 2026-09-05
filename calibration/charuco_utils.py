"""OpenCV-version-independent ChArUco helpers used by calibration tools."""

from __future__ import annotations

import cv2
import numpy as np


def detector_parameters():
    factory = getattr(cv2.aruco, "DetectorParameters_create", None)
    return factory() if factory is not None else cv2.aruco.DetectorParameters()


def create_board(squares_x, squares_y, square_length, marker_length,
                 dictionary):
    factory = getattr(cv2.aruco, "CharucoBoard_create", None)
    if factory is not None:
        return factory(squares_x, squares_y, square_length, marker_length,
                       dictionary)
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_length, marker_length, dictionary)


def detect(gray, board, dictionary, parameters=None):
    """Return marker corners/ids and interpolated ChArUco corners/ids."""
    parameters = parameters or detector_parameters()
    legacy = getattr(cv2.aruco, "detectMarkers", None)
    if legacy is not None:
        marker_corners, marker_ids, rejected = legacy(
            gray, dictionary, parameters=parameters)
    else:
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    charuco_corners = charuco_ids = None
    if marker_ids is not None and len(marker_ids) >= 2:
        interpolate = getattr(cv2.aruco, "interpolateCornersCharuco", None)
        if interpolate is not None:
            _, charuco_corners, charuco_ids = interpolate(
                marker_corners, marker_ids, gray, board)
        else:
            charuco = cv2.aruco.CharucoDetector(board)
            (charuco_corners, charuco_ids,
             marker_corners, marker_ids) = charuco.detectBoard(gray)
    return (marker_corners, marker_ids, rejected,
            charuco_corners, charuco_ids)


def estimate_pose(charuco_corners, charuco_ids, board, camera_matrix,
                  distortion):
    """Estimate camera->board pose across OpenCV 4.5 and 4.11+ APIs."""
    legacy = getattr(cv2.aruco, "estimatePoseCharucoBoard", None)
    if legacy is not None:
        return legacy(charuco_corners, charuco_ids, board, camera_matrix,
                      distortion, None, None)
    object_points, image_points = board.matchImagePoints(
        charuco_corners, charuco_ids)
    if object_points is None or len(object_points) < 4:
        return False, None, None
    return cv2.solvePnP(object_points, image_points, camera_matrix,
                        distortion)


def calibrate_camera(charuco_corners, charuco_ids, board, image_size,
                     flags=0):
    """Calibrate a camera from ChArUco observations on OpenCV 4.5--4.11.

    ``cv2.aruco.calibrateCameraCharuco`` was removed from newer OpenCV
    contrib wheels.  The replacement is mathematically equivalent: map each
    detected ChArUco ID to its known chessboard 3D point, then use the normal
    OpenCV calibrator over those object/image point pairs.
    """
    legacy = getattr(cv2.aruco, "calibrateCameraCharuco", None)
    if legacy is not None:
        return legacy(charuco_corners, charuco_ids, board, image_size,
                      None, None, flags=flags)

    chessboard_points = np.asarray(
        board.getChessboardCorners(), dtype=np.float32)
    object_points = []
    image_points = []
    for corners, ids in zip(charuco_corners, charuco_ids):
        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        if len(ids) != len(corners) or np.any(ids < 0) or \
                np.any(ids >= len(chessboard_points)):
            raise ValueError("invalid ChArUco corner IDs for this board")
        object_points.append(chessboard_points[ids].reshape(-1, 1, 3))
        image_points.append(corners)
    return cv2.calibrateCamera(object_points, image_points, image_size,
                               None, None, flags=flags)
