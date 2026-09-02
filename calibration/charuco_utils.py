"""OpenCV-version-independent ChArUco helpers used by calibration tools."""

from __future__ import annotations

import cv2


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
