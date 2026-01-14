#!/usr/bin/env python3
"""
Test script to verify rotation mathematics are consistent
"""

import numpy as np
from tf_transformations import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_multiply,
    quaternion_inverse
)


def test_rotation_delta():
    """Test that rotation delta computation is correct"""
    
    # Start with a known orientation
    initial_rpy = np.array([0.1, 0.2, 0.3])  # radians
    initial_quat = quaternion_from_euler(initial_rpy[0], initial_rpy[1], initial_rpy[2])
    
    # Apply a known delta
    delta_rpy = np.array([0.05, 0.0, 0.0])  # Small roll rotation
    delta_quat = quaternion_from_euler(delta_rpy[0], delta_rpy[1], delta_rpy[2])
    
    # Compute final orientation
    final_quat = quaternion_multiply(delta_quat, initial_quat)
    
    # Recover the delta
    q_initial_inv = quaternion_inverse(initial_quat)
    recovered_delta_quat = quaternion_multiply(final_quat, q_initial_inv)
    recovered_delta_rpy = np.array(euler_from_quaternion(recovered_delta_quat))
    
    print("Rotation Delta Test")
    print("=" * 60)
    print(f"Initial RPY (deg): {np.rad2deg(initial_rpy)}")
    print(f"Applied delta RPY (deg): {np.rad2deg(delta_rpy)}")
    print(f"Recovered delta RPY (deg): {np.rad2deg(recovered_delta_rpy)}")
    print(f"Error (deg): {np.rad2deg(recovered_delta_rpy - delta_rpy)}")
    print(f"Error magnitude (deg): {np.rad2deg(np.linalg.norm(recovered_delta_rpy - delta_rpy)):.6f}")
    print()
    
    # Test with larger rotations
    print("Testing with larger rotations:")
    delta_rpy_large = np.array([0.3, 0.2, 0.1])
    delta_quat_large = quaternion_from_euler(delta_rpy_large[0], delta_rpy_large[1], delta_rpy_large[2])
    final_quat_large = quaternion_multiply(delta_quat_large, initial_quat)
    recovered_delta_quat_large = quaternion_multiply(final_quat_large, q_initial_inv)
    recovered_delta_rpy_large = np.array(euler_from_quaternion(recovered_delta_quat_large))
    
    print(f"Applied delta RPY (deg): {np.rad2deg(delta_rpy_large)}")
    print(f"Recovered delta RPY (deg): {np.rad2deg(recovered_delta_rpy_large)}")
    print(f"Error (deg): {np.rad2deg(recovered_delta_rpy_large - delta_rpy_large)}")
    print(f"Error magnitude (deg): {np.rad2deg(np.linalg.norm(recovered_delta_rpy_large - delta_rpy_large)):.6f}")
    print()
    
    # Test that naive subtraction fails
    print("Comparison with naive Euler subtraction:")
    final_rpy = np.array(euler_from_quaternion(final_quat))
    naive_delta = final_rpy - initial_rpy
    # Wrap to [-pi, pi]
    naive_delta = np.array([
        (naive_delta[i] + np.pi) % (2 * np.pi) - np.pi
        for i in range(3)
    ])
    print(f"Naive subtraction delta (deg): {np.rad2deg(naive_delta)}")
    print(f"Correct delta (deg): {np.rad2deg(recovered_delta_rpy)}")
    print(f"Naive method error (deg): {np.rad2deg(np.abs(naive_delta - recovered_delta_rpy))}")


if __name__ == '__main__':
    test_rotation_delta()
