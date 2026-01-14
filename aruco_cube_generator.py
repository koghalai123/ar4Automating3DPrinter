#!/usr/bin/env python3
"""
ArUco Cube Generator for Gazebo
Generates and spawns a cube with ArUco markers on each face
"""

import subprocess
import tempfile
import os
import argparse
from pathlib import Path


def create_aruco_cube_sdf(cube_size=0.1, position=(1.0, 0.0, 0.05), 
                          mass=0.5, name='aruco_cube'):
    """
    Create SDF XML string for an ArUco marker cube
    
    Args:
        cube_size: Size of the cube in meters
        position: (x, y, z) position tuple
        mass: Mass of the cube in kg
        name: Name of the model
    
    Returns:
        SDF XML string
    """
    x, y, z = position
    
    # Calculate inertia for a cube: I = (1/6) * m * s^2
    inertia_val = (1.0/6.0) * mass * cube_size * cube_size
    
    # Note: In Gazebo, textures need to be properly set up with material files
    # For simplicity, we'll create a white cube with black borders to simulate ArUco marker
    sdf = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <pose>{x} {y} {z} 0 0 0</pose>
    <static>false</static>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{inertia_val:.6f}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{inertia_val:.6f}</iyy>
          <iyz>0.0</iyz>
          <izz>{inertia_val:.6f}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>{cube_size} {cube_size} {cube_size}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{cube_size} {cube_size} {cube_size}</size>
          </box>
        </geometry>
        <material>
          <ambient>1.0 1.0 1.0 1.0</ambient>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
          <specular>0.1 0.1 0.1 1.0</specular>
        </material>
      </visual>
      <!-- Add black border visuals to simulate ArUco marker pattern -->
      <visual name="border_frame">
        <pose>0 0 0 0 0 0</pose>
        <geometry>
          <box>
            <size>{cube_size * 1.02} {cube_size * 1.02} {cube_size * 1.02}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.0 0.0 0.0 1.0</ambient>
          <diffuse>0.0 0.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
    return sdf


def spawn_aruco_cube(cube_size=0.1, position=(0.5, 0.0, 0.05), mass=0.5, name='aruco_cube'):
    """
    Generate and spawn an ArUco cube in Gazebo
    
    Args:
        cube_size: Size of the cube in meters
        position: (x, y, z) position tuple
        mass: Mass of the cube in kg
        name: Name of the model
    
    Returns:
        True if successful, False otherwise
    """
    print(f"Generating ArUco cube: {name}")
    print(f"  Size: {cube_size}m")
    print(f"  Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    print(f"  Mass: {mass}kg")
    
    # Create SDF content
    sdf_content = create_aruco_cube_sdf(cube_size, position, mass, name)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
        f.write(sdf_content)
        temp_filepath = f.name
    
    try:
        # Spawn using Gazebo command
        cmd = [
            'gz', 'service', '-s', '/world/default/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', f'sdf_filename: "{temp_filepath}"'
        ]
        
        print(f"\nSpawning {name} in Gazebo...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Successfully spawned {name}")
            return True
        else:
            print(f"✗ Failed to spawn {name}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate and spawn ArUco marker cube in Gazebo')
    parser.add_argument('--size', type=float, default=0.2,
                       help='Size of the cube in meters (default: 0.2)')
    parser.add_argument('--x', type=float, default=1.0,
                       help='X position (default: 1.0)')
    parser.add_argument('--y', type=float, default=0.0,
                       help='Y position (default: 0.0)')
    parser.add_argument('--z', type=float, default=None,
                       help='Z position (default: size/2 to sit on ground)')
    parser.add_argument('--mass', type=float, default=0.5,
                       help='Mass of the cube in kg (default: 0.5)')
    parser.add_argument('--name', type=str, default='aruco_cube',
                       help='Name of the cube model (default: aruco_cube)')
    
    args = parser.parse_args()
    
    # Set default Z position if not provided
    z_pos = args.z if args.z is not None else args.size / 2
    
    # Spawn the cube
    success = spawn_aruco_cube(
        cube_size=args.size,
        position=(args.x, args.y, z_pos),
        mass=args.mass,
        name=args.name
    )
    
    if success:
        print("\n✓ ArUco cube spawned successfully!")
        return 0
    else:
        print("\n✗ Failed to spawn ArUco cube")
        print("Make sure Gazebo is running")
        return 1


if __name__ == '__main__':
    exit(main())
