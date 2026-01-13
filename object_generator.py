#!/usr/bin/env python3
"""
Object Generator for Gazebo Simulation
Generates random colored objects with various shapes and positions
"""

import random
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple
import math


@dataclass
class ObjectConfig:
    """Configuration for a spawned object"""
    name: str
    shape: str  # 'box', 'sphere', 'cylinder'
    position: Tuple[float, float, float]  # x, y, z
    size: Tuple[float, ...]  # depends on shape
    color: Tuple[float, float, float, float]  # r, g, b, a
    mass: float = 1.0


class ObjectGenerator:
    """Generate random objects for Gazebo spawning"""
    
    def __init__(self, seed=None):
        """Initialize generator with optional seed for reproducibility"""
        if seed is not None:
            random.seed(seed)
        
        self.temp_dir = None
        self.generated_objects = []
        
    def generate_color(self, preset=None):
        """Generate a random color or use preset"""
        if preset:
            colors = {
                'red': (1.0, 0.0, 0.0, 1.0),
                'green': (0.0, 1.0, 0.0, 1.0),
                'blue': (0.0, 0.0, 1.0, 1.0),
                'yellow': (1.0, 1.0, 0.0, 1.0),
                'magenta': (1.0, 0.0, 1.0, 1.0),
                'cyan': (0.0, 1.0, 1.0, 1.0),
                'orange': (1.0, 0.5, 0.0, 1.0),
                'purple': (0.5, 0.0, 0.5, 1.0),
                'white': (1.0, 1.0, 1.0, 1.0),
                'gray': (0.5, 0.5, 0.5, 1.0),
                'pink': (1.0, 0.4, 0.7, 1.0),
                'lime': (0.5, 1.0, 0.0, 1.0),
                'teal': (0.0, 0.8, 0.8, 1.0),
                'brown': (0.6, 0.3, 0.1, 1.0),
            }
            return colors.get(preset, (1.0, 1.0, 1.0, 1.0))
        else:
            # Generate random bright color
            r = random.uniform(0.0, 1.0)
            g = random.uniform(0.0, 1.0)
            b = random.uniform(0.0, 1.0)
            return (r, g, b, 1.0)
    
    def generate_position_circular(self, radius_min=0.2, radius_max=1.5, height=0.04):
        """Generate a position in a circular pattern around origin"""
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(radius_min, radius_max)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return (x, y, height)
    
    def generate_position_grid(self, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), height=0.04):
        """Generate a position in a rectangular grid"""
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        return (x, y, height)
    
    def generate_box(self, name=None, position=None, size_range=(0.05, 0.12), color=None):
        """Generate a box object"""
        if name is None:
            name = f"box_{random.randint(1000, 9999)}"
        if position is None:
            position = self.generate_position_circular()
        if color is None:
            color = self.generate_color()
        
        size = random.uniform(size_range[0], size_range[1])
        obj = ObjectConfig(
            name=name,
            shape='box',
            position=position,
            size=(size, size, size),
            color=color
        )
        self.generated_objects.append(obj)
        return obj
    
    def generate_sphere(self, name=None, position=None, radius_range=(0.03, 0.06), color=None):
        """Generate a sphere object"""
        if name is None:
            name = f"sphere_{random.randint(1000, 9999)}"
        if position is None:
            position = self.generate_position_circular()
        if color is None:
            color = self.generate_color()
        
        radius = random.uniform(radius_range[0], radius_range[1])
        obj = ObjectConfig(
            name=name,
            shape='sphere',
            position=position,
            size=(radius,),
            color=color
        )
        self.generated_objects.append(obj)
        return obj
    
    def generate_cylinder(self, name=None, position=None, radius_range=(0.02, 0.05), 
                         length_range=(0.08, 0.15), color=None):
        """Generate a cylinder object"""
        if name is None:
            name = f"cylinder_{random.randint(1000, 9999)}"
        if position is None:
            position = self.generate_position_circular()
        if color is None:
            color = self.generate_color()
        
        radius = random.uniform(radius_range[0], radius_range[1])
        length = random.uniform(length_range[0], length_range[1])
        
        # Adjust z position for cylinder height
        x, y, z = position
        z = length / 2  # Center cylinder vertically
        
        obj = ObjectConfig(
            name=name,
            shape='cylinder',
            position=(x, y, z),
            size=(radius, length),
            color=color
        )
        self.generated_objects.append(obj)
        return obj
    
    def generate_random_objects(self, count=40, shape_distribution=None):
        """
        Generate a collection of random objects
        
        Args:
            count: Number of objects to generate
            shape_distribution: Dict with 'box', 'sphere', 'cylinder' percentages
                               Default: {'box': 0.4, 'sphere': 0.3, 'cylinder': 0.3}
        """
        if shape_distribution is None:
            shape_distribution = {'box': 0.4, 'sphere': 0.3, 'cylinder': 0.3}
        
        for i in range(count):
            rand = random.random()
            
            if rand < shape_distribution['box']:
                self.generate_box()
            elif rand < shape_distribution['box'] + shape_distribution['sphere']:
                self.generate_sphere()
            else:
                self.generate_cylinder()
    
    def create_sdf(self, obj: ObjectConfig) -> str:
        """Create SDF XML string for an object"""
        r, g, b, a = obj.color
        x, y, z = obj.position
        
        # Create geometry based on shape
        if obj.shape == 'box':
            sx, sy, sz = obj.size
            geometry = f"<box><size>{sx} {sy} {sz}</size></box>"
        elif obj.shape == 'sphere':
            radius = obj.size[0]
            geometry = f"<sphere><radius>{radius}</radius></sphere>"
        elif obj.shape == 'cylinder':
            radius, length = obj.size
            geometry = f"<cylinder><radius>{radius}</radius><length>{length}</length></cylinder>"
        else:
            raise ValueError(f"Unknown shape: {obj.shape}")
        
        sdf = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{obj.name}">
    <pose>{x} {y} {z} 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>{obj.mass}</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          {geometry}
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          {geometry}
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        return sdf
    
    def save_to_temp(self, obj: ObjectConfig) -> str:
        """Save object SDF to a temporary file and return path"""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix='gz_objects_')
        
        filepath = os.path.join(self.temp_dir, f"{obj.name}.sdf")
        with open(filepath, 'w') as f:
            f.write(self.create_sdf(obj))
        
        return filepath
    
    def save_all_to_temp(self) -> List[Tuple[str, str]]:
        """Save all generated objects to temp files. Returns list of (name, filepath) tuples"""
        return [(obj.name, self.save_to_temp(obj)) for obj in self.generated_objects]
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def clear(self):
        """Clear all generated objects"""
        self.generated_objects = []


# Preset configurations for common scenarios
def generate_demo_scene(seed=None):
    """Generate a demo scene with diverse objects"""
    gen = ObjectGenerator(seed=seed)
    
    # Add some preset colored boxes in cardinal directions
    gen.generate_box("red_box", (0.4, 0.35, 0.04), color=gen.generate_color('red'))
    gen.generate_box("green_box", (0.4, -0.35, 0.04), color=gen.generate_color('green'))
    gen.generate_box("blue_box", (0.0, -0.45, 0.04), color=gen.generate_color('blue'))
    gen.generate_box("yellow_box", (0.0, 0.45, 0.04), color=gen.generate_color('yellow'))
    
    # Add some spheres
    gen.generate_sphere("magenta_sphere", (0.55, 0.4, 0.04), color=gen.generate_color('magenta'))
    gen.generate_sphere("cyan_sphere", (0.55, -0.4, 0.04), color=gen.generate_color('cyan'))
    
    # Add some cylinders
    gen.generate_cylinder("red_cylinder", (0.3, 0.2, 0.05), color=gen.generate_color('red'))
    gen.generate_cylinder("blue_cylinder", (0.3, -0.2, 0.05), color=gen.generate_color('blue'))
    
    # Add tall landmarks
    gen.generate_cylinder("white_tower", (0.6, 0.0, 0.15), 
                         radius_range=(0.05, 0.05), length_range=(0.3, 0.3),
                         color=gen.generate_color('white'))
    gen.generate_cylinder("gray_tower", (-0.5, 0.0, 0.15), 
                         radius_range=(0.05, 0.05), length_range=(0.3, 0.3),
                         color=gen.generate_color('gray'))
    
    # Fill in with random objects
    gen.generate_random_objects(count=10)
    
    return gen


def generate_random_scene(count=30, seed=None):
    """Generate a completely random scene"""
    gen = ObjectGenerator(seed=seed)
    gen.generate_random_objects(count=count)
    return gen


if __name__ == '__main__':
    import sys
    import subprocess
    import time
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate and spawn objects in Gazebo')
    parser.add_argument('--scene', choices=['demo', 'random'], default='random',
                       help='Scene type: demo (preset + random) or random (all random)')
    parser.add_argument('--count', type=int, default=40,
                       help='Number of objects to generate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: generate and print objects without spawning')
    parser.add_argument('--no-spawn', action='store_true',
                       help='Generate objects but do not spawn them')
    
    args = parser.parse_args()
    
    # Generate scene
    print(f"Generating {args.scene} scene with {args.count} objects...")
    if args.seed:
        print(f"Using seed: {args.seed}")
    
    if args.scene == 'demo':
        gen = generate_demo_scene(seed=args.seed)
    else:
        gen = generate_random_scene(count=args.count, seed=args.seed)
    
    print(f"Generated {len(gen.generated_objects)} objects:")
    for obj in gen.generated_objects:
        print(f"  - {obj.name}: {obj.shape} at ({obj.position[0]:.2f}, {obj.position[1]:.2f}, {obj.position[2]:.2f})")
    
    # Test mode - just show what would be generated
    if args.test:
        print("\nTest mode - not spawning objects")
        files = gen.save_all_to_temp()
        print(f"Would save to: {gen.temp_dir}")
        gen.cleanup()
        sys.exit(0)
    
    # No spawn mode - generate but don't spawn
    if args.no_spawn:
        print("\nNo-spawn mode - objects generated but not spawned")
        gen.cleanup()
        sys.exit(0)
    
    # Spawn objects in Gazebo
    print("\nSpawning objects into Gazebo...")
    files = gen.save_all_to_temp()
    
    spawned = 0
    failed = 0
    
    for name, filepath in files:
        cmd = [
            'gz', 'service', '-s', '/world/default/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', f'sdf_filename: "{filepath}"'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Spawned {name}")
            spawned += 1
        else:
            print(f"✗ Failed to spawn {name}")
            failed += 1
            if result.stderr and '--verbose' in sys.argv:
                print(f"  Error: {result.stderr[:100]}")
        
        time.sleep(0.1)  # Small delay between spawns
    
    # Cleanup
    gen.cleanup()
    
    print(f"\nFinished!")
    print(f"Successfully spawned: {spawned}/{len(files)}")
    if failed > 0:
        print(f"Failed: {failed}")
    if args.seed:
        print(f"Seed used: {args.seed} (use '--seed {args.seed}' to reproduce)")
