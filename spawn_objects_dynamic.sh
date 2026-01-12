#!/bin/bash

# Script to spawn objects using Python object generator
# This version is flexible and can be easily customized

echo "Generating and spawning objects into Gazebo..."

# Configuration
SCENE_TYPE="${1:-demo}"  # 'demo', 'random', or 'custom'
OBJECT_COUNT="${2:-20}"   # Number of objects for random scene
SEED="${3:-}"             # Optional seed for reproducibility

# Path to Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/object_generator.py"

# Generate Python code to create and spawn objects
python3 << EOF
import sys
sys.path.insert(0, '$SCRIPT_DIR')

from object_generator import generate_demo_scene, generate_random_scene, ObjectGenerator
import subprocess
import time

# Generate scene based on type
scene_type = '$SCENE_TYPE'
seed = None if '$SEED' == '' else int('$SEED')

if scene_type == 'demo':
    print("Generating demo scene...")
    gen = generate_demo_scene(seed=seed)
elif scene_type == 'random':
    print(f"Generating random scene with $OBJECT_COUNT objects...")
    gen = generate_random_scene(count=$OBJECT_COUNT, seed=seed)
else:
    print("Generating custom scene...")
    gen = ObjectGenerator(seed=seed)
    # Add custom object generation here
    gen.generate_random_objects(count=$OBJECT_COUNT)

print(f"Generated {len(gen.generated_objects)} objects")

# Save all objects to temp files
files = gen.save_all_to_temp()

# Spawn each object using gz service
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
    else:
        print(f"✗ Failed to spawn {name}")
        if result.stderr:
            print(f"  Error: {result.stderr[:100]}")
    
    time.sleep(0.1)  # Small delay between spawns

# Cleanup
gen.cleanup()
print("\nFinished spawning all objects!")
print(f"Scene type: {scene_type}")
if seed:
    print(f"Seed used: {seed} (use this to reproduce the same scene)")
EOF

echo "Done!"
