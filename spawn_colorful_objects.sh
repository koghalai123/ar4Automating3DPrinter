#!/bin/bash

# Script to spawn colorful objects into Gazebo using gz service
# This version saves SDF to temp files first

echo "Spawning colorful objects into Gazebo..."

TEMP_DIR="/tmp/gz_spawn_$$"
mkdir -p "$TEMP_DIR"

# Function to spawn from file
spawn_from_file() {
    local filepath=$1
    local name=$2
    
    gz service -s /world/default/create \
        --reqtype gz.msgs.EntityFactory \
        --reptype gz.msgs.Boolean \
        --timeout 1000 \
        --req "sdf_filename: \"$filepath\""
    
    echo "Spawned $name"
    sleep 0.1
}

# Create and spawn red box (front right)
cat > "$TEMP_DIR/red_box.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="red_box">
    <pose>0.4 0.35 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/red_box.sdf" "red_box"

# Create and spawn green box (front left)
cat > "$TEMP_DIR/green_box.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="green_box">
    <pose>0.4 -0.35 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.0 1.0</ambient>
          <diffuse>0.0 1.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/green_box.sdf" "green_box"

# Create and spawn blue box (left side)
cat > "$TEMP_DIR/blue_box.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="blue_box">
    <pose>0.0 -0.45 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>0.0 0.0 1.0 1.0</ambient>
          <diffuse>0.0 0.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/blue_box.sdf" "blue_box"

# Create and spawn yellow box (right side)
cat > "$TEMP_DIR/yellow_box.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="yellow_box">
    <pose>0.0 0.45 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>1.0 1.0 0.0 1.0</ambient>
          <diffuse>1.0 1.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/yellow_box.sdf" "yellow_box"

# Create and spawn pink box (back right)
cat > "$TEMP_DIR/pink_box.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="pink_box">
    <pose>-0.3 0.3 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>1.0 0.4 0.7 1.0</ambient>
          <diffuse>1.0 0.4 0.7 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/pink_box.sdf" "pink_box"

# Create and spawn lime box (back left)
cat > "$TEMP_DIR/lime_box.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="lime_box">
    <pose>-0.3 -0.3 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.08 0.08 0.08</size></box>
        </geometry>
        <material>
          <ambient>0.5 1.0 0.0 1.0</ambient>
          <diffuse>0.5 1.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/lime_box.sdf" "lime_box"

# Create and spawn magenta sphere (front far right)
cat > "$TEMP_DIR/magenta_sphere.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="magenta_sphere">
    <pose>0.55 0.4 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
        <material>
          <ambient>1.0 0.0 1.0 1.0</ambient>
          <diffuse>1.0 0.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/magenta_sphere.sdf" "magenta_sphere"

# Create and spawn cyan sphere (front far left)
cat > "$TEMP_DIR/cyan_sphere.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="cyan_sphere">
    <pose>0.55 -0.4 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
        <material>
          <ambient>0.0 1.0 1.0 1.0</ambient>
          <diffuse>0.0 1.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/cyan_sphere.sdf" "cyan_sphere"

# Create and spawn orange sphere (back far left)
cat > "$TEMP_DIR/orange_sphere.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="orange_sphere">
    <pose>-0.4 -0.4 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
        <material>
          <ambient>1.0 0.5 0.0 1.0</ambient>
          <diffuse>1.0 0.5 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/orange_sphere.sdf" "orange_sphere"

# Create and spawn purple sphere (back far right)
cat > "$TEMP_DIR/purple_sphere.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="purple_sphere">
    <pose>-0.4 0.4 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
        <material>
          <ambient>0.5 0.0 0.5 1.0</ambient>
          <diffuse>0.5 0.0 0.5 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/purple_sphere.sdf" "purple_sphere"

# Create and spawn teal sphere (front center)
cat > "$TEMP_DIR/teal_sphere.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="teal_sphere">
    <pose>0.5 0.0 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
        <material>
          <ambient>0.0 0.8 0.8 1.0</ambient>
          <diffuse>0.0 0.8 0.8 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/teal_sphere.sdf" "teal_sphere"

# Create and spawn brown sphere (back center)
cat > "$TEMP_DIR/brown_sphere.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="brown_sphere">
    <pose>-0.35 0.0 0.04 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <sphere><radius>0.04</radius></sphere>
        </geometry>
        <material>
          <ambient>0.6 0.3 0.1 1.0</ambient>
          <diffuse>0.6 0.3 0.1 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/brown_sphere.sdf" "brown_sphere"

# Create and spawn red cylinder (front right quadrant)
cat > "$TEMP_DIR/red_cylinder.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="red_cylinder">
    <pose>0.3 0.2 0.05 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1.0</ambient>
          <diffuse>1.0 0.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/red_cylinder.sdf" "red_cylinder"

# Create and spawn blue cylinder (front left quadrant)
cat > "$TEMP_DIR/blue_cylinder.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="blue_cylinder">
    <pose>0.3 -0.2 0.05 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
        <material>
          <ambient>0.0 0.0 1.0 1.0</ambient>
          <diffuse>0.0 0.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/blue_cylinder.sdf" "blue_cylinder"

# Create and spawn green cylinder (back left quadrant)
cat > "$TEMP_DIR/green_cylinder.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="green_cylinder">
    <pose>-0.2 -0.25 0.05 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.0 1.0</ambient>
          <diffuse>0.0 1.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/green_cylinder.sdf" "green_cylinder"

# Create and spawn yellow cylinder (back right quadrant)
cat > "$TEMP_DIR/yellow_cylinder.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="yellow_cylinder">
    <pose>-0.2 0.25 0.05 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
        <material>
          <ambient>1.0 1.0 0.0 1.0</ambient>
          <diffuse>1.0 1.0 0.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/yellow_cylinder.sdf" "yellow_cylinder"

# Create and spawn magenta cylinder (right side)
cat > "$TEMP_DIR/magenta_cylinder.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="magenta_cylinder">
    <pose>0.15 0.4 0.05 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.0 1.0 1.0</ambient>
          <diffuse>1.0 0.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/magenta_cylinder.sdf" "magenta_cylinder"

# Create and spawn cyan cylinder (left side)
cat > "$TEMP_DIR/cyan_cylinder.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="cyan_cylinder">
    <pose>0.15 -0.4 0.05 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.03</radius><length>0.1</length></cylinder>
        </geometry>
        <material>
          <ambient>0.0 1.0 1.0 1.0</ambient>
          <diffuse>0.0 1.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/cyan_cylinder.sdf" "cyan_cylinder"

# Create and spawn white tower (front center - tall landmark)
cat > "$TEMP_DIR/white_tower.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="white_tower">
    <pose>0.6 0.0 0.15 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.05</radius><length>0.3</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.05</radius><length>0.3</length></cylinder>
        </geometry>
        <material>
          <ambient>1.0 1.0 1.0 1.0</ambient>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/white_tower.sdf" "white_tower"

# Create and spawn gray tower (back center - tall landmark)
cat > "$TEMP_DIR/gray_tower.sdf" << 'EOF'
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="gray_tower">
    <pose>-0.5 0.0 0.15 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder><radius>0.05</radius><length>0.3</length></cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder><radius>0.05</radius><length>0.3</length></cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1.0</ambient>
          <diffuse>0.5 0.5 0.5 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF
spawn_from_file "$TEMP_DIR/gray_tower.sdf" "gray_tower"

# Cleanup
rm -rf "$TEMP_DIR"

echo "Finished spawning all objects!"
