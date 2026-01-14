#!/usr/bin/env bash

ros2 launch annin_ar4_gazebo gazebo.launch.py &
sleep 6

# Camera is now included directly in the world file (empty.world)
# No need to spawn it separately

sleep 2

ros2 launch annin_ar4_moveit_config moveit.launch.py use_sim_time:=true include_gripper:=True




<!-- Gazebo Sim RGB-D camera sensor attached to ee_camera_link -->
  <!--
  <gazebo reference="$(arg tf_prefix)ee_camera_link">
    <sensor type="rgbd_camera" name="ee_rgbd_sensor">
      <pose>0 0 0 1.5708 -1.5708 0</pose>
      <camera>
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>320</width>
          <height>240</height>
          <!-- Force RGB format to avoid stride/step mismatches -->
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </camera>
      <!-- Explicit depth camera configuration to ensure 32-bit float depths -->
      <depth_camera>
        <image>
          <width>320</width>
          <height>240</height>
          <!-- Use FLOAT32 so ROS sees encoding 32FC1 without padding issues -->
          <format>FLOAT32</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
      </depth_camera>
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
      <topic>rgbd_camera</topic>
      <enable_metrics>false</enable_metrics>
    </sensor>
  </gazebo>

  -->