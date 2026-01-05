sudo apt update
sudo apt install -y \
  ros-jazzy-moveit-servo \
  ros-jazzy-rtabmap-ros \
  ros-jazzy-realsense2-camera \
  ros-jazzy-gazebo-ros-pkgs \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-tf-transformations

sudo apt update
sudo apt install \
  ros-jazzy-controller-manager \
  ros-jazzy-hardware-interface \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-pluginlib \
  ros-jazzy-rclcpp \

# Gazebo ros2_control system plugin so controllers work in simulation
sudo apt install -y ros-jazzy-gz-ros2-control
  ros-jazzy-rclcpp-lifecycle

sudo apt install -y \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-launch-param-builder

sudo apt update
sudo apt install -y \
  ros-jazzy-gz-transport-vendor \
  ros-jazzy-gz-msgs-vendor \
  ros-jazzy-gz-common-vendor \
  ros-jazzy-gz-tools-vendor \
  ros-jazzy-gz-sim-vendor \
  ros-jazzy-moveit
sudo ldconfig

export LD_LIBRARY_PATH=/opt/ros/jazzy/opt/gz-transport-vendor/lib:/opt/ros/jazzy/opt/gz-msgs-vendor/lib:$LD_LIBRARY_PATH
# Install MoveIt meta-package to provide move_group
sudo apt install -y ros-jazzy-moveit

git clone https://github.com/ycheng517/ar4_ros_driver
git clone https://github.com/ycheng517/ar4_hand_eye_calibration

vcs import . --input ar4_hand_eye_calibration/hand_eye_calibration.repos
sudo apt update && rosdep install --from-paths . --ignore-src -y

source /opt/ros/jazzy/setup.bash
cd ~/ar4_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
sudo adduser "$USER" dialout



