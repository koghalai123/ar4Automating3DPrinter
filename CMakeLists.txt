cmake_minimum_required(VERSION 3.10)
project(ar4Automating3DPrinter)

find_package(ament_cmake REQUIRED)

# Install local models to the package share directory
install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

# Environment hook to add models path to Gazebo Sim resource paths
ament_environment_hooks(env-hooks/gz_resource_path.dsv)

ament_package()
