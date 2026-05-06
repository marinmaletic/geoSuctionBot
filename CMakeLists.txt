cmake_minimum_required(VERSION 3.8)
project(ur_suctionbot)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SuctionScore.msg"
  "msg/GraspCandidate.msg"
  "srv/ComputeSuction.srv"
  "srv/ExecuteGrasp.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Install launch, config and Python nodes
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python nodes
install(PROGRAMS
  ur_suctionbot/camera_node.py
  ur_suctionbot/suction_node.py
  ur_suctionbot/gui_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  ur_suctionbot/__init__.py
  ur_suctionbot/knn.py
  ur_suctionbot/sobel.py
  ur_suctionbot/ransac.py
  DESTINATION ${PYTHON_INSTALL_DIR}/ur_suctionbot
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
