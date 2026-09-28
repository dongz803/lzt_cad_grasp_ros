cmake_minimum_required(VERSION 3.8)
project(lzt_system_tests)

find_package(ament_cmake REQUIRED)
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(package_boundaries test/test_package_boundaries.py)
  ament_add_pytest_test(
    repository_regression
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests
    TIMEOUT 120
  )
endif()
ament_package()
