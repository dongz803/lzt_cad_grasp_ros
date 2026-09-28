cmake_minimum_required(VERSION 3.8)
project(lzt_description)

find_package(ament_cmake REQUIRED)
install(DIRECTORY assets scenes config DESTINATION share/${PROJECT_NAME})
ament_package()

