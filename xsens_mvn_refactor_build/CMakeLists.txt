cmake_minimum_required(VERSION 3.14)
project(xsens_mvn_refactor_build LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread chrono)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
find_package(GTest REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enable_testing()

set(PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto)
set(PROTO_FILES ${PROTO_DIR}/link_states.proto)

add_subdirectory(src)
add_subdirectory(tests)
