file(GLOB_RECURSE XSENS_SDK_SOURCES xsens_mvn_sdk/*.cpp)
add_library(xsens_sdk ${XSENS_SDK_SOURCES})
target_include_directories(xsens_sdk PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(xsens_sdk PUBLIC
  Boost::system
  Boost::thread
  Boost::chrono
  Eigen3::Eigen
)

add_library(xsens_core
  xsens_core/HumanDataHandler.cpp
  xsens_core/Socket.cpp
  xsens_core/XSensClient.cpp
)
target_include_directories(xsens_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(xsens_core PUBLIC
  xsens_sdk
  Boost::system
  Boost::thread
  Boost::chrono
  Eigen3::Eigen
)

add_library(xsens_transport
  xsens_transport/LinkStateProtoSerializer.cpp
  xsens_transport/LinkStatePublisher.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/link_states.pb.cc
)
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/link_states.pb.cc
    ${CMAKE_CURRENT_BINARY_DIR}/link_states.pb.h
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
  ARGS
    --proto_path=${PROTO_DIR}
    --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
    ${PROTO_FILES}
  DEPENDS ${PROTO_FILES}
  COMMENT "Generating C++ protobuf sources for link_states.proto"
  VERBATIM
)
add_custom_target(xsens_transport_proto DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/link_states.pb.cc
  ${CMAKE_CURRENT_BINARY_DIR}/link_states.pb.h
)
add_dependencies(xsens_transport xsens_transport_proto)
target_include_directories(xsens_transport PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Protobuf_INCLUDE_DIRS}
  ${ZMQ_INCLUDE_DIRS}
)
target_link_libraries(xsens_transport PUBLIC
  xsens_core
  protobuf::libprotobuf
  ${ZMQ_LIBRARIES}
  Eigen3::Eigen
)

add_library(xsens_app_support
  apps/LinkStateFrameFilter.cpp
  apps/XsensLinkStateRunner.cpp
  apps/TestFrameFactory.cpp
)
target_include_directories(xsens_app_support PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(xsens_app_support PUBLIC
  xsens_transport
  Eigen3::Eigen
)

add_executable(link_states_test_publisher
  apps/link_states_test_publisher.cpp
)
target_include_directories(link_states_test_publisher PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(link_states_test_publisher PRIVATE
  xsens_app_support
)

add_executable(link_states_zmq_subscriber
  apps/link_states_zmq_subscriber.cpp
)
target_include_directories(link_states_zmq_subscriber PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(link_states_zmq_subscriber PRIVATE
  xsens_transport
)

add_executable(xsens_link_states_publisher
  apps/xsens_link_states_publisher.cpp
)
target_include_directories(xsens_link_states_publisher PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(xsens_link_states_publisher PRIVATE
  xsens_app_support
)
