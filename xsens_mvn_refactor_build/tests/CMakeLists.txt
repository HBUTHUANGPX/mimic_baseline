add_executable(test_parsermanager_cache test_parsermanager_cache.cpp)
target_include_directories(test_parsermanager_cache PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(
  test_parsermanager_cache
  PRIVATE
    xsens_sdk
    GTest::gtest_main
)
add_test(NAME test_parsermanager_cache COMMAND test_parsermanager_cache)

add_executable(test_human_data_handler test_human_data_handler.cpp)
target_include_directories(test_human_data_handler PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(
  test_human_data_handler
  PRIVATE
    xsens_core
    GTest::gtest_main
)
add_test(NAME test_human_data_handler COMMAND test_human_data_handler)

add_executable(test_link_state_proto_serializer test_link_state_proto_serializer.cpp)
target_include_directories(test_link_state_proto_serializer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../src
)
target_link_libraries(
  test_link_state_proto_serializer
  PRIVATE
    xsens_transport
    GTest::gtest_main
)
add_test(NAME test_link_state_proto_serializer COMMAND test_link_state_proto_serializer)

add_executable(test_zmq_loopback test_zmq_loopback.cpp)
target_include_directories(test_zmq_loopback PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../src
)
target_link_libraries(
  test_zmq_loopback
  PRIVATE
    xsens_app_support
    xsens_transport
    GTest::gtest_main
)
add_test(NAME test_zmq_loopback COMMAND test_zmq_loopback)

add_executable(test_link_states_test_publisher_process test_link_states_test_publisher_process.cpp)
target_include_directories(test_link_states_test_publisher_process PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../src
)
target_compile_definitions(test_link_states_test_publisher_process PRIVATE
  LINK_STATES_TEST_PUBLISHER_PATH="$<TARGET_FILE:link_states_test_publisher>"
)
target_link_libraries(
  test_link_states_test_publisher_process
  PRIVATE
    xsens_transport
    GTest::gtest_main
)
add_test(
  NAME test_link_states_test_publisher_process
  COMMAND test_link_states_test_publisher_process
)

add_executable(test_link_states_zmq_subscriber_process test_link_states_zmq_subscriber_process.cpp)
target_include_directories(test_link_states_zmq_subscriber_process PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../src
)
target_compile_definitions(test_link_states_zmq_subscriber_process PRIVATE
  LINK_STATES_ZMQ_SUBSCRIBER_PATH="$<TARGET_FILE:link_states_zmq_subscriber>"
)
target_link_libraries(
  test_link_states_zmq_subscriber_process
  PRIVATE
    xsens_app_support
    xsens_transport
    GTest::gtest_main
)
add_test(
  NAME test_link_states_zmq_subscriber_process
  COMMAND test_link_states_zmq_subscriber_process
)

add_executable(test_xsens_link_state_runner test_xsens_link_state_runner.cpp)
target_include_directories(test_xsens_link_state_runner PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
  ${CMAKE_CURRENT_BINARY_DIR}/../src
)
target_link_libraries(
  test_xsens_link_state_runner
  PRIVATE
    xsens_app_support
    xsens_transport
    GTest::gtest_main
)
add_test(NAME test_xsens_link_state_runner COMMAND test_xsens_link_state_runner)
