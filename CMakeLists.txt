cmake_minimum_required(VERSION 2.8)

project(boat-detection)

find_package(OpenCV REQUIRED)

message(${CMAKE_CURRENT_SOURCE_DIR})
set(SOURCE_FILES
        src/main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})
