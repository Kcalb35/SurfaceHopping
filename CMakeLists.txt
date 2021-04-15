cmake_minimum_required(VERSION 3.17)
project(FSSH)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_subdirectory(src)

add_executable(${PROJECT_NAME} main.cpp include/gslExtra.h)

target_link_libraries(${PROJECT_NAME} FSSHLib)