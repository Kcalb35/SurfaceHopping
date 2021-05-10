cmake_minimum_required(VERSION 2.8)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message("Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()

project(SH)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()

# target log lib
add_library(log ${PROJECT_SOURCE_DIR}/lib/easyloggingpp/src/easylogging++.cc)
target_include_directories(log PUBLIC ${PROJECT_SOURCE_DIR}/lib/easyloggingpp/src)

# target SH lib
add_library(SHLib STATIC
        src/FSSHMath.cpp
        src/gslExtra.cpp src/ModelBase.cpp)
target_include_directories(SHLib
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(SHLib PUBLIC gsl.a gslcblas.a log)

# target MF lib
add_library(MFLib STATIC
        src/MF.cpp
        )
target_include_directories(MFLib PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(MFLib PUBLIC gsl.a gslcblas.a log)

# target executable
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        SHLib pthread)

add_executable(surface ${PROJECT_SOURCE_DIR}/tests/energySurface.cpp)
target_link_libraries(surface PUBLIC SHLib)

add_executable(diagonal ${PROJECT_SOURCE_DIR}/tests/TestDiagonal.cpp)
target_link_libraries(diagonal SHLib)

add_executable(MF ${PROJECT_SOURCE_DIR}/MeanField.cpp)
target_link_libraries(MF MFLib)