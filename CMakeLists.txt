cmake_minimum_required(VERSION 2.8)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message("Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()

project(FSSH)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

find_library(GSL libgsl)
# find_package(GSL REQUIRED COMPONENTS gsl gslcblas)

# if (GSL_FOUND)
#     message("GSL FOUND")
# else ()
#     message(FATAL_ERROR "Cannot found GSL")
# endif ()

set(SOURCES
        main.cpp
        lib/easyloggingpp/src/easylogging++.cc)


add_library(FSSHLib STATIC
        src/FSSHMath.cpp
        src/gslExtra.cpp
        )
target_include_directories(FSSHLib
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        )

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${PROJECT_SOURCE_DIR}/lib/easyloggingpp/src)

target_link_libraries(${PROJECT_NAME}
        PRIVATE
        FSSHLib gsl gslcblas -static)