cmake_minimum_required(VERSION 3.10)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message("Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()

project(SH)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

add_subdirectory(${PROJECT_SOURCE_DIR}/lib/toml11)

# target log lib
add_library(log ${PROJECT_SOURCE_DIR}/lib/easyloggingpp/src/easylogging++.cc)
target_include_directories(log PUBLIC ${PROJECT_SOURCE_DIR}/lib/easyloggingpp/src)

#target model base lib
add_library(ModelBase STATIC
        src/ModelBase.cpp)
target_include_directories(ModelBase PUBLIC include)
target_link_libraries(ModelBase PUBLIC gsl.a)

# target SH lib
add_library(SHLib STATIC
        src/FSSHMath.cpp)
target_include_directories(SHLib
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/lib/QUtil/include)
target_link_libraries(SHLib PUBLIC ModelBase gsl.a gslcblas.a log fmt.a)

# target MF lib
add_library(MFLib STATIC
        src/MFMath.cpp)
target_include_directories(MFLib PUBLIC ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/lib/QUtil/include)
target_link_libraries(MFLib PUBLIC gsl.a gslcblas.a log fmt.a)


# target executable
add_executable(${PROJECT_NAME} SurfaceHopping.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/lib/toml11)
target_link_libraries(${PROJECT_NAME} PUBLIC SHLib)

add_executable(surface ${PROJECT_SOURCE_DIR}/tests/energySurface.cpp)
target_link_libraries(surface PUBLIC SHLib)

add_executable(diagonal ${PROJECT_SOURCE_DIR}/tests/TestDiagonal.cpp)
target_link_libraries(diagonal SHLib)

add_executable(MF ${PROJECT_SOURCE_DIR}/MeanField.cpp)
target_include_directories(MF PRIVATE ${PROJECT_SOURCE_DIR}/lib/toml11)
target_link_libraries(MF MFLib SHLib toml11::toml11)