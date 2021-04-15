project(FSSHLib)

find_package(GSL REQUIRED)

set(SOURCES
        FSSHMath.cpp gslExtra.cpp)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

target_link_libraries(${PROJECT_NAME} GSL::gsl)