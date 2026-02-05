cmake_minimum_required(VERSION 3.18)

project(get_gro_connectivity
    VERSION 1.0
    DESCRIPTION "Molecular connectivity analysis tool"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /W3 /DEBUG")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    
    option(ENABLE_EXTRA_OPTIMIZATIONS "Enable extra compiler optimizations" OFF)
    if(ENABLE_EXTRA_OPTIMIZATIONS)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
    endif()
    
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.4 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found. Please set EIGEN3_INCLUDE_DIR or CPLUS_INCLUDE_PATH")
endif()

find_package(fmt 8.0 REQUIRED CONFIG)
if(fmt_FOUND)
    message(STATUS "fmt found: ${fmt_DIR}")
else()
    message(FATAL_ERROR "fmt not found. Please set fmt_DIR or CPLUS_INCLUDE_PATH and LIBRARY_PATH")
endif()

set(SOURCES
    src/gro_file_parser.cpp
    src/distance_calculator.cpp
    src/connectivity_analysis.cpp
    src/molecule_utils.cpp
    src/selection_utils.cpp
    src/molecule_generator.cpp
    src/utils.cpp
    src/output_writer.cpp
    src/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${EIGEN3_INCLUDE_DIR}
    ${fmt_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    Eigen3::Eigen
    fmt::fmt
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "fmt: ${fmt_DIR}")
message(STATUS "Extra optimizations: ${ENABLE_EXTRA_OPTIMIZATIONS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=================================")
