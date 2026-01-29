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
    if(DEFINED ENV{CPLUS_INCLUDE_PATH})
        string(REPLACE ":" ";" INCLUDE_PATHS $ENV{CPLUS_INCLUDE_PATH})
        find_path(EIGEN3_INCLUDE_DIR
            NAMES Eigen/Core
            PATHS ${INCLUDE_PATHS}
            PATH_SUFFIXES eigen3 eigen
            NO_DEFAULT_PATH
        )
        if(EIGEN3_INCLUDE_DIR)
            set(Eigen3_FOUND TRUE)
            message(STATUS "Eigen3 found via CPLUS_INCLUDE_PATH: ${EIGEN3_INCLUDE_DIR}")
        endif()
    endif()
    
    if(NOT Eigen3_FOUND)
        message(FATAL_ERROR "Eigen3 not found. Please set EIGEN3_INCLUDE_DIR or CPLUS_INCLUDE_PATH")
    endif()
endif()

find_package(fmt 8.0 REQUIRED)
if(fmt_FOUND)
    message(STATUS "fmt found: ${fmt_DIR}")
else()
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(PC_FMT QUIET fmt)
        if(PC_FMT_FOUND)
            set(fmt_FOUND TRUE)
            set(fmt_INCLUDE_DIRS ${PC_FMT_INCLUDE_DIRS})
            set(fmt_LIBRARIES ${PC_FMT_LINK_LIBRARIES})
            message(STATUS "fmt found via pkg-config: ${fmt_INCLUDE_DIRS}")
        endif()
    endif()
    
    if(NOT fmt_FOUND AND DEFINED ENV{CPLUS_INCLUDE_PATH})
        string(REPLACE ":" ";" INCLUDE_PATHS $ENV{CPLUS_INCLUDE_PATH})
        find_path(fmt_INCLUDE_DIR
            NAMES fmt/format.h
            PATHS ${INCLUDE_PATHS}
            NO_DEFAULT_PATH
        )
        if(fmt_INCLUDE_DIR)
            set(fmt_FOUND TRUE)
            set(fmt_INCLUDE_DIRS ${fmt_INCLUDE_DIR})
            
            if(DEFINED ENV{LIBRARY_PATH})
                string(REPLACE ":" ";" LIBRARY_PATHS $ENV{LIBRARY_PATH})
                find_library(fmt_LIBRARY
                    NAMES fmt
                    PATHS ${LIBRARY_PATHS}
                    NO_DEFAULT_PATH
                )
                if(fmt_LIBRARY)
                    set(fmt_LIBRARIES ${fmt_LIBRARY})
                endif()
            endif()
            
            message(STATUS "fmt found via environment variables: ${fmt_INCLUDE_DIRS}")
        endif()
    endif()
    
    if(NOT fmt_FOUND)
        message(FATAL_ERROR "fmt not found. Please install fmt library or set FMT_DIR")
    endif()
endif()

set(SOURCES
    src/get_gro_connectivity.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${EIGEN3_INCLUDE_DIR}
    ${fmt_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    Eigen3::Eigen
    ${fmt_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "fmt_INCLUDE_DIRS: ${fmt_INCLUDE_DIRS}")
message(STATUS "fmt_LIBRARIES: ${fmt_LIBRARIES}")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "fmt: ${fmt_INCLUDE_DIRS} ${fmt_LIBRARIES}")
message(STATUS "Extra optimizations: ${ENABLE_EXTRA_OPTIMIZATIONS}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=================================")
