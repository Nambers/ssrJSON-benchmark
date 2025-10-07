cmake_minimum_required(VERSION 3.18)
project(ssrjson_benchmark LANGUAGES C)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# ------------------------------------------------------------------------------
# Build Options for tests and docs

# ------------------------------------------------------------------------------
# Project Config
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(XcodeProperty)

# ------------------------------------------------------------------------------
# Search Python Package
find_package(Python3 COMPONENTS Development)

# check for python3
if((NOT Python3_INCLUDE_DIRS) OR(NOT Python3_LIBRARIES))
    message(FATAL_ERROR "Python3 not found")
endif()

message("Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message("Python3_LIBRARIES = ${Python3_LIBRARIES}")
message("Python3_LIBRARY_DIRS = ${Python3_LIBRARY_DIRS}")

# ------------------------------------------------------------------------------
# Build Type
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "No build type selected, default to: Release")
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

# ------------------------------------------------------------------------------
# Global settings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Libraries

# common definitions
add_library(commonBuild INTERFACE)
target_link_libraries(commonBuild INTERFACE ${Python3_LIBRARIES})
target_include_directories(commonBuild INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${Python3_INCLUDE_DIRS})
if(MSVC)
    target_compile_options(commonBuild INTERFACE /FAcs)
endif()

message("CMAKE_C_COMPILER_ID = ${CMAKE_C_COMPILER_ID}")
message("CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
message("CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")


# ------------------------------------------------------------------------------
# setting up ssrjson_benchmark

# set ssrjson_benchmark src files
set(SRC_FILES
    src/_ssrjson_benchmark.c
)

add_library(ssrjson_benchmark SHARED ${SRC_FILES})
target_link_libraries(ssrjson_benchmark PUBLIC ${Python3_LIBRARIES})
set_target_properties(ssrjson_benchmark PROPERTIES PREFIX "")
target_include_directories(ssrjson_benchmark PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> ${Python3_INCLUDE_DIRS})
set_target_properties(ssrjson_benchmark PROPERTIES OUTPUT_NAME "_ssrjson_benchmark")
# ------------------------------------------------------------------------------
if(XCODE)
    set(SSRJSON_BENCHMARK_FLAGS)
    list(APPEND SSRJSON_BENCHMARK_FLAGS "-Wall" "-Wextra" "-Werror" "-pedantic" "-pedantic-errors" "-Wno-psabi")

    set_default_xcode_property(commonBuild)
    set_xcode_deployment_version(commonBuild "10.13" "12.0" "12.0" "4.0")

    set_xcode_property(commonBuild GCC_C_LANGUAGE_STANDARD "c17")
    set_xcode_property(commonBuild CLANG_C_LANGUAGE_STANDARD "c17")

    # set_xcode_property(commonBuild CLANG_CXX_LANGUAGE_STANDARD "c++98")
    set_xcode_property(commonBuild OTHER_CFLAGS[variant=Debug] ${SSRJSON_BENCHMARK_FLAGS})
    set_xcode_property(commonBuild OTHER_CFLAGS[variant=MinSizeRel] ${SSRJSON_BENCHMARK_FLAGS})
    set_xcode_property(commonBuild OTHER_CFLAGS[variant=RelWithDebInfo] ${SSRJSON_BENCHMARK_FLAGS})
    set_xcode_property(commonBuild OTHER_CFLAGS[variant=Release] ${SSRJSON_BENCHMARK_FLAGS})

elseif(MSVC)
    set(SSRJSON_BENCHMARK_FLAGS)
    list(APPEND SSRJSON_BENCHMARK_FLAGS "/utf-8")

    target_compile_options(commonBuild INTERFACE ${SSRJSON_BENCHMARK_FLAGS})

elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
    set(SSRJSON_BENCHMARK_FLAGS)

    if(CMAKE_C_COMPILER_ID MATCHES Clang)
        # https://github.com/llvm/llvm-project/issues/63963
        list(APPEND SSRJSON_BENCHMARK_FLAGS "-Wno-constant-logical-operand")
    endif()

    if(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        message("Release mode with debug info, enabling maximal optimization and debug symbols")
        target_compile_options(commonBuild INTERFACE -Werror -Wno-psabi -pedantic -pedantic-errors)
    elseif(CMAKE_BUILD_TYPE MATCHES ".*Rel.*")
        message("Release mode, enabling maximal optimization")
        target_compile_options(commonBuild INTERFACE -Werror -Wno-psabi -pedantic -pedantic-errors)
    else()
        target_compile_options(commonBuild INTERFACE -O0 -Wno-psabi)

        message("Debug mode, enabling debug symbols")
    endif()

    target_compile_options(commonBuild INTERFACE ${SSRJSON_BENCHMARK_FLAGS})
endif()

# ------------------------------------------------------------------------------
# Install
if(APPLE)
    set_target_properties(ssrjson_benchmark PROPERTIES SUFFIX ".so")
endif(APPLE)

install(TARGETS ssrjson_benchmark LIBRARY DESTINATION .)
