cmake_minimum_required(VERSION 3.15)
project(dr_mmse_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# --- Eigen (header-only). Use system package if present, fetch 3.4 otherwise.
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found on system; fetching 3.4.0")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(eigen)
endif()

# --- pybind11. Recommended: `pip install pybind11` then -DCMAKE_PREFIX_PATH=$(python -m pybind11 --cmakedir)
find_package(pybind11 REQUIRED)

set(DR_MMSE_SOURCES
    cpp/src/kalman_utils.cpp
    cpp/src/fw_oracle.cpp
    cpp/src/dr_mmse_tac.cpp
    cpp/src/bindings.cpp
)

pybind11_add_module(dr_mmse_cpp ${DR_MMSE_SOURCES})

target_include_directories(dr_mmse_cpp PRIVATE cpp/include)
target_link_libraries(dr_mmse_cpp PRIVATE Eigen3::Eigen)

# Optimization flags. Pick the right native-tuning syntax per architecture:
# aarch64 (Cortex-A class, Apple Silicon) needs -mcpu=native; x86_64 needs
# -march=native. Both enable NEON/SVE/AVX as available.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(dr_mmse_cpp PRIVATE -O3 -DNDEBUG)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(arm|aarch64)")
        target_compile_options(dr_mmse_cpp PRIVATE -mcpu=native)
    else()
        target_compile_options(dr_mmse_cpp PRIVATE -march=native)
    endif()
elseif(MSVC)
    target_compile_options(dr_mmse_cpp PRIVATE /O2 /DNDEBUG)
endif()

# Place the .so next to the Python package so `python -m dr_mmse` works
# from the source tree without an install step.
set_target_properties(dr_mmse_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/dr_mmse"
)

install(TARGETS dr_mmse_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/python/dr_mmse)
