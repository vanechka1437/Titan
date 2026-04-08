cmake_minimum_required(VERSION 3.20)
project(TitanSimulator VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# 1. Global C++ Standards
# ==============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# 2. Compiler Flags (Cross-Platform for Windows/MSVC and Linux/GCC/Clang)
# ==============================================================================
if(MSVC)
    # Windows / MSVC specific optimization flags
    # /O2: Maximize speed, /Ot: Favor fast code, /GL: Whole program optimization, /arch:AVX2: Vectorization
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ot /GL /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /fsanitize=address")
else()
    # GCC / Clang optimization flags
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
endif()

include(FetchContent)

# ==============================================================================
# 3. Dependencies: Boost (Auto-fetch header-only mode for Windows)
# ==============================================================================
FetchContent_Declare(
    boost
    URL https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz
)
FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    message(STATUS "Fetching Boost headers from official archives...")
    FetchContent_Populate(boost)
endif()

file(GLOB BOOST_ROOT_DIRS "${boost_SOURCE_DIR}/boost_*")
if(BOOST_ROOT_DIRS)
    list(GET BOOST_ROOT_DIRS 0 BOOST_ACTUAL_ROOT)
else()
    set(BOOST_ACTUAL_ROOT "${boost_SOURCE_DIR}")
endif()

add_library(Boost::boost INTERFACE IMPORTED)
set_target_properties(Boost::boost PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${BOOST_ACTUAL_ROOT}"
)

# ==============================================================================
# 4. Dependencies: Google Benchmark
# ==============================================================================
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)

set(HAVE_STD_REGEX TRUE CACHE BOOL "Force std::regex support" FORCE)
set(HAVE_POSIX_REGEX FALSE CACHE BOOL "Force disable POSIX" FORCE)
set(HAVE_GNU_POSIX_REGEX FALSE CACHE BOOL "Force disable GNU POSIX" FORCE)

FetchContent_MakeAvailable(googlebenchmark)

# ==============================================================================
# 5. Core Library (Titan)
# ==============================================================================
add_library(titan_core STATIC)

target_include_directories(titan_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add source files
target_sources(titan_core PRIVATE
    src/core/memory.cpp
    # Future modules:
    # src/core/lob_state.cpp
    # src/core/matching_engine.cpp
)

# ==============================================================================
# 6. Subdirectories
# ==============================================================================
add_subdirectory(benchmarks)