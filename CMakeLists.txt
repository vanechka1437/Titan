cmake_minimum_required(VERSION 3.20)
project(TitanSimulator VERSION 0.1.0 LANGUAGES CXX)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# 1. Global C++ Standards
# ==============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# 2. Hardware-Aware Configuration (L3 Cache Detection)
# ==============================================================================
if(CMAKE_HOST_WIN32)
    # Windows: Returns value in Megabytes
    execute_process(COMMAND wmic cpu get L3CacheSize OUTPUT_VARIABLE WMIC_OUTPUT ERROR_QUIET)
    string(REGEX MATCHALL "[0-9]+" L3_MB_LIST "${WMIC_OUTPUT}")
    list(GET L3_MB_LIST 0 L3_MB)
    if(L3_MB)
        math(EXPR L3_BYTES "${L3_MB} * 1024 * 1024")
    endif()
elseif(CMAKE_HOST_UNIX)
    # Linux: Returns value in Bytes
    execute_process(COMMAND getconf LEVEL3_CACHE_SIZE OUTPUT_VARIABLE GETCONF_OUTPUT ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(GETCONF_OUTPUT MATCHES "^[0-9]+$")
        set(L3_BYTES ${GETCONF_OUTPUT})
    endif()
endif()

# Safe fallback if hardware query fails (Default: 16 MB)
if(NOT L3_BYTES OR L3_BYTES EQUAL 0)
    set(L3_BYTES 16777216)
    message(WARNING "[Titan] Failed to detect L3 Cache size. Defaulting to 16 MB.")
endif()

# Expose to user for manual override if necessary
set(TITAN_SYSTEM_L3_BYTES ${L3_BYTES} CACHE STRING "L3 Cache size in bytes for Hot Zone optimization")
message(STATUS "[Titan] Hardware Profile -> L3 Cache: ${TITAN_SYSTEM_L3_BYTES} bytes")

# ==============================================================================
# 3. Compiler Flags (HFT Profile)
# ==============================================================================
if(MSVC)
    # /O2: Max speed, /Ot: Fast code, /GL: Whole program optimization, /arch:AVX2: Vectorization
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ot /GL /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /fsanitize=address")
else()
    # GCC / Clang: Native architecture tuning, Link-Time Optimization
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
endif()

include(FetchContent)

# ==============================================================================
# 4. Dependencies: Boost (Header-only)
# ==============================================================================
FetchContent_Declare(
    boost
    URL https://archives.boost.io/release/1.84.0/source/boost_1_84_0.tar.gz
)
FetchContent_GetProperties(boost)
if(NOT boost_POPULATED)
    message(STATUS "[Titan] Fetching Boost headers...")
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
# 5. Dependencies: Google Benchmark
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

message(STATUS "[Titan] Fetching Google Benchmark...")
FetchContent_MakeAvailable(googlebenchmark)

# ==============================================================================
# 6. Dependencies: Google Test
# ==============================================================================
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

message(STATUS "[Titan] Fetching Google Test...")
FetchContent_MakeAvailable(googletest)

# ==============================================================================
# 7. Dependencies: Google Abseil (B-Tree Integration)
# ==============================================================================
FetchContent_Declare(
    abseil_cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG 20240116.2
)

# Force Abseil to compile against the project's native C++ standard
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Use the main project's C++ standard" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable Abseil internal tests" FORCE)

message(STATUS "[Titan] Fetching Google Abseil...")
FetchContent_MakeAvailable(abseil_cpp)

# ==============================================================================
# 8. Core Library (Titan)
# ==============================================================================
add_library(titan_core STATIC)

target_include_directories(titan_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(titan_core PRIVATE
    src/core/memory.cpp
    src/core/lob_state.cpp
    src/core/matching_engine.cpp
)

# Inject the detected L3 Cache size directly into the compiler
target_compile_definitions(titan_core PUBLIC TITAN_SYSTEM_L3_BYTES=${TITAN_SYSTEM_L3_BYTES})

target_link_libraries(titan_core PUBLIC Boost::boost)

# ==============================================================================
# 9. Subdirectories
# ==============================================================================

# ==============================================================================
# 10. Tests
# ==============================================================================
add_executable(titan_lob_tests
    tests/lob_state_tests.cpp
)

target_link_libraries(titan_lob_tests PRIVATE
    titan_core
    gtest_main
)

add_executable(titan_memory_tests
    tests/memory_tests.cpp
)

target_link_libraries(titan_memory_tests PRIVATE titan_core gtest_main)

add_executable(titan_state_tests
    tests/state_tests.cpp
)

target_link_libraries(titan_state_tests PRIVATE titan_core gtest_main)

add_executable(titan_matching_engine_tests
    tests/matching_engine_tests.cpp
)

target_link_libraries(titan_matching_engine_tests PRIVATE titan_core gtest_main)

add_executable(titan_scheduler_tests
    tests/scheduler_tests.cpp
)

target_link_libraries(titan_scheduler_tests PRIVATE titan_core gtest_main)

include(GoogleTest)
gtest_discover_tests(titan_lob_tests)
gtest_discover_tests(titan_memory_tests)
gtest_discover_tests(titan_state_tests)
gtest_discover_tests(titan_matching_engine_tests)

# ==============================================================================
# 11. Benchmarks
# ==============================================================================
add_executable(titan_lob_bench
    benchmarks/lob_state_bench.cpp
)

target_link_libraries(titan_lob_bench PRIVATE
    titan_core
    benchmark::benchmark
    absl::btree
)

add_executable(titan_memory_bench
    benchmarks/memory_bench.cpp
)

target_link_libraries(titan_memory_bench PRIVATE
    titan_core
    benchmark::benchmark
    Boost::boost
)

add_executable(titan_scheduler_bench
    benchmarks/scheduler_bench.cpp
)

target_link_libraries(titan_scheduler_bench PRIVATE
    titan_core
    benchmark::benchmark
    Boost::boost
)