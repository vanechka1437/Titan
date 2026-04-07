cmake_minimum_required(VERSION 3.20)
project(TitanSimulator VERSION 0.1.0 LANGUAGES CXX)

# Устанавливаем стандарт C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Флаги оптимизации
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")

# Подключение Google Benchmark 
include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)

# Отключаем тесты самого бенчмарка, чтобы не мусорить в сборке
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Основная библиотека Titan
add_library(titan_core STATIC)

target_include_directories(titan_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Подключаем исходники
target_sources(titan_core PRIVATE
    src/core/memory.cpp
    # В будущем добавим:
    # src/core/lob_state.cpp
    # src/core/matching_engine.cpp
)

# Подключаем папку с бенчмарками
add_subdirectory(benchmarks)