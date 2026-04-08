# Create the executable for memory benchmarking
add_executable(memory_bench memory_bench.cpp)

# Link Titan core, Google Benchmark, and Boost
target_link_libraries(memory_bench 
    PRIVATE 
    titan_core 
    benchmark::benchmark 
    benchmark::benchmark_main 
    Boost::boost
)

# Enable Link-Time Optimization (LTO/GL) specifically for this benchmark
# to ensure the allocator functions are inlined into the loop.
set_property(TARGET memory_bench PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)