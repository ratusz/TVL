add_executable(tester)

target_link_libraries(tester PRIVATE
    tvl
    ${PROJECT_SOURCE_DIR}/../generator/supplementary/fpga_rtl/lib.a
)

target_sources(tester PRIVATE
    src/main.cpp
)
