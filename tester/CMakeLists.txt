cmake_minimum_required(VERSION 3.10)

project(tester LANGUAGES CXX)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/2022.0.1/linux/bin/dpcpp)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(target "emulator" CACHE STRING "")

# # Emulator
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
#     -fintelfpga \
#     -fsycl-device-code-split=off \
#     -DFPGA_EMULATOR \
#     -I${PROJECT_SOURCE_DIR}/../generator/supplementary/fpga_rtl \
#     -I${PROJECT_SOURCE_DIR}/lib/include \
# ")

# # FPGA
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
#     -fintelfpga \
#     -fsycl-device-code-split=off \
#     -Xshardware \
#     -Xsboard=intel_s10sx_pac:pac_s10_usm \
#     -I${PROJECT_SOURCE_DIR}/../generator/supplementary/fpga_rtl \
#     -I${PROJECT_SOURCE_DIR}/lib/include \
# ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -fintelfpga \
    -fsycl-device-code-split=off \
    -I${PROJECT_SOURCE_DIR}/../generator/supplementary/fpga_rtl \
    -I${PROJECT_SOURCE_DIR}/lib/include \
")

if (${target} STREQUAL "fpga")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -Xshardware \
        -Xsboard=intel_s10sx_pac:pac_s10_usm \
    ")
elseif (${target} STREQUAL "emulator")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -DFPGA_EMULATOR \
    ")
else()
    message(SEND_ERROR "Incorrect target: ${target}")
    return()
endif()

add_subdirectory(app)
add_subdirectory(lib)
