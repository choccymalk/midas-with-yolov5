cmake_minimum_required(VERSION 3.6)

project(rockchip_npu)
set(CMAKE_CROSSCOMPILING TRUE)

# (Optional) Check if we are cross-compiling
if (CMAKE_CROSSCOMPILING)
    message(STATUS "Cross-compiling for aarch64 Linux")
    # Set any target-specific flags here
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8") # adjust for your architecture
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -Ofast -march=armv8")
else()
    message(STATUS "Not cross-compiling (building for host)")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -Ofast")
endif()

# opencv
set(OpenCV_DIR 3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
  message(STATUS "OpenCV library status:")
  message(STATUS "    version: ${OpenCV_VERSION}")
  message(STATUS "    libraries: ${OpenCV_LIBS}")
  message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
  include_directories(${OpenCV_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "OpenCV not found")
endif()

# rknn api
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB "${CMAKE_SOURCE_DIR}/3rdparty/static/librknnrt.so")
include_directories(include)
include_directories(3rdparty)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rockchip_npu
    src/main.cpp
)

target_link_libraries(rockchip_npu
  "${RKNN_RT_LIB}"
  ${OpenCV_LIBS}
)

# install target and libraries
install(TARGETS rockchip_npu DESTINATION ./)