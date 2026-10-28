cmake_minimum_required(VERSION 3.16)
project(GST_PLUGIN_QTI_OSS_TOF
  VERSION ${GST_PLUGINS_QTI_OSS_VERSION}
  LANGUAGES C
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(PkgConfig)

pkg_check_modules(GST
  REQUIRED gstreamer-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_BASE
  REQUIRED gstreamer-base-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_ALLOC
  REQUIRED gstreamer-allocators-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_VIDEO
  REQUIRED gstreamer-video-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_QCOM_VIDEO
  REQUIRED gstreamer-qcom-oss-video-1.0>=1.0.0)
pkg_check_modules(GST_QCOM_ALLOC
  REQUIRED gstreamer-qcom-oss-allocators-1.0>=1.0.0)
# qtitofdepth "pointcloud": the QIM SDK tensor caps and GstMLTensorMeta
pkg_check_modules(GST_QCOM_ML
  REQUIRED gstreamer-qcom-oss-ml-1.0>=1.0.0)

# qtitofdepth: calibration key file used when the property is unset (none is installed with the plugin)
set(TOF_DEFAULT_CALIBRATION "/usr/share/arducam-tof/tof-depth-cal-v1.conf"
    CACHE STRING "Calibration key file qtitofdepth loads when calibration is unset")

# Generate configuration header file with plugin describing definitions
configure_file(config.h.in config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Precompiler definitions.
add_definitions(-DHAVE_CONFIG_H)

# Common compiler flags.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")

# GStreamer plugin: qtitofsrc and qtitofdepth.
set(GST_QTI_TOF gstqtitof)

add_library(${GST_QTI_TOF} SHARED
  qtitof.c
  tofsrc.c
  tofdepth.c
)

target_include_directories(${GST_QTI_TOF} PUBLIC
  ${GST_INCLUDE_DIRS}
  ${GST_BASE_INCLUDE_DIRS}
  ${GST_VIDEO_INCLUDE_DIRS}
  ${GST_QCOM_VIDEO_INCLUDE_DIRS}
  ${GST_QCOM_ALLOC_INCLUDE_DIRS}
  ${GST_QCOM_ML_INCLUDE_DIRS}
)

target_link_libraries(${GST_QTI_TOF} PRIVATE
  ${GST_LIBRARIES}
  ${GST_BASE_LIBRARIES}
  ${GST_ALLOC_LIBRARIES}
  ${GST_VIDEO_LIBRARIES}
  ${GST_QCOM_VIDEO_LIBRARIES}
  ${GST_QCOM_ALLOC_LIBRARIES}
  ${GST_QCOM_ML_LIBRARIES}
  m
)

install(
  TARGETS ${GST_QTI_TOF}
  LIBRARY DESTINATION ${GST_PLUGINS_QTI_OSS_INSTALL_LIBDIR}/gstreamer-1.0
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
)
