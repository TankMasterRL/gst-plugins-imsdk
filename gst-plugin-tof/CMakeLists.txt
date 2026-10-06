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

# Generate configuration header file with plugin describing definitions
configure_file(config.h.in config.h @ONLY)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Precompiler definitions.
add_definitions(-DHAVE_CONFIG_H)

# Common compiler flags.
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter")

# GStreamer plugin: qtitofsrc (T5.2); qtitofdepth follows in T5.3 (DEC-0020).
set(GST_QTI_TOF gstqtitof)

add_library(${GST_QTI_TOF} SHARED
  qtitof.c
  tofsrc.c
)

target_include_directories(${GST_QTI_TOF} PUBLIC
  ${GST_INCLUDE_DIRS}
  ${GST_BASE_INCLUDE_DIRS}
)

target_link_libraries(${GST_QTI_TOF} PRIVATE
  ${GST_LIBRARIES}
  ${GST_BASE_LIBRARIES}
)

install(
  TARGETS ${GST_QTI_TOF}
  LIBRARY DESTINATION ${GST_PLUGINS_QTI_OSS_INSTALL_LIBDIR}/gstreamer-1.0
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
)
