cmake_minimum_required(VERSION 3.16)
project(GST_TEST_FRAMEWORK LANGUAGES C)

set(GST_TEST_FRAMEWORK gst-test-framework)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(PkgConfig)

# Get the pkgconfigs exported by the automake tools
pkg_check_modules(GST
  REQUIRED gstreamer-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_CHECK
  REQUIRED gstreamer-check-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_VIDEO
  REQUIRED gstreamer-video-1.0>=${GST_VERSION_REQUIRED})
pkg_check_modules(GST_PBUTILS
  REQUIRED gstreamer-pbutils-1.0>=${GST_VERSION_REQUIRED})
# suite-tof "pointcloud": parse the tensor caps and meta with the QIM ML base library
pkg_check_modules(GST_QCOM_ML
  REQUIRED gstreamer-qcom-oss-ml-1.0>=1.0.0)

include_directories(utils)

add_executable(${GST_TEST_FRAMEWORK}
  main.c
  utils/suite-utils.c
  suite-camera/suite-camera-case.c
  suite-camera/suite-camera-pipeline.c
  suite-ML/suite-ml-case.c
  suite-ML/suite-ml-pipeline.c
  suite-tof/suite-tof-case.c
)

target_include_directories(${GST_TEST_FRAMEWORK} PRIVATE
  ${GST_INCLUDE_DIRS}
  ${GST_CHECK_INCLUDE_DIRS}
  ${GST_QCOM_ML_INCLUDE_DIRS}
)

target_link_libraries(${GST_TEST_FRAMEWORK} PRIVATE
  ${GST_LIBRARIES}
  ${GST_CHECK_LIBRARIES}
  ${GST_PBUTILS_LIBRARIES}
  ${GST_VIDEO_LIBRARIES}
  ${GST_QCOM_ML_LIBRARIES}
  m
)

install(
  TARGETS ${GST_TEST_FRAMEWORK}
  RUNTIME DESTINATION ${GST_PLUGINS_QTI_OSS_INSTALL_BINDIR}
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
)
