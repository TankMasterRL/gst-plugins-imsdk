set(GST_CAMERA_SAMPLE_APPS
    gst-camera-fd-stream-example
    gst-camera-per-port-example
    gst-camera-burst-capture-example
    gst-camera-metadata-example
)

if (ENABLE_GST_SAMPLE_APPS_CAMERA)
  foreach(dir ${GST_CAMERA_SAMPLE_APPS})
    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
      message(STATUS "Directory exists, adding subdirectory: ${dir}")
      add_subdirectory(${dir})
    else()
      message(STATUS "Directory does NOT exist: ${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
    endif()
  endforeach()
else()
  message(STATUS "ENABLE_GST_SAMPLE_APPS_CAMERA is disabled")
endif()

# Optional subset, e.g. -DGST_SAMPLE_APPS_ONLY="gst-qimsdk-tof-depth;gst-appsink-example" (empty = all)
set(GST_SAMPLE_APPS_ONLY "" CACHE STRING "Build only these sample apps (empty: all)")

if (ENABLE_GST_SAMPLE_APPS)
  file(GLOB GST_SAMPLE_APPS_DIRS
       RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
       gst-*
       GstD-*)

  foreach(dir ${GST_SAMPLE_APPS_DIRS})

    if(NOT IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
      continue()
    endif()

    list(FIND GST_CAMERA_SAMPLE_APPS ${dir} _camera_index)
    if (_camera_index GREATER -1)
      continue()
    endif()

    if (dir STREQUAL "gst-sample-apps-utils")
      continue()
    endif()

    if (GST_SAMPLE_APPS_ONLY)
      list(FIND GST_SAMPLE_APPS_ONLY ${dir} _only_index)
      if (_only_index EQUAL -1)
        continue()
      endif()
    endif()

    add_subdirectory(${dir})

  endforeach()
endif()
