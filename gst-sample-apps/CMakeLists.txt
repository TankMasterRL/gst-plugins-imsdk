# Add all immediate subdirectories automatically, except excluded ones
file(GLOB SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

set(EXCLUDED_DIRS
    gst-sample-apps-utils
    gst-camera-fd-stream-example
    gst-camera-metadata-example
    gst-camera-per-port-example
    gst-camera-burst-capture-example
    gst-webrtc-sendrecv-example
)

# Optional subset, e.g. -DGST_SAMPLE_APPS_ONLY="gst-qimsdk-tof-depth;gst-appsink-example" (empty = all)
set(GST_SAMPLE_APPS_ONLY "" CACHE STRING "Build only these sample apps (empty: all)")

foreach(subdir IN LISTS SUBDIRS)
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
        list(FIND EXCLUDED_DIRS "${subdir}" EXCLUDED_INDEX)
        if(GST_SAMPLE_APPS_ONLY)
            list(FIND GST_SAMPLE_APPS_ONLY "${subdir}" ONLY_INDEX)
            if(ONLY_INDEX EQUAL -1)
                set(EXCLUDED_INDEX 0)
            endif()
        endif()
        if(EXCLUDED_INDEX EQUAL -1)
            add_subdirectory("${subdir}")
        endif()
    endif()
endforeach()
