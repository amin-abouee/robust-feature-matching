# relase binary folder
set(CMAKE_RELEASE_DIR "../bin/release")
# debug binary folder
set(CMAKE_DEBUG_DIR "../bin/debug")

# find openCV
if( APPLE )
    set( OpenCV_DIR "/opt/local/lib/cmake" )
endif()
find_package( OpenCV REQUIRED )

add_executable (robust_feature_matching main.cpp)
target_link_libraries (robust_feature_matching ${OpenCV_LIBS})

if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(robust_feature_matching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_DEBUG_DIR})
	set_target_properties(robust_feature_matching PROPERTIES OUTPUT_NAME robust_feature_matching${BUILD_PREFIX})
else()
	set_target_properties(robust_feature_matching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RELEASE_DIR})
endif()