# relase binary folder
set(CMAKE_RELEASE_DIR "../build/release")
# debug binary folder
set(CMAKE_DEBUG_DIR "../build/debug")

find_package( OpenCV REQUIRED )

set(HEADER_PATH ${PROJECT_SOURCE_DIR}/headers)
include_directories(${HEADER_PATH})

set(HEADER_FILES ${HEADER_PATH}/RobustFeatureMatching.hpp)
set(SOURCES_FILES main.cpp RobustFeatureMatching.cpp)

add_library(RobustFeatureMatchingLib SHARED RobustFeatureMatching.cpp)
target_link_libraries (RobustFeatureMatchingLib ${OpenCV_LIBS})

add_executable (RobustFeatureMatching ${SOURCES_FILES} ${HEADER_FILES})
target_link_libraries (RobustFeatureMatching ${OpenCV_LIBS})

if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(RobustFeatureMatching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_DEBUG_DIR})
	set_target_properties(RobustFeatureMatching PROPERTIES OUTPUT_NAME RobustFeatureMatching${BUILD_PREFIX})
else()
	set_target_properties(RobustFeatureMatching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RELEASE_DIR})
endif()
