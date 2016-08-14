find_package( OpenCV REQUIRED )

set(HEADER_PATH ${PROJECT_SOURCE_DIR}/include)
include_directories(${HEADER_PATH})

file(GLOB HEADER_FILES ${HEADER_PATH}/*.hpp)
file(GLOB SOURCES_FILES *.cpp)

add_library(RobustFeatureMatchingLib SHARED RobustFeatureMatching.cpp)
target_link_libraries (RobustFeatureMatchingLib ${OpenCV_LIBS})

set(BUILD_PATH ${PROJECT_SOURCE_DIR}/build)

add_executable (RobustFeatureMatching ${SOURCES_FILES} ${HEADER_FILES})
target_link_libraries (RobustFeatureMatching ${OpenCV_LIBS})

file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(LIB_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RELEASE_DIR ${BUILD_PATH}/release)
set(CMAKE_DEBUG_DIR ${BUILD_PATH}/debug)

set_target_properties(RobustFeatureMatchingLib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${LIB_PATH})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(RobustFeatureMatching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_DEBUG_DIR})
	set_target_properties(RobustFeatureMatching PROPERTIES OUTPUT_NAME RobustFeatureMatching${BUILD_PREFIX})
else()
	set_target_properties(RobustFeatureMatching PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RELEASE_DIR})
endif()
