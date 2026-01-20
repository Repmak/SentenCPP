cmake_minimum_required(VERSION 3.20)
set(CMAKE_POLICY_DEFAULT_CMP0000 NEW)
project(sentencpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ICU
find_package(ICU COMPONENTS uc i18n REQUIRED)

# nlohmann_json
include(FetchContent)
FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime root")
find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_ROOT}
        PATH_SUFFIXES include
        REQUIRED
)
message(STATUS "LOCATED ONNXRUNTIME AT: ${ONNXRUNTIME_INCLUDE_DIR}")
find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        HINTS ${ONNXRUNTIME_ROOT}
        PATH_SUFFIXES lib
        REQUIRED
)

# Wrapper for ONNX Runtime
add_library(onnxruntime_external INTERFACE)
target_include_directories(onnxruntime_external INTERFACE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(onnxruntime_external INTERFACE ${ONNXRUNTIME_LIB})

# Library target
add_library(sentencpp STATIC
        tokenizer/src/VocabList.cpp
        tokenizer/src/WordPiece.cpp
        inference/src/OnnxEngine.cpp
        embedding_utils/src/VectorMaths.cpp
)

target_include_directories(sentencpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inference/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/embedding_utils/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(sentencpp
        PUBLIC
            nlohmann_json::nlohmann_json
            onnxruntime_external
        PRIVATE
            ICU::uc
            ICU::i18n
)

add_custom_command(TARGET sentencpp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_LIB}
        $<TARGET_FILE_DIR:sentencpp>
        COMMENT "Copying ONNX Runtime binary to build directory"
)

target_compile_definitions(sentencpp PUBLIC
        SENTENCPP_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(example_1 examples/example_usage_1.cpp)

target_link_libraries(example_1 PRIVATE sentencpp)
