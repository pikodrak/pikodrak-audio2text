cmake_minimum_required(VERSION 3.22.1)
project(whisper_android)

# whisper.cpp source tree is a git submodule at this path.
# Run `git submodule update --init --recursive` before the first build.
set(WHISPER_DIR ${CMAKE_SOURCE_DIR}/whisper.cpp)
if(NOT EXISTS ${WHISPER_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
        "whisper.cpp submodule is missing at ${WHISPER_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# Disable everything we don't need
set(BUILD_SHARED_LIBS OFF)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)

add_subdirectory(${WHISPER_DIR} whisper_build)

add_library(whisper-jni SHARED whisper_jni.cpp)

target_include_directories(whisper-jni PRIVATE
    ${WHISPER_DIR}/include
    ${WHISPER_DIR}/ggml/include
)

# ggml is split into its own target in current whisper.cpp; link both.
target_link_libraries(whisper-jni
    whisper
    ggml
    android
    log
)
