cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Turing, Ampere, Ada, Blackwell (+ PTX for newer GPUs).
    set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real 89-real 120)
endif()

project(gltf-texture-compressor VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Static CUDA runtime: the executable only needs the NVIDIA driver at run time.
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# tinygltf is header only (it also bundles nlohmann/json and stb_image); download it without adding its CMake project.
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG v2.9.6
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR do-not-add)
FetchContent_MakeAvailable(tinygltf)

# KTX-Software (libktx with the Basis Universal encoders), built as a static library.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_TOOLS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_TESTS OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_DOC OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_LOADTEST_APPS "" CACHE STRING "" FORCE)
set(KTX_FEATURE_GL_UPLOAD OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_VK_UPLOAD OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_JNI OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_PY OFF CACHE BOOL "" FORCE)
set(KTX_FEATURE_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(BASISU_SUPPORT_OPENCL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(ktx
    GIT_REPOSITORY https://github.com/KhronosGroup/KTX-Software.git
    GIT_TAG v4.4.2
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(ktx)

add_executable(gltf-texture-compressor
    src/main.cpp
    src/gltf_io.cpp
    src/image.cpp
    src/ktx_encoder.cpp
    src/neural_pass.cpp
    src/third_party_impl.cpp
    src/neural/babylon_ntc.cpp
    src/neural/preset.cpp
    src/uv_mask.cpp
    src/neural/ntc_cuda.cu
    src/neural/ntc_shape_0.cu
    src/neural/ntc_shape_1.cu
    src/neural/ntc_shape_2.cu
    src/neural/ntc_shape_3.cu
    src/neural/ntc_shape_4.cu
    src/neural/ntc_shape_5.cu
    src/neural/ntc_shape_6.cu
    src/neural/ntc_shape_7.cu)

option(GTC_WITH_NVIDIA_NTC "Enable the nvidia-ntc comparison codec (runs the external RTXNTC ntc-cli; nothing is linked)" OFF)
if(GTC_WITH_NVIDIA_NTC)
    target_sources(gltf-texture-compressor PRIVATE src/neural/nvidia_ntc.cpp)
    target_compile_definitions(gltf-texture-compressor PRIVATE GTC_WITH_NVIDIA_NTC=1)
endif()

target_include_directories(gltf-texture-compressor PRIVATE src ${tinygltf_SOURCE_DIR})
target_link_libraries(gltf-texture-compressor PRIVATE ktx)
target_compile_definitions(gltf-texture-compressor PRIVATE
    GTC_VERSION="${PROJECT_VERSION}" _CRT_SECURE_NO_WARNINGS NOMINMAX)
if(MSVC)
    target_compile_options(gltf-texture-compressor PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /utf-8 /bigobj>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
target_compile_options(gltf-texture-compressor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

install(TARGETS gltf-texture-compressor RUNTIME DESTINATION bin)

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GTC_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(GTC_GIT_SHA)
    set_source_files_properties(src/neural_pass.cpp PROPERTIES COMPILE_DEFINITIONS GTC_GIT_SHA="${GTC_GIT_SHA}")
endif()
