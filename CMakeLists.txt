cmake_minimum_required(VERSION 3.26)
project(MemoryManager CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Download CPM
set(CPM_DOWNLOAD_VERSION 0.34.0)
if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()
if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION})
endif ()
include(${CPM_DOWNLOAD_LOCATION})

# Add CCCL
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY NVIDIA/cccl
        GIT_TAG main)

set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -lineinfo")

add_library(MemoryManager SHARED library.cu
        include/memmove.cuh
        include/kernels/ScanTileStateInit.cuh
        include/kernels/MemMoveSimple.cuh
        include/kernels/MemMoveComplex.h
)

set_target_properties(MemoryManager PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(MemoryManager PRIVATE CCCL::CCCL)
