CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT ( OutOfCoreSparseVoxelOctree )

FIND_PACKAGE ( OpenMP REQUIRED )
FIND_PACKAGE(CUDA)
if (CUDA_SDK_ROOT_DIR MATCHES "CUDA_SDK_ROOT_DIR-NOTFOUND")
        IF(WIN32)
                # Look for the SDK stuff
                find_path(CUDA_SDK_ROOT_DIR C/common/inc/cutil.h
                  "$ENV{NVSDKCUDA_ROOT}"
                  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]"
                "/Developer/GPU\ Computing/C"
                )

                # Keep the CUDA_SDK_ROOT_DIR first in order to be able to override the
                # environment variables.
                set(CUDA_SDK_SEARCH_PATH
                  "${CUDA_SDK_ROOT_DIR}"
                  "${CUDA_TOOLKIT_ROOT_DIR}/local/NVSDK0.2"
                  "${CUDA_TOOLKIT_ROOT_DIR}/NVSDK0.2"
                  "${CUDA_TOOLKIT_ROOT_DIR}/NV_CUDA_SDK"
                  "$ENV{HOME}/NVIDIA_CUDA_SDK"
                  "$ENV{HOME}/NVIDIA_CUDA_SDK_MACOSX"
                  "/Developer/CUDA"
                )

        ELSE(WIN32)
                find_path(CUDA_SDK_ROOT_DIR C/common/inc/cutil.h
                                $ENV{NVSDKCUDA_ROOT}
                                ~/NVIDIA_GPU_Computing_SDK
                )
                SET(CUDA_SDK_SEARCH_PATH
                        "${CUDA_SDK_ROOT_DIR}"
                        "~/NVIDIA_GPU_Computing_SDK"
                )
        ENDIF(WIN32)
 endif (CUDA_SDK_ROOT_DIR MATCHES "CUDA_SDK_ROOT_DIR-NOTFOUND")
INCLUDE_DIRECTORIES(
    ${CUDA_SDK_ROOT_DIR}/common/inc
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    )
if(WIN32)
    SET(CUTIL_LIB_DIRECTORY
        "${CUDA_SDK_ROOT_DIR}/common/lib/"
    )
else(WIN32)
    SET(CUTIL_LIB_DIRECTORY
        "${CUDA_SDK_ROOT_DIR}/C/lib/"
    )
endif(WIN32)
#set(CUDA_NVCC_FLAGS "-arch;sm_20;--ptxas-options=-v;-use_fast_math")
#for 650 ti boost
set(CUDA_NVCC_FLAGS "-arch;sm_30")
#for k20
#set(CUDA_NVCC_FLAGS " -gencode arch=compute_35,code=sm_35;--ptxas-options=-v;-use_fast_math")

SET(Trimesh2_INCLUDE_DIR "" CACHE PATH "Path to Trimesh2 includes")
FIND_FILE(Trimesh2_TriMesh_h TriMesh.h ${Trimesh2_INCLUDE_DIR})
IF(NOT Trimesh2_TriMesh_h)
	message(SEND_ERROR "Can't find TriMesh.h in ${Trimesh2_INCLUDE_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_TriMesh_h)
INCLUDE_DIRECTORIES(${Trimesh2_INCLUDE_DIR})

SET(Trimesh2_LINK_DIR "" CACHE PATH "Path to Trimesh2 libraries")
FIND_LIBRARY(Trimesh2_LIBRARY trimesh ${Trimesh2_LINK_DIR})
IF(NOT Trimesh2_LIBRARY)
	message(SEND_ERROR "Can't find libtrimesh.a in ${Trimesh2_LINK_DIR}")
ENDIF()
MARK_AS_ADVANCED(Trimesh2_LIBRARY)

IF (MSVC)
  ADD_DEFINITIONS ( -openmp )
ELSE ()
  # Apple and Linux both uses GCC
  ADD_DEFINITIONS ( -fopenmp )
ENDIF (MSVC)

INCLUDE_DIRECTORIES (
  ./src/ooc_svo_builder/tri_convert
  ./src/ooc_svo_builder/tri_tools/include
  )


SET(SVO_BUILDER_SRCS
  ./src/ooc_svo_builder/svo_builder/main.cpp
  ./src/ooc_svo_builder/svo_builder/OctreeBuilder.cpp
  ./src/ooc_svo_builder/svo_builder/partitioner.cpp
  ./src/ooc_svo_builder/svo_builder/voxelizer.cpp
  ./src/ooc_svo_builder/svo_builder/cuda.cu
  ./src/ooc_svo_builder/svo_builder/ErrorCheck.cpp
)
CUDA_ADD_EXECUTABLE (  svo_builder ${SVO_BUILDER_SRCS} )



SET(TRI_CONVERT_SRCS
  ./src/ooc_svo_builder/tri_convert/tri_convert.cpp
)
ADD_EXECUTABLE ( tri_convert ${TRI_CONVERT_SRCS} )

TARGET_LINK_LIBRARIES ( svo_builder
  gomp
tbb
	tbbmalloc_proxy
)
TARGET_LINK_LIBRARIES ( tri_convert
  ${Trimesh2_LIBRARY}
  gomp
)
