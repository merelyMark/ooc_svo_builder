SET(SVO_BUILDER_CUDA_SRCS
main.cpp
OctreeBuilder.cpp
partitioner.cpp
voxelizer.cpp
cuda.cu
ErrorCheck.cpp
MThread.h
)
CUDA_ADD_EXECUTABLE (  svo_builder_cuda ${SVO_BUILDER_CUDA_SRCS} )

TARGET_LINK_LIBRARIES ( svo_builder_cuda
  gomp
tbb
        tbbmalloc_proxy
boost_system
boost_thread
)
