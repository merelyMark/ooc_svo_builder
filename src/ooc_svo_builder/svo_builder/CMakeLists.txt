
SET(SVO_BUILDER_SRCS
  main.cpp
  OctreeBuilder.cpp
  partitioner.cpp
  voxelizer.cpp

)
ADD_EXECUTABLE (  svo_builder ${SVO_BUILDER_SRCS} )


TARGET_LINK_LIBRARIES ( svo_builder
  gomp
tbb
        tbbmalloc_proxy
)

