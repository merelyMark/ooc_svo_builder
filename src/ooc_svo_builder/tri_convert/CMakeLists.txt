SET(TRI_CONVERT_SRCS
tri_convert.cpp
)
ADD_EXECUTABLE ( tri_convert ${TRI_CONVERT_SRCS} )
TARGET_LINK_LIBRARIES ( tri_convert
  ${Trimesh2_LIBRARY}
  gomp
)
