#ifndef VOXELIZER_H_
#define VOXELIZER_H_

#include <tri_tools.h>
#include <TriReaderIter.h>
#include <tbb/atomic.h>
#include <tbb/concurrent_vector.h>
#include <cuda_runtime.h>
#include "globals.h"
#include "intersection.h"
#include "morton.h"
#include "VoxelData.h"

// Voxelization-related stuff
typedef Vec<3, unsigned int> uivec3;
typedef unsigned long long int uint64;

using namespace std;

#define EMPTY_VOXEL 0
#define FULL_VOXEL 1
#define WORKING_VOXEL 2


extern "C"
void cudaRun(const float3* d_v0, const float3*d_v1, const float3*d_v2,const uint64 morton_start, const uint64 morton_end, const float unitlength, tbb::atomic<char> *voxels, tbb::concurrent_vector<uint64> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled,
             const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items, size_t num_triangles);


void voxelize_schwarz_method(TriReaderIter &reader, const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<char>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled);


#endif // VOXELIZER_H_
