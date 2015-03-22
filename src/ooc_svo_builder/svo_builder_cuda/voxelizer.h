#ifndef VOXELIZER_H_
#define VOXELIZER_H_

#include <tbb/atomic.h>

#include <cuda_runtime.h>
#include <vector>
#include "morton.h"

// Voxelization-related stuff
typedef unsigned long long int uint64;
typedef int voxel_t;
using namespace std;

#define EMPTY_VOXEL 0
#define FULL_VOXEL 1
#define WORKING_VOXEL 2
class TriReaderIter;

extern "C"
void cudaRun(const float3* d_v0, const float3*d_v1, const float3*d_v2, unsigned int *d_tri_idx, unsigned int *d_nfilled, int *d_voxels, uint64 *d_data,
             const uint64 morton_start, const uint64 morton_end, const float unitlength, voxel_t *voxels, std::vector<uint64> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled,
             const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items, unsigned int num_triangles);


void voxelize_schwarz_method(TriReaderIter *reader, TriReaderIter *orig_reader,
                             float3 *&d_v0, float3 *&d_v1, float3 *&d_v2, int *&d_voxels,
                             const mort_t morton_start, const mort_t morton_end, const mort_t morton_length, const float unitlength, voxel_t* voxels, std::vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled);


#endif // VOXELIZER_H_
