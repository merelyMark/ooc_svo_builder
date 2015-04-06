#ifndef VOXELIZER_H_
#define VOXELIZER_H_

#include <tbb/atomic.h>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <vector>
#include "morton.h"

// Voxelization-related stuff
typedef unsigned long long int uint64;
typedef char voxel_t;
using namespace std;

#define EMPTY_VOXEL 0
#define FULL_VOXEL 1
#define WORKING_VOXEL 2
class TriReaderIter;

extern "C"
void cudaCount(const float3* d_v0, const float3*d_v1, const float3*d_v2, uint *d_tri_idx, uint *d_nfilled, uint nfilled_idx, voxel_t *d_voxels, uint64 *d_data,
             const uint64 morton_start, const uint64 morton_end, const float unitlength,  bool &use_data, tbb::atomic<size_t> &nfilled,
             const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items, uint num_triangles);


extern "C"
void cudaRun(const float3* d_v0, const float3*d_v1, const float3*d_v2, uint *d_tri_idx, uint *d_nfilled, voxel_t *d_voxels, uint64 *d_data,
             const uint64 morton_start, const uint64 morton_end, const float unitlength, uint64 *data, uint &data_size, bool &use_data,
             const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items, uint num_triangles,
             uint partition_idx,
             uint idx);

extern "C"
uint cudaCountFinalize(uint nfilled_blocks, uint*);
extern "C"
uint cudaFinalize(uint cnt, uint64 *d_data);

void voxelize_schwarz_method(TriReaderIter *reader, TriReaderIter *orig_reader,
                             uint64 *&d_data,
                             float3 *&d_v0, float3 *&d_v1, float3 *&d_v2,
                             voxel_t *&d_voxels, size_t data_max_items,
                             const mort_t morton_start, const mort_t morton_end, const mort_t morton_length,
                             const float unitlength, voxel_t*&voxels, mort_t *&data, uint &data_size, float sparseness_limit,
                             bool &use_data, tbb::atomic<size_t> &nfilled,
                             uint nfilled_blocks,
                             uint partition_idx,
                            uint idx, uint size);

void voxelize_schwarz_count(TriReaderIter *reader, TriReaderIter *orig_reader,
                            uint64 *&d_data,
                             float3 *&d_v0, float3 *&d_v1, float3 *&d_v2,
                             voxel_t *&d_voxels, size_t data_max_items,
                             const mort_t morton_start, const mort_t morton_end,
                            const mort_t morton_length, const float unitlength,
                            voxel_t*&voxels, mort_t *&data, uint &data_size,
                            float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled,
                            uint nfilled_blocks,
                            uint nfilled_idx);

size_t voxelize_count_finalize(uint nfilled_blocks, std::vector<uint> &output);
uint voxelize_finalize(uint cnt, uint64 *d_data);

#endif // VOXELIZER_H_
