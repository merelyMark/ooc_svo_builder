#include "voxelizer.h"
#include <tbb/atomic.h>
#include <omp.h>
#include <typeinfo>
#include <nmmintrin.h>
#include <TriReaderIter.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include "ErrorCheck.h"

#include "intersection.h"
#include "partitioner.h"
#include <vector>
using namespace std;
using namespace trimesh;

#define X 0
#define Y 1
#define Z 2
uint *d_nfilled;
uint *d_tri_idx;
ErrorCheck ec;

void cudaLoadMem(TriReaderIter *reader, const AABox<uivec3> &p_bbox_grid, const vec3 &delta_p,
                 uint3 &p_bbox_grid_min, uint3 &p_bbox_grid_max,
                 float3 &cuda_delta_p)
{
    ec.chk("cudaLoadMem start");
    vector<Triangle> tris = reader->triangles;

    uint *tri_idx = new uint[tris.size()];

    for (int i=0; i<tris.size(); i++){
        tri_idx[i] = tris[i].idx;
    }


    cudaMemcpy(d_tri_idx, tri_idx, sizeof(uint)*tris.size(), cudaMemcpyHostToDevice);
    ec.chk("cudaLoadMem");
    delete tri_idx;

    p_bbox_grid_min.x =  p_bbox_grid.min[0];
    p_bbox_grid_min.y =  p_bbox_grid.min[1];
    p_bbox_grid_min.z =  p_bbox_grid.min[2];
    p_bbox_grid_max.x =  p_bbox_grid.max[0];
    p_bbox_grid_max.y =  p_bbox_grid.max[1];
    p_bbox_grid_max.z =  p_bbox_grid.max[2];

    cuda_delta_p.x = delta_p[0];
    cuda_delta_p.y = delta_p[1];
    cuda_delta_p.z = delta_p[2];

}

void cudaLoadFullTri(TriReaderIter *orig_reader, uint64 *&d_data,
                 float3 *&d_v0, float3 *&d_v1, float3 *&d_v2, voxel_t *&h_voxels, voxel_t *&d_voxels,
                     mort_t morton_length,
                     int nfilled_blocks
                     )
{
    ErrorCheck ec;
    vector<Triangle> tris = orig_reader->triangles;

    float3 *v0, *v1, *v2;
    v0 = new float3[tris.size()];
    v1 = new float3[tris.size()];
    v2 = new float3[tris.size()];

    for (int i=0; i<tris.size(); i++){
        v0[i].x = tris[i].v0[0];
        v0[i].y = tris[i].v0[1];
        v0[i].z = tris[i].v0[2];

        v1[i].x = tris[i].v1[0];
        v1[i].y = tris[i].v1[1];
        v1[i].z = tris[i].v1[2];

        v2[i].x = tris[i].v2[0];
        v2[i].y = tris[i].v2[1];
        v2[i].z = tris[i].v2[2];

    }

    cudaMalloc((void**)&d_v0, tris.size() * sizeof(float3));
    cudaMalloc((void**)&d_v1, tris.size() * sizeof(float3));
    cudaMalloc((void**)&d_v2, tris.size() * sizeof(float3));

    cudaMemcpy(d_v0, v0, sizeof(float3)*tris.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1, v1, sizeof(float3)*tris.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, sizeof(float3)*tris.size(), cudaMemcpyHostToDevice);
    delete []v0;
    delete []v1;
    delete []v2;

    cudaHostAlloc((void**)&h_voxels, sizeof(voxel_t)*morton_length, cudaHostAllocDefault);

    cudaMalloc( (void**) &d_nfilled, sizeof(uint)*nfilled_blocks);
    cudaMalloc( (void**) &d_voxels, sizeof(voxel_t)*morton_length);
    cudaMalloc( (void**) &d_data, sizeof(uint64)*morton_length);

    cudaMalloc((void**)&d_tri_idx, tris.size() * sizeof(uint));

    cudaMemset(d_nfilled, 0, sizeof(uint)*nfilled_blocks);

    ec.chk("loadfulltri");
}



void runCUDA(float3 *d_v0, float3 *d_v1, float3 *d_v2, voxel_t *d_voxels, uint64 *d_data,
             uint num_tris, const mort_t morton_start, const mort_t morton_end, const float unitlength,
             voxel_t * voxels, mort_t *data, uint &data_size, float sparseness_limit, bool &use_data,
             uint3 &p_bbox_grid_min, uint3 &p_bbox_grid_max, const float unit_div,
             float3 &cuda_delta_p,	size_t data_max_items, uint partition_idx, uint idx)
{


    cudaRun(d_v0, d_v1, d_v2, d_tri_idx, d_nfilled, d_voxels, d_data, morton_start, morton_end,
            unitlength, data, data_size, use_data, p_bbox_grid_min, p_bbox_grid_max,
            unit_div, cuda_delta_p, data_max_items, num_tris, partition_idx, idx);

}

void runCount(const float3* d_v0, const float3*d_v1, const float3*d_v2, uint nfilled_idx, voxel_t *d_voxels, uint64 *d_data,
              const uint64 morton_start, const uint64 morton_end, const float unitlength,  bool &use_data, tbb::atomic<size_t> &nfilled,
              const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items, uint num_triangles)
{
    cudaCount(d_v0, d_v1, d_v2, d_tri_idx, d_nfilled, nfilled_idx, d_voxels, d_data,
                 morton_start,  morton_end, unitlength,  use_data, nfilled,
                 p_bbox_grid_min, p_bbox_grid_max, unit_div, delta_p, data_max_items, num_triangles);
}

bool first_cnt_time = true;

void voxelize_schwarz_count(TriReaderIter *reader, TriReaderIter *orig_reader,
                            uint64 *&d_data,
                             float3 *&d_v0, float3 *&d_v1, float3 *&d_v2,
                             voxel_t *&d_voxels, size_t data_max_items,
                             const mort_t morton_start, const mort_t morton_end,
                            const mort_t morton_length, const float unitlength,
                            voxel_t*&voxels, mort_t *&data, uint &data_size,
                            float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled,
                            uint nfilled_blocks,
                            uint nfilled_idx)
{
    vox_algo_timer.start();

    AABox<uivec3> p_bbox_grid;
    mortonDecode(morton_start, p_bbox_grid.min[2], p_bbox_grid.min[1], p_bbox_grid.min[0]);
    mortonDecode(morton_end - 1, p_bbox_grid.max[2], p_bbox_grid.max[1], p_bbox_grid.max[0]);


    // COMMON PROPERTIES FOR ALL TRIANGLES
    float unit_div = 1.0f / unitlength;
    vec3 delta_p = vec3(unitlength, unitlength, unitlength);

    if (first_cnt_time){
        first_cnt_time = false;
        cudaLoadFullTri(orig_reader, d_data, d_v0, d_v1, d_v2,voxels, d_voxels, morton_length, nfilled_blocks);
    }

    float3 cuda_delta_p;
    uint3 p_bbox_grid_min, p_bbox_grid_max;
    cudaLoadMem(reader, p_bbox_grid, delta_p, p_bbox_grid_min, p_bbox_grid_max, cuda_delta_p);


    runCount(d_v0, d_v1, d_v2, nfilled_idx, d_voxels, d_data,
            morton_start, morton_end, unitlength,
            use_data, nfilled, p_bbox_grid_min, p_bbox_grid_max,
            unit_div, cuda_delta_p, data_max_items, reader->triangles.size());

    vox_algo_timer.stop();


}

// Implementation of algorithm from http://research.michael-schwarz.com/publ/2010/vox/ (Schwarz & Seidel)
// Adapted for mortoncode -based subgrids
bool first_time = 1;
void voxelize_schwarz_method(TriReaderIter *reader, TriReaderIter *orig_reader,
                             uint64 *&d_data,
                             float3 *&d_v0, float3 *&d_v1, float3 *&d_v2,
                             voxel_t *&d_voxels, size_t data_max_items,
                             const mort_t morton_start, const mort_t morton_end, const mort_t morton_length,
                             const float unitlength, voxel_t*&voxels, mort_t *&data, uint &data_size, float sparseness_limit,
                             bool &use_data, tbb::atomic<size_t> &nfilled,
                             uint nfilled_blocks,
                             uint partition_idx,
                            uint idx, uint size)
{
    ec.chk("start voxelize_method");

    vox_algo_timer.start();

	// compute partition min and max in grid coords
	AABox<uivec3> p_bbox_grid;
	mortonDecode(morton_start, p_bbox_grid.min[2], p_bbox_grid.min[1], p_bbox_grid.min[0]);
	mortonDecode(morton_end - 1, p_bbox_grid.max[2], p_bbox_grid.max[1], p_bbox_grid.max[0]);



    // COMMON PROPERTIES FOR ALL TRIANGLES
    float unit_div = 1.0f / unitlength;
    vec3 delta_p = vec3(unitlength, unitlength, unitlength);

    if (first_time){
        first_time = false;
        //d_data should already be malloced
        //cudaMalloc((void**)&d_data, sizeof(uint64)*size);

        cudaMemset(d_nfilled, 0, sizeof(uint)*nfilled_blocks);
    }
    cudaMemset(d_data + idx, 0, sizeof(uint64)*size);

    float3 cuda_delta_p;
    uint3 p_bbox_grid_min, p_bbox_grid_max;
    cudaLoadMem(reader, p_bbox_grid, delta_p, p_bbox_grid_min, p_bbox_grid_max, cuda_delta_p);
    runCUDA(d_v0, d_v1, d_v2, d_voxels, d_data, reader->triangles.size(), morton_start, morton_end, unitlength, voxels, data, data_size, sparseness_limit, use_data, p_bbox_grid_min, p_bbox_grid_max, unit_div, cuda_delta_p, data_max_items, partition_idx, idx);

    vox_algo_timer.stop();
}

size_t voxelize_count_finalize(uint nfilled_blocks, std::vector<uint> &output)
{
    output.resize(nfilled_blocks);
    size_t tot_filled = cudaCountFinalize(nfilled_blocks, d_nfilled);
    cudaMemcpy((void**)&output[0], d_nfilled, sizeof(uint)*nfilled_blocks, cudaMemcpyDeviceToHost);

    return tot_filled;

}

uint voxelize_finalize(uint cnt, uint64 *d_data)
{
    return cudaFinalize(cnt, d_data);

}
