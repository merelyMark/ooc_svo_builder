#include "voxelizer.h"
#include <tbb/atomic.h>
#include <omp.h>
#include <typeinfo>
#include <nmmintrin.h>
#include <TriReaderIter.h>
#include "intersection.h"
#include "partitioner.h"
#include <vector>
using namespace std;
using namespace trimesh;

#define X 0
#define Y 1
#define Z 2
uint *d_nfilled;
uint64 *d_data;
uint *d_tri_idx;

void cudaLoadMem(TriReaderIter *reader, const AABox<uivec3> &p_bbox_grid, const vec3 &delta_p,
                 uint3 &p_bbox_grid_min, uint3 &p_bbox_grid_max,
                 float3 &cuda_delta_p)
{
    vector<Triangle> tris = reader->triangles;

    uint *tri_idx = new uint[tris.size()];

    for (int i=0; i<tris.size(); i++){
        tri_idx[i] = tris[i].idx;
    }


    cudaMemcpy(d_tri_idx, tri_idx, sizeof(uint)*tris.size(), cudaMemcpyHostToDevice);

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

void cudaLoadFullTri(TriReaderIter *orig_reader,
                 float3 *&d_v0, float3 *&d_v1, float3 *&d_v2, voxel_t *&h_voxels, voxel_t *&d_voxels,
                     mort_t morton_length)
{
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

    cudaMalloc( (void**) &d_nfilled, sizeof(uint));
    cudaMalloc( (void**) &d_voxels, sizeof(voxel_t)*morton_length);
    cudaMalloc( (void**) &d_data, sizeof(uint64)*morton_length);

    cudaMalloc((void**)&d_tri_idx, tris.size() * sizeof(uint));

}


void runCUDA(float3 *d_v0, float3 *d_v1, float3 *d_v2, voxel_t *d_voxels, uint64 *d_data,
             uint num_tris, const mort_t morton_start, const mort_t morton_end, const float unitlength, voxel_t * voxels, mort_t *data, uint &data_size, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled, uint3 &p_bbox_grid_min, uint3 &p_bbox_grid_max, const float unit_div, float3 &cuda_delta_p,	size_t data_max_items)
{


    cudaRun(d_v0, d_v1, d_v2, d_tri_idx, d_nfilled, d_voxels, d_data, morton_start, morton_end, unitlength, voxels, data, data_size, sparseness_limit, use_data, nfilled, p_bbox_grid_min, p_bbox_grid_max, unit_div, cuda_delta_p, data_max_items, num_tris);

}

// Implementation of algorithm from http://research.michael-schwarz.com/publ/2010/vox/ (Schwarz & Seidel)
// Adapted for mortoncode -based subgrids

bool first_time = true;
void voxelize_schwarz_method(TriReaderIter *reader, TriReaderIter *orig_reader,
                             float3 *&d_v0, float3 *&d_v1, float3 *&d_v2,
                             voxel_t *&d_voxels, size_t data_max_items,
                             const mort_t morton_start, const mort_t morton_end, const mort_t morton_length, const float unitlength, voxel_t*&voxels, mort_t *&data, uint &data_size, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled)
{

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
        cudaLoadFullTri(orig_reader, d_v0, d_v1, d_v2, voxels, d_voxels, morton_length);
    }

    float3 cuda_delta_p;
    uint3 p_bbox_grid_min, p_bbox_grid_max;
    cudaLoadMem(reader, p_bbox_grid, delta_p, p_bbox_grid_min, p_bbox_grid_max, cuda_delta_p);
    runCUDA(d_v0, d_v1, d_v2, d_voxels, d_data, reader->triangles.size(), morton_start, morton_end, unitlength, voxels, data, data_size, sparseness_limit, use_data, nfilled, p_bbox_grid_min, p_bbox_grid_max, unit_div, cuda_delta_p, data_max_items);

    vox_algo_timer.stop();
}

