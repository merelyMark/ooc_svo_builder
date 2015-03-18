#include "voxelizer.h"
#include <tbb/atomic.h>
#include <omp.h>
#include <typeinfo>
#include <nmmintrin.h>
#include <TriReaderIter.h>
#include "intersection.h"
#include "partitioner.h"

using namespace std;
using namespace trimesh;

#define X 0
#define Y 1
#define Z 2

typedef union{
    uint2 u2;
    mort_t l;
} cpu_uint64;

/************************************************************
 * From Bonsai treecode
 * ********************************************************/
uint2 cpu_dilate3(int value) {
  unsigned int x;
  uint2 key;

  // dilate first 10 bits

  x = value & 0x03FF;
  x = ((x << 16) + x) & 0xFF0000FF;
  x = ((x <<  8) + x) & 0x0F00F00F;
  x = ((x <<  4) + x) & 0xC30C30C3;
  x = ((x <<  2) + x) & 0x49249249;
  key.x = x;

  // dilate second 10 bits

  x = (value >> 10) & 0x03FF;
  x = ((x << 16) + x) & 0xFF0000FF;
  x = ((x <<  8) + x) & 0x0F00F00F;
  x = ((x <<  4) + x) & 0xC30C30C3;
  x = ((x <<  2) + x) & 0x49249249;
  key.y = x;

  return key;
}

//#if 0
//Morton order
 cpu_uint64 cpu_get_key_morton(int4 crd) {
  cpu_uint64 key, key1;
  key.u2  = cpu_dilate3(crd.x);

  key1.u2 = cpu_dilate3(crd.y);
  key.u2.x = key.u2.x | (key1.u2.x << 1);
  key.u2.y = key.u2.y | (key1.u2.y << 1);

  key1.u2 = cpu_dilate3(crd.z);
  key.u2.x = key.u2.x | (key1.u2.x << 2);
  key.u2.y = key.u2.y | (key1.u2.y << 2);

  return key;
}

bool compare_and_swap(tbb::atomic<voxel_t> *voxels, mort_t idx)
{
    return voxels[idx].compare_and_swap(FULL_VOXEL, EMPTY_VOXEL) == EMPTY_VOXEL;
}

template<char COUNT_ONLY, char CUDA_PARALLEL>
void voxelize_triangle(const Triangle &t,const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<voxel_t>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled, const AABox<uivec3> &p_bbox_grid, const float unit_div, const vec3 &delta_p,	size_t data_max_items, int tid = 0)

{
    // read triangle

    vec3 v0 = t.v0;
    vec3 v1 = t.v1;
    vec3 v2 = t.v2;

    if (use_data){
        if (data.size() > data_max_items){
            if (verbose){
                cout << "Sparseness optimization side-array overflowed, reverting to slower voxelization." << endl;
                cout << data.size() << " > " << data_max_items << endl;
            }
            use_data = false;
        }
    }


    // compute triangle bbox in world and grid
    const AABox<vec3> t_bbox_world = computeBoundingBox(t.v0, t.v1, t.v2);
    const ivec3 grid_min((int)(t_bbox_world.min[0] * unit_div),(int)(t_bbox_world.min[1] * unit_div),(int)(t_bbox_world.min[2] * unit_div));
    const ivec3 grid_max((int)(t_bbox_world.max[0] * unit_div),(int)(t_bbox_world.max[1] * unit_div),(int)(t_bbox_world.max[2] * unit_div));
    // clamp
    const ivec3 clamp_grid_min(clampval<int>(grid_min[0], p_bbox_grid.min[0], p_bbox_grid.max[0]),
            clampval<int>(grid_min[1], p_bbox_grid.min[1], p_bbox_grid.max[1]),
            clampval<int>(grid_min[2], p_bbox_grid.min[2], p_bbox_grid.max[2]));
    const ivec3 clamp_grid_max(clampval<int>(grid_max[0], p_bbox_grid.min[0], p_bbox_grid.max[0]),
            clampval<int>(grid_max[1], p_bbox_grid.min[1], p_bbox_grid.max[1]),
            clampval<int>(grid_max[2], p_bbox_grid.min[2], p_bbox_grid.max[2]));
    const AABox<ivec3> t_bbox_grid(clamp_grid_min, clamp_grid_max);

    // COMMON PROPERTIES FOR THE TRIANGLE
    const vec3 e0 = v1 - v0;
    const vec3 e1 = v2 - v1;
    const vec3 e2 = v0 - v2;
    vec3 to_normalize = e0 CROSS e1;
    const vec3 n = normalize(to_normalize); // triangle normal
    // PLANE TEST PROPERTIES
    const vec3 c = vec3(n[X] > 0 ? unitlength : 0.0f,
                        n[Y] > 0 ? unitlength : 0.0f,
                        n[Z] > 0 ? unitlength : 0.0f); // critical point
    const float d1 = n DOT(c - v0);
    const float d2 = n DOT((delta_p - c) - v0);

    // PROJECTION TEST PROPERTIES
    // XY plane
    const vec2 n_xy_e0 = n[Z] < 0.0f ? -1.0f * vec2(-1.0f*e0[Y], e0[X]) : vec2(-1.0f*e0[Y], e0[X]);
    const vec2 n_xy_e1 = n[Z] < 0.0f ? -1.0f * vec2(-1.0f*e1[Y], e1[X]) : vec2(-1.0f*e1[Y], e1[X]);
    const vec2 n_xy_e2 = n[Z] < 0.0f ? -1.0f * vec2(-1.0f*e2[Y], e2[X]) : vec2(-1.0f*e2[Y], e2[X]);

    const float d_xy_e0 = (-1.0f * (n_xy_e0 DOT vec2(v0[X], v0[Y]))) + max(0.0f, unitlength*n_xy_e0[0]) + max(0.0f, unitlength*n_xy_e0[1]);
    const float d_xy_e1 = (-1.0f * (n_xy_e1 DOT vec2(t.v1[X], t.v1[Y]))) + max(0.0f, unitlength*n_xy_e1[0]) + max(0.0f, unitlength*n_xy_e1[1]);
    const float d_xy_e2 = (-1.0f * (n_xy_e2 DOT vec2(t.v2[X], t.v2[Y]))) + max(0.0f, unitlength*n_xy_e2[0]) + max(0.0f, unitlength*n_xy_e2[1]);
    // YZ plane
    const vec2 n_yz_e0 = n[X] < 0.0f ? -1.0f * vec2(-1.0f*e0[Z], e0[Y]) : vec2(-1.0f*e0[Z], e0[Y]);
    const vec2 n_yz_e1 = n[X] < 0.0f ? -1.0f * vec2(-1.0f*e1[Z], e1[Y]) : vec2(-1.0f*e1[Z], e1[Y]);
    const vec2 n_yz_e2 = n[X] < 0.0f ? -1.0f * vec2(-1.0f*e2[Z], e2[Y]) : vec2(-1.0f*e2[Z], e2[Y]);

    const float d_yz_e0 = (-1.0f * (n_yz_e0 DOT vec2(t.v0[Y], t.v0[Z]))) + max(0.0f, unitlength*n_yz_e0[0]) + max(0.0f, unitlength*n_yz_e0[1]);
    const float d_yz_e1 = (-1.0f * (n_yz_e1 DOT vec2(t.v1[Y], t.v1[Z]))) + max(0.0f, unitlength*n_yz_e1[0]) + max(0.0f, unitlength*n_yz_e1[1]);
    const float d_yz_e2 = (-1.0f * (n_yz_e2 DOT vec2(t.v2[Y], t.v2[Z]))) + max(0.0f, unitlength*n_yz_e2[0]) + max(0.0f, unitlength*n_yz_e2[1]);
    // ZX plane
    const vec2 n_zx_e0 = n[Y] < 0.0f ? -1.0f * vec2(-1.0f*e0[X], e0[Z]) : vec2(-1.0f*e0[X], e0[Z]);
    const vec2 n_zx_e1 = n[Y] < 0.0f ? -1.0f * vec2(-1.0f*e1[X], e1[Z]) : vec2(-1.0f*e1[X], e1[Z]);
    const vec2 n_zx_e2 = n[Y] < 0.0f ? -1.0f * vec2(-1.0f*e2[X], e2[Z]) : vec2(-1.0f*e2[X], e2[Z]);

    const float d_xz_e0 = (-1.0f * (n_zx_e0 DOT vec2(t.v0[Z], t.v0[X]))) + max(0.0f, unitlength*n_zx_e0[0]) + max(0.0f, unitlength*n_zx_e0[1]);
    const float d_xz_e1 = (-1.0f * (n_zx_e1 DOT vec2(t.v1[Z], t.v1[X]))) + max(0.0f, unitlength*n_zx_e1[0]) + max(0.0f, unitlength*n_zx_e1[1]);
    const float d_xz_e2 = (-1.0f * (n_zx_e2 DOT vec2(t.v2[Z], t.v2[X]))) + max(0.0f, unitlength*n_zx_e2[0]) + max(0.0f, unitlength*n_zx_e2[1]);

    // test possible grid boxes for overlap
    const ivec3 bbox_size((t_bbox_grid.max[0] - t_bbox_grid.min[0] + 1), (t_bbox_grid.max[1] - t_bbox_grid.min[1] + 1), (t_bbox_grid.max[2] - t_bbox_grid.min[2] + 1));

#if 0
    const int z = t_bbox_grid.min[2];
    const int y = t_bbox_grid.min[1];
    const int x = t_bbox_grid.min[0];
    const cpu_uint64 index = cpu_get_key_morton(make_int4(x,y,z,0));//mortonEncode_LUT(z, y, x);
    nfilled++;
    data.push_back(index.l);

#else
//    for (int x=t_bbox_grid.min[0]; x<t_bbox_grid.max[0]+1; x++){
//    for (int y=t_bbox_grid.min[1]; y<t_bbox_grid.max[1]+1; y++){
//    for (int z=t_bbox_grid.min[2]; z<t_bbox_grid.max[2]+1; z++){
    const int idx_cnt =  bbox_size[0] * bbox_size[1] * bbox_size[2];
    for (int i=0; i<idx_cnt; i++){
        const int z = t_bbox_grid.min[2] + i / (bbox_size[1] * bbox_size[0]);
        const int rem = i % (bbox_size[1] * bbox_size[0]);
        const int y = t_bbox_grid.min[1] + (rem / bbox_size[0]);
        const int x = t_bbox_grid.min[0] + (rem % bbox_size[0]);
        const cpu_uint64 index = cpu_get_key_morton(make_int4(z,y,x,0));//mortonEncode_LUT(z, y, x);
        assert(index.l < morton_end);
        // TRIANGLE PLANE THROUGH BOX TEST
        const vec3 p = vec3(x*unitlength, y*unitlength, z*unitlength);
        const float nDOTp = n DOT p;

        // PROJECTION TESTS
        // XY
        const vec2 p_xy = vec2(p[X], p[Y]);
        // YZ
        const vec2 p_yz = vec2(p[Y], p[Z]);
        // XZ
        const vec2 p_zx = vec2(p[Z], p[X]);

        if (!((nDOTp + d1) * (nDOTp + d2) > 0.0f
                || (((n_xy_e0 DOT p_xy) + d_xy_e0) < 0.0f)
                || (((n_xy_e1 DOT p_xy) + d_xy_e1) < 0.0f)
                || (((n_xy_e2 DOT p_xy) + d_xy_e2) < 0.0f)
                || (((n_yz_e0 DOT p_yz) + d_yz_e0) < 0.0f)
                || (((n_yz_e1 DOT p_yz) + d_yz_e1) < 0.0f)
                || (((n_yz_e2 DOT p_yz) + d_yz_e2) < 0.0f)
                || (((n_zx_e0 DOT p_zx) + d_xz_e0) < 0.0f)
                || (((n_zx_e1 DOT p_zx) + d_xz_e1) < 0.0f)
                || (((n_zx_e2 DOT p_zx) + d_xz_e2) < 0.0f)
                )){
            if (COUNT_ONLY == 0){
                if (compare_and_swap(voxels, index.l - morton_start)){
                    if (use_data){
                        nfilled++;
                        data.push_back(index.l);
                    }
                }
            }
        }
//    }
//    }
//    }
    }
#endif
}

void runCPUCUDAStyle(TriReaderIter *reader, const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<voxel_t>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled, const AABox<uivec3> &p_bbox_grid, const float unit_div, const vec3 &delta_p,	size_t data_max_items)
{
    //this is
#pragma omp parallel for
    for (int i=0; i<reader->triangles.size(); i++){
        Triangle t = reader->triangles[i];

        voxelize_triangle<1,1>(t, morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);
    }

    data.resize(nfilled);
    nfilled = 0;
    memset(voxels, EMPTY_VOXEL, (morton_end - morton_start)*sizeof(char));

#pragma omp parallel for
    for (int i=0; i<reader->triangles.size(); i++){
        Triangle t = reader->triangles[i];
        voxelize_triangle<0,1>(t, morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);

    }

}

void runCPUParallel(TriReaderIter *reader, const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<voxel_t>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled, const AABox<uivec3> &p_bbox_grid, const float unit_div, const vec3 &delta_p,	size_t data_max_items)
{
//#pragma omp parallel for
    for (int i=0; i<reader->triangles.size(); i++){
        Triangle t = reader->triangles[i];
        voxelize_triangle<0,0>(t, morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);
//        const int z = i / (128*128);
//        const int rem = i % (128*128);
//        const int y = (rem / 128);
//        const int x = (rem % 128);
//        data.push_back(mortonEncode_for(x,y,z));

    }
}

void runCUDA(TriReaderIter *reader, const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<voxel_t>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled, const AABox<uivec3> &p_bbox_grid, const float unit_div, const vec3 &delta_p,	size_t data_max_items)
{
    vector<Triangle> tris = reader->triangles;
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

    float3 *d_v0, *d_v1, *d_v2;
    cudaMalloc((void**)&d_v0, tris.size() * sizeof(float3));
    cudaMalloc((void**)&d_v1, tris.size() * sizeof(float3));
    cudaMalloc((void**)&d_v2, tris.size() * sizeof(float3));

    cudaMemcpy(d_v0, v0, sizeof(float3)*tris.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1, v1, sizeof(float3)*tris.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2, sizeof(float3)*tris.size(), cudaMemcpyHostToDevice);

    uint3 p_bbox_grid_min, p_bbox_grid_max;
    p_bbox_grid_min.x =  p_bbox_grid.min[0];
    p_bbox_grid_min.y =  p_bbox_grid.min[1];
    p_bbox_grid_min.z =  p_bbox_grid.min[2];
    p_bbox_grid_max.x =  p_bbox_grid.max[0];
    p_bbox_grid_max.y =  p_bbox_grid.max[1];
    p_bbox_grid_max.z =  p_bbox_grid.max[2];

    float3 cuda_delta_p;
    cuda_delta_p.x = delta_p[0];
    cuda_delta_p.y = delta_p[1];
    cuda_delta_p.z = delta_p[2];

    cudaRun(d_v0, d_v1, d_v2, morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid_min, p_bbox_grid_max, unit_div, cuda_delta_p, data_max_items, tris.size());

    cudaFree(d_v0);
    cudaFree(d_v1);
    cudaFree(d_v2);
}

// Implementation of algorithm from http://research.michael-schwarz.com/publ/2010/vox/ (Schwarz & Seidel)
// Adapted for mortoncode -based subgrids

void voxelize_schwarz_method(TriReaderIter *reader, const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<voxel_t>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled) {

    vox_algo_timer.start();
	memset(voxels, EMPTY_VOXEL, (morton_end - morton_start)*sizeof(char));
	data.clear();

	// compute partition min and max in grid coords
	AABox<uivec3> p_bbox_grid;
	mortonDecode(morton_start, p_bbox_grid.min[2], p_bbox_grid.min[1], p_bbox_grid.min[0]);
	mortonDecode(morton_end - 1, p_bbox_grid.max[2], p_bbox_grid.max[1], p_bbox_grid.max[0]);

	// compute maximum grow size for data array

    size_t data_max_items;
	if (use_data){
        mort_t max_bytes_data = (mort_t) (((morton_end - morton_start)*sizeof(char)) * sparseness_limit);

        data_max_items = max_bytes_data / sizeof(mort_t);
    }


    // COMMON PROPERTIES FOR ALL TRIANGLES
    float unit_div = 1.0f / unitlength;
    vec3 delta_p = vec3(unitlength, unitlength, unitlength);

    // voxelize every triangle
    //while (reader.hasNext()) {
//    std::vector<Triangle>::iterator iter;
//    for (iter = reader.triangles.begin();
//         iter != reader.triangles.end(); ++iter){

    //runCPUCUDAStyle(reader,morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);
    runCPUParallel(reader,morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);
    //runCUDA(reader,morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);

    vox_algo_timer.stop();
}

