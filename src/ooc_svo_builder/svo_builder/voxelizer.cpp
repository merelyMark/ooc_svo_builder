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

bool compare_and_swap(tbb::atomic<voxel_t> *voxels, mort_t idx)
{
    //tbb does not have atomic bit operations: but c++11 does
    //but version 6.0 of CUDA does not support c++11. *sigh*
    //so I can't try my idea for atomicAnd and atomicCAS
    //for CUDA char atomicCAS

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
            //if (verbose){
                cout << "\t Sparseness optimization side-array overflowed, reverting to slower voxelization." << endl;
                cout << "\t" << data.size() << " > " << data_max_items << endl;
            //}
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

    for (int x=t_bbox_grid.min[0]; x<t_bbox_grid.max[0]+1; x++){
    for (int y=t_bbox_grid.min[1]; y<t_bbox_grid.max[1]+1; y++){
    for (int z=t_bbox_grid.min[2]; z<t_bbox_grid.max[2]+1; z++){

        const uint64 index = mortonEncode_LUT(z, y, x);
        if (voxels[ index - morton_start ] == FULL_VOXEL){continue;}
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

        if (!(((nDOTp + d1) * (nDOTp + d2) > 0.0f)
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
                if (compare_and_swap(voxels, index - morton_start)){
                //voxels[ index - morton_start ] = FULL_VOXEL;
                    if (use_data){
                        nfilled++;
                        data.push_back(index);
                    }
                }
            }
        }
    }
    }
    }
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

void runCPUParallel(TriReaderIter *reader, TriReaderIter *orig_reader, const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<voxel_t>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled, const AABox<uivec3> &p_bbox_grid, const float unit_div, const vec3 &delta_p,	size_t data_max_items)
{
    int num_threads = 0;
#pragma omp parallel
    num_threads = omp_get_num_threads();
    size_t start_idx[num_threads];
    size_t num_tris_per_thread = reader->triangles.size()/num_threads + 1;

    for (int i=0; i<num_threads; i++){
        start_idx[i] = num_tris_per_thread * i;
    }
#pragma omp parallel
    {
        const int start = start_idx[omp_get_thread_num()];
        const int end = std::min(reader->triangles.size(), start_idx[omp_get_thread_num()]+num_tris_per_thread);

        for (int i=start; i<end; i++){
            int idx = reader->triangles[i].idx;
            Triangle t = orig_reader->triangles[idx];
            voxelize_triangle<0,0>(t, morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);
        }
    }
}


// Implementation of algorithm from http://research.michael-schwarz.com/publ/2010/vox/ (Schwarz & Seidel)
// Adapted for mortoncode -based subgrids
bool first_time = false;
void voxelize_schwarz_method(TriReaderIter *reader, TriReaderIter *orig_reader, const mort_t morton_start, const mort_t morton_end, const float unitlength, tbb::atomic<voxel_t>* voxels, tbb::concurrent_vector<mort_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled) {

    vox_algo_timer.start();
    orig_reader->resetCount();

	memset(voxels, EMPTY_VOXEL, (morton_end - morton_start)*sizeof(char));

    if (first_time){
        first_time = true;
    }
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
    runCPUParallel(reader, orig_reader, morton_start, morton_end, unitlength, voxels, data, sparseness_limit, use_data, nfilled, p_bbox_grid, unit_div, delta_p, data_max_items);

    vox_algo_timer.stop();
}

