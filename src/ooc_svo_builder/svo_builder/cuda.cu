
#include <helper_math.h>

typedef unsigned long long int uint64;
#define EMPTY_VOXEL 0
#define FULL_VOXEL 1
#define WORKING_VOXEL 2


//Let's go with magic bits instead of LUT
// VERSION WITH MAGIC BITS
// -----------------------

__device__ uint64 splitBy3(int a){
        uint64 x = a & 0x1fffff;
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8) & 0x100f00f00f00f00f;
        x = (x | x << 4) & 0x10c30c30c30c30c3;
        x = (x | x << 2) & 0x1249249249249249;
        return x;
}

__device__
uint64 mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
        uint64 answer = 0;
        answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
        return answer;
}

template <typename T>
struct CAABox {
    T min;
    T max;
};

template<typename T>
__device__
CAABox<T> make_CAABox(const T &min, const T &max){CAABox<T> tmp; tmp.min = min; tmp.max = max; return tmp;}

template <typename T>
__device__ T d_clampval(const T& value, const T& low, const T& high) {
  return value < low ? low : (value > high ? high : value);
}

__device__
CAABox<float3> cudaComputeBoundingBox(const float3 &v0, const float3 &v1, const float3 &v2){
        CAABox<float3> answer;
        answer.min.x = min(v0.x,min(v1.x,v2.x));
        answer.min.y = min(v0.y,min(v1.y,v2.y));
        answer.min.z = min(v0.z,min(v1.z,v2.z));
        answer.max.x = max(v0.x,max(v1.x,v2.x));
        answer.max.y = max(v0.y,max(v1.y,v2.y));
        answer.max.z = max(v0.z,max(v1.z,v2.z));
        return answer;
}


template<char COUNT_ONLY>
__device__
void voxelize_triangle(float3 v0, float3 v1, float3 v2,const uint64 morton_start, const uint64 morton_end, const float unitlength, int* voxels, uint64 *data, float sparseness_limit, uint *nfilled,
                       const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items)

{
    // read triangle



//    if (use_data){
//        if (data.size() > data_max_items){
//            if (verbose){
//                cout << "Sparseness optimization side-array overflowed, reverting to slower voxelization." << endl;
//                cout << data.size() << " > " << data_max_items << endl;
//            }
//            use_data = false;
//        }
//    }


    // compute triangle bbox in world and grid
    const CAABox<float3> t_bbox_world = cudaComputeBoundingBox(v0, v1, v2);
    const int3 grid_min = make_int3((int)(t_bbox_world.min.x * unit_div),(int)(t_bbox_world.min.y * unit_div),(int)(t_bbox_world.min.z * unit_div));
    const int3 grid_max = make_int3((int)(t_bbox_world.max.x * unit_div),(int)(t_bbox_world.max.y * unit_div),(int)(t_bbox_world.max.z * unit_div));
    // clamp2
    const int3 clamp_grid_min = make_int3(d_clampval<int>(grid_min.x, p_bbox_grid_min.x, p_bbox_grid_max.x),
            d_clampval<int>(grid_min.y, p_bbox_grid_min.y, p_bbox_grid_max.y),
            d_clampval<int>(grid_min.z, p_bbox_grid_min.z, p_bbox_grid_max.z));
    const int3 clamp_grid_max = make_int3(d_clampval<int>(grid_max.x, p_bbox_grid_min.x, p_bbox_grid_max.x),
            d_clampval<int>(grid_max.y, p_bbox_grid_min.y, p_bbox_grid_max.y),
            d_clampval<int>(grid_max.z, p_bbox_grid_min.z, p_bbox_grid_max.z));
    const CAABox<int3> t_bbox_grid = make_CAABox<int3>(clamp_grid_min, clamp_grid_max);

    // COMMON PROPERTIES FOR THE TRIANGLE
    const float3 e0 = v1 - v0;
    const float3 e1 = v2 - v1;
    const float3 e2 = v0 - v2;
    float3 to_normalize = cross(e0,e1);
    const float3  n = normalize(to_normalize); // triangle normal
    // PLANE TEST PROPERTIES
    const float3 c = make_float3(n.x > 0 ? unitlength : 0.0f,
                        n.y > 0 ? unitlength : 0.0f,
                        n.z > 0 ? unitlength : 0.0f); // critical point
    const float d1 = dot(n , c - v0);
    const float d2 = dot(n, ((delta_p - c) - v0));
    // PROJECTION TEST PROPERTIES
    // XY plane
    const float2 n_xy_e0 = n.z < 0.0f ? -1.0f * make_float2(-1.0f*e0.y, e0.x) : make_float2(-1.0f*e0.y, e0.x);
    const float2 n_xy_e1 = n.z < 0.0f ? -1.0f * make_float2(-1.0f*e1.y, e1.x) : make_float2(-1.0f*e1.y, e1.x);
    const float2 n_xy_e2 = n.z < 0.0f ? -1.0f * make_float2(-1.0f*e2.y, e2.x) : make_float2(-1.0f*e2.y, e2.x);

    const float d_xy_e0 = (-1.0f * dot(n_xy_e0, make_float2(v0.x, v0.y))) + max(0.0f, unitlength*n_xy_e0.x) + max(0.0f, unitlength*n_xy_e0.y);
    const float d_xy_e1 = (-1.0f * dot(n_xy_e1, make_float2(v1.x, v1.y))) + max(0.0f, unitlength*n_xy_e1.x) + max(0.0f, unitlength*n_xy_e1.y);
    const float d_xy_e2 = (-1.0f * dot(n_xy_e2, make_float2(v2.x, v2.y))) + max(0.0f, unitlength*n_xy_e2.x) + max(0.0f, unitlength*n_xy_e2.y);
    // YZ plane
    const float2 n_yz_e0 = n.x < 0.0f ? -1.0f * make_float2(-1.0f*e0.z, e0.y) : make_float2(-1.0f*e0.z, e0.y);
    const float2 n_yz_e1 = n.x < 0.0f ? -1.0f * make_float2(-1.0f*e1.z, e1.y) : make_float2(-1.0f*e1.z, e1.y);
    const float2 n_yz_e2 = n.x < 0.0f ? -1.0f * make_float2(-1.0f*e2.z, e2.y) : make_float2(-1.0f*e2.z, e2.y);

    const float d_yz_e0 = (-1.0f * dot(n_yz_e0, make_float2(v0.y, v0.z))) + max(0.0f, unitlength*n_yz_e0.x) + max(0.0f, unitlength*n_yz_e0.y);
    const float d_yz_e1 = (-1.0f * dot(n_yz_e1, make_float2(v1.y, v1.z))) + max(0.0f, unitlength*n_yz_e1.x) + max(0.0f, unitlength*n_yz_e1.y);
    const float d_yz_e2 = (-1.0f * dot(n_yz_e2, make_float2(v2.y, v2.z))) + max(0.0f, unitlength*n_yz_e2.x) + max(0.0f, unitlength*n_yz_e2.y);
    // ZX plane
    const float2 n_zx_e0 = n.y < 0.0f ? -1.0f * make_float2(-1.0f*e0.x, e0.z) : make_float2(-1.0f*e0.x, e0.z);
    const float2 n_zx_e1 = n.y < 0.0f ? -1.0f * make_float2(-1.0f*e1.x, e1.z) : make_float2(-1.0f*e1.x, e1.z);
    const float2 n_zx_e2 = n.y < 0.0f ? -1.0f * make_float2(-1.0f*e2.x, e2.z) : make_float2(-1.0f*e2.x, e2.z);

    const float d_xz_e0 = (-1.0f * dot(n_zx_e0, make_float2(v0.z, v0.x))) + max(0.0f, unitlength*n_zx_e0.x) + max(0.0f, unitlength*n_zx_e0.y);
    const float d_xz_e1 = (-1.0f * dot(n_zx_e1, make_float2(v1.z, v1.x))) + max(0.0f, unitlength*n_zx_e1.x) + max(0.0f, unitlength*n_zx_e1.y);
    const float d_xz_e2 = (-1.0f * dot(n_zx_e2, make_float2(v2.z, v2.x))) + max(0.0f, unitlength*n_zx_e2.x) + max(0.0f, unitlength*n_zx_e2.y);

    // test possible grid boxes for overlap
    const int3 bbox_size = make_int3((t_bbox_grid.max.x - t_bbox_grid.min.x + 1), (t_bbox_grid.max.y - t_bbox_grid.min.y + 1), (t_bbox_grid.max.z - t_bbox_grid.min.z + 1));

    const int idx_cnt =  bbox_size.x * bbox_size.y * bbox_size.z;
//    for (int x = t_bbox_grid.min.x; x <= t_bbox_grid.max.x; x++){
//        for (int y = t_bbox_grid.min.y; y <= t_bbox_grid.max.y; y++){
//            for (int z = t_bbox_grid.min.z; z <= t_bbox_grid.max.z;){
    for (int i=0; i<idx_cnt;){
        const int z = t_bbox_grid.min.z + i / (bbox_size.y * bbox_size.x);
        const int rem = i % (bbox_size.y * bbox_size.x);
        const int y = t_bbox_grid.min.y + (rem / bbox_size.x);
        const int x = t_bbox_grid.min.x + (rem % bbox_size.x);

        const uint64 index = mortonEncode_magicbits(z, y, x);
        //if (voxels[index - morton_start].compare_and_swap(WORKING_VOXEL, EMPTY_VOXEL) != WORKING_VOXEL){
        if (atomicCAS(&voxels[index - morton_start], WORKING_VOXEL, EMPTY_VOXEL) != WORKING_VOXEL){

            if (voxels[index - morton_start] != FULL_VOXEL){
                // TRIANGLE PLANE THROUGH BOX TEST
                const float3  p = make_float3(x*unitlength, y*unitlength, z*unitlength);
                const float nDOTp = dot(n , p);

                // PROJECTION TESTS
                // XY
                const float2 p_xy = make_float2(p.x, p.y);
                // YZ
                const float2 p_yz = make_float2(p.y, p.z);
                // XZ
                const float2 p_zx = make_float2(p.z, p.x);

                if ((nDOTp + d1) * (nDOTp + d2) > 0.0f
                        || ((dot(n_xy_e0 , p_xy) + d_xy_e0) < 0.0f)
                        || ((dot(n_xy_e1 , p_xy) + d_xy_e1) < 0.0f)
                        || ((dot(n_xy_e2 , p_xy) + d_xy_e2) < 0.0f)
                        || ((dot(n_yz_e0 , p_yz) + d_yz_e0) < 0.0f)
                        || ((dot(n_yz_e1 , p_yz) + d_yz_e1) < 0.0f)
                        || ((dot(n_yz_e2 , p_yz) + d_yz_e2) < 0.0f)
                        || ((dot(n_zx_e0 , p_zx) + d_xz_e0) < 0.0f)
                        || ((dot(n_zx_e1 , p_zx) + d_xz_e1) < 0.0f)
                        || ((dot(n_zx_e2 , p_zx) + d_xz_e2) < 0.0f)
                        ){
                    voxels[index - morton_start] = EMPTY_VOXEL;
                }
                else{
                    size_t idx = atomicInc(&nfilled[0], 1000000000);//nfilled++;
                    if (COUNT_ONLY == 0){

                        //if (use_data){
                             data[idx] = index;
                        //}

                    }
                    voxels[index - morton_start] = FULL_VOXEL;
                }
            } // else, it's already marked, continue
            i++;
        }
        __syncthreads();
    }

}


__global__
void voxelize(const float3 *v0, const float3 *v1, const float3 *v2,const uint64 morton_start, const uint64 morton_end, const float unitlength, int* voxels, uint64 *data, float sparseness_limit, uint *nfilled,
               const uint3 p_bbox_grid_min, const uint3 p_bbox_grid_max, const float unit_div, const float3 delta_p,	size_t data_max_items, size_t num_triangles)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_triangles){
        voxelize_triangle<1>(v0[idx],v1[idx], v2[idx], morton_start, morton_end, unitlength, voxels, data, sparseness_limit, nfilled,
                          p_bbox_grid_min, p_bbox_grid_max, unit_div, delta_p, data_max_items);
    }
}


#include <tbb/atomic.h>
#include <tbb/tbb.h>
extern "C"
void cudaRun(const float3* d_v0, const float3*d_v1, const float3*d_v2,const uint64 morton_start, const uint64 morton_end, const float unitlength, tbb::atomic<char> *voxels, tbb::concurrent_vector<uint64> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled,
             const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items, size_t num_triangles)
{
    int *d_voxels, *h_voxels;
    cudaMalloc( (void**) &d_voxels, sizeof(int)*(morton_end - morton_start));
    h_voxels = new int[morton_end - morton_start];

    uint64 *d_data, *h_data;
    cudaMalloc( (void**) &d_data, sizeof(uint64)*data.size());
    h_data = new uint64[data.size()];

    uint *d_nfilled;
    cudaMalloc( (void**) &d_nfilled, sizeof(uint));

    cudaMemset(d_nfilled,0, sizeof(uint));
    cudaMemset(d_voxels, 0, sizeof(int)*(morton_end - morton_start));

    //get count
    voxelize<<<1024,1024>>>(d_v0, d_v1, d_v2, morton_start, morton_end, unitlength, d_voxels, d_data, use_data, d_nfilled,
                    p_bbox_grid_min, p_bbox_grid_max, unit_div, delta_p, data_max_items, num_triangles);

    cudaThreadSynchronize();
    cudaMemcpy(&nfilled, d_nfilled, sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_voxels, d_voxels, sizeof(int) * (morton_end - morton_start), cudaMemcpyDeviceToHost);
    for (int i=0; i<(morton_end - morton_start); i++){
        voxels[i] = (char)h_voxels[i];
    }

    cudaMemcpy(h_data, d_data, data.size()*sizeof(uint64), cudaMemcpyDeviceToHost);
    for (int i=0; i<data.size(); i++){
        data[i] = h_data[i];
    }


    cudaFree(d_voxels);
    cudaFree(d_data);
    cudaFree(d_nfilled);
    delete h_voxels;
    delete h_data;

}
