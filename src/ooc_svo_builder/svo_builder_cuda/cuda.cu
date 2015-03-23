#include <assert.h>
#include <helper_math.h>
#include "ErrorCheck.h"
#include "voxelizer.h"

__constant__ u_int32_t c_morton256_x[256];
__constant__ u_int32_t c_morton256_y[256];
__constant__ u_int32_t c_morton256_z[256];

typedef union {
  uint2 u2;
  long long unsigned int l;
} _uint64;



extern "C"
void cudaConstants(const uint *x, const uint *y, const uint *z)
{
    ErrorCheck ec;
    ec.chk("setupConst start");
    cudaMemcpyToSymbol(c_morton256_x, x, sizeof(u_int32_t)*256,0); ec.chk("setupConst: c_morton256_x");
    cudaMemcpyToSymbol(c_morton256_y, y, sizeof(u_int32_t)*256,0); ec.chk("setupConst: c_morton256_y");
    cudaMemcpyToSymbol(c_morton256_z, z, sizeof(u_int32_t)*256,0); ec.chk("setupConst: c_morton256_z");

}

#define EMPTY_VOXEL 0
#define FULL_VOXEL 1
#define WORKING_VOXEL 2


__device__ u_int64_t cuda_splitBy3(int a){
        u_int64_t x = a & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

__device__ u_int64_t cuda_mortonEncode_magicbits(unsigned int x, unsigned int y, unsigned int z){
        u_int64_t answer = 0;
    answer |= cuda_splitBy3(x) | cuda_splitBy3(y) << 1 | cuda_splitBy3(z) << 2;
    return answer;
}

__device__ u_int64_t cuda_mortonEncode_LUT(unsigned int x, unsigned int y, unsigned int z){
        u_int64_t answer = 0;
    answer =	c_morton256_z[(z >> 16) & 0xFF ] |
                c_morton256_y[(y >> 16) & 0xFF ] |
                c_morton256_x[(x >> 16) & 0xFF ];
    answer = answer << 48 |
                c_morton256_z[(z >> 8) & 0xFF ] |
                c_morton256_y[(y >> 8) & 0xFF ] |
                c_morton256_x[(x >> 8) & 0xFF ];
    answer = answer << 24 |
                c_morton256_z[(z) & 0xFF ] |
                c_morton256_y[(y) & 0xFF ] |
                c_morton256_x[(x) & 0xFF ];
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
        answer.min.x = fminf(v0.x,fminf(v1.x,v2.x));
        answer.min.y = fminf(v0.y,fminf(v1.y,v2.y));
        answer.min.z = fminf(v0.z,fminf(v1.z,v2.z));
        answer.max.x = fmaxf(v0.x,fmaxf(v1.x,v2.x));
        answer.max.y = fmaxf(v0.y,fmaxf(v1.y,v2.y));
        answer.max.z = fmaxf(v0.z,fmaxf(v1.z,v2.z));
        return answer;
}


template<bool COUNT_ONLY>
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

#if 0
    const int z = t_bbox_grid.min.z;
    const int y = t_bbox_grid.min.y;
    const int x = t_bbox_grid.min.x;
    const _uint64 index = get_key_morton(make_int4(x,y,z,0));//cuda_mortonEncode_for(x,y,z);
    int idx = atomicInc(&nfilled[0], 100000000);//nfilled++;
    data[idx] = index.l;
#else

    for (int i=0; i<idx_cnt; i++){
        const int z = t_bbox_grid.min.z + i / (bbox_size.y * bbox_size.x);
        const int rem = i % (bbox_size.y * bbox_size.x);
        const int y = t_bbox_grid.min.y + (rem / bbox_size.x);
        const int x = t_bbox_grid.min.x + (rem % bbox_size.x);

        const u_int64_t index = cuda_mortonEncode_LUT(z, y, x);
        //const u_int64_t index = cuda_mortonEncode_magicbits(z,y,x);//cuda_mortonEncode_for(x,y,z);
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

        if (!(((nDOTp + d1) * (nDOTp + d2) > 0.0f)
                || ((dot(n_xy_e0 , p_xy) + d_xy_e0) < 0.0f)
                || ((dot(n_xy_e1 , p_xy) + d_xy_e1) < 0.0f)
                || ((dot(n_xy_e2 , p_xy) + d_xy_e2) < 0.0f)
                || ((dot(n_yz_e0 , p_yz) + d_yz_e0) < 0.0f)
                || ((dot(n_yz_e1 , p_yz) + d_yz_e1) < 0.0f)
                || ((dot(n_yz_e2 , p_yz) + d_yz_e2) < 0.0f)
                || ((dot(n_zx_e0 , p_zx) + d_xz_e0) < 0.0f)
                || ((dot(n_zx_e1 , p_zx) + d_xz_e1) < 0.0f)
                || ((dot(n_zx_e2 , p_zx) + d_xz_e2) < 0.0f)
                )){
            if (atomicCAS(&voxels[index - morton_start], EMPTY_VOXEL,FULL_VOXEL) == EMPTY_VOXEL){
                uint idx = atomicInc(&nfilled[0], 1000000000000);//nfilled++;
                if (COUNT_ONLY == false){

                    //if (use_data){
                         data[idx] = index;
                    //}
                }
            }
        }

        __syncthreads();
    }
#endif
}

template<bool COUNT_ONLY>
__global__
void voxelize(const float3 *v0, const float3 *v1, const float3 *v2, const uint *tri_idx, const uint64 morton_start, const uint64 morton_end, const float unitlength, int* voxels, uint64 *data, float sparseness_limit, uint *nfilled,
               const uint3 p_bbox_grid_min, const uint3 p_bbox_grid_max, const float unit_div, const float3 delta_p,	size_t data_max_items, size_t num_triangles)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < num_triangles){
        uint idx = tri_idx[tid];
        voxelize_triangle<COUNT_ONLY>(v0[idx],v1[idx], v2[idx], morton_start, morton_end, unitlength, voxels, data, sparseness_limit, nfilled,
                          p_bbox_grid_min, p_bbox_grid_max, unit_div, delta_p, data_max_items);
    }
//    const int z = idx / (128*128);
//    const int rem = idx % (128*128);
//    const int y = (rem / 128);
//    const int x = (rem % 128);
//    data[idx] = cuda_mortonEncode_for(x,y,z);
}


#include <tbb/atomic.h>
#include <tbb/tbb.h>
extern "C"
void cudaRun(const float3* d_v0, const float3*d_v1, const float3*d_v2, uint *d_tri_idx, uint *d_nfilled, int *d_voxels, uint64 *d_data,
             const uint64 morton_start, const uint64 morton_end, const float unitlength, voxel_t *voxels, uint64 *data, uint &data_size, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled,
             const uint3 &p_bbox_grid_min, const uint3 &p_bbox_grid_max, const float unit_div, const float3 &delta_p,	size_t data_max_items, uint num_triangles)
{
    ErrorCheck ec;
    ec.chk("cudaRun");



    cudaMemset(d_nfilled,0, sizeof(uint));  ec.chk("memory nfilled");
    cudaMemset(d_voxels, EMPTY_VOXEL, sizeof(int)*(morton_end - morton_start)); ec.chk("memory voxels");

    //get count
    voxelize<true><<<5000,64>>>(d_v0, d_v1, d_v2, d_tri_idx, morton_start, morton_end, unitlength, d_voxels, d_data, use_data, d_nfilled,
                    p_bbox_grid_min, p_bbox_grid_max, unit_div, delta_p, data_max_items, num_triangles);

    cudaThreadSynchronize();

    ec.chk("voxelize count" );
    uint h_nfilled = 0;
    cudaMemcpy(&h_nfilled, d_nfilled, sizeof(uint), cudaMemcpyDeviceToHost);
    if (h_nfilled > 0){

        cudaMemset(d_data, 0, sizeof(uint64)*(morton_end - morton_start));
        cudaMemset(d_voxels, 0, sizeof(int)*(morton_end - morton_start));
        cudaMemset(d_nfilled, 0, sizeof(uint));

        //fill d_data
        voxelize<false><<<5000,64>>>(d_v0, d_v1, d_v2, d_tri_idx, morton_start, morton_end, unitlength, d_voxels, d_data, use_data, d_nfilled,
                        p_bbox_grid_min, p_bbox_grid_max, unit_div, delta_p, data_max_items, num_triangles);


        cudaMemcpy(&h_nfilled, d_nfilled, sizeof(uint), cudaMemcpyDeviceToHost);
        nfilled += h_nfilled;
        //cudaMemcpy(&voxels[0], d_voxels, sizeof(int) * (morton_end - morton_start), cudaMemcpyDeviceToHost);

        data_size = h_nfilled;
        cudaMemcpy(&data[0], d_data, h_nfilled*sizeof(uint64), cudaMemcpyDeviceToHost);
        cudaThreadSynchronize();
        ec.chk("voxelize output data");
    }


    ec.chk("final cuda check");
}
