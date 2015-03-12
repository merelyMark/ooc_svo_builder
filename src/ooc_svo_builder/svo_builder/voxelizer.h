#ifndef VOXELIZER_H_
#define VOXELIZER_H_

#include <tri_tools.h>
#include <TriReaderIter.h>
#include <tbb/atomic.h>
#include <tbb/concurrent_vector.h>
#include "globals.h"
#include "intersection.h"
#include "morton.h"
#include "VoxelData.h"

// Voxelization-related stuff
typedef Vec<3, unsigned int> uivec3;
using namespace std;

#define EMPTY_VOXEL 0
#define FULL_VOXEL 1
#define WORKING_VOXEL 2




void voxelize_schwarz_method(TriReaderIter &reader, const uint64_t morton_start, const uint64_t morton_end, const float unitlength, tbb::atomic<char>* voxels, tbb::concurrent_vector<uint64_t> &data, float sparseness_limit, bool &use_data, tbb::atomic<size_t> &nfilled);


#endif // VOXELIZER_H_
