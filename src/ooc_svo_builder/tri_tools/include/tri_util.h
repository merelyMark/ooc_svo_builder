#ifndef TRI_UTIL_H_
#define TRI_UTIL_H_
#include <TriMesh.h>
#include <sstream>

using namespace std;
using namespace trimesh;

#define TRIANGLE_SIZE 10 // just the vertices


// Various helper methods / structs

// Custom value to string method to avoid C++11 dependency causing fopenmp problems in OSX
template <typename T>
string val_to_string( T Number ) {
	stringstream ss;
	ss << Number;
	return ss.str();
}

template <typename T>
struct AABox {
	T min;
	T max;
	AABox(): min(T()), max(T()){}
	AABox(T min, T max): min(min), max(max){}
    void clamp(const T &a, const T &b){


    }
};

struct Triangle {
	vec3 v0;
	vec3 v1;
        vec3 v2;
        int idx;
        Triangle(): v0(vec3()), v1(vec3()), v2(vec3()) {}
	Triangle(vec3 v0, vec3 v1, vec3 v2): v0(v0), v1(v1), v2(v2) {}
};
#endif
