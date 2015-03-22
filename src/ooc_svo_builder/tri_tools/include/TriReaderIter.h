#ifndef TRIREADERITER_H
#define TRIREADERITER_H
#include "TriReader.h"
#include <stdio.h>

using namespace std;
using namespace trimesh;

// A class to read triangles from a .tridata file.
class TriReaderIter : public TriReader{
public:
    TriReaderIter() : TriReader(){}
    TriReaderIter(const TriReader&tr) : TriReader(tr){}
    TriReaderIter(const std::string &filename, size_t n_triangles, size_t buffersize);
    Triangle getTriangle();
    void getTriangle(Triangle& t);
    void resetCount(){ n_served = 0;current_tri = 0;}

    virtual bool hasNext();
    ~TriReaderIter();
    std::vector<Triangle> triangles;
private:
    void fillBuffer();
};

inline TriReaderIter::TriReaderIter(const std::string &filename, size_t n_triangles, size_t buffersize):
    TriReader(filename, n_triangles, buffersize)
{

    triangles.reserve(n_triangles);
    fillBuffer();
    n_served = 0;
    current_tri = 0;
}



inline bool TriReaderIter::hasNext(){
    return (n_served < n_triangles);
}

inline void TriReaderIter::fillBuffer(){
    while(TriReader::hasNext()){
        Triangle t = TriReader::getTriangle();
        triangles.push_back(t);
    }

}

inline TriReaderIter::~TriReaderIter(){
}

inline Triangle TriReaderIter::getTriangle(){
    Triangle t = triangles[current_tri]; // assign triangle from buffer
    current_tri++; // set index for next triangle
    n_served++;
    return t;
}
inline void TriReaderIter::getTriangle(Triangle& t){

    t = getTriangle();
}

#endif // TRIREADERITER_H
