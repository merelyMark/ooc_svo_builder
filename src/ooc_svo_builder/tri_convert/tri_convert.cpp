#include <TriMesh.h>
#include <vector>
#include <string>
#include <sstream>
#include <tri_tools.h>
#include "tri_convert_util.h"

using namespace std;
using namespace trimesh;

// Program version
string version = "1.2";

// Program parameters
string filename = "";
bool recompute_normals = false;
vec3 fixed_color = vec3(1.0f, 1.0f, 1.0f);

void printInfo(){
	cout << "-------------------------------------------------------------" << endl;
	cout << "Tri Converter " << version << " - BINARY VOXELIZATION"<< endl;

#if defined(_WIN32) || defined(_WIN64)
	cout << "Windows ";
#endif
#ifdef __linux__
	cout << "Linux ";
#endif
#ifdef _WIN64
	cout << "64-bit version" << endl;
#endif
	cout << "Jeroen Baert - jeroen.baert@cs.kuleuven.be - www.forceflow.be" << endl;
	cout << "-------------------------------------------------------------" << endl << endl;
}

void printHelp(){
	std::cout << "Example: tri_convert -f /home/jeroen/bunny.ply" << endl;
	std::cout << "" << endl;
	std::cout << "All available program options:" << endl;
	std::cout << "" << endl;
	std::cout << "-f <filename>         Path to a model input file (.ply, .obj, .3ds, .sm, .ray or .off)." << endl;
	std::cout << "-r                    Recompute face normals." << endl;
	std::cout << "-h                    Print help and exit." << endl;
}

void printInvalid(){
	std::cout << "Not enough or invalid arguments, please try again.\n" << endl; 
	printHelp();
}

void parseProgramParameters(int argc, char* argv[]){
	// Input argument validation
	if(argc<3){ // not enough arguments
		printInvalid(); exit(0);
	} 
	for (int i = 1; i < argc; i++) {
			if (string(argv[i]) == "-f") {
				filename = argv[i + 1]; 
				i++;
			} else if (string(argv[i]) == "-r") {
				recompute_normals = true;
			} else if(string(argv[i]) == "-h") {
				printHelp(); exit(0);
			} else {
				printInvalid(); exit(0);
			}
	}
	cout << "  filename: " << filename << endl;
	cout << "  recompute normals: " << recompute_normals << endl;
}

int main(int argc, char *argv[]){
	printInfo();

	// Parse parameters
	parseProgramParameters(argc,argv);

	// Read mesh
	TriMesh *themesh = TriMesh::read(filename.c_str());
	themesh->need_faces(); // unpack triangle strips so we have faces
	themesh->need_bbox(); // compute the bounding box
	AABox<vec3> mesh_bbox = createMeshBBCube(themesh); // pad the mesh BBOX out to be a cube

	// Moving mesh to origin
	cout << "Moving mesh to origin ... "; 
	Timer timer = Timer();
	for(size_t i = 0; i < themesh->vertices.size() ; i++){
		themesh->vertices[i] = themesh->vertices[i] - mesh_bbox.min;
	}
	cout << "done in " << timer.getTotalTimeSeconds() << " s." << endl;

	// Write mesh to format we can stream in
	string base = filename.substr(0,filename.find_last_of("."));
	std::string tri_header_out_name = base + string(".tri");
	std::string tri_out_name = base + string(".tridata");

	FILE* tri_out = fopen(tri_out_name.c_str(), "wb");

	cout << "Writing mesh triangles ... "; timer.reset();
	Triangle t = Triangle();
	// Write all triangles to data file
	for(size_t i = 0; i < themesh->faces.size(); i++){
		t.v0 = themesh->vertices[themesh->faces[i][0]];
		t.v1 = themesh->vertices[themesh->faces[i][1]];
		t.v2 = themesh->vertices[themesh->faces[i][2]];
		writeTriangle(tri_out,t);
	}
	cout << "done in " << timer.getTotalTimeSeconds() << " ms." << endl;

	// Prepare tri_info and write header
	cout << "Writing header to " << tri_header_out_name << " ... " << endl;
	TriInfo tri_info;
	tri_info.version = 1;
	tri_info.mesh_bbox = mesh_bbox;
	tri_info.n_triangles = themesh->faces.size();
	tri_info.geometry_only = 1;
	writeTriHeader(tri_header_out_name, tri_info);
	tri_info.print();
	cout << "Done." << endl;
}
