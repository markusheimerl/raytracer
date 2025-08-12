#ifndef MESH_H
#define MESH_H

#include "triangle.h"
#include "math/ray.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Forward declarations
typedef struct BVH BVH;
typedef struct GPUBVHNode GPUBVHNode;

typedef struct {
    Triangle* triangles;
    size_t triangle_count;
    unsigned char* texture_data;
    int texture_width;
    int texture_height;
    BVH bvh;
    Transform transform;
} Mesh;

typedef struct GPUMesh {
    Triangle* d_triangles;
    size_t triangle_count;
    unsigned char* d_texture_data;
    int texture_width;
    int texture_height;
    GPUBVHNode* d_bvh_root;
    Transform transform;
} GPUMesh;

// Include after GPUMesh is defined
#include "accel/bvh.h"

// Mesh operations
Mesh create_mesh(const char* obj_filename, const char* texture_filename);
void set_mesh_position(Mesh* mesh, Vec3 position);
void set_mesh_rotation(Mesh* mesh, Vec3 rotation);
void destroy_mesh(Mesh* mesh);
Vec3 sample_mesh_texture(const Mesh* mesh, float u, float v);

// GPU mesh operations
void convert_mesh_to_gpu(const Mesh* cpu_mesh, GPUMesh* gpu_mesh);
__device__ Vec3 sample_mesh_texture_gpu(const GPUMesh* mesh, float u, float v);

#endif