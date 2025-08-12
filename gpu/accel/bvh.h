#ifndef BVH_H
#define BVH_H

#include "geometry/aabb.h"
#include "geometry/triangle.h"
#include <stdlib.h>

// Forward declarations
typedef struct GPUMesh GPUMesh;

typedef struct BVHNode {
    AABB bounds;
    struct BVHNode* left;
    struct BVHNode* right;
    int start_idx;
    int triangle_count;
} BVHNode;

typedef struct {
    BVHNode* root;
    Triangle* triangles;
    size_t triangle_count;
} BVH;

typedef struct {
    AABB bounds;
    int left_idx;   // -1 if leaf
    int right_idx;  // -1 if leaf
    int start_idx;
    int triangle_count;
} GPUBVHNode;

// BVH operations
BVHNode* create_bvh_node(Triangle* triangles, int start, int count);
BVH create_bvh(Triangle* triangles, size_t count);
void destroy_bvh_node(BVHNode* node);
void destroy_bvh(BVH* bvh);
bool intersect_bvh(BVHNode* node, Ray ray, const Triangle* triangles,
                   float* t_out, float* u_out, float* v_out, int* tri_idx);

// GPU BVH operations
GPUBVHNode* copy_bvh_to_gpu(BVHNode* cpu_node);
__device__ bool intersect_bvh_gpu(const GPUMesh* mesh, Ray ray,
                                  float* t_out, float* u_out, float* v_out, int* tri_idx);

#endif