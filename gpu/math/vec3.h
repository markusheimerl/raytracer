#ifndef VEC3_H
#define VEC3_H

#include <math.h>

typedef struct { float x, y, z; } Vec3;
typedef struct { float u, v; } Vec2;

// Vec3 operations (CPU)
Vec3 vec3_add(Vec3 a, Vec3 b);
Vec3 vec3_sub(Vec3 a, Vec3 b);
Vec3 vec3_mul(Vec3 a, float t);
Vec3 vec3_div(Vec3 a, float t);
float vec3_dot(Vec3 a, Vec3 b);
Vec3 vec3_cross(Vec3 a, Vec3 b);
float vec3_length(Vec3 v);
Vec3 vec3_normalize(Vec3 v);
Vec3 vec3_mul_vec3(Vec3 a, Vec3 b);

// Vec3 operations (GPU)
__device__ Vec3 vec3_add_gpu(Vec3 a, Vec3 b);
__device__ Vec3 vec3_sub_gpu(Vec3 a, Vec3 b);
__device__ Vec3 vec3_mul_gpu(Vec3 a, float t);
__device__ float vec3_dot_gpu(Vec3 a, Vec3 b);
__device__ Vec3 vec3_cross_gpu(Vec3 a, Vec3 b);
__device__ float vec3_length_gpu(Vec3 v);
__device__ Vec3 vec3_normalize_gpu(Vec3 v);
__device__ Vec3 vec3_mul_vec3_gpu(Vec3 a, Vec3 b);

#endif