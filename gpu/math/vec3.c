#include "vec3.h"

// CPU implementations
Vec3 vec3_add(Vec3 a, Vec3 b) { 
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; 
}

Vec3 vec3_sub(Vec3 a, Vec3 b) { 
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; 
}

Vec3 vec3_mul(Vec3 a, float t) { 
    return (Vec3){a.x * t, a.y * t, a.z * t}; 
}

Vec3 vec3_div(Vec3 a, float t) { 
    return vec3_mul(a, 1.0f/t); 
}

float vec3_dot(Vec3 a, Vec3 b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float vec3_length(Vec3 v) { 
    return sqrtf(vec3_dot(v, v)); 
}

Vec3 vec3_normalize(Vec3 v) { 
    return vec3_div(v, vec3_length(v)); 
}

Vec3 vec3_mul_vec3(Vec3 a, Vec3 b) { 
    return (Vec3){a.x * b.x, a.y * b.y, a.z * b.z}; 
}

// GPU implementations
__device__ Vec3 vec3_add_gpu(Vec3 a, Vec3 b) { 
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; 
}

__device__ Vec3 vec3_sub_gpu(Vec3 a, Vec3 b) { 
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; 
}

__device__ Vec3 vec3_mul_gpu(Vec3 a, float t) { 
    return (Vec3){a.x * t, a.y * t, a.z * t}; 
}

__device__ float vec3_dot_gpu(Vec3 a, Vec3 b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

__device__ Vec3 vec3_cross_gpu(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ float vec3_length_gpu(Vec3 v) { 
    return sqrtf(vec3_dot_gpu(v, v)); 
}

__device__ Vec3 vec3_normalize_gpu(Vec3 v) { 
    return vec3_mul_gpu(v, 1.0f/vec3_length_gpu(v)); 
}

__device__ Vec3 vec3_mul_vec3_gpu(Vec3 a, Vec3 b) { 
    return (Vec3){a.x * b.x, a.y * b.y, a.z * b.z}; 
}