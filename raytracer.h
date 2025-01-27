#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <math.h>
#include <stdbool.h>

typedef struct { float x, y, z; } Vec3;
typedef struct { float u, v; } Vec2;
typedef struct { Vec3 origin, direction; } Ray;
typedef struct { Vec3 position; Vec3 look_at; Vec3 up; float fov; } Camera;
typedef struct {
    Vec3 v0, v1, v2;      // Vertices
    Vec2 t0, t1, t2;      // Texture coordinates
    Vec3 n0, n1, n2;      // Vertex normals
} Triangle;

Vec3 vec3_add(Vec3 a, Vec3 b) { return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 vec3_sub(Vec3 a, Vec3 b) { return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 vec3_mul(Vec3 a, float t) { return (Vec3){a.x * t, a.y * t, a.z * t}; }
Vec3 vec3_div(Vec3 a, float t) { return vec3_mul(a, 1.0f/t); }
float vec3_dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
float vec3_length(Vec3 v) { return sqrtf(vec3_dot(v, v)); }
Vec3 vec3_normalize(Vec3 v) { return vec3_div(v, vec3_length(v)); }

bool ray_triangle_intersect(Ray ray, Vec3 v0, Vec3 v1, Vec3 v2, 
                          float* t, float* u_out, float* v_out) {
    const float EPSILON = 0.0000001f;
    Vec3 edge1 = vec3_sub(v1, v0);
    Vec3 edge2 = vec3_sub(v2, v0);
    Vec3 h = vec3_cross(ray.direction, edge2);
    float a = vec3_dot(edge1, h);

    if (a > -EPSILON && a < EPSILON) return false;

    float f = 1.0f / a;
    Vec3 s = vec3_sub(ray.origin, v0);
    float u = f * vec3_dot(s, h);

    if (u < 0.0f || u > 1.0f) return false;

    Vec3 q = vec3_cross(s, edge1);
    float v = f * vec3_dot(ray.direction, q);

    if (v < 0.0f || u + v > 1.0f) return false;

    *t = f * vec3_dot(edge2, q);
    *u_out = u;
    *v_out = v;
    return *t > EPSILON;
}

Camera create_camera(Vec3 position, Vec3 look_at, Vec3 up, float fov) {
    return (Camera){position, look_at, up, fov};
}

Ray get_camera_ray(const Camera* camera, float x, float y, float aspect) {
    Vec3 forward = vec3_normalize(vec3_sub(camera->look_at, camera->position));
    Vec3 right = vec3_normalize(vec3_cross(forward, camera->up));
    Vec3 camera_up = vec3_cross(right, forward);
    
    float scale = tanf((camera->fov * 0.5f) * M_PI / 180.0f);
    float ray_x = (2.0f * x - 1.0f) * aspect * scale;
    float ray_y = (1.0f - 2.0f * y) * scale;

    Vec3 ray_dir = vec3_normalize(vec3_add(
        vec3_add(
            vec3_mul(right, ray_x),
            vec3_mul(camera_up, ray_y)
        ),
        forward
    ));

    return (Ray){camera->position, ray_dir};
}

#endif