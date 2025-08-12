#ifndef SCENE_H
#define SCENE_H

#include "geometry/mesh.h"
#include "render/camera.h"
#include "render/light.h"
#include "utils/progress.h"
#include "utils/image.h"
#include <webp/encode.h>
#include <webp/mux.h>
#include <cuda_runtime.h>
#include <time.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    Mesh* meshes;
    size_t mesh_count;
    Camera camera;
    DirectionalLight light;
    unsigned char** frames;
    unsigned char* d_frame_buffer;  // GPU frame buffer
    int frame_count;
    int current_frame;
    int width;
    int height;
    float scale_factor;
    int duration_ms;
    int fps;
} Scene;

// Scene management
Scene create_scene(int width, int height, int duration_ms, int fps, float scale_factor);
void add_mesh_to_scene(Scene* scene, Mesh mesh);
void set_scene_camera(Scene* scene, Vec3 position, Vec3 look_at, Vec3 up, float fov);
void set_scene_light(Scene* scene, Vec3 direction, Vec3 color);
void next_frame(Scene* scene);
void render_scene(Scene* scene);
void save_scene(Scene* scene, const char* filename);
void destroy_scene(Scene* scene);

// CUDA rendering function
void render_scene_cuda(Scene* scene, float aspect);

#endif