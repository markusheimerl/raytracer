#include "scene.h"
#include "accel/bvh.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Scene create_scene(int width, int height, int duration_ms, int fps, float scale_factor) {
    int frame_count = (duration_ms * fps) / 1000;

    Scene scene;
    scene.meshes = NULL;
    scene.mesh_count = 0;
    scene.width = (int)(width * scale_factor);
    scene.height = (int)(height * scale_factor);
    scene.scale_factor = scale_factor;
    scene.frame_count = frame_count;
    scene.current_frame = 0;
    scene.duration_ms = duration_ms;
    scene.fps = fps;
    scene.frames = (unsigned char**)malloc(frame_count * sizeof(unsigned char*));
    
    // Allocate memory for each frame
    for (int i = 0; i < frame_count; i++) {
        scene.frames[i] = (unsigned char*)malloc(width * height * 3);
    }
    
    // Allocate GPU frame buffer
    CHECK_CUDA(cudaMalloc(&scene.d_frame_buffer, scene.width * scene.height * 3 * sizeof(unsigned char)));
    
    return scene;
}

void add_mesh_to_scene(Scene* scene, Mesh mesh) {
    scene->meshes = (Mesh*)realloc(scene->meshes, (scene->mesh_count + 1) * sizeof(Mesh));
    scene->meshes[scene->mesh_count] = mesh;
    scene->mesh_count++;
}

void set_scene_camera(Scene* scene, Vec3 position, Vec3 look_at, Vec3 up, float fov) {
    scene->camera = create_camera(position, look_at, up, fov);
}

void set_scene_light(Scene* scene, Vec3 direction, Vec3 color) {
    scene->light = create_directional_light(direction, color);
}

void next_frame(Scene* scene) {
    scene->current_frame++;
    if (scene->current_frame >= scene->frame_count) {
        scene->current_frame = scene->frame_count - 1;
    }
}

void render_scene(Scene* scene) {
    float aspect = (float)scene->width / scene->height;
    unsigned char* current_frame = scene->frames[scene->current_frame];

    // Launch CUDA kernel for rendering
    render_scene_cuda(scene, aspect);
    
    // Copy result from GPU to CPU
    CHECK_CUDA(cudaMemcpy(current_frame, scene->d_frame_buffer, 
                         scene->width * scene->height * 3 * sizeof(unsigned char), 
                         cudaMemcpyDeviceToHost));
}

void save_scene(Scene* scene, const char* filename) {
    // Calculate scaled dimensions
    int scaled_width = (int)(scene->width / scene->scale_factor + 0.5f);
    int scaled_height = (int)(scene->height / scene->scale_factor + 0.5f);

    // Prepare WebP animation configuration
    WebPAnimEncoderOptions anim_config;
    WebPAnimEncoderOptionsInit(&anim_config);
    WebPAnimEncoder* enc = WebPAnimEncoderNew(scaled_width, scaled_height, &anim_config);
    
    // Configure each frame
    WebPConfig config;
    WebPConfigInit(&config);
    config.image_hint = WEBP_HINT_GRAPH;
    
    // Prepare picture with scaled dimensions
    WebPPicture pic;
    WebPPictureInit(&pic);
    pic.width = scaled_width;
    pic.height = scaled_height;
    pic.use_argb = 1;
    WebPPictureAlloc(&pic);

    // Add each frame to the animation
    for (int frame = 0; frame < scene->frame_count; frame++) {
        // Scale up the frame using bicubic interpolation
        for (int y = 0; y < scaled_height; y++) {
            for (int x = 0; x < scaled_width; x++) {
                float src_x = x * (scene->width - 1.0f) / (scaled_width - 1.0f);
                float src_y = y * (scene->height - 1.0f) / (scaled_height - 1.0f);
                
                pic.argb[y * scaled_width + x] = bicubic_interpolate(
                    scene->frames[frame],
                    src_x,
                    src_y,
                    scene->width,
                    scene->height
                );
            }
        }
        
        int timestamp = frame * (scene->duration_ms / scene->frame_count);
        WebPAnimEncoderAdd(enc, &pic, timestamp, &config);
    }
    
    // Finalize animation
    WebPAnimEncoderAdd(enc, NULL, scene->duration_ms, NULL);
    WebPData webp_data;
    WebPDataInit(&webp_data);
    WebPAnimEncoderAssemble(enc, &webp_data);
    
    // Save to file
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(webp_data.bytes, webp_data.size, 1, fp);
        fclose(fp);
    }
    
    // Cleanup
    WebPDataClear(&webp_data);
    WebPAnimEncoderDelete(enc);
    WebPPictureFree(&pic);
}

void destroy_scene(Scene* scene) {
    free(scene->meshes);
    
    // Free all frame buffers
    for (int i = 0; i < scene->frame_count; i++) {
        free(scene->frames[i]);
    }
    free(scene->frames);
    
    // Free GPU memory
    CHECK_CUDA(cudaFree(scene->d_frame_buffer));
    
    scene->meshes = NULL;
    scene->frames = NULL;
    scene->mesh_count = 0;
}

// CUDA kernel for scene rendering
__global__ void render_kernel(unsigned char* frame_buffer, int width, int height, float aspect,
                             Camera camera, DirectionalLight light, 
                             GPUMesh* meshes, int mesh_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    Ray ray = get_camera_ray_gpu(&camera, 
                                (x + 0.5f) / width, 
                                (y + 0.5f) / height, 
                                aspect);
    
    float closest_t = 1e30f;
    bool hit = false;
    Vec2 hit_uv = {0, 0};
    Vec3 hit_normal = {0, 0, 0};
    int hit_mesh_idx = -1;
    
    // Check intersection with all meshes using BVH
    for (int m = 0; m < mesh_count; m++) {
        GPUMesh* current_mesh = &meshes[m];
        float t = closest_t;
        float u, v;
        int tri_idx;
        
        // Transform ray to mesh local space
        Ray transformed_ray = transform_ray_gpu(ray, current_mesh->transform);
        
        if (intersect_bvh_gpu(current_mesh, transformed_ray, &t, &u, &v, &tri_idx) && t < closest_t) {
            closest_t = t;
            hit = true;
            hit_mesh_idx = m;
            float w = 1.0f - u - v;

            // Interpolate texture coordinates
            hit_uv.u = w * current_mesh->d_triangles[tri_idx].t0.u + 
                      u * current_mesh->d_triangles[tri_idx].t1.u + 
                      v * current_mesh->d_triangles[tri_idx].t2.u;
            hit_uv.v = w * current_mesh->d_triangles[tri_idx].t0.v + 
                      u * current_mesh->d_triangles[tri_idx].t1.v + 
                      v * current_mesh->d_triangles[tri_idx].t2.v;

            // Interpolate normal
            hit_normal = vec3_normalize_gpu(vec3_add_gpu(
                vec3_add_gpu(
                    vec3_mul_gpu(current_mesh->d_triangles[tri_idx].n0, w),
                    vec3_mul_gpu(current_mesh->d_triangles[tri_idx].n1, u)
                ),
                vec3_mul_gpu(current_mesh->d_triangles[tri_idx].n2, v)
            ));

            // Transform the interpolated normal according to the mesh's transformation
            hit_normal = transform_normal_gpu(hit_normal, current_mesh->transform);
        }
    }

    int idx = (y * width + x) * 3;
    if (hit && hit_mesh_idx >= 0) {
        GPUMesh* hit_mesh = &meshes[hit_mesh_idx];
        Vec3 color = sample_mesh_texture_gpu(hit_mesh, hit_uv.u, hit_uv.v);
        
        // Calculate diffuse lighting
        float diffuse = 0.2f;  // Ambient light level
        
        // Calculate hit point in world space using original ray
        Vec3 hit_point = vec3_add_gpu(ray.origin, vec3_mul_gpu(ray.direction, closest_t));
        Vec3 shadow_origin = vec3_add_gpu(hit_point, vec3_mul_gpu(hit_normal, 0.001f));
        Ray shadow_ray = {shadow_origin, light.direction};
        
        // Check if point is in shadow
        bool in_shadow = false;
        for (int m = 0; m < mesh_count && !in_shadow; m++) {
            GPUMesh* current_mesh = &meshes[m];
            float shadow_t = 1e30f;
            float shadow_u, shadow_v;
            int shadow_tri_idx;
            
            // Transform shadow ray to mesh local space
            Ray transformed_shadow_ray = transform_ray_gpu(shadow_ray, current_mesh->transform);
            
            if (intersect_bvh_gpu(current_mesh, transformed_shadow_ray, 
                                &shadow_t, &shadow_u, &shadow_v, 
                                &shadow_tri_idx)) {
                in_shadow = true;
            }
        }
        
        // Add direct lighting if not in shadow
        if (!in_shadow) {
            diffuse = fmaxf(diffuse, 
                vec3_dot_gpu(hit_normal, light.direction));
        }
        
        // Apply lighting
        color = vec3_mul_vec3_gpu(color, light.color);
        color = vec3_mul_gpu(color, diffuse);
        
        // Convert to RGB bytes
        frame_buffer[idx] = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
        frame_buffer[idx + 1] = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
        frame_buffer[idx + 2] = (unsigned char)(fminf(color.z * 255.0f, 255.0f));
    } else {
        frame_buffer[idx] = frame_buffer[idx + 1] = frame_buffer[idx + 2] = 50;
    }
}

void render_scene_cuda(Scene* scene, float aspect) {
    // Copy mesh data to GPU if needed
    static GPUMesh* d_meshes = NULL;
    static int prev_mesh_count = 0;
    
    if (d_meshes == NULL || prev_mesh_count != (int)scene->mesh_count) {
        if (d_meshes) CHECK_CUDA(cudaFree(d_meshes));
        CHECK_CUDA(cudaMalloc(&d_meshes, scene->mesh_count * sizeof(GPUMesh)));
        
        // Convert and copy mesh data
        GPUMesh* h_meshes = (GPUMesh*)malloc(scene->mesh_count * sizeof(GPUMesh));
        for (size_t i = 0; i < scene->mesh_count; i++) {
            convert_mesh_to_gpu(&scene->meshes[i], &h_meshes[i]);
        }
        CHECK_CUDA(cudaMemcpy(d_meshes, h_meshes, scene->mesh_count * sizeof(GPUMesh), cudaMemcpyHostToDevice));
        free(h_meshes);
        prev_mesh_count = scene->mesh_count;
    }
    
    // Launch kernel
    dim3 block_size(16, 16);
    dim3 grid_size((scene->width + block_size.x - 1) / block_size.x,
                   (scene->height + block_size.y - 1) / block_size.y);
    
    render_kernel<<<grid_size, block_size>>>(
        scene->d_frame_buffer, scene->width, scene->height, aspect,
        scene->camera, scene->light, d_meshes, scene->mesh_count
    );
    
    CHECK_CUDA(cudaDeviceSynchronize());
}