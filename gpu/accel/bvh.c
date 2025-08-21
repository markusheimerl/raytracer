#include "bvh.h"
#include "math/ray.h"

BVHNode* create_bvh_node(Triangle* triangles, int start, int count) {
    BVHNode* node = (BVHNode*)malloc(sizeof(BVHNode));
    node->start_idx = start;
    node->triangle_count = count;
    node->left = node->right = NULL;

    // Calculate bounds
    node->bounds = create_empty_aabb();
    for (int i = 0; i < count; i++) {
        node->bounds = expand_aabb(node->bounds, triangles[start + i].v0);
        node->bounds = expand_aabb(node->bounds, triangles[start + i].v1);
        node->bounds = expand_aabb(node->bounds, triangles[start + i].v2);
    }

    // Split if more than 4 triangles
    if (count > 4) {
        // Find longest axis
        Vec3 extent = vec3_sub(node->bounds.max, node->bounds.min);
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > extent.x && extent.z > extent.y) axis = 2;

        // Sort triangles based on centroid
        float split = 0.0f;
        for (int i = 0; i < count; i++) {
            Triangle* tri = &triangles[start + i];
            Vec3 centroid = vec3_mul(vec3_add(vec3_add(tri->v0, tri->v1), tri->v2), 1.0f/3.0f);
            split += axis == 0 ? centroid.x : (axis == 1 ? centroid.y : centroid.z);
        }
        split /= count;

        // Partition triangles
        int mid = start;
        for (int i = 0; i < count; i++) {
            Triangle* tri = &triangles[start + i];
            Vec3 centroid = vec3_mul(vec3_add(vec3_add(tri->v0, tri->v1), tri->v2), 1.0f/3.0f);
            float value = axis == 0 ? centroid.x : (axis == 1 ? centroid.y : centroid.z);
            
            if (value < split) {
                Triangle temp = triangles[start + i];
                triangles[start + i] = triangles[mid];
                triangles[mid] = temp;
                mid++;
            }
        }

        // Create children
        int left_count = mid - start;
        if (left_count > 0 && left_count < count) {
            node->left = create_bvh_node(triangles, start, left_count);
            node->right = create_bvh_node(triangles, mid, count - left_count);
        }
    }

    return node;
}

BVH create_bvh(Triangle* triangles, size_t count) {
    BVH bvh;
    bvh.triangles = triangles;
    bvh.triangle_count = count;
    bvh.root = create_bvh_node(triangles, 0, count);
    return bvh;
}

void destroy_bvh_node(BVHNode* node) {
    if (node->left) destroy_bvh_node(node->left);
    if (node->right) destroy_bvh_node(node->right);
    free(node);
}

void destroy_bvh(BVH* bvh) {
    if (bvh->root) destroy_bvh_node(bvh->root);
    bvh->root = NULL;
}

bool intersect_bvh(BVHNode* node, Ray ray, const Triangle* triangles,
                   float* t_out, float* u_out, float* v_out, int* tri_idx) {
    if (!ray_aabb_intersect(ray, node->bounds)) return false;

    bool hit = false;
    float closest_t = *t_out;

    if (node->left == NULL && node->right == NULL) {
        // Leaf node - test all triangles
        for (int i = 0; i < node->triangle_count; i++) {
            float t, u, v;
            if (ray_triangle_intersect(ray,
                triangles[node->start_idx + i].v0,
                triangles[node->start_idx + i].v1,
                triangles[node->start_idx + i].v2,
                &t, &u, &v) && t < closest_t) {
                closest_t = t;
                *t_out = t;
                *u_out = u;
                *v_out = v;
                *tri_idx = node->start_idx + i;
                hit = true;
            }
        }
    } else {
        // Internal node - recurse
        float t1 = 1e30f, t2 = 1e30f;
        float u1, v1, u2, v2;
        int idx1, idx2;

        bool hit1 = node->left ? intersect_bvh(node->left, ray, triangles, &t1, &u1, &v1, &idx1) : false;
        bool hit2 = node->right ? intersect_bvh(node->right, ray, triangles, &t2, &u2, &v2, &idx2) : false;

        if (hit1 && (!hit2 || t1 < t2)) {
            *t_out = t1;
            *u_out = u1;
            *v_out = v1;
            *tri_idx = idx1;
            hit = true;
        } else if (hit2) {
            *t_out = t2;
            *u_out = u2;
            *v_out = v2;
            *tri_idx = idx2;
            hit = true;
        }
    }

    return hit;
}

// GPU BVH functions (simplified implementation)
GPUBVHNode* copy_bvh_to_gpu(BVHNode* cpu_node) {
    if (!cpu_node) return NULL;
    
    GPUBVHNode* gpu_node;
    CHECK_CUDA(cudaMalloc(&gpu_node, sizeof(GPUBVHNode)));
    
    GPUBVHNode h_node;
    h_node.bounds = cpu_node->bounds;
    h_node.start_idx = cpu_node->start_idx;
    h_node.triangle_count = cpu_node->triangle_count;
    h_node.left_idx = -1;
    h_node.right_idx = -1;
    
    CHECK_CUDA(cudaMemcpy(gpu_node, &h_node, sizeof(GPUBVHNode), cudaMemcpyHostToDevice));
    
    return gpu_node;
}

__device__ bool intersect_bvh_gpu(const GPUMesh* mesh, Ray ray,
                                  float* t_out, float* u_out, float* v_out, int* tri_idx) {
    // Simplified GPU BVH traversal - in practice you'd want a more sophisticated implementation
    // For now, just test all triangles
    bool hit = false;
    float closest_t = *t_out;
    
    for (int i = 0; i < mesh->triangle_count; i++) {
        float t, u, v;
        if (ray_triangle_intersect_gpu(ray,
            mesh->d_triangles[i].v0,
            mesh->d_triangles[i].v1,
            mesh->d_triangles[i].v2,
            &t, &u, &v) && t < closest_t) {
            closest_t = t;
            *t_out = t;
            *u_out = u;
            *v_out = v;
            *tri_idx = i;
            hit = true;
        }
    }
    
    return hit;
}