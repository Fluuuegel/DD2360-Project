#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// --- AABB Helper Struct ---
struct aabb {
    vec3 min, max;
    __device__ aabb() {}
    __device__ aabb(const vec3& a, const vec3& b) { min = a; max = b; }

    __device__ bool hit(const ray& r, float tmin, float tmax) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (min[a] - r.origin()[a]) * invD;
            float t1 = (max[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) { float temp = t0; t0 = t1; t1 = temp; }
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }
        return true;
    }
    
    // Check if other box is fully contained in this box
    __device__ bool contains(const aabb& other) const {
        return (other.min.x() >= min.x() && other.max.x() <= max.x() &&
                other.min.y() >= min.y() && other.max.y() <= max.y() &&
                other.min.z() >= min.z() && other.max.z() <= max.z());
    }
};

__device__ aabb get_box(hitable* obj) {
    sphere* s = (sphere*)obj; 
    return aabb(s->center - vec3(s->radius,s->radius,s->radius), s->center + vec3(s->radius,s->radius,s->radius));
}

// --- Octree Node ---
class octree_node : public hitable {
public:
    aabb box;
    hitable** objects; // Objects that stay at this level (don't fit in children)
    int obj_count;
    octree_node* children[8]; // 8 Octants

    __device__ octree_node(aabb b) : box(b), objects(nullptr), obj_count(0) {
        for(int i=0; i<8; i++) children[i] = nullptr;
    }

    __device__ void insert(hitable** list, int n, int depth);
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
};

__device__ void octree_node::insert(hitable** list, int n, int depth) {
    if (depth >= 6 || n == 0) { // Max depth or empty
        objects = new hitable*[n];
        for(int i=0; i<n; i++) objects[i] = list[i];
        obj_count = n;
        return;
    }

    vec3 center = (box.min + box.max) * 0.5f;
    
    // Temporary lists for children
    hitable** lists[8];
    int counts[8] = {0};
    hitable** self_list = new hitable*[n]; // Objects staying here
    int self_count = 0;

    // Allocate max possible for temporary bins (optimization: can be dynamic but simpler here)
    for(int i=0; i<8; i++) lists[i] = new hitable*[n];

    for (int i = 0; i < n; i++) {
        aabb obj_box = get_box(list[i]);
        int index = -1;
        
        // Determine which octant the object fits in perfectly
        for(int j=0; j<8; j++) {
            vec3 child_min, child_max;
            child_min.e[0] = (j & 1) ? center.x() : box.min.x();
            child_max.e[0] = (j & 1) ? box.max.x() : center.x();
            child_min.e[1] = (j & 2) ? center.y() : box.min.y();
            child_max.e[1] = (j & 2) ? box.max.y() : center.y();
            child_min.e[2] = (j & 4) ? center.z() : box.min.z();
            child_max.e[2] = (j & 4) ? box.max.z() : center.z();
            
            aabb child_box(child_min, child_max);
            if (child_box.contains(obj_box)) {
                index = j;
                break;
            }
        }

        if (index != -1) {
            lists[index][counts[index]++] = list[i];
        } else {
            self_list[self_count++] = list[i];
        }
    }

    // Save objects that stayed at this level
    if (self_count > 0) {
        objects = new hitable*[self_count];
        for(int i=0; i<self_count; i++) objects[i] = self_list[i];
        obj_count = self_count;
    }
    delete[] self_list;

    // Recurse for children
    for(int i=0; i<8; i++) {
        if (counts[i] > 0) {
            vec3 child_min, child_max;
            child_min.e[0] = (i & 1) ? center.x() : box.min.x();
            child_max.e[0] = (i & 1) ? box.max.x() : center.x();
            child_min.e[1] = (i & 2) ? center.y() : box.min.y();
            child_max.e[1] = (i & 2) ? box.max.y() : center.y();
            child_min.e[2] = (i & 4) ? center.z() : box.min.z();
            child_max.e[2] = (i & 4) ? box.max.z() : center.z();

            children[i] = new octree_node(aabb(child_min, child_max));
            children[i]->insert(lists[i], counts[i], depth + 1);
        }
        delete[] lists[i];
    }
}

__device__ bool octree_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    if (!box.hit(r, t_min, t_max)) return false;

    bool hit_anything = false;
    float closest_so_far = t_max;

    // Check objects at this node
    if (obj_count > 0) {
        for (int i = 0; i < obj_count; i++) {
            if (objects[i]->hit(r, t_min, closest_so_far, rec)) {
                hit_anything = true;
                closest_so_far = rec.t;
            }
        }
    }

    // Check children (order doesn't matter for correctness, only performance)
    for (int i = 0; i < 8; i++) {
        if (children[i]) {
            hit_record temp_rec;
            if (children[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }
    }
    return hit_anything;
}

// --- Standard Ray Tracing Functions ---

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            } else {
                return vec3(0.0,0.0,0.0);
            }
        } else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0);
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state, int grid_radius) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000, new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -grid_radius; a < grid_radius; a++) {
            for(int b = -grid_radius; b < grid_radius; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2, new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                } else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2, new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                } else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        
        // Create Octree Root
        // Bounds covering the whole scene (ground is at -1000 radius 1000, so min y is -2000)
        aabb root_box(vec3(-100, -2000, -100), vec3(100, 100, 100));
        octree_node* root = new octree_node(root_box);
        root->insert(d_list, i, 0);
        *d_world = root;

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, float(nx)/float(ny), aperture, dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    delete *d_world;
    delete *d_camera;
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    // CRITICAL FIX: Increase Heap and Stack for Octree construction and recursion
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
    cudaDeviceSetLimit(cudaLimitStackSize, 16*1024);

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	const int GRID_RADIUS = 11; // original 11  22*22+1+3
	const int NUM_EXTRA_SPHERES = 3;    // big ball
	const int NUM_GROUND = 1;

	const int NUM_GRID_SPHERES = (2 * GRID_RADIUS) * (2 * GRID_RADIUS);
	const int NUM_HITABLES     = NUM_GRID_SPHERES + NUM_EXTRA_SPHERES + NUM_GROUND;
    hitable **d_list;
    int num_hitables = NUM_HITABLES;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, GRID_RADIUS);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
