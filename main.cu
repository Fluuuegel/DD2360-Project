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

// ---------------------- 简易 Octree 支持（最小增量实现） ----------------------
struct aabb {
    vec3 _min;
    vec3 _max;
    __device__ aabb() {}
    __device__ aabb(const vec3& a, const vec3& b) : _min(a), _max(b) {}
    __device__ bool hit(const ray& r, float t_min, float t_max) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (_min[a] - r.origin()[a]) * invD;
            float t1 = (_max[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) {
                float tmp = t0; t0 = t1; t1 = tmp;
            }
            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;
            if (t_max <= t_min) return false;
        }
        return true;
    }
};

__device__ inline aabb surrounding_box(const aabb& b0, const aabb& b1) {
    vec3 small(fminf(b0._min.x(), b1._min.x()),
               fminf(b0._min.y(), b1._min.y()),
               fminf(b0._min.z(), b1._min.z()));
    vec3 big(fmaxf(b0._max.x(), b1._max.x()),
             fmaxf(b0._max.y(), b1._max.y()),
             fmaxf(b0._max.z(), b1._max.z()));
    return aabb(small, big);
}

__device__ inline aabb sphere_box(const sphere* s) {
    vec3 rvec(s->radius, s->radius, s->radius);
    return aabb(s->center - rvec, s->center + rvec);
}

class octree_node : public hitable {
public:
    __device__ octree_node() : leaf(nullptr) {
        for (int i = 0; i < 8; i++) children[i] = nullptr;
    }
    aabb box;
    octree_node* children[8];
    hitable_list* leaf; // 叶子节点复用已有 hitable_list

    __device__ bool is_leaf() const { return leaf != nullptr; }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        if (!box.hit(r, t_min, t_max)) return false;
        bool hit_anything = false;
        float closest = t_max;
        hit_record temp;
        if (is_leaf()) {
            if (leaf->hit(r, t_min, closest, temp)) {
                hit_anything = true;
                closest = temp.t;
                rec = temp;
            }
        } else {
            for (int i = 0; i < 8; i++) {
                if (children[i] && children[i]->hit(r, t_min, closest, temp)) {
                    hit_anything = true;
                    closest = temp.t;
                    rec = temp;
                }
            }
        }
        return hit_anything;
    }
};

__device__ octree_node* build_octree(sphere** objs, int count, int depth, int max_depth = 8, int leaf_size = 8) {
    octree_node* node = new octree_node();
    // 计算当前节点包围盒
    aabb cur_box = sphere_box(objs[0]);
    for (int i = 1; i < count; i++) {
        cur_box = surrounding_box(cur_box, sphere_box(objs[i]));
    }
    node->box = cur_box;

    // 叶子条件
    if (count <= leaf_size || depth >= max_depth) {
        sphere** leaf_objs = new sphere*[count];
        for (int i = 0; i < count; i++) leaf_objs[i] = objs[i];
        node->leaf = new hitable_list((hitable**)leaf_objs, count);
        return node;
    }

    // 划分 8 个子空间
    vec3 center = 0.5f * (cur_box._min + cur_box._max);
    int child_counts[8] = {0};
    // 预分配临时桶
    sphere** child_objs[8];
    for (int i = 0; i < 8; i++) child_objs[i] = new sphere*[count];

    for (int i = 0; i < count; i++) {
        int oct = 0;
        if (objs[i]->center.x() >= center.x()) oct |= 1;
        if (objs[i]->center.y() >= center.y()) oct |= 2;
        if (objs[i]->center.z() >= center.z()) oct |= 4;
        child_objs[oct][child_counts[oct]++] = objs[i];
    }

    for (int i = 0; i < 8; i++) {
        if (child_counts[i] > 0) {
            node->children[i] = build_octree(child_objs[i], child_counts[i], depth + 1, max_depth, leaf_size);
        }
        delete [] child_objs[i];
    }
    return node;
}

__device__ void free_octree(octree_node* node) {
    if (!node) return;
    if (node->is_leaf()) {
        // leaf->list 在 hitable_list 内部，不需要单独释放元素指针（由外部统一释放）
        delete[] (sphere**)node->leaf->list;
        delete node->leaf;
    } else {
        for (int i = 0; i < 8; i++) {
            free_octree(node->children[i]);
        }
    }
    delete node;
}

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
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
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
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
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
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

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        // 使用 Octree 作为世界加速结构
        *d_world  = build_octree((sphere**)d_list, 22*22+1+3, 0);

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    // 释放 Octree 节点
    free_octree((octree_node*)(*d_world));
    // 释放所有球体及材质
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_camera;
}

int main() {
    int nx = 1200;
    int ny = 800;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int num_hitables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
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

    // Output FB as Image
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

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}
