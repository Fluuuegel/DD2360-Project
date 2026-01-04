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
/*******************************************************************************************/
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include "lbvh.h"
#include "morton.h"
#include "aabb.h"

__global__ void compute_sphere_aabb_centroid(hitable **d_list, int n, aabb *d_aabb, vec3 *d_centroid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    sphere* sp = (sphere*)d_list[idx];
    const vec3 c = sp->center;
    const float r = sp->radius;

    d_centroid[idx] = c;

    const vec3 mn(c.x() - r, c.y() - r, c.z() - r);
    const vec3 mx(c.x() + r, c.y() + r, c.z() + r);
    d_aabb[idx] = aabb(mn, mx);
}

__global__ void init_leaf_aabb(int n, const int* primIdxSorted, const aabb* primAabb,
                               aabb* nodeAabb) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;
    int leafNode = i + (n - 1);
    int prim = primIdxSorted[i];
    nodeAabb[leafNode] = primAabb[prim];
}

__global__ void build_aabb_bottom_up(int n,
                                    const int* left, const int* right,
                                    int* parent, int* visit,
                                    aabb* nodeAabb) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    int node = i + (n - 1); // leaf node index
    int p = parent[node];

    while (p != -1) {
        int old = atomicAdd(&visit[p], 1);
        if (old == 0) {
            return;
        } else {
            const int lc = left[p];
            const int rc = right[p];
            nodeAabb[p] = surrounding_box(nodeAabb[lc], nodeAabb[rc]);
            node = p;
            p = parent[node];
        }
    }
}

__global__ void find_root(int n, const int* parent, int* root_out) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n - 1) return;
    if (parent[i] == -1) {
        atomicExch(root_out, i);
    }
}

__global__ void compute_lbvh_keys(const vec3 *d_centroid, int n,
                                  vec3 scene_min, vec3 scene_max,
                                  unsigned long long *d_keys, int *d_primIdx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
	
	//const vec3 c = d_centroid[idx];
    const int prim = idx + 1;
	const vec3 c = d_centroid[prim];

    const float ex = scene_max.x() - scene_min.x();
    const float ey = scene_max.y() - scene_min.y();
    const float ez = scene_max.z() - scene_min.z();

    float nx = (ex > 0.0f) ? ((c.x() - scene_min.x()) / ex) : 0.0f;
    float ny = (ey > 0.0f) ? ((c.y() - scene_min.y()) / ey) : 0.0f;
    float nz = (ez > 0.0f) ? ((c.z() - scene_min.z()) / ez) : 0.0f;

    const unsigned int m = morton3D(nx, ny, nz);

    //d_primIdx[idx] = idx;
    //d_keys[idx] = ( (unsigned long long)m << 32 ) | (unsigned int)idx;
    d_primIdx[idx] = prim;
	d_keys[idx] = ( (unsigned long long) m << 32) | (unsigned int)prim;
}

/**********************************************
__global__ void compute_morton_codes(const vec3 *d_centroid, int n, vec3 scene_min, vec3 scene_max,
                                     unsigned int *d_morton, int *d_primIdx) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    const vec3 c = d_centroid[idx];

    const float ex = scene_max.x() - scene_min.x();
    const float ey = scene_max.y() - scene_min.y();
    const float ez = scene_max.z() - scene_min.z();

    float nx = (ex > 0.0f) ? ( (c.x() - scene_min.x()) / ex ) : 0.0f;
    float ny = (ey > 0.0f) ? ( (c.y() - scene_min.y()) / ey ) : 0.0f;
    float nz = (ez > 0.0f) ? ( (c.z() - scene_min.z()) / ez ) : 0.0f;

    d_morton[idx] = morton3D(nx, ny, nz);
    d_primIdx[idx] = idx;
}
**********************************************/

__device__ __forceinline__ int clzll_u64(unsigned long long x) {
    return (x == 0ull) ? 64 : __clzll(x);
}

__device__ __forceinline__ int delta(const unsigned long long* keys, int n, int i, int j) {
    if (j < 0 || j >= n) return -1;
    const unsigned long long a = keys[i];
    const unsigned long long b = keys[j];
    return clzll_u64(a ^ b);
}

__device__ __forceinline__ int2 determine_range(const unsigned long long* keys, int n, int idx) {
    const int d1 = delta(keys, n, idx, idx + 1);
    const int d2 = delta(keys, n, idx, idx - 1);
    const int dir = (d1 - d2) >= 0 ? 1 : -1;

    const int delta_min = delta(keys, n, idx, idx - dir);

    int lmax = 2;
    while (delta(keys, n, idx, idx + lmax * dir) > delta_min) {
        lmax *= 2;
    }

    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2) {
        if (delta(keys, n, idx, idx + (l + t) * dir) > delta_min) {
            l += t;
        }
    }

    int j = idx + l * dir;
    int first = (idx < j) ? idx : j;
    int last  = (idx > j) ? idx : j;
    return make_int2(first, last);
}

__device__ __forceinline__ int find_split(const unsigned long long* keys, int n, int first, int last) {
    const int common_prefix = delta(keys, n, first, last);
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        const int new_split = split + step;
        if (new_split < last) {
            const int split_prefix = delta(keys, n, first, new_split);
            if (split_prefix > common_prefix) {
                split = new_split;
            }
        }
    } while (step > 1);

    return split;
}

__global__ void build_lbvh_internal(const unsigned long long* keys, int n,
                                    int* left, int* right, int* parent) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n - 1) return;

    const int2 range = determine_range(keys, n, idx);
    const int first = range.x;
    const int last  = range.y;

    const int split = find_split(keys, n, first, last);

    const int left_child  = (split == first) ? (split + (n - 1)) : split;
    const int right_child = (split + 1 == last) ? (split + 1 + (n - 1)) : (split + 1);

    left[idx] = left_child;
    right[idx] = right_child;

    parent[left_child] = idx;
    parent[right_child] = idx;
}

__device__ inline bool hit_lbvh(const ray& r,
                                const LBVH& bvh,
                                const hitable** d_list,
                                float tmin,
                                float tmax,
                                hit_record& out_rec)
{
    bool hit_anything = false;
    float closest = tmax;

    const int leaf_base = bvh.n - 1;

    int stack[128];
    int sp = 0;
    stack[sp++] = bvh.root;

    while (sp > 0) {
        const int node = stack[--sp];

        if (!hit_aabb(bvh.node_aabb[node], r, tmin, closest)) continue;

        if (node >= leaf_base) {
            const int leaf_i = node - leaf_base;
            const int prim = bvh.prim[leaf_i]; // 1..487
            hit_record rec;
            if (d_list[prim]->hit(r, tmin, closest, rec)) {
                hit_anything = true;
                closest = rec.t;
                out_rec = rec;
            }
        } else {
            const int l = bvh.left[node];
            const int rr = bvh.right[node];
            if (rr >= 0) stack[sp++] = rr;
            if (l >= 0)  stack[sp++] = l;
        }
    }

    return hit_anything;
}

/*******************************************************************************************/
/*******************************************************************************************/
/*******************************************************************************************/

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
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state, const LBVH& bvh, hitable** d_list) { //modified by YQWang
ray  cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f, 1.0f, 1.0f);
    for (int bounce = 0; bounce < 50; bounce++) {
		/*
        // If you want to keep exact original behavior as a "gate", keep this block.
        // If not needed, you can remove it entirely and only rely on hit_small/hit_ground.
        if (world != nullptr) {
            hit_record tmp;
            if (!(*world)->hit(cur_ray, 0.001f, FLT_MAX, tmp)) {
                vec3 unit_direction = unit_vector(cur_ray.direction());
                float t = 0.5f * (unit_direction.y() + 1.0f);
                vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
                return cur_attenuation * c;
            }
        }
		*/
        // 1) LBVH hit for small spheres
        hit_record rec_small;
        bool hit_small = hit_lbvh(cur_ray, bvh, (const hitable**)d_list,
                                  0.001f, FLT_MAX, rec_small);

        // 2) Ground hit (brute force), but cap tmax if LBVH already hit something
        hit_record rec_ground;
        bool hit_ground = d_list[0]->hit(cur_ray, 0.001f,
                                         hit_small ? rec_small.t : FLT_MAX,
                                         rec_ground);

        bool hit_anything = hit_small || hit_ground;

        if (!hit_anything) {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
            return cur_attenuation * c;
        }

        // 3) Choose nearest hit between ground and LBVH result
        hit_record rec;
        if (hit_small && hit_ground) {
            rec = (rec_ground.t < rec_small.t) ? rec_ground : rec_small;
        } else {
            rec = hit_small ? rec_small : rec_ground;
        }

        // 4) Scatter as in original
        ray  scattered;
        vec3 attenuation;
        if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        } else {
            return vec3(0.0, 0.0, 0.0);
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

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state, LBVH bvh, hitable **d_list) { //modified by YQWang
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
        //col += color(r, world, &local_rand_state);
        col += color(r, world, &local_rand_state, bvh, d_list);
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
        *d_world  = new hitable_list(d_list, 22*22+1+3);

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
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
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
    
    /*******************************************************************************************/
    // --- LBVH prep step 1: extract per-sphere AABB and centroid ---
    aabb *d_aabb;
    vec3 *d_centroid;
    checkCudaErrors(cudaMalloc((void **)&d_aabb, num_hitables * sizeof(aabb)));
    checkCudaErrors(cudaMalloc((void **)&d_centroid, num_hitables * sizeof(vec3)));

    int threads1 = 256;
    int blocks1 = (num_hitables + threads1 - 1) / threads1;
    compute_sphere_aabb_centroid<<<blocks1, threads1>>>(d_list, num_hitables, d_aabb, d_centroid);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // test point
    aabb *h_aabb = (aabb*)malloc(num_hitables * sizeof(aabb));
    checkCudaErrors(cudaMemcpy(h_aabb, d_aabb, num_hitables * sizeof(aabb), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "AABB[0] mn: " << h_aabb[0].mn << " mx: " << h_aabb[0].mx << "\n";
    std::cerr << "AABB[1] mn: " << h_aabb[1].mn << " mx: " << h_aabb[1].mx << "\n";

	// compute scene bbox on CPU using h_aabb
	std::cerr << "num_hitables = " << num_hitables << "\n";
	std::cerr << "h_aabb[0] mn: " << h_aabb[0].mn << " mx: " << h_aabb[0].mx << "\n";

	vec3 scene_min( FLT_MAX, FLT_MAX, FLT_MAX );
	vec3 scene_max(-FLT_MAX,-FLT_MAX,-FLT_MAX );

	for (int i = 1; i < num_hitables; i++) { // exclude the ground: i = 0
		const vec3 mn = h_aabb[i].mn;
		const vec3 mx = h_aabb[i].mx;

		if (i < 5) {
		    std::cerr << "bbox input i=" << i
		              << " mn: " << mn
		              << " mx: " << mx << "\n";
		}

		if (mn.x() < scene_min[0]) scene_min[0] = mn.x();
		if (mn.y() < scene_min[1]) scene_min[1] = mn.y();
		if (mn.z() < scene_min[2]) scene_min[2] = mn.z();
		if (mx.x() > scene_max[0]) scene_max[0] = mx.x();
		if (mx.y() > scene_max[1]) scene_max[1] = mx.y();
		if (mx.z() > scene_max[2]) scene_max[2] = mx.z();

		if (i < 5) {
		    std::cerr << "bbox running i=" << i
		              << " scene_min: " << scene_min
		              << " scene_max: " << scene_max << "\n";
		}
	}

	std::cerr << "Scene bbox mn: " << scene_min << " mx: " << scene_max << "\n";
	
	free(h_aabb);
	/*
	// allocate morton + primIdx
	unsigned int *d_morton;
	int *d_primIdx;
	checkCudaErrors(cudaMalloc((void **)&d_morton, num_hitables * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void **)&d_primIdx, num_hitables * sizeof(int)));

	compute_morton_codes<<<blocks1, threads1>>>(d_centroid, num_hitables, scene_min, scene_max, d_morton, d_primIdx);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// sort by morton
	thrust::device_ptr<unsigned int> morton_ptr(d_morton);
	thrust::device_ptr<int> prim_ptr(d_primIdx);
	thrust::sort_by_key(morton_ptr, morton_ptr + num_hitables, prim_ptr);
	
	// test point
	unsigned int h_morton[10];
	int h_idx[10];
	checkCudaErrors(cudaMemcpy(h_morton, d_morton, 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_idx, d_primIdx, 10 * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	for (int k = 0; k < 10; k++) {
		std::cerr << "sorted[" << k << "] morton=" << h_morton[k] << " prim=" << h_idx[k] << "\n";
	}
	*/
	const int n_bvh = num_hitables - 1; // 487, exclude prim 0
	
	unsigned long long* d_keys = nullptr;
	int* d_primIdx = nullptr;

	checkCudaErrors(cudaMalloc((void **)&d_keys, n_bvh * sizeof(unsigned long long)));
	checkCudaErrors(cudaMalloc((void **)&d_primIdx, n_bvh * sizeof(int)));

	compute_lbvh_keys<<<blocks1, threads1>>>(d_centroid, n_bvh, scene_min, scene_max, d_keys, d_primIdx);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::device_ptr<unsigned long long> keys_ptr(d_keys);
	thrust::device_ptr<int> prim_ptr(d_primIdx);
	thrust::sort_by_key(keys_ptr, keys_ptr + n_bvh, prim_ptr);

	unsigned long long h_keys[10];
	int h_idx[10];
	checkCudaErrors(cudaMemcpy(h_keys, d_keys, 10 * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_idx, d_primIdx, 10 * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	for (int k = 0; k < 10; k++) {
		unsigned int m = (unsigned int)(h_keys[k] >> 32);
		std::cerr << "sorted[" << k << "] morton=" << m << " prim=" << h_idx[k] << "\n";
	}

	LBVH bvh;
	bvh.n = n_bvh;

	checkCudaErrors(cudaMalloc((void**)&bvh.keys, bvh.n * sizeof(unsigned long long)));
	checkCudaErrors(cudaMalloc((void**)&bvh.prim, bvh.n * sizeof(int)));
	checkCudaErrors(cudaMemcpy(bvh.keys, d_keys, bvh.n * sizeof(unsigned long long), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(bvh.prim, d_primIdx, bvh.n * sizeof(int), cudaMemcpyDeviceToDevice));

	checkCudaErrors(cudaMalloc((void**)&bvh.left, (bvh.n - 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&bvh.right, (bvh.n - 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&bvh.parent, (2 * bvh.n - 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&bvh.node_aabb, (2 * bvh.n - 1) * sizeof(aabb)));
	checkCudaErrors(cudaMalloc((void**)&bvh.visit, (bvh.n - 1) * sizeof(int)));

	checkCudaErrors(cudaMemset(bvh.parent, 0xFF, (2 * bvh.n - 1) * sizeof(int))); // set to -1
	checkCudaErrors(cudaMemset(bvh.visit, 0, (bvh.n - 1) * sizeof(int)));

	int threads2 = 256;
	int blocks_internal = (bvh.n - 1 + threads2 - 1) / threads2;
	build_lbvh_internal<<<blocks_internal, threads2>>>(bvh.keys, bvh.n, bvh.left, bvh.right, bvh.parent);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int blocks_leaf = (bvh.n + threads2 - 1) / threads2;
	init_leaf_aabb<<<blocks_leaf, threads2>>>(bvh.n, bvh.prim, d_aabb, bvh.node_aabb);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	build_aabb_bottom_up<<<blocks_leaf, threads2>>>(bvh.n, bvh.left, bvh.right, bvh.parent, bvh.visit, bvh.node_aabb);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// test point: find root
	int *d_root;
	checkCudaErrors(cudaMalloc((void**)&d_root, sizeof(int)));
	int minus1 = -1;
	checkCudaErrors(cudaMemcpy(d_root, &minus1, sizeof(int), cudaMemcpyHostToDevice));

	find_root<<<blocks_internal, threads2>>>(bvh.n, bvh.parent, d_root);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(&bvh.root, d_root, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_root));

	std::cerr << "LBVH root internal index = " << bvh.root << "\n";

	aabb rootBox;
	checkCudaErrors(cudaMemcpy(&rootBox, &bvh.node_aabb[bvh.root], sizeof(aabb), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaDeviceSynchronize());
	std::cerr << "LBVH root AABB mn: " << rootBox.mn << " mx: " << rootBox.mx << "\n";
    /*******************************************************************************************/
    
    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    //render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state, bvh, d_list);

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
    /*******************************************************************************************/
    checkCudaErrors(cudaFree(d_aabb));
    checkCudaErrors(cudaFree(d_centroid));
	//checkCudaErrors(cudaFree(d_morton));
	checkCudaErrors(cudaFree(d_keys));
	checkCudaErrors(cudaFree(d_primIdx));
	
	checkCudaErrors(cudaFree(bvh.keys));
	checkCudaErrors(cudaFree(bvh.prim));
	checkCudaErrors(cudaFree(bvh.left));
	checkCudaErrors(cudaFree(bvh.right));
	checkCudaErrors(cudaFree(bvh.parent));
	checkCudaErrors(cudaFree(bvh.node_aabb));
	checkCudaErrors(cudaFree(bvh.visit));



    cudaDeviceReset();
}
