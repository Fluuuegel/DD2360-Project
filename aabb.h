#ifndef AABBH
#define AABBH

#include <float.h>
#include "vec3.h"
#include "ray.h"

struct aabb {
    vec3 mn;
    vec3 mx;

    __host__ __device__ aabb() {}
    __host__ __device__ aabb(const vec3& a, const vec3& b) : mn(a), mx(b) {}
};

__host__ __device__ inline aabb surrounding_box(const aabb& box0, const aabb& box1) {
    const vec3 small(
        (box0.mn.x() < box1.mn.x()) ? box0.mn.x() : box1.mn.x(),
        (box0.mn.y() < box1.mn.y()) ? box0.mn.y() : box1.mn.y(),
        (box0.mn.z() < box1.mn.z()) ? box0.mn.z() : box1.mn.z()
    );
    const vec3 big(
        (box0.mx.x() > box1.mx.x()) ? box0.mx.x() : box1.mx.x(),
        (box0.mx.y() > box1.mx.y()) ? box0.mx.y() : box1.mx.y(),
        (box0.mx.z() > box1.mx.z()) ? box0.mx.z() : box1.mx.z()
    );
    return aabb(small, big);
}

__device__ inline bool hit_aabb(const aabb& box, const ray& r, float tmin, float tmax) {
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (box.mn[a] - r.origin()[a]) * invD;
        float t1 = (box.mx[a] - r.origin()[a]) * invD;
        if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
        tmin = t0 > tmin ? t0 : tmin;
        tmax = t1 < tmax ? t1 : tmax;
        if (tmax <= tmin) return false;
    }
    return true;
}

#endif

