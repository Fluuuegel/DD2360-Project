#ifndef LBVHH
#define LBVHH

#include <stdint.h>
#include "aabb.h"

struct LBVH {
    int n;                 // number of leaves
    int root;              // internal node index of root (0..n-2)
    unsigned long long* keys; // sorted unique keys, size n
    int* prim;             // sorted primitive indices, size n

    int* left;             // internal nodes: size n-1, child indices in [0..2n-2)
    int* right;            // internal nodes: size n-1
    int* parent;           // all nodes: size 2n-1, parent index in internal space [0..n-2], root=-1

    aabb* node_aabb;       // all nodes: size 2n-1
    int* visit;            // internal nodes: size n-1 for bottom-up build
};

#endif

