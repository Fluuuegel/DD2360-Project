#ifndef MORTONH
#define MORTONH

__device__ __forceinline__ unsigned int expand_bits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __forceinline__ unsigned int morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);

    const unsigned int xx = expand_bits((unsigned int)x);
    const unsigned int yy = expand_bits((unsigned int)y);
    const unsigned int zz = expand_bits((unsigned int)z);

    return (xx << 2) | (yy << 1) | zz;
}

#endif

