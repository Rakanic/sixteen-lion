// Probe: NVFP4 variant, kind::mxf4nvf4 with scale_vec::4X UE4M3 (one scale per 16 K values).
// One warp computes D[16x8] = A[16x64] * B[64x8] with every element = 1.0 (E2M1 code 0x2).
#include <cstdio>
#include <cstdint>

__global__ void probe(float* out, uint32_t scale_a, uint32_t scale_b) {
  uint32_t a[4], b[2];
  const uint32_t ones = 0x22222222u;  // eight E2M1 values of 1.0 per 32-bit register
  for (int i = 0; i < 4; ++i) a[i] = ones;
  for (int i = 0; i < 2; ++i) b[i] = ones;
  float d[4] = {0, 0, 0, 0};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(scale_a), "r"(scale_b));
  for (int i = 0; i < 4; ++i) out[threadIdx.x * 4 + i] = d[i];
}

int main() {
  float* d_out; float h[128];
  cudaMalloc(&d_out, sizeof(h));
  // UE4M3 scale bytes: 0x38 = 1.0, 0x40 = 2.0 (one per 16 K values); replicated in all 4 bytes of the register.
  uint32_t s1 = 0x38383838u, s2 = 0x40404040u;
  for (auto [sa, label] : {std::pair{s1, "scale 1"}, std::pair{s2, "scale_a 2"}}) {
    probe<<<1, 32>>>(d_out, sa, s1);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("%s: CUDA error %s\n", label, cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost);
    float mn = h[0], mx = h[0];
    for (float v : h) { mn = v < mn ? v : mn; mx = v > mx ? v : mx; }
    printf("%s: D in [%g, %g] (expect %g)\n", label, mn, mx, sa == s1 ? 64.0 : 128.0);
  }
  return 0;
}
