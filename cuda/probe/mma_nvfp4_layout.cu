// Probe: register layout of mma.m16n8k64 kind::mxf4nvf4 scale_vec::4X (NVFP4).
// Random E2M1 A[16x64], B[64x8] and UE4M3 scales SA[16x4], SB[8x4] (one per
// 16 K values); each lane packs its registers from the full matrices using a
// hypothesized layout, and the result is compared with a CPU reference.
// Several hypotheses for the scale-A lane mapping are tried; exactly one
// should match.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

__device__ __host__ inline float e2m1(int c) {
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (c & 8) ? -lut[c & 7] : lut[c & 7];
}
__host__ inline float ue4m3(uint8_t b) {  // only the codes used here: 2^((b>>3)-7), mantissa 0
  return std::ldexp(1.0f, (b >> 3) - 7);
}

// hypothesis for which lane holds the 4 scale bytes of A row r (thread-id-a = 0)
__device__ int sa_lane(int hyp, int r) {
  switch (hyp) {
    case 0: return 4 * (r % 8) + (r / 8);      // rows 8-15 from the next thread of the quad
    case 1: return 4 * (r % 8) + 2 * (r / 8);  // rows 8-15 two threads over
    case 2: return 4 * (r % 8) + 3 * (r / 8);  // rows 8-15 three threads over
    default: return 4 * (r % 8);               // all rows from thread 0 of the quad (rows 8-15 alias)
  }
}

__global__ void run(const uint8_t* A, const uint8_t* B, const uint8_t* SA, const uint8_t* SB, float* D, int hyp) {
  int lane = threadIdx.x, gq = lane >> 2, tq = lane & 3;
  uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 8; ++i) {
      int row = gq + 8 * (j & 1), col = tq * 8 + i + 32 * (j >> 1);
      a[j] |= uint32_t(A[row * 64 + col]) << (4 * i);
    }
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 8; ++i) {
      int k = tq * 8 + i + 32 * j, n = gq;
      b[j] |= uint32_t(B[k * 8 + n]) << (4 * i);
    }
  // scale registers: lanes named by the hypothesis get their row's 4 bytes; others get 8.0 (a valid, wrong scale)
  uint32_t sa = 0x50505050u, sb = 0;
  for (int r = 0; r < 16; ++r)
    if (sa_lane(hyp, r) == lane) {
      sa = 0;
      for (int k = 0; k < 4; ++k) sa |= uint32_t(SA[r * 4 + k]) << (8 * k);
    }
  for (int k = 0; k < 4; ++k) sb |= uint32_t(SB[gq * 4 + k]) << (8 * k);  // every lane of quad gq: column gq
  float d[4] = {0, 0, 0, 0};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
  for (int i = 0; i < 4; ++i) {
    int row = gq + 8 * (i >> 1), col = tq * 2 + (i & 1);
    D[row * 8 + col] = d[i];
  }
}

int main() {
  srand(1);
  uint8_t A[16 * 64], B[64 * 8], SA[16 * 4], SB[8 * 4];
  const uint8_t scodes[4] = {0x30, 0x38, 0x40, 0x48};  // 0.5, 1, 2, 4
  for (auto& x : A) x = rand() % 16;
  for (auto& x : B) x = rand() % 16;
  for (auto& x : SA) x = scodes[rand() % 4];
  for (auto& x : SB) x = scodes[rand() % 4];
  float ref[16 * 8];
  for (int r = 0; r < 16; ++r)
    for (int n = 0; n < 8; ++n) {
      double s = 0;
      for (int k = 0; k < 64; ++k) s += e2m1(A[r * 64 + k]) * e2m1(B[k * 8 + n]) * ue4m3(SA[r * 4 + k / 16]) * ue4m3(SB[n * 4 + k / 16]);
      ref[r * 8 + n] = float(s);
    }
  uint8_t *dA, *dB, *dSA, *dSB; float* dD;
  cudaMalloc(&dA, sizeof(A)); cudaMalloc(&dB, sizeof(B)); cudaMalloc(&dSA, sizeof(SA)); cudaMalloc(&dSB, sizeof(SB));
  cudaMalloc(&dD, sizeof(ref));
  cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice); cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice);
  cudaMemcpy(dSA, SA, sizeof(SA), cudaMemcpyHostToDevice); cudaMemcpy(dSB, SB, sizeof(SB), cudaMemcpyHostToDevice);
  for (int hyp = 0; hyp < 4; ++hyp) {
    run<<<1, 32>>>(dA, dB, dSA, dSB, dD, hyp);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("hyp %d: CUDA error\n", hyp); return 1; }
    float D[16 * 8];
    cudaMemcpy(D, dD, sizeof(D), cudaMemcpyDeviceToHost);
    double err = 0; int bad_rows = 0;
    for (int r = 0; r < 16; ++r) {
      double re = 0;
      for (int n = 0; n < 8; ++n) { double e = fabs(D[r * 8 + n] - ref[r * 8 + n]); re = (e != e || e > re) ? (e != e ? 1e30 : e) : re; }
      err = fmax(err, re); bad_rows += re > 1e-3;
    }
    printf("scale-A hypothesis %d: max |D - ref| = %g, rows wrong = %d/16\n", hyp, err, bad_rows);
  }
  return 0;
}
