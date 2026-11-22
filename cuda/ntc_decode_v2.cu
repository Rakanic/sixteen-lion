// Neural texture decoder, v2: 4-neighbour (concat4) model, NVFP4 latents.
//
// Layer 1 has two interchangeable paths (template FP4):
//   FP4 = true  "FP4-direct": per grid, the 4 neighbouring latent texels
//               (4 x 16 channels = K 64, each texel one NVFP4 block) go into
//               mma.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X as the
//               stored E2M1 codes and UE4M3 scales; the weights are NVFP4 too.
//               The four neighbouring latent texels are the four scale blocks
//               of one FP4 MMA.
//   FP4 = false "FP16-unpack": the same values dequantized to FP16 and
//               multiplied with mma.m16n8k16 (8 K-tiles for the latents).
// Positional encoding + fractional positions (12 values, one K-16 tile) and
// layers 2-3 use FP16 MMA in both paths.
//
// Filtering is in output space: each pixel picks one full-resolution texel
// per frame (STF, probability = bilinear weight) and decodes its center.
//
// Register layouts (verified by cuda/probe/mma_nvfp4_layout.cu), lane =
// 4 * gq + tq:
//   A (16 x 64 E2M1): a0 row gq, cols tq*8..+7; a1 row gq+8; a2/a3 cols +32
//   B (64 x 8 E2M1):  b0 k = tq*8..+7, col gq;  b1 k + 32
//   scale A: row r's 4 UE4M3 bytes (one per 16 K) in lane 4*(r%8) + r/8
//   scale B: column gq's 4 bytes in every lane of quad gq
// With 16 channels per texel, cols tq*8..+7 are 8 consecutive channels of
// texel tq/2: one aligned 32-bit load of packed codes.

#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int OUT_CH = 9;
constexpr int N1 = 8;                                   // 64 hidden / 8
constexpr int K2 = 4, N2 = 8, K3 = 4, N3 = 2;           // layers 2-3 as in v1
constexpr int L23_FRAGS = K2 * N2 + K3 * N3;            // uint2 per lane
constexpr int W1_FP16_FRAGS = 8 * N1;                   // 8 latent K-tiles x 8 N-tiles
constexpr int PE_FRAGS = N1;                            // 1 K-tile x 8 N-tiles

struct Grid {
  const uint8_t* codes;   // [h*w*8], two E2M1 codes per byte, low nibble first
  const uint8_t* scales;  // [h*w] UE4M3, one per texel
  float gscale;
  int w, h;
};

struct Weights {
  const uint2* w1_fp4;      // [2 grids][8 n][32 lanes] B registers (FP4 path)
  const uint32_t* w1_sf;    // [2][8][32] B scale registers (FP4 path)
  const uint2* w1_fp16;     // [8 k][8 n][32] FP16 B fragments (unpack path)
  const uint2* w1_pe;       // [8 n][32] FP16 B fragments for PE + fractions
  const uint2* w23;         // layers 2-3, as v1
  const float* bias;        // [64 + 64 + 16]
  float k0, k1;             // FP4 path: per-tensor scales, latent grid g x weights
};

__device__ __forceinline__ uint32_t pcg(uint32_t v) {
  uint32_t s = v * 747796405u + 2891336453u;
  uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
__device__ __forceinline__ float uniform(uint32_t seed) { return (pcg(seed) >> 8) * (1.0f / 16777216.0f); }

__device__ __forceinline__ float e2m1(uint32_t code) {
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float m = lut[code & 7u];
  return (code & 8u) ? -m : m;
}

__device__ __forceinline__ int wrap(int i, int n) { return ((i % n) + n) % n; }

// position x in texel units (x = u * res - 0.5): floor and fraction, IEEE-rounded to match PyTorch
__device__ __forceinline__ float texel_coord(float u, int res) { return __fadd_rn(__fmul_rn(u, float(res)), -0.5f); }

struct Taps {
  int t[2][4];      // per grid: (x0,y0), (x1,y0), (x0,y1), (x1,y1)
  float f[2][2];    // per grid: fx, fy
  float u, v;       // decoded texel center
};

__device__ __forceinline__ Taps make_taps(const Grid* g, float u, float v, int W, int H, uint32_t seed) {
  Taps T;
  // output-space STF: one full-resolution texel, chosen with its bilinear weight
  float x = texel_coord(u, W), y = texel_coord(v, H);
  float x0 = floorf(x), y0 = floorf(y);
  int tx = wrap((int)x0 + (uniform(seed) < x - x0 ? 1 : 0), W);
  int ty = wrap((int)y0 + (uniform(seed ^ 0x9e3779b9u) < y - y0 ? 1 : 0), H);
  T.u = __fdiv_rn(float(tx) + 0.5f, float(W));
  T.v = __fdiv_rn(float(ty) + 0.5f, float(H));
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    float gx = texel_coord(T.u, g[k].w), gy = texel_coord(T.v, g[k].h);
    float gx0 = floorf(gx), gy0 = floorf(gy);
    T.f[k][0] = gx - gx0;
    T.f[k][1] = gy - gy0;
    int xa = wrap((int)gx0, g[k].w), ya = wrap((int)gy0, g[k].h);
    int xb = (xa + 1) % g[k].w, yb = (ya + 1) % g[k].h;
    T.t[k][0] = ya * g[k].w + xa;
    T.t[k][1] = ya * g[k].w + xb;
    T.t[k][2] = yb * g[k].w + xa;
    T.t[k][3] = yb * g[k].w + xb;
  }
  return T;
}

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ uint32_t relu_pack(float a, float b) {
  return h2u(__floats2half2_rn(fmaxf(a, 0.f), fmaxf(b, 0.f)));
}

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}

__device__ __forceinline__ void mma_nvfp4(float* d, const uint32_t* a, uint2 b, uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "r"(sa), "r"(sb));
}

__device__ __forceinline__ __half2 latent_pair(const Grid& g, int texel, int c) {
  uint8_t byte = g.codes[texel * 8 + (c >> 1)];
  __nv_fp8_e4m3 s8;
  s8.__x = g.scales[texel];
  float s = float(s8) * g.gscale;
  return __floats2half2_rn(e2m1(byte & 0xfu) * s, e2m1(byte >> 4) * s);
}

// Fast unpack for the FP16 baseline: one byte (2 E2M1 codes) -> half2 with the
// hardware conversion, then one half2 multiply by the texel's scale.
__device__ __forceinline__ uint32_t cvt_e2m1x2(uint32_t byte) {
  uint32_t out;
  asm("{\n .reg .b8 t;\n cvt.u8.u32 t, %1;\n cvt.rn.f16x2.e2m1x2 %0, t;\n}\n" : "=r"(out) : "r"(byte));
  return out;
}

__device__ __forceinline__ __half2 texel_scale(const Grid& g, int texel) {
  __nv_fp8_e4m3 s8;
  s8.__x = g.scales[texel];
  return __float2half2_rn(float(s8) * g.gscale);
}

template <int N>
__device__ __forceinline__ void to_next(const float (*acc)[4], uint32_t (*a)[4]) {
#pragma unroll
  for (int j = 0; j < N / 2; ++j) {
    a[j][0] = relu_pack(acc[2 * j][0], acc[2 * j][1]);
    a[j][1] = relu_pack(acc[2 * j][2], acc[2 * j][3]);
    a[j][2] = relu_pack(acc[2 * j + 1][0], acc[2 * j + 1][1]);
    a[j][3] = relu_pack(acc[2 * j + 1][2], acc[2 * j + 1][3]);
  }
}

template <int K, int N>
__device__ __forceinline__ void layer(const uint32_t (*a)[4], const uint2* wfrag, const float* bias, float (*acc)[4],
                                      int lane) {
  const int tq = lane & 3;
#pragma unroll
  for (int n = 0; n < N; ++n) {
    float b0 = bias[n * 8 + tq * 2], b1 = bias[n * 8 + tq * 2 + 1];
    acc[n][0] = b0; acc[n][1] = b1; acc[n][2] = b0; acc[n][3] = b1;
#pragma unroll
    for (int k = 0; k < K; ++k) mma16816(acc[n], a[k], wfrag[(k * N + n) * 32 + lane]);
  }
}

#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1  // blocks per SM the compiler must fit (caps registers; set with -DMIN_BLOCKS=n)
#endif

template <bool FP4>
__global__ void __launch_bounds__(256, MIN_BLOCKS) decode_v2(Grid g0, Grid g1, int W, int H, const float2* __restrict__ uv, int n_pix,
                                                 uint32_t frame, Weights wt, float* __restrict__ out) {
  constexpr int W1_WORDS = FP4 ? (2 * N1 * 32) : (W1_FP16_FRAGS * 32);
  __shared__ uint2 s_w1[W1_WORDS];
  __shared__ uint32_t s_sf[FP4 ? 2 * N1 * 32 : 1];
  __shared__ uint2 s_pe[PE_FRAGS * 32];
  __shared__ uint2 s_w23[L23_FRAGS * 32];
  __shared__ float s_bias[144];
  for (int i = threadIdx.x; i < W1_WORDS; i += blockDim.x) s_w1[i] = FP4 ? wt.w1_fp4[i] : wt.w1_fp16[i];
  if (FP4)
    for (int i = threadIdx.x; i < 2 * N1 * 32; i += blockDim.x) s_sf[i] = wt.w1_sf[i];
  for (int i = threadIdx.x; i < PE_FRAGS * 32; i += blockDim.x) s_pe[i] = wt.w1_pe[i];
  for (int i = threadIdx.x; i < L23_FRAGS * 32; i += blockDim.x) s_w23[i] = wt.w23[i];
  for (int i = threadIdx.x; i < 144; i += blockDim.x) s_bias[i] = wt.bias[i];
  __syncthreads();

  const Grid grids[2] = {g0, g1};
  const int lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
  const int warps = gridDim.x * (blockDim.x >> 5);
  const float two_pi = 6.283185307f;

  for (int base = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 16; base < n_pix; base += warps * 16) {
    Taps T[2];  // rows gq and gq + 8
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int p = base + gq + 8 * h;
      float2 q = p < n_pix ? uv[p] : make_float2(0.f, 0.f);
      T[h] = make_taps(grids, q.x, q.y, W, H, pcg(uint32_t(p) * 2654435761u ^ frame));
    }

    // ---- PE + fractions tile (K 16): cols 0-7 PE [sin x, sin y, cos x, cos y] x freq 1, 2;
    //      cols 8-11 fractions fx0, fy0, fx1, fy1; cols 12-15 zero
    uint32_t ape[4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float px = T[h].u * g0.w * two_pi, py = T[h].v * g0.h * two_pi;
      float f = (tq < 2) ? 1.f : 2.f;
      float e0, e1;
      if ((tq & 1) == 0) { e0 = __sinf(px * f); e1 = __sinf(py * f); }
      else               { e0 = __cosf(px * f); e1 = __cosf(py * f); }
      ape[h] = h2u(__floats2half2_rn(e0, e1));
      ape[h + 2] = tq < 2 ? h2u(__floats2half2_rn(T[h].f[tq][0], T[h].f[tq][1])) : 0u;
    }

    float acc1[N1][4];
    if constexpr (FP4) {
      // ---- FP4-direct: one m16n8k64 per grid per 8 neurons
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const Grid& g = grids[k];
        const uint32_t* codes32 = reinterpret_cast<const uint32_t*>(g.codes);
        uint32_t a[4];
        a[0] = codes32[T[0].t[k][tq >> 1] * 2 + (tq & 1)];
        a[1] = codes32[T[1].t[k][tq >> 1] * 2 + (tq & 1)];
        a[2] = codes32[T[0].t[k][2 + (tq >> 1)] * 2 + (tq & 1)];
        a[3] = codes32[T[1].t[k][2 + (tq >> 1)] * 2 + (tq & 1)];
        uint32_t sa = 0;
        if (tq < 2) {  // lane 4*gq + 0 holds row gq, lane 4*gq + 1 holds row gq + 8
          const Taps& R = T[tq];
#pragma unroll
          for (int j = 0; j < 4; ++j) sa |= uint32_t(g.scales[R.t[k][j]]) << (8 * j);
        }
        // grid 0 starts from zero; grid 1 continues from grid 0 rescaled, then x k1
        const float carry = (k == 0) ? 0.f : wt.k0 / wt.k1;
#pragma unroll
        for (int n = 0; n < N1; ++n) {
#pragma unroll
          for (int i = 0; i < 4; ++i) acc1[n][i] = (k == 0) ? 0.f : acc1[n][i] * carry;
          int f = (k * N1 + n) * 32 + lane;
          mma_nvfp4(acc1[n], a, s_w1[f], sa, s_sf[f]);
        }
      }
#pragma unroll
      for (int n = 0; n < N1; ++n) {
        float b0 = s_bias[n * 8 + tq * 2], b1 = s_bias[n * 8 + tq * 2 + 1];
        acc1[n][0] = acc1[n][0] * wt.k1 + b0; acc1[n][1] = acc1[n][1] * wt.k1 + b1;
        acc1[n][2] = acc1[n][2] * wt.k1 + b0; acc1[n][3] = acc1[n][3] * wt.k1 + b1;
        mma16816(acc1[n], ape, s_pe[n * 32 + lane]);
      }
    } else {
      // ---- FP16-unpack: K-tile kt = grid kt/4, texel kt%4 (16 channels = one texel)
#pragma unroll
      for (int n = 0; n < N1; ++n) {
        float b0 = s_bias[n * 8 + tq * 2], b1 = s_bias[n * 8 + tq * 2 + 1];
        acc1[n][0] = b0; acc1[n][1] = b1; acc1[n][2] = b0; acc1[n][3] = b1;
      }
#pragma unroll
      for (int kt = 0; kt < 8; ++kt) {
        const Grid& g = grids[kt >> 2];
        uint32_t a[4];
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          int texel = T[h].t[kt >> 2][kt & 3];
          __half2 sc = texel_scale(g, texel);  // one scale load per texel
          uint32_t lo = cvt_e2m1x2(g.codes[texel * 8 + tq]);      // channels tq*2, tq*2+1
          uint32_t hi = cvt_e2m1x2(g.codes[texel * 8 + 4 + tq]);  // channels tq*2+8, +9
          a[h] = h2u(__hmul2(*reinterpret_cast<__half2*>(&lo), sc));
          a[h + 2] = h2u(__hmul2(*reinterpret_cast<__half2*>(&hi), sc));
        }
#pragma unroll
        for (int n = 0; n < N1; ++n) mma16816(acc1[n], a, s_w1[(kt * N1 + n) * 32 + lane]);
      }
#pragma unroll
      for (int n = 0; n < N1; ++n) mma16816(acc1[n], ape, s_pe[n * 32 + lane]);
    }

    float acc2[N2][4], acc3[N3][4];
    uint32_t a2[K2][4], a3[K3][4];
    to_next<N1>(acc1, a2);
    layer<K2, N2>(a2, s_w23, s_bias + 64, acc2, lane);
    to_next<N2>(acc2, a3);
    layer<K3, N3>(a3, s_w23 + K2 * N2 * 32, s_bias + 128, acc3, lane);

#pragma unroll
    for (int n = 0; n < N3; ++n)
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int row = gq + (i >> 1) * 8, col = n * 8 + tq * 2 + (i & 1);
        int p = base + row;
        if (p < n_pix && col < OUT_CH) out[p * OUT_CH + col] = acc3[n][i];
      }
  }
}

Grid make_grid(const torch::Tensor& codes, const torch::Tensor& scales, double gscale, int64_t w, int64_t h) {
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.numel() == w * h * 8, "bad codes");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 && scales.numel() == w * h, "bad scales");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(codes.data_ptr()) % 4 == 0, "codes must be 4-byte aligned");
  return Grid{codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), float(gscale), int(w), int(h)};
}

template <typename T>
const T* ptr(const torch::Tensor& t, int64_t bytes, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.numel() * t.element_size() == bytes, name, ": wrong size");
  return reinterpret_cast<const T*>(t.data_ptr());
}

}  // namespace

torch::Tensor decode(torch::Tensor codes0, torch::Tensor scales0, double gscale0, int64_t w0, int64_t h0,
                     torch::Tensor codes1, torch::Tensor scales1, double gscale1, int64_t w1, int64_t h1,
                     int64_t W, int64_t H, torch::Tensor uv, int64_t frame, torch::Tensor w1_fp4, torch::Tensor w1_sf,
                     torch::Tensor w1_fp16, torch::Tensor w1_pe, torch::Tensor w23, torch::Tensor bias, double k0,
                     double k1, bool fp4, int64_t blocks) {
  TORCH_CHECK(uv.is_cuda() && uv.dtype() == torch::kFloat32 && uv.dim() == 2 && uv.size(1) == 2, "uv: [N, 2] float32");
  Weights wt{ptr<uint2>(w1_fp4, 2 * N1 * 32 * 8, "w1_fp4"), ptr<uint32_t>(w1_sf, 2 * N1 * 32 * 4, "w1_sf"),
             ptr<uint2>(w1_fp16, W1_FP16_FRAGS * 32 * 8, "w1_fp16"), ptr<uint2>(w1_pe, PE_FRAGS * 32 * 8, "w1_pe"),
             ptr<uint2>(w23, L23_FRAGS * 32 * 8, "w23"), ptr<float>(bias, 144 * 4, "bias"), float(k0), float(k1)};
  int n = uv.size(0);
  auto out = torch::empty({n, OUT_CH}, uv.options());
  Grid a = make_grid(codes0, scales0, gscale0, w0, h0), b = make_grid(codes1, scales1, gscale1, w1, h1);
  auto q = reinterpret_cast<const float2*>(uv.data_ptr<float>());
  if (fp4)
    decode_v2<true><<<int(blocks), 256>>>(a, b, int(W), int(H), q, n, uint32_t(frame), wt, out.data_ptr<float>());
  else
    decode_v2<false><<<int(blocks), 256>>>(a, b, int(W), int(H), q, n, uint32_t(frame), wt, out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode", &decode, "concat4 NVFP4 neural texture decode (FP4-direct or FP16-unpack layer 1)");
}
