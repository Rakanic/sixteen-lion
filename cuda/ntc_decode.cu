// Neural texture decoder, v1: NVFP4 latents, one stochastic (STF) tap per
// grid, MLP on Tensor Cores with FP16 mma.sync.m16n8k16.
//
// One warp decodes 16 pixels per iteration (the M dimension of the MMA).
// Layer sizes are fixed to the compressor's default model:
//   input 40 = grid0 16 latents + grid1 16 latents + 8 positional encodings
//   (padded to K = 48), hidden 64, hidden 64, output 9 (padded to N = 16).
// The output of one layer's MMAs (16x8 FP32 tiles) is reused directly as the
// next layer's A fragments (16x16 FP16): tile pairs (2j, 2j+1) form K-tile j,
// so no shuffles are needed between layers.
//
// Latent storage per grid (see python/sixteen/decoder.py, pack_nvfp4):
//   codes  uint8 [h*w*8]   two E2M1 codes per byte, low nibble first, texel-major
//   scales uint8 [h*w]     one E4M3 block scale per texel (16 values = 1 block)
//   gscale float           per-tensor scale
// Weights are pre-arranged on the host in MMA-fragment order (one uint2 per
// lane per (k-tile, n-tile)), so each lane loads its B fragment with one
// 8-byte, conflict-free shared-memory read.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int K1 = 3, N1 = 8;  // layer 1: 48 x 64
constexpr int K2 = 4, N2 = 8;  // layer 2: 64 x 64
constexpr int K3 = 4, N3 = 2;  // layer 3: 64 x 16 (9 used)
constexpr int W_FRAGS = K1 * N1 + K2 * N2 + K3 * N3;  // uint2 per lane
constexpr int OUT_CH = 9;

struct Grid {
  const uint8_t* codes;
  const uint8_t* scales;
  float gscale;
  int w, h;
};

__device__ __forceinline__ uint32_t pcg(uint32_t v) {
  uint32_t s = v * 747796405u + 2891336453u;
  uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// uniform in [0, 1) from 24 random bits; matches sixteen.decoder.stf_uniform
__device__ __forceinline__ float uniform(uint32_t seed) { return (pcg(seed) >> 8) * (1.0f / 16777216.0f); }

__device__ __forceinline__ float e2m1(uint32_t code) {
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float m = lut[code & 7u];
  return (code & 8u) ? -m : m;
}

// STF: pick one of the 4 bilinear taps with probability equal to its weight
// (separably: x0 + [u < fx], y0 + [v < fy]), wrapping at the borders.
__device__ __forceinline__ int stf_texel(const Grid& g, float u, float v, uint32_t seed) {
  // explicit rounding (no FMA contraction) so the tap matches the PyTorch reference bit for bit
  float x = __fadd_rn(__fmul_rn(u, float(g.w)), -0.5f), y = __fadd_rn(__fmul_rn(v, float(g.h)), -0.5f);
  float x0 = floorf(x), y0 = floorf(y);
  int xi = (int)x0 + (uniform(seed) < x - x0 ? 1 : 0);
  int yi = (int)y0 + (uniform(seed ^ 0x9e3779b9u) < y - y0 ? 1 : 0);
  xi = ((xi % g.w) + g.w) % g.w;
  yi = ((yi % g.h) + g.h) % g.h;
  return yi * g.w + xi;
}

// Two latents (channels c, c+1) of a texel, dequantized, as half2.
__device__ __forceinline__ __half2 latent_pair(const Grid& g, int texel, int c) {
  uint8_t byte = g.codes[texel * 8 + (c >> 1)];
  __nv_fp8_e4m3 s8;
  s8.__x = g.scales[texel];
  float s = float(s8) * g.gscale;
  return __floats2half2_rn(e2m1(byte & 0xfu) * s, e2m1(byte >> 4) * s);
}

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, uint2 b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y));
}

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }

__device__ __forceinline__ uint32_t relu_pack(float a, float b) {
  return h2u(__floats2half2_rn(fmaxf(a, 0.f), fmaxf(b, 0.f)));
}

// acc[n][4] (FP32 D tiles) -> A fragments a[k][4] for the next layer, with ReLU
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

__global__ void __launch_bounds__(256) decode_kernel(Grid g0, Grid g1, const float2* __restrict__ uv, int n_pix,
                                                     uint32_t frame, const uint2* __restrict__ wfrag_g,
                                                     const float* __restrict__ bias_g, float* __restrict__ out) {
  __shared__ uint2 wfrag[W_FRAGS * 32];
  __shared__ float bias[64 + 64 + 16];
  for (int i = threadIdx.x; i < W_FRAGS * 32; i += blockDim.x) wfrag[i] = wfrag_g[i];
  for (int i = threadIdx.x; i < 144; i += blockDim.x) bias[i] = bias_g[i];
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int gq = lane >> 2, tq = lane & 3;  // MMA fragment row group / thread-in-group
  const int warps = gridDim.x * (blockDim.x >> 5);
  const float two_pi = 6.283185307f;

  for (int base = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 16; base < n_pix; base += warps * 16) {
    // ---- layer-1 A fragments: rows gq and gq+8 of this 16-pixel tile
    uint32_t a1[K1][4];
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      int p = base + gq + half * 8;
      bool valid = p < n_pix;
      float2 q = valid ? uv[p] : make_float2(0.f, 0.f);
      uint32_t seed = pcg(uint32_t(p) * 2654435761u ^ frame);
      int t0 = stf_texel(g0, q.x, q.y, seed);
      int t1 = stf_texel(g1, q.x, q.y, pcg(seed));
      // K-tile 0: grid0 channels; K-tile 1: grid1 channels. Lane holds cols tq*2(+1) and tq*2+8(+9).
      a1[0][half] = h2u(latent_pair(g0, t0, tq * 2));
      a1[0][half + 2] = h2u(latent_pair(g0, t0, tq * 2 + 8));
      a1[1][half] = h2u(latent_pair(g1, t1, tq * 2));
      a1[1][half + 2] = h2u(latent_pair(g1, t1, tq * 2 + 8));
      // K-tile 2: positional encoding [sin px, sin py, cos px, cos py] for freq 1, 2; cols 8-15 are zero.
      float px = q.x * g0.w * two_pi, py = q.y * g0.h * two_pi;
      float f = (tq < 2) ? 1.f : 2.f;
      float e0, e1;
      if ((tq & 1) == 0) { e0 = __sinf(px * f); e1 = __sinf(py * f); }
      else               { e0 = __cosf(px * f); e1 = __cosf(py * f); }
      a1[2][half] = h2u(__floats2half2_rn(e0, e1));
      a1[2][half + 2] = 0u;
    }

    float acc1[N1][4], acc2[N2][4], acc3[N3][4];
    uint32_t a2[K2][4], a3[K3][4];
    layer<K1, N1>(a1, wfrag, bias, acc1, lane);
    to_next<N1>(acc1, a2);
    layer<K2, N2>(a2, wfrag + K1 * N1 * 32, bias + 64, acc2, lane);
    to_next<N2>(acc2, a3);
    layer<K3, N3>(a3, wfrag + (K1 * N1 + K2 * N2) * 32, bias + 128, acc3, lane);

    // ---- write 9 channels: D tile n covers cols n*8 + tq*2 (+1), rows gq / gq+8
#pragma unroll
    for (int n = 0; n < N3; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        int row = gq + (i >> 1) * 8, col = n * 8 + tq * 2 + (i & 1);
        int p = base + row;
        if (p < n_pix && col < OUT_CH) out[p * OUT_CH + col] = acc3[n][i];
      }
    }
  }
}

Grid make_grid(const torch::Tensor& codes, const torch::Tensor& scales, double gscale, int64_t w, int64_t h) {
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 && codes.numel() == w * h * 8, "bad codes");
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 && scales.numel() == w * h, "bad scales");
  return Grid{codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), float(gscale), int(w), int(h)};
}

}  // namespace

torch::Tensor decode(torch::Tensor codes0, torch::Tensor scales0, double gscale0, int64_t w0, int64_t h0,
                     torch::Tensor codes1, torch::Tensor scales1, double gscale1, int64_t w1, int64_t h1,
                     torch::Tensor uv, int64_t frame, torch::Tensor wfrag, torch::Tensor bias, int64_t blocks) {
  TORCH_CHECK(uv.is_cuda() && uv.dtype() == torch::kFloat32 && uv.dim() == 2 && uv.size(1) == 2, "uv: [N, 2] float32");
  TORCH_CHECK(wfrag.is_cuda() && wfrag.numel() * wfrag.element_size() == W_FRAGS * 32 * 8, "bad weight fragments");
  TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32 && bias.numel() == 144, "bad bias");
  int n = uv.size(0);
  auto out = torch::empty({n, OUT_CH}, uv.options());
  decode_kernel<<<int(blocks), 256>>>(make_grid(codes0, scales0, gscale0, w0, h0),
                                      make_grid(codes1, scales1, gscale1, w1, h1),
                                      reinterpret_cast<const float2*>(uv.data_ptr<float>()), n, uint32_t(frame),
                                      reinterpret_cast<const uint2*>(wfrag.data_ptr()), bias.data_ptr<float>(),
                                      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("decode", &decode, "NVFP4 neural texture decode (STF tap, FP16 MMA)"); }
