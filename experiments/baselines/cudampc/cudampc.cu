// CUDA MPC -- a re-implementation of
//   B. Akbari and M. Greeff, "CUDA MPC: A GPU-Native Solver for Model
//   Predictive Control", arXiv:2608.03051, 2026.
// The authors' code is not public ("released upon acceptance"), so this is
// written from Sec. V of the paper:
//
//   * algorithm   parallel-in-horizon ADMM of piMPC (Wu et al., 2026),
//                 eqs. (8)-(13): stagewise primal update, projection/consensus
//                 update, scaled dual update;
//   * execution   ONE persistent kernel per solve, launched with
//                 cudaLaunchCooperativeKernel so every block is co-resident;
//                 no host involvement between ADMM iterations;
//   * memory      each block owns a contiguous sub-horizon of M stages and
//                 keeps every iterate (and, when they fit, the system
//                 matrices) in shared memory for the whole solve;
//   * sync        no grid-wide barrier: each block publishes its boundary
//                 values and bumps a 64-bit counter; neighbours spin only on
//                 the counters of blocks b-1 and b+1.
//
// The neighbour protocol below waits for the *current* boundary values, so the
// iterates are identical to synchronous (Jacobi) piMPC -- the paper's variant
// admits a one-iteration delay.  Identical iterates let us validate the kernel
// against a NumPy reference to float32 round-off.
//
// LTI specialisation: the paper caches per-stage matrices because its SCP
// outer loop re-linearises at every stage.  For the linear MPC problems used
// here the matrices are stage-invariant, so ONE copy per block is kept in
// shared memory.  That only lowers the shared-memory footprint, i.e. it can
// only help CUDA MPC.
//
// Iteration (scaled variables, stage k = 0..N-1, xb_k = [x_k; u_{k-1}]):
//   du_k     = J (v_k - beta_k)
//   xb_{k+1} = H  (q_k + rho (z_k - th_k + Ab' (z_{k+1} - v_{k+1} + la_{k+1} - e)))   k < N-1
//   xb_N     = HN (q_{N-1} + rho (z_{N-1} - th_{N-1}))
//   bu = Bb du_k,  ax = Ab xb_k
//   z_k   = Proj( (2 (xb_{k+1} + th_k) + bu + beta_k + ax + e - la_k) / 3 )
//   v_k   = (z_k + bu + beta_k - ax - e + la_k) / 2
//   th_k += xb_{k+1} - z_k;  beta_k += bu - v_k;  la_k += z_k - ax - v_k - e

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "cudampc: %s at %s:%d\n", cudaGetErrorString(e_),        \
              __FILE__, __LINE__);                                             \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static constexpr int MAX_NB = 256;  // nb = nx + nu upper bound (kernel arg)

template <typename T>
struct X0Arg {
  T v[MAX_NB];
};

template <typename T>
struct Params {
  int nx, nu, nb, N, M, B;
  T rho;
  int iters;
  int warm;
  int mats_shared;
  unsigned long long base;  // counter value every block holds at launch
  // device buffers
  // Matrices are stored TRANSPOSED (column-major) except Ab, so that thread r
  // reads word j*nb + r: consecutive threads hit consecutive banks.
  const T *Ab, *AbT, *BbT, *JT, *HT, *HNT, *lo, *hi, *e, *q;  // q: N x nb
  T *Xs, *Zs, *Vs, *Ths, *Bes, *Las, *DUs;           // persistent state
  T *gX;    // B x nb   : last primal xb of each block
  T *gZVL;  // B x 3nb  : first-stage z, v, la of each block
  unsigned long long *flag;  // B counters
  T *uout;  // N x nu (scaled u part of xb_{k+1})
};

__device__ __forceinline__ unsigned long long ld_acquire(
    const unsigned long long *p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release(unsigned long long *p,
                                           unsigned long long v) {
  asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

template <typename T>
__device__ __forceinline__ T ldcg(const T *p) { return __ldcg(p); }

template <typename T>
__global__ void __launch_bounds__(1024)
cudampc_kernel(Params<T> P, X0Arg<T> x0) {
  extern __shared__ unsigned char smem_raw[];
  T *sm = reinterpret_cast<T *>(smem_raw);

  const int nb = P.nb, nu = P.nu, nx = P.nx, M = P.M, N = P.N;
  const int b = blockIdx.x;
  const int tid = threadIdx.x;
  const int s = tid / nb;        // local stage
  const int r = tid - s * nb;    // row
  const int k = b * M + s;       // global stage
  const bool act = (s < M) && (k < N);
  const int Mloc = min(M, N - b * M);  // stages owned by this block
  const int klast = b * M + Mloc - 1;

  // ---- shared-memory carve-up -------------------------------------------
  T *p = sm;
  const T *Ab, *AbT, *BbT, *JT, *HT, *HNT;
  if (P.mats_shared) {
    T *a = p; p += nb * nb;
    T *at = p; p += nb * nb;
    T *bb = p; p += nb * nu;
    T *j = p; p += nu * nb;
    T *h = p; p += nb * nb;
    T *hn = p; p += nb * nb;
    for (int i = tid; i < nb * nb; i += blockDim.x) {
      a[i] = P.Ab[i]; at[i] = P.AbT[i]; h[i] = P.HT[i]; hn[i] = P.HNT[i];
    }
    for (int i = tid; i < nb * nu; i += blockDim.x) { bb[i] = P.BbT[i]; j[i] = P.JT[i]; }
    Ab = a; AbT = at; BbT = bb; JT = j; HT = h; HNT = hn;
  } else {
    Ab = P.Ab; AbT = P.AbT; BbT = P.BbT; JT = P.JT; HT = P.HT; HNT = P.HNT;
  }
  T *lo = p; p += nb;
  T *hi = p; p += nb;
  T *e = p; p += nb;
  T *xprev = p; p += nb;            // xb_k of the first local stage
  T *X = p; p += M * nb;            // xb_{k+1}
  T *Z = p; p += (M + 1) * nb;      // slot M = halo (next block's stage 0)
  T *V = p; p += (M + 1) * nb;
  T *La = p; p += (M + 1) * nb;
  T *Th = p; p += M * nb;
  T *Be = p; p += M * nb;
  T *Q = p; p += M * nb;
  T *t1 = p; p += M * nb;           // scratch
  T *t2 = p; p += M * nb;           // scratch
  T *DU = p; p += M * nu;

  for (int i = tid; i < nb; i += blockDim.x) {
    lo[i] = P.lo[i]; hi[i] = P.hi[i]; e[i] = P.e[i];
  }
  if (b == 0)
    for (int i = tid; i < nb; i += blockDim.x) xprev[i] = x0.v[i];

  const int gi = k * nb + r;
  if (act) {
    Q[s * nb + r] = P.q[gi];
    if (P.warm) {
      // warm == 1: previous solution as is; warm == 2: shifted by one stage,
      // last stage repeated (piMPC's receding-horizon warm start).  Stage k+1
      // may belong to block b+1, which writes it back only after it has
      // passed the first neighbour barrier -- i.e. after this load.
      const int ks = (P.warm == 2) ? min(k + 1, N - 1) : k;
      const int gs = ks * nb + r;
      X[s * nb + r] = P.Xs[gs]; Z[s * nb + r] = P.Zs[gs];
      V[s * nb + r] = P.Vs[gs]; La[s * nb + r] = P.Las[gs];
      Th[s * nb + r] = P.Ths[gs]; Be[s * nb + r] = P.Bes[gs];
      if (r < nu) DU[s * nu + r] = P.DUs[ks * nu + r];
    } else {
      X[s * nb + r] = 0; Z[s * nb + r] = 0; V[s * nb + r] = 0;
      La[s * nb + r] = 0; Th[s * nb + r] = 0; Be[s * nb + r] = 0;
      if (r < nu) DU[s * nu + r] = 0;
    }
  }
  __syncthreads();

  unsigned long long *myflag = P.flag + b;
  const unsigned long long base = P.base;

  // publish initial first-stage consensus vars; counter -> base+1
  if (s == 0 && r < nb) {
    T *g = P.gZVL + (size_t)b * 3 * nb;
    g[r] = Z[r]; g[nb + r] = V[r]; g[2 * nb + r] = La[r];
    __threadfence();
  }
  __syncthreads();
  if (tid == 0) st_release(myflag, base + 1);

  const T rho = P.rho;
  const bool has_next = (b + 1 < P.B);
  const bool has_prev = (b > 0);

  for (int it = 0; it < P.iters; ++it) {
    const unsigned long long c = base + 2ull * it;
    // ---------------- primal phase ----------------
    // needs next block's first-stage (z, v, la) from the previous iteration
    if (has_next) {
      if (tid == 0) {
        while (ld_acquire(P.flag + b + 1) < c + 1) { }
      }
      __syncthreads();
      if (tid < nb) {
        const T *g = P.gZVL + (size_t)(b + 1) * 3 * nb;
        Z[M * nb + tid] = ldcg(g + tid);
        V[M * nb + tid] = ldcg(g + nb + tid);
        La[M * nb + tid] = ldcg(g + 2 * nb + tid);
      }
    }
    __syncthreads();

    if (act) {
      // du_k = J (v_k - beta_k)
      if (r < nu) {
        T acc = 0;
        for (int j = 0; j < nb; ++j)
          acc += JT[j * nu + r] * (V[s * nb + j] - Be[s * nb + j]);
        DU[s * nu + r] = acc;
      }
      // t1 = z_{k+1} - v_{k+1} + la_{k+1} - e     (stage s+1 may be the halo)
      if (k < N - 1) {
        const int s1 = s + 1;
        t1[s * nb + r] = Z[s1 * nb + r] - V[s1 * nb + r] + La[s1 * nb + r] - e[r];
      }
    }
    __syncthreads();
    if (act) {
      T rhs = Z[s * nb + r] - Th[s * nb + r];
      if (k < N - 1) {
        T acc = 0;
        for (int j = 0; j < nb; ++j) acc += Ab[j * nb + r] * t1[s * nb + j];  // Ab'
        rhs += acc;
      }
      t2[s * nb + r] = Q[s * nb + r] + rho * rhs;
    }
    __syncthreads();
    if (act) {
      const T *HH = (k < N - 1) ? HT : HNT;
      T acc = 0;
      for (int j = 0; j < nb; ++j) acc += HH[j * nb + r] * t2[s * nb + j];
      X[s * nb + r] = acc;
      if (k == klast && has_next) {
        P.gX[(size_t)b * nb + r] = acc;
        __threadfence();
      }
    }
    __syncthreads();
    if (tid == 0) st_release(myflag, c + 2);

    // ---------------- consensus phase ----------------
    // needs previous block's last xb from THIS iteration
    if (has_prev) {
      if (tid == 0) {
        while (ld_acquire(P.flag + b - 1) < c + 2) { }
      }
      __syncthreads();
      if (tid < nb) xprev[tid] = ldcg(P.gX + (size_t)(b - 1) * nb + tid);
      __syncthreads();
    }
    if (act) {
      const T *xk = (s == 0) ? xprev : (X + (s - 1) * nb);
      T bu = 0, ax = 0;
      for (int j = 0; j < nu; ++j) bu += BbT[j * nb + r] * DU[s * nu + j];
      for (int j = 0; j < nb; ++j) ax += AbT[j * nb + r] * xk[j];
      const int i = s * nb + r;
      const T th = Th[i], be = Be[i], la = La[i], xn = X[i], ee = e[r];
      T z = (T(2) * (xn + th) + bu + be + ax + ee - la) / T(3);
      z = fmin(fmax(z, lo[r]), hi[r]);
      const T v = T(0.5) * (z + bu + be - ax - ee + la);
      Th[i] = th + xn - z;
      Be[i] = be + bu - v;
      La[i] = la + z - ax - v - ee;
      Z[i] = z;
      V[i] = v;
    }
    __syncthreads();  // all reads of xk (= X of stage s-1) done before next primal writes
    if (s == 0 && act && has_prev) {
      T *g = P.gZVL + (size_t)b * 3 * nb;
      g[r] = Z[r]; g[nb + r] = V[r]; g[2 * nb + r] = La[r];
      __threadfence();
    }
    __syncthreads();
    if (tid == 0) st_release(myflag, c + 3);
  }

  // ---- write back --------------------------------------------------------
  if (act) {
    P.Xs[gi] = X[s * nb + r]; P.Zs[gi] = Z[s * nb + r];
    P.Vs[gi] = V[s * nb + r]; P.Las[gi] = La[s * nb + r];
    P.Ths[gi] = Th[s * nb + r]; P.Bes[gi] = Be[s * nb + r];
    if (r < nu) P.DUs[k * nu + r] = DU[s * nu + r];
    if (r >= nx) P.uout[k * nu + (r - nx)] = X[s * nb + r];
  }
}

// ===========================================================================
// host side
// ===========================================================================
struct Handle {
  int dtype;  // 0 = float, 1 = double
  int nx, nu, nb, N, M, B, mats_shared;
  size_t smem;
  double rho;
  unsigned long long base;
  void *buf[21];
  void *h_u;  // pinned host output (double after conversion)
  void *h_uT; // pinned host output (T)
  cudaStream_t stream;
  cudaEvent_t ev0, ev1;
};

template <typename T>
static size_t smem_bytes(int nb, int nu, int M, int mats) {
  size_t w = 0;
  if (mats) w += (size_t)4 * nb * nb + 2 * (size_t)nb * nu;
  w += 4 * (size_t)nb;              // lo, hi, e, xprev
  w += (size_t)M * nb * 9;          // X, Th, Be, Q, t1, t2 + Z,V,La
  w += 3 * (size_t)nb;              // halo of Z, V, La
  w += (size_t)M * nu;              // DU
  return w * sizeof(T);
}

template <typename T>
static T *upload(const double *src, size_t n) {
  std::vector<T> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = (T)src[i];
  T *d = nullptr;
  if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemcpy(d, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}

template <typename T>
static int create_impl(Handle *h, const double *Ab, const double *Bb,
                       const double *J, const double *H, const double *HN,
                       const double *lo, const double *hi, const double *e,
                       const double *q, int force_global, int max_stages) {
  const int nb = h->nb, nu = h->nu, N = h->N;
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  const size_t Smax = prop.sharedMemPerBlockOptin;
  const int Tmax = prop.maxThreadsPerBlock;

  // eqs. (14)-(17) of the paper, with the matrices either block-resident or not
  int mats = force_global ? 0 : 1;
  int M = 0;
  for (; mats >= 0; --mats) {
    int Mthreads = Tmax / nb;
    int Mmem = 0;
    while (smem_bytes<T>(nb, nu, Mmem + 1, mats) <= Smax) ++Mmem;
    M = std::min(N, std::min(Mthreads, Mmem));
    if (max_stages > 0) M = std::min(M, max_stages);
    if (M >= 1) break;
  }
  if (M < 1) { fprintf(stderr, "cudampc: one stage does not fit in shared memory\n"); return -2; }
  h->mats_shared = mats;
  h->M = M;
  h->B = (N + M - 1) / M;
  h->smem = smem_bytes<T>(nb, nu, M, mats);

  CK(cudaFuncSetAttribute(cudampc_kernel<T>,
                          cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)h->smem));
  int per_sm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, cudampc_kernel<T>,
                                                   M * nb, h->smem));
  if (per_sm * prop.multiProcessorCount < h->B) {
    fprintf(stderr, "cudampc: %d blocks exceed cooperative residency %d\n", h->B,
            per_sm * prop.multiProcessorCount);
    return -3;
  }

  auto tr = [](const double *A, int r, int c) {  // r x c row-major -> c x r
    std::vector<double> t((size_t)r * c);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) t[(size_t)j * r + i] = A[(size_t)i * c + j];
    return t;
  };
  h->buf[0] = upload<T>(Ab, (size_t)nb * nb);
  h->buf[20] = upload<T>(tr(Ab, nb, nb).data(), (size_t)nb * nb);
  h->buf[1] = upload<T>(tr(Bb, nb, nu).data(), (size_t)nb * nu);
  h->buf[2] = upload<T>(tr(J, nu, nb).data(), (size_t)nu * nb);
  h->buf[3] = upload<T>(tr(H, nb, nb).data(), (size_t)nb * nb);
  h->buf[4] = upload<T>(tr(HN, nb, nb).data(), (size_t)nb * nb);
  h->buf[5] = upload<T>(lo, nb);
  h->buf[6] = upload<T>(hi, nb);
  h->buf[7] = upload<T>(e, nb);
  h->buf[8] = upload<T>(q, (size_t)N * nb);
  for (int i = 9; i <= 14; ++i) {  // X Z V Th Be La
    CK(cudaMalloc(&h->buf[i], (size_t)N * nb * sizeof(T)));
    CK(cudaMemset(h->buf[i], 0, (size_t)N * nb * sizeof(T)));
  }
  CK(cudaMalloc(&h->buf[15], (size_t)N * nu * sizeof(T)));  // DU
  CK(cudaMemset(h->buf[15], 0, (size_t)N * nu * sizeof(T)));
  CK(cudaMalloc(&h->buf[16], (size_t)h->B * nb * sizeof(T)));      // gX
  CK(cudaMalloc(&h->buf[17], (size_t)h->B * 3 * nb * sizeof(T)));  // gZVL
  CK(cudaMalloc(&h->buf[18], (size_t)h->B * sizeof(unsigned long long)));
  CK(cudaMemset(h->buf[18], 0, (size_t)h->B * sizeof(unsigned long long)));
  CK(cudaMalloc(&h->buf[19], (size_t)N * nu * sizeof(T)));  // uout
  CK(cudaMallocHost(&h->h_uT, (size_t)N * nu * sizeof(T)));
  h->base = 0;
  CK(cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking));
  CK(cudaEventCreate(&h->ev0));
  CK(cudaEventCreate(&h->ev1));
  return 0;
}

template <typename T>
static int solve_impl(Handle *h, const double *xb0, int iters, int warm,
                      double *u_out, float *kernel_ms) {
  Params<T> P;
  P.nx = h->nx; P.nu = h->nu; P.nb = h->nb; P.N = h->N; P.M = h->M; P.B = h->B;
  P.rho = (T)h->rho; P.iters = iters; P.warm = warm;
  P.mats_shared = h->mats_shared; P.base = h->base;
  P.Ab = (T *)h->buf[0]; P.AbT = (T *)h->buf[20]; P.BbT = (T *)h->buf[1];
  P.JT = (T *)h->buf[2]; P.HT = (T *)h->buf[3]; P.HNT = (T *)h->buf[4];
  P.lo = (T *)h->buf[5];
  P.hi = (T *)h->buf[6]; P.e = (T *)h->buf[7]; P.q = (T *)h->buf[8];
  P.Xs = (T *)h->buf[9]; P.Zs = (T *)h->buf[10]; P.Vs = (T *)h->buf[11];
  P.Ths = (T *)h->buf[12]; P.Bes = (T *)h->buf[13]; P.Las = (T *)h->buf[14];
  P.DUs = (T *)h->buf[15]; P.gX = (T *)h->buf[16]; P.gZVL = (T *)h->buf[17];
  P.flag = (unsigned long long *)h->buf[18]; P.uout = (T *)h->buf[19];
  X0Arg<T> x0;
  for (int i = 0; i < h->nb; ++i) x0.v[i] = (T)xb0[i];

  void *args[] = {&P, &x0};
  if (kernel_ms) CK(cudaEventRecord(h->ev0, h->stream));
  CK(cudaLaunchCooperativeKernel((void *)cudampc_kernel<T>, dim3(h->B),
                                 dim3(h->M * h->nb), args, h->smem, h->stream));
  if (kernel_ms) CK(cudaEventRecord(h->ev1, h->stream));
  h->base += 1ull + 2ull * (unsigned long long)iters;
  const size_t nuN = (size_t)h->N * h->nu;
  CK(cudaMemcpyAsync(h->h_uT, P.uout, nuN * sizeof(T), cudaMemcpyDeviceToHost,
                     h->stream));
  CK(cudaStreamSynchronize(h->stream));
  CK(cudaGetLastError());
  if (kernel_ms) CK(cudaEventElapsedTime(kernel_ms, h->ev0, h->ev1));
  const T *src = (const T *)h->h_uT;
  for (size_t i = 0; i < nuN; ++i) u_out[i] = (double)src[i];
  return 0;
}

extern "C" {

void *cmpc_create(int dtype, int nx, int nu, int N, double rho,
                  const double *Ab, const double *Bb, const double *J,
                  const double *H, const double *HN, const double *lo,
                  const double *hi, const double *e, const double *q,
                  int force_global, int max_stages) {
  Handle *h = new Handle();
  memset(h, 0, sizeof(Handle));
  h->dtype = dtype; h->nx = nx; h->nu = nu; h->nb = nx + nu; h->N = N;
  h->rho = rho;
  if (h->nb > MAX_NB) { delete h; return nullptr; }
  int rc = dtype ? create_impl<double>(h, Ab, Bb, J, H, HN, lo, hi, e, q,
                                       force_global, max_stages)
                 : create_impl<float>(h, Ab, Bb, J, H, HN, lo, hi, e, q,
                                      force_global, max_stages);
  if (rc != 0) { delete h; return nullptr; }
  return h;
}

int cmpc_info(void *hp, int *out) {
  Handle *h = (Handle *)hp;
  out[0] = h->M; out[1] = h->B; out[2] = h->mats_shared; out[3] = (int)h->smem;
  out[4] = h->M * h->nb;
  return 0;
}

int cmpc_set_q(void *hp, const double *q) {
  Handle *h = (Handle *)hp;
  size_t n = (size_t)h->N * h->nb;
  if (h->dtype) {
    std::vector<double> t(q, q + n);
    CK(cudaMemcpy(h->buf[8], t.data(), n * sizeof(double), cudaMemcpyHostToDevice));
  } else {
    std::vector<float> t(n);
    for (size_t i = 0; i < n; ++i) t[i] = (float)q[i];
    CK(cudaMemcpy(h->buf[8], t.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  }
  return 0;
}

int cmpc_solve(void *hp, const double *xb0, int iters, int warm, double *u_out,
               float *kernel_ms) {
  Handle *h = (Handle *)hp;
  return h->dtype ? solve_impl<double>(h, xb0, iters, warm, u_out, kernel_ms)
                  : solve_impl<float>(h, xb0, iters, warm, u_out, kernel_ms);
}

void cmpc_destroy(void *hp) {
  Handle *h = (Handle *)hp;
  if (!h) return;
  for (int i = 0; i < 21; ++i)
    if (h->buf[i]) cudaFree(h->buf[i]);
  if (h->h_uT) cudaFreeHost(h->h_uT);
  if (h->stream) cudaStreamDestroy(h->stream);
  if (h->ev0) cudaEventDestroy(h->ev0);
  if (h->ev1) cudaEventDestroy(h->ev1);
  delete h;
}

}  // extern "C"
