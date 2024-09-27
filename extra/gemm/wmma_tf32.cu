
#include <mma.h>
using namespace nvcuda;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;

extern "C" __global__ void wmma_example(float *a, float *b, float *c)
{
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
  warpM *= 4;
  warpN *= 4;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> a_frag[4];
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> b_frag[4];
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[4][4];
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      wmma::fill_fragment(acc_frag[i][j], 0.0f);
    }
  }

  for (int k = 0; k < 4096; k += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = k;
    int bRow = k;
    int bCol = warpN * WMMA_N;

    wmma::load_matrix_sync(a_frag[0], a + aRow + 0 * WMMA_M + aCol * 4096, 4096);
    wmma::load_matrix_sync(a_frag[1], a + aRow + 1 * WMMA_M + aCol * 4096, 4096);
    wmma::load_matrix_sync(a_frag[2], a + aRow + 2 * WMMA_M + aCol * 4096, 4096);
    wmma::load_matrix_sync(a_frag[3], a + aRow + 3 * WMMA_M + aCol * 4096, 4096);

    wmma::load_matrix_sync(b_frag[0], b + bRow + (0 * WMMA_N + bCol) * 4096, 4096);
    wmma::load_matrix_sync(b_frag[1], b + bRow + (1 * WMMA_N + bCol) * 4096, 4096);
    wmma::load_matrix_sync(b_frag[2], b + bRow + (2 * WMMA_N + bCol) * 4096, 4096);
    wmma::load_matrix_sync(b_frag[3], b + bRow + (3 * WMMA_N + bCol) * 4096, 4096);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
      #pragma unroll
      for (int t = 0; t < a_frag[i].num_elements; t++) { a_frag[i].x[t] =  wmma::__float_to_tf32(a_frag[i].x[t]); }
      #pragma unroll
      for (int t = 0; t < b_frag[i].num_elements; t++) { b_frag[i].x[t] =  wmma::__float_to_tf32(b_frag[i].x[t]); }
    }

    #pragma unroll
    for (int j = 0; j < 4; j++) {
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        wmma::mma_sync(acc_frag[i][j], a_frag[i], b_frag[j], acc_frag[i][j]);
      }
    }
  }

  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_store;
      for (int t = 0; t < acc_frag[i][j].num_elements; t++) acc_store.x[t] = acc_frag[i][j].x[t];
      int cRow = (warpM + i) * WMMA_M;
      int cCol = (warpN + j) * WMMA_N;
      wmma::store_matrix_sync(c + cRow + cCol * 4096, acc_store, 4096, wmma::mem_col_major);
    }
  }
}
