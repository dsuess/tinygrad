#define INFINITY (__int_as_float(0x7f800000))
#define NAN (__int_as_float(0x7fffffff))
#include <cuda_fp16.h>

struct __align__(8) half4 { half x, y, z, w; };
__device__ half4 make_half4(half x, half y, half z, half w)
{
  half4 r = {x, y, z, w};
  return r;
}

struct __align__(16) half8 { half x, y, z, w, a, b, c, d; };
__device__ half8 make_half8(half x, half y, half z, half w, half a, half b, half c, half d)
{
  half8 r = {x, y, z, w, a, b, c, d};
  return r;
}
__device__ float4 __WMMA_8_16_16_half_float(half8 a, half4 b, float4 c)
{
  // TODO Replace with
  // mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
  // Note that k is halved, so we probably need to change the strides somewhere
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
  int *a_pk = (int *)(&a), *b_pk = (int *)(&b);
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 { %0, %1, %2, %3 }, { %4, %5, %6, %7 }, { %8, %9 }, { %0, %1, %2, %3 };"
      : "+f"(c.x), "+f"(c.y), "+f"(c.z), "+f"(c.w) : "r"(a_pk[0]), "r"(a_pk[1]), "r"(a_pk[2]), "r"(a_pk[3]), "r"(b_pk[0]), "r"(b_pk[1]));
  return c;
}
extern "C" __global__ void __launch_bounds__(32) wmma_example(float *data0, const float *data1, const float *data2)
{
  int gidx0 = blockIdx.x;  /* 512 */
  int gidx1 = blockIdx.y;  /* 256 */
  int lidx0 = threadIdx.x; /* 8 */
  int lidx1 = threadIdx.y; /* 2 */
  int lidx2 = threadIdx.z; /* 2 */
  int alu0 = (lidx0 % 2);
  int alu1 = ((lidx0 >> 1) % 2);
  int alu2 = (alu0 << 1);
  int alu3 = (alu1 << 2);
  int alu4 = (lidx0 >> 2);
  int alu5 = (gidx0 << 3);
  int alu6 = (alu4 << 13);
  int alu7 = (lidx1 << 14);
  int alu8 = (lidx2 << 15);
  int alu9 = (gidx1 << 16);
  int alu10 = (alu9 + alu5 + alu2 + alu3 + alu6 + alu7 + alu8);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;
  for (int ridx0 = 0; ridx0 < 256; ridx0++)
  {
    int alu11 = ((ridx0 << 4) + alu9 + alu2 + alu3 + alu6 + alu7 + alu8);
    int alu12 = (alu5 + (alu0 << 13) + (alu1 << 14) + alu4 + (lidx1 << 1) + (lidx2 << 2) + (ridx0 << 16));
    half val0 = (half)data2[alu12 + 4096];
    half val1 = (half)data2[alu12 + 32768];
    half val2 = (half)data2[alu12 + 36864];
    half val3 = (half)data2[alu12];
    half2 val4 = *((half2 *)(data1 + alu11 + 8));
    half2 val5 = *((half2 *)(data1 + alu11 + 4096));
    half2 val6 = *((half2 *)(data1 + alu11 + 4104));
    half2 val7 = *((half2 *)(data1 + alu11));
    float4 wmma0 = __WMMA_8_16_16_half_float(make_half8(val7.x, val7.y, val5.x, val5.y, val4.x, val4.y, val6.x, val6.y), make_half4(val3, val0, val1, val2), make_float4(acc0, acc1, acc2, acc3));
    acc0 = wmma0.x;
    acc1 = wmma0.y;
    acc2 = wmma0.z;
    acc3 = wmma0.w;
  }
  //*((half2 *)(data0 + alu10 + 4096)) = make_half2((half)(acc2), (half)(acc3));
  //*((half2 *)(data0 + alu10)) = make_half2((half)(acc0), (half)(acc1));
  *(data0 + alu10 + 4096) = acc2;
  *(data0 + alu10 + 4096 + 1) = acc3;
  *(data0 + alu10) = acc0;
  *(data0 + alu10 + 1) = acc1;
}