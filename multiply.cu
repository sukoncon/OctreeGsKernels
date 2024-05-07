#include <vector>
#include <cmath>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <c10/util/complex.h>

#include <ATen/ATen.h>

#include <cuda.h>
#include <mma.h>

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line) {
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err) {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

#define M 16
#define N 16
#define K 8

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define blockDim_x 32*2
#define blockDim_y 4
#define warp_x 4 // X coordinate of warps in a block
#define warp_y 4 // Y coordinate of warps in a block
#define warp_size 32

using namespace nvcuda;

template<typename T, int pack_size>
struct GetPackType {
  using type = typename std::aligned_storage<pack_size * sizeof(T), pack_size * sizeof(T)>::type;
};

template<typename T, int pack_size>
using PackType = typename GetPackType<T, pack_size>::type;

template<typename T, int pack_size>
union Pack {

  PackType<T, pack_size> storage;
  T elem[pack_size];
};


#define colMajorIdx(rows, cols, i, j)    i + j*rows
#define rowMajorIdx(rows, cols, i, j)    i * cols + j


__global__ void simple_fused_gemm_wmma(
                float* input, float* weight0, float* bias0, float* weight1, float* bias1,
                float* output,
                std::string activation0, std::string activation1,
                int Mblock, int M0, int N0, int K0, int N1, int K1,
                int N0pad, int K0pad, int N1pad, int K1pad,
                int lda0Pad, int ldb0Pad, int ldb1Pad,
                int lda0, int ldb0, int ldb1){
  extern __shared__ float buffer[];
  float* inSmem = buffer;
  float* w0Smem = buffer + Mblock * K0pad;
  float* w1Smem = w0Smem + K0pad * N0pad;
  float* out0Smem = w1Smem + K1pad * N1pad;
  float* out1Smem = out0Smem + Mblock * K1pad;

  // Tile using a 2D grid
  int warpM = threadIdx.x / warpSize;
  int warpN = threadIdx.y;
  int idx = threadIdx.y * blockDim.x + threadIdx.x; // local index within a block

  // load input into shared memory
  int offset = blockIdx.x * Mblock; // row offset of this block
  for (int i = idx; i < Mblock * K0pad; i += blockDim.x * blockDim.y){
    int row = i / K0pad;
    int col = i % K0pad;
    if (col < K0) inSmem[row * K0pad + col] = input[(row + offset) * K0 + col];
    else inSmem[row * K0pad + col] = 0;
  }

  // load weights into shared memory
  for (int i = idx; i < N0pad * K0pad; i += blockDim.x * blockDim.y){
    int row = i % K0pad;
    int col = i / K0pad;
    if (row < K0 & col < N0) w0Smem[row + col * K0pad] = weight0[row + col * K0];
    else w0Smem[row + col * K0pad] = 0;
  }

  for (int i = idx; i < N1pad * K1pad; i += blockDim.x * blockDim.y){
    int row = i % K1pad;
    int col = i / K1pad;
    if (row < K1 & col < N1) w1Smem[row + col * K1pad] = weight1[row + col * K1];
    else w1Smem[row + col * K1pad] = 0;
  }
  
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>  in0_frag;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major>  in1_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major>  w0_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major>  w1_frag;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc1_frag;
  // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc0_frag;
  // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out1_frag;

  wmma::fill_fragment(acc0_frag, 0.0f);
  wmma::fill_fragment(acc1_frag, 0.0f);
  
  __syncthreads();

  // First layer
  int aRow = warpM * WMMA_M;
  int bCol = warpN * WMMA_N;
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  for (int i = 0; i < K0pad; i += WMMA_K) {
    int aCol = i;
    int bRow = i;
    // Bounds checking
    if (bCol < N0pad & (aRow + offset) < M0) {
      // Load the inputs
      wmma::load_matrix_sync(in0_frag, inSmem + aCol + aRow * K0pad, K0pad);
      wmma::load_matrix_sync(w0_frag, w0Smem + bRow + bCol * K0pad, K0pad);

      #pragma unroll
      for (int t = 0; t < in0_frag.num_elements; t++) {
              in0_frag.x[t] =  wmma::__float_to_tf32(in0_frag.x[t]);
      }

      #pragma unroll
      for (int t = 0; t < w0_frag.num_elements; t++) {
              w0_frag.x[t] =  wmma::__float_to_tf32(w0_frag.x[t]);
      }

      // Perform the matrix multiplication
      wmma::mma_sync(acc0_frag, in0_frag, w0_frag, acc0_frag);
    }
  }

  if (bCol < N0pad & (aRow + offset) < M0) {
    wmma::store_matrix_sync(out0Smem  + cCol + cRow * N0pad, acc0_frag, N0pad,
                            wmma::mem_row_major);
  }
  __syncthreads();
  float * in1Smem = out0Smem;
  // bias0
  for (int i = idx; i < Mblock * N0pad; i += blockDim.x * blockDim.y){
    int row = i / N0pad;
    int col = i % N0pad;
    if (col < N0) in1Smem[row*N0pad + col] +=  bias0[col]; // problem
  }

  // __syncthreads();
  // for (int i = idx; i < Mblock * K0pad; i += blockDim.x * blockDim.y){
  //   int row = i / K0pad;
  //   int col = i % K0pad;
    
  //   if (blockIdx.x ==0){
  //     printf("idx %d, row %d, col %d, offset %d, inSmem(after) %f\n", 
  //     idx, row, col, offset, inSmem[row*K0pad + col]);
  //   }

  // }
  // __syncthreads();
  // for (int i = idx; i < K0pad * N0pad; i += blockDim.x * blockDim.y){
  //   int row = i % K0pad;
  //   int col = i / K0pad;
    
  //   if (blockIdx.x ==0){
  //     printf("idx %d, row %d, col %d, offset %d, w0smem %f\n", 
  //     idx, row, col, offset, w0Smem[row + K0pad * col]);
  //   }

  // }
  // __syncthreads();
  // for (int i = idx; i < K1pad * N1pad; i += blockDim.x * blockDim.y){
  //   int row = i % K1pad;
  //   int col = i / K1pad;
    
  //   // if (blockIdx.x ==0){
  //     if (row < 17 &  w1Smem[row + K1pad * col] != 1) printf("row < 17 &  w1Smem[row + K1pad * col] != 1");
  //     if (row >= 17 & w1Smem[row + K1pad * col] != 0) printf("row >= 17 & w1Smem[row + K1pad * col] != 0");
  //     // printf("idx %d, row %d, col %d, offset %d, w1smem %f\n", 
  //     // idx, row, col, offset, w1Smem[row + K1pad * col]);
  //   // }

  // }
  // __syncthreads();
  // for (int i = idx; i < Mblock * N0pad; i += blockDim.x * blockDim.y){
  //   int row = i / N0pad;
  //   int col = i % N0pad;
    
  //   // if (blockIdx.x ==0){
  //     if (col < 17 & in1Smem[row*N0pad + col] != 35) printf("col < 17 & in1Smem[row*N0pad + col] != 35");
  //     if (col >= 17 & in1Smem[row*N0pad + col] != 0) printf("col >= 17 & in1Smem[row*N0pad + col] != 0");
  //     // printf("idx %d, row %d, col %d, offset %d, out0Smem %f, bias0 %f\n", 
  //     // idx, row, col, offset, in1Smem[row*N0pad + col], bias0[col]);
  //   // }

  // }
  
  __syncthreads();
  // activation0
  // Second layer
  
  for (int i = 0; i < K1pad; i += WMMA_K) {
    int aCol = i;
    int bRow = i;
    // Bounds checking
    if (bCol < N1pad & (aRow + offset) < M0) {
      // Load the inputs
      wmma::load_matrix_sync(in1_frag, in1Smem + aCol + aRow * K1pad, K1pad);
      wmma::load_matrix_sync(w1_frag, w1Smem + bRow + bCol * K1pad, K1pad); // b multiply.cu:221 if idx == 192 || idx == 256
      #pragma unroll
      for (int t = 0; t < in1_frag.num_elements; t++) {
              in1_frag.x[t] =  wmma::__float_to_tf32(in1_frag.x[t]);
      }

      #pragma unroll
      for (int t = 0; t < w1_frag.num_elements; t++) {
              w1_frag.x[t] =  wmma::__float_to_tf32(w1_frag.x[t]);
      }

      // Perform the matrix multiplication
      wmma::mma_sync(acc1_frag, in1_frag, w1_frag, acc1_frag);
    }
  }


  // for (int i = idx; i < Mblock * N1pad; i += blockDim.x * blockDim.y){
  //   out1Smem[i] = 0;
  // }
  // __syncthreads();
  // for (int i = idx; i < Mblock * N1pad; i += blockDim.x * blockDim.y){
  //   int row = i / N1pad;
  //   int col = i % N1pad;
    
  //   if (blockIdx.x ==0){
  //     printf("idx %d, row %d, col %d, offset %d, out1Smem(before) %f\n", 
  //     idx, row, col, offset, out1Smem[row*N1pad + col]);
  //   }

  // }

  // __syncthreads();
  
  if (cCol < N1pad & (aRow + offset) < M0) {
    wmma::store_matrix_sync(out1Smem  + cCol + cRow * N1pad, acc1_frag, N1pad,
                            wmma::mem_row_major);
  }
  __syncthreads();
  // bias0
  for (int i = idx; i < Mblock * N1pad; i += blockDim.x * blockDim.y){
    int row = i / N1pad;
    int col = i % N1pad;
    
    if (col<N1 & (row + offset) < M0) output[(row + offset) * N1 + col] = out1Smem[row*N1pad + col] + bias1[col];

    // if (blockIdx.x ==0 & col<N1){
    //   printf("idx %d, row %d, col %d, offset %d, out1Smem %f, output %f\n", 
    //   idx, row, col, offset, out1Smem[row*N1pad + col], output[(row + offset) * N1 + col]);
    // }
  }

}


torch::Tensor  simple_fused_gemm(torch::Tensor& input, 
                        torch::Tensor& weight0, 
                        torch::Tensor& bias0, 
                        std::string activation0,
                        torch::Tensor& weight1, 
                        torch::Tensor& bias1, 
                        std::string& activation1) {

    int M0; int N0; int K0; int K0pad; int N0pad;
    int M1; int N1; int K1; int K1pad; int N1pad;
    int lda0Pad; int ldb0Pad; int ldb1Pad;
    int lda0; int ldb0; int ldb1;
    // Assertion

    if (input.is_contiguous()){
      M0 = input.size(0); K0 = input.size(1); lda0 = input.strides()[0];
      K0pad = ((K0 + WMMA_K - 1) / WMMA_K) * WMMA_K;
      lda0Pad = K0pad;
    }
    else {throw std::runtime_error("Not implemented for discontiguous INPUT tensor.");}

    if (weight0.is_contiguous()){
      N0 = weight0.size(0); ldb0 = weight0.strides()[0];
      N0pad = ((N0 + WMMA_N - 1) / WMMA_N) * WMMA_N;
      K1pad = N0pad;
      ldb0Pad = K0pad;
    }
    else {throw std::runtime_error("Not implemented for discontiguous WEIGHT0 tensor.");}

    if (weight1.is_contiguous()){
      N1 = weight1.size(0); K1 = weight1.size(1); ldb1 = weight1.strides()[0];
      N1pad = ((N1 + WMMA_N - 1) / WMMA_N) * WMMA_N;
      ldb1Pad = K1pad;
    }
    else {throw std::runtime_error("Not implemented for discontiguous WEIGHT1 tensor.");}

    torch::Tensor output = torch::empty({M0, N1}, input.options());

    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = blockDim_x;
    blockDim.y = int((max(N0, N1) + WMMA_N - 1) / WMMA_N);

    gridDim.x = int((M0 + (WMMA_M * blockDim.x / warp_size - 1)) /
                (WMMA_M * blockDim.x / warp_size));
    gridDim.y = 1; // assume N and K are small

    // store input block + weights + intermediate result in shared memory
    int Mblock = (blockDim_x/warp_size) * WMMA_M; // how many rows a block can deal with
    int smem_size = (Mblock * K0pad + 
                      K0pad * N0pad +  K1pad * N1pad +
                      Mblock * N1pad + Mblock * K1pad) * sizeof(float);

    // printf("Computing... using simple_fused_gemm_wmma kernel, blockDim.x %d, blockDim.y %d, gridDim.x %d, gridDim.y %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    simple_fused_gemm_wmma<<<gridDim, blockDim, smem_size>>>(input.data_ptr<float>(), 
                                                  weight0.data_ptr<float>(), 
                                                  bias0.data_ptr<float>(), 
                                                  weight1.data_ptr<float>(),
                                                  bias1.data_ptr<float>(),
                                                  output.data_ptr<float>(),
                                                  activation0, 
                                                  activation1,
                                                  Mblock, M0, N0, K0, N1, K1, 
                                                  N0pad,  K0pad, N1pad, K1pad, 
                                                  lda0Pad, ldb0Pad, ldb1Pad,
                                                  lda0,
                                                  ldb0,
                                                  ldb1);

    getLastCudaError("simple_fused_gemm_wmma launch failed\n");
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("simple_gemm", &simple_gemm, "CUDA kernel: simple_gemm");
    // m.def("smem_gemm", &smem_gemm, "CUDA kernel: smem_gemm");
    m.def("simple_fused_gemm", &simple_fused_gemm, "CUDA kernel: simple_fused_gemm");

}