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
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

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


// template <typename scalar_t>
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {

// Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>  a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>  b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i;
    int aRow = warpM * WMMA_M;
    int bCol = warpN * N;
    int bRow = i;

    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                           wmma::mem_row_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                            wmma::mem_row_major);
  }
}

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
  extern __shared__ __half buffer[];
  __half* inSmem = buffer;
  __half* w0Smem = buffer + Mblock * K0pad;
  __half* w1Smem = w0Smem + K0pad * N0pad;
  __half* out0Smem = w1Smem + K1pad * N1pad;

  // Tile using a 2D grid
  int warpM = threadIdx.x / warpSize;
  int warpN = threadIdx.y;
  int idx = threadIdx.y * blockDim.x + threadIdx.x; // local index within a block

  // load input into shared memory
  int offset = blockIdx.x * Mblock; // row offset of this block
  for (int i = idx; i < Mblock * K0; i += blockDim.x * blockDim.y){
    int row = i / K0;
    int col = i % K0;
    inSmem[row * K0pad + col] = __float2half(input[(row + offset) * K0 + col]); // Why?!!
  }


  // load weights into shared memory
  for (int i = idx; i < N0 * K0; i += blockDim.x * blockDim.y){
    int row = i % K0;
    int col = i / K0;
    w0Smem[colMajorIdx(K0pad, N0pad, row, col)] = __float2half(weight0[i]);
  }

  for (int i = idx; i < N1 * K1; i += blockDim.x * blockDim.y){
    int row = i % K1;
    int col = i / K1;
    w1Smem[colMajorIdx(K1pad, N1pad, row, col)] = __float2half(weight1[i]);
  }
  
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>  in_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>  w0_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>  w1_frag;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc0_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc1_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> out1_frag;

  wmma::fill_fragment(acc0_frag, __float2half(0.0f));
  wmma::fill_fragment(acc1_frag, __float2half(0.0f));

  // First layer
  int aRow = warpM * WMMA_M;
  int bCol = warpN * WMMA_N;
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;
  for (int i = 0; i < K0pad; i += WMMA_K) {
    int aCol = i;
    int bRow = i;
    // Bounds checking
    if (bCol < N0pad) {
      // Load the inputs
      wmma::load_matrix_sync(in_frag, inSmem + aCol + aRow * K0pad, K0pad);
      wmma::load_matrix_sync(w0_frag, w0Smem + bRow + bCol * K0pad, K0pad);
      // Perform the matrix multiplication
      wmma::mma_sync(acc0_frag, in_frag, w0_frag, acc0_frag);
    }
  }
  // Store intermediate result into shared memory
  if (bCol < N0pad) {
    wmma::store_matrix_sync(out0Smem  + cCol + cRow * N0pad, acc0_frag, N0pad,
                            wmma::mem_row_major);
  }
  __syncthreads();
  __half * int1Smem = out0Smem;
  // bias0
  for (int i = idx; i < Mblock * N0; i += blockDim.x * blockDim.y){
    int row = i / N0;
    int col = i % N0;
    int1Smem[rowMajorIdx(Mblock, K1pad, row, col)] =  __hadd(int1Smem[rowMajorIdx(Mblock, K1pad, row, col)], __float2half(bias0[col])); // problemm
  }

  __syncthreads();
  // activation0
  // Second layer
  wmma::fill_fragment(in_frag, __float2half(0.0f));
  wmma::fill_fragment(acc1_frag, __float2half(0.0f));
  for (int i = 0; i < K1pad; i += WMMA_K) {
    int aCol = i;
    int bRow = i;
    // Bounds checking
    if (bCol < N1pad) {
      // Load the inputs
      wmma::load_matrix_sync(in_frag, int1Smem + aCol + aRow * K1pad, K1pad);
      wmma::load_matrix_sync(w1_frag, w1Smem + bRow + bCol * K1pad, K1pad); // b multiply.cu:221 if idx == 192 || idx == 256
      // Perform the matrix multiplication
      wmma::mma_sync(acc1_frag, in_frag, w1_frag, acc1_frag);
    }
  }
  __syncthreads();
  __half* outSmem = buffer;
  // Store intermediate result into shared memory
  if (bCol < N1pad) {
    wmma::store_matrix_sync(outSmem  + cCol + cRow * N1pad, acc1_frag, N1pad,
                            wmma::mem_row_major); //problem
  }
  __syncthreads();
  // bias0
  for (int i = idx; i < Mblock * N1; i += blockDim.x * blockDim.y){
    int row = i / N1;
    int col = i % N1;
    outSmem[rowMajorIdx(Mblock, N1pad, row, col)] =__hadd(outSmem[rowMajorIdx(Mblock, N1pad, row, col)], __float2half(bias1[col]));
  }
  __syncthreads();

  // activation1

  // Store output into global mem
  // int offset = blockIdx.x * Mblock; // row offset of this block
  for (int i = idx; i < Mblock * N1; i += blockDim.x * blockDim.y){
    int row = i / N1;
    int col = i % N1;
    output[(row + offset) * N1 + col] = __half2float(outSmem[row * N1pad + col]); 
      // rowMajorIdx(M0, N1, , col)] = __half2float(outSmem[rowMajorIdx(Mblock, N1pad, row, col)]); // problem outSmem right, but global wrong
  }

}

__global__ void smem_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta) {

// Leading dimensions. Packed with no transpositions.
  int lda = k_ld;
  int ldb = k_ld;
  int ldc = n_ld;
  const int packed_size = 8;
  const int grp = 4; //(blockDim_x*blockDim_y*packed_size) / (warp_x*WMMA_M*WMMA_K);
  __shared__ half Asmem[warp_x*WMMA_M*WMMA_K*grp*2];
  __shared__ half Bsmem[warp_y*WMMA_N*WMMA_K*grp*2];
  // __shared__ half Csmem[warp_x*WMMA_M][warp_y*WMMA_N];
  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  int blockM = warpM/warp_y;
  int blockN = warpN/warp_x;

  int idx = threadIdx.y * blockDim.x + threadIdx.x; // local index within a block
  // if (threadIdx.x == 0 & threadIdx.y == 0){
  //   printf("blockM %d, blockN %d\n", blockM, blockN);
  // }

  // load data into the shared memory
  
  
  int a_row = blockM * warp_y * WMMA_M;
  int b_col = blockN * warp_x * WMMA_N;

  // copy M
  for (int j = idx*packed_size; j < warp_y*WMMA_M*WMMA_K*grp; j += blockDim.x*blockDim.y*packed_size){
    int row = j/(WMMA_K*grp); int col = j%(WMMA_K*grp);
    reinterpret_cast<Pack<half, packed_size>*> (&Asmem[j])[0] = reinterpret_cast<Pack<half, packed_size>*> (a + (a_row+row) * k_ld + col)[0];
    // Asmem[i] = *(a + a_row * k_ld + i);
  }

  // copy N
  for (int j = idx*packed_size ; j < warp_x*WMMA_N*WMMA_K*grp; j += blockDim.x*blockDim.y*packed_size){
    int row = j%(WMMA_K*grp); int col = j/(WMMA_K*grp);
    reinterpret_cast<Pack<half, packed_size>*> (&Bsmem[j])[0] = reinterpret_cast<Pack<half, packed_size>*> (b + (b_col+col) * k_ld + row)[0];
  }
   __syncthreads();

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>  a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>  b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
  // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);
  int ptr = 0;
  // Loop over k
  for (int i = 0; i < k_ld; i += WMMA_K) {
    int aCol = i%(WMMA_K*grp);
    int aRow = (warpM % warp_y) * WMMA_M;// int aRow = warpM * WMMA_M;
    int bCol = (warpN % warp_x) * N; //int bCol = warpN * N;
    int bRow = i%(WMMA_K*grp);

    half* Aptr = Asmem + ptr*warp_x*WMMA_M*WMMA_K*grp; 
    half* Bptr = Bsmem + ptr*warp_y*WMMA_N*WMMA_K*grp;

    half* AptrN = Asmem + (1-ptr)*warp_x*WMMA_M*WMMA_K*grp; 
    half* BptrN = Bsmem + (1-ptr)*warp_y*WMMA_N*WMMA_K*grp;
    // Bounds checking
    if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld) {
      // Load the inputs
      // wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
      // wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);
      wmma::load_matrix_sync(a_frag, Aptr + aRow * WMMA_K *grp + aCol, WMMA_K *grp);
      wmma::load_matrix_sync(b_frag, Bptr + bCol * WMMA_K *grp + bRow, WMMA_K *grp);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    if ( (i%(WMMA_K*grp) == 0) & (i < k_ld-WMMA_K*grp)){
      ptr = 1-ptr;
      // copy M
      for (int j = idx*packed_size; j < warp_y*WMMA_M*WMMA_K*grp; j += blockDim.x*blockDim.y*packed_size){
        int row = j/(WMMA_K*grp); int col = j%(WMMA_K*grp);
        reinterpret_cast<Pack<half, packed_size>*> (AptrN + j)[0] = reinterpret_cast<Pack<half, packed_size>*> (a + (a_row+row) * k_ld + col + i+WMMA_K*grp)[0];
        // Asmem[i] = *(a + a_row * k_ld + i);
      }

      // copy N
      for (int j = idx*packed_size ; j < warp_x*WMMA_N*WMMA_K*grp; j += blockDim.x*blockDim.y*packed_size){
        int row = j%(WMMA_K*grp); int col = j/(WMMA_K*grp);
        reinterpret_cast<Pack<half, packed_size>*> (BptrN + j)[0] = reinterpret_cast<Pack<half, packed_size>*> (b + (b_col+col) * k_ld + row + i+WMMA_K*grp)[0];
      }
      // __syncthreads();
    }
    
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cCol = warpN * WMMA_N;
  int cRow = warpM * WMMA_M;

  if (cRow < m_ld && cCol < n_ld) {
    // wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
    //                        wmma::mem_row_major);

    // for (int i = 0; i < c_frag.num_elements; i++) {
    //   c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    // }

    // Store the output
    wmma::store_matrix_sync(d + cCol + cRow * ldc, acc_frag, ldc,
                            wmma::mem_row_major);
  }
}


// void simple_gemm(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C, torch::Tensor& D, float alpha, float beta) {

//     //开启线程数量
//     int M0; int N0; int K0; int M1; int N1; int K1; 
//     dim3 gridDim;
//     dim3 blockDim;

//     // blockDim.x must be a multple of warpSize
//     // 128x4 means we have 16 warps and a block computes a 64x64 output tile
//     blockDim.x = blockDim_x;
//     blockDim.y = (N_GLOBAL + WMMA_M - 1) / WMMA_M;

//     gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / warp_size - 1)) /
//                 (WMMA_M * blockDim.x / warp_size);
//     gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

//     printf("Computing... using simple_wmma_gemm kernel, blockDim.x %d, blockDim.y %d, gridDim.x %d, gridDim.y %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
//     // AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.type(), "simple_wmma_gemm", ([&] {
//       simple_wmma_gemm<<<gridDim, blockDim>>>(reinterpret_cast<half*>(A.data_ptr<torch::Half>()), 
//                                               reinterpret_cast<half*>(B.data_ptr<torch::Half>()), 
//                                               C.data_ptr<float>(),
//                                               D.data_ptr<float>(), M_GLOBAL, N_GLOBAL,
//                                               K_GLOBAL, alpha, beta);
//     // }));
// }

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
      ldb0Pad = K0pad;
    }
    else {throw std::runtime_error("Not implemented for discontiguous WEIGHT0 tensor.");}

    if (weight1.is_contiguous()){
      N1 = weight1.size(0); K1 = weight1.size(1); ldb1 = weight1.strides()[0];
      K1pad = ((K1 + WMMA_K - 1) / WMMA_K) * WMMA_K;
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
                      Mblock * N0pad ) * sizeof(half);

    printf("Computing... using simple_fused_gemm_wmma kernel, blockDim.x %d, blockDim.y %d, gridDim.x %d, gridDim.y %d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.type(), "simple_wmma_gemm", ([&] {
    simple_fused_gemm_wmma<<<gridDim, blockDim, max(smem_size, int(Mblock * N1pad * sizeof(half)))>>>(input.data_ptr<float>(), 
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

void smem_gemm(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C, torch::Tensor& D, float alpha, float beta) {

   //开启线程数量
    int M_GLOBAL = A.size(0);
    int N_GLOBAL = D.size(0);
    int K_GLOBAL = A.size(1);
    dim3 gridDim;
    dim3 blockDim;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = blockDim_x;
    blockDim.y = (N_GLOBAL + WMMA_M - 1) / WMMA_M;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / warp_size - 1)) /
                (WMMA_M * blockDim.x / warp_size);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_wmma_gemm kernel, gridDim.x %d, gridDim.y %d\n", gridDim.x, gridDim.y);
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.type(), "simple_wmma_gemm", ([&] {
      smem_wmma_gemm<<<gridDim, blockDim>>>(reinterpret_cast<half*>(A.data_ptr<torch::Half>()), 
                                              reinterpret_cast<half*>(B.data_ptr<torch::Half>()), 
                                              C.data_ptr<float>(),
                                              D.data_ptr<float>(), M_GLOBAL, N_GLOBAL,
                                              K_GLOBAL, alpha, beta);
    // }));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("simple_gemm", &simple_gemm, "CUDA kernel: simple_gemm");
    // m.def("smem_gemm", &smem_gemm, "CUDA kernel: smem_gemm");
    m.def("simple_fused_gemm", &simple_fused_gemm, "CUDA kernel: simple_fused_gemm");

}