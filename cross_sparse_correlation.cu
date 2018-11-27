/*!
 * Copyright (c) 2018 by Contributors
 * Copyright (c) 2018 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file cross_sparse_correlation-inl.h
 * \brief cross_sparse_correlation operator
 * \author Jiang ZhengKai
*/

#include "./cross_sparse_correlation-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
namespace cuda {
inline __device__ int offset(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n*C*H*W + c*H*W + h*W + w;
}

template<typename Dtype>
__global__ void CrossSparseCorrelationForwardKernel(const int count, int N, int Data_C, int Out_C, int H, int W,
                                                    Dtype* top_data, const Dtype* bottom_data1, const Dtype* bottom_data2) {
  // C is the channel of input data
  // Out_C is the channel of output data, Out_C is H + W - 1 in default.
  CUDA_KERNEL_LOOP(index, count) {
      const int w = index % W;
      const int h = (index / W) % H;
      const int c = (index / (H * W)) % Out_C;
      const int n = (index / (Out_C * H * W));

      for(int channel=0; channel < Data_C; channel++) {
          if (c < W) {
          // c is the same height as (h,w) in data2
          top_data[offset(n,c,h,w,N,Out_C,H,W)] += bottom_data1[offset(n,channel,h,w,N,Data_C,H,W)] * bottom_data2[offset(n,channel,h,c,N,Data_C,H,W)];
          }
          else {
              // the index is in the height
              int data2_h = c - W;
              // upper than h or lower than h
              int data2h = data2_h < h ? data2_h : data2_h + 1;  
              top_data[offset(n,c,h,w,N,Out_C,H,W)] += bottom_data1[offset(n,channel,h,w,N,Data_C,H,W)] * bottom_data2[offset(n,channel,data2h,w,N,Data_C,H,W)];
          }
      }
  } // cuda_kernel_loop
}

// cross sparse correlation forward
template<typename Dtype>
inline void CrossSparseCorrelationForward(const Tensor<gpu, 4, Dtype> &out,
                                          const Tensor<gpu, 4, Dtype> &data1,
                                          const Tensor<gpu, 4, Dtype> &data2) {
  const Dtype *bottom_data1 = data1.dptr_;
  const Dtype *bottom_data2 = data2.dptr_;
  Dtype *top_data = out.dptr_;

  const int count = out.shape_.Size(); // the number of threads
  int N = data1.size(0);
  int Data_C = data1.size(1);
  int H = data1.size(2);
  int W = data1.size(3);
  int Out_C = H + W - 1;

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Cross Sparse Correlation Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); 
  CrossSparseCorrelationForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(count, N, Data_C, Out_C, H, W, top_data, bottom_data1, bottom_data2);
}

template<typename Dtype>
__global__ void CrossSparseCorrelationData1BackwardKernel(const int count, int N, int Data_C, int Out_C, int H, int W,
                                                          const Dtype* bottom_data2, 
                                                          const Dtype* top_grad, Dtype* bottom_grad_data1) {
  CUDA_KERNEL_LOOP(index, count) {
      const int w = index % W;
      const int h = (index / W) % H;
      const int c = (index / (H * W)) % Out_C;
      const int n = (index / (Out_C * H * W));

      for(int channel=0; channel < Data_C; channel++) {
          if (c < W) {
          // c is the same height as (h,w) in data2
          atomicAdd(bottom_grad_data1+offset(n,channel,h,w,N,Data_C,H,W), top_grad[index]*bottom_data2[offset(n,channel,h,c,N,Data_C,H,W)]);
          }
          else {
              // the index is in the height
              int data2_h = c - W;
              // upper than h or lower than h
              int data2h = data2_h < h ? data2_h : data2_h + 1;
              atomicAdd(bottom_grad_data1+offset(n,channel,h,w,N,Data_C,H,W), top_grad[index]*bottom_data2[offset(n,channel,data2h,w,N,Data_C,H,W)]);
          }
      }
  }
}


template<typename Dtype>
__global__ void CrossSparseCorrelationData2BackwardKernel(const int count, int N, int Data_C, int Out_C, int H, int W,
                                                          const Dtype* bottom_data1, 
                                                          const Dtype* top_grad, Dtype* bottom_grad_data2) {
  CUDA_KERNEL_LOOP(index, count) {
      const int w = index % W;
      const int h = (index / W) % H;
      const int c = (index / (H * W)) % Out_C;
      const int n = (index / (Out_C * H * W));

      for(int channel=0; channel < Data_C; channel++) {
          if (c < W) {
          // c is the same height as (h,w) in data2
          atomicAdd(bottom_grad_data2+offset(n,channel,h,c,N,Data_C,H,W), top_grad[index]*bottom_data1[offset(n,channel,h,w,N,Data_C,H,W)]);
          }
          else {
              // the index is in the height
              int data2_h = c - W;
              // upper than h or lower than h
              int data2h = data2_h < h ? data2_h : data2_h + 1;  
              atomicAdd(bottom_grad_data2+offset(n,channel,data2h,w,N,Data_C,H,W), top_grad[index]*bottom_data1[offset(n,channel,h,w,N,Data_C,H,W)]);
          }
      }
  }
}

// cross sparse correlation backward
template<typename Dtype>
inline void CrossSparseCorrelationBackward(const Tensor<gpu, 4, Dtype> &grad_data1,
                                           const Tensor<gpu, 4, Dtype> &grad_data2,
                                           const Tensor<gpu, 4, Dtype> &grad_out,
                                           const Tensor<gpu, 4, Dtype> &data1,
                                           const Tensor<gpu, 4, Dtype> &data2) {
  const Dtype *bottom_data1 = data1.dptr_;
  const Dtype *bottom_data2 = data2.dptr_;
  const Dtype *top_grad = grad_out.dptr_;
  Dtype *bottom_grad_data1 = grad_data1.dptr_;
  Dtype *bottom_grad_data2 = grad_data2.dptr_;

  const int count = grad_out.shape_.Size(); // the number of threads
  int N = data1.size(0);
  int Data_C = data1.size(1);
  int H = data1.size(2);
  int W = data1.size(3);

  int Out_C = H + W - 1;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Cross Sparse Correlation Forward");
  cudaStream_t stream_data1 = Stream<gpu>::GetStream(grad_data1.stream_);
  CrossSparseCorrelationData1BackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream_data1>>>(count, N, Data_C, Out_C, H, W, bottom_data2, top_grad, bottom_grad_data1);
  cudaStream_t stream_data2 = Stream<gpu>::GetStream(grad_data2.stream_);
  CrossSparseCorrelationData2BackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream_data2>>>(count, N, Data_C, Out_C, H, W, bottom_data1, top_grad, bottom_grad_data2);
}

} // namespace cuda
template<typename Dtype>
inline void CrossSparseCorrelationForward(const Tensor<gpu, 4, Dtype> &out,
                                          const Tensor<gpu, 4, Dtype> &data1,
                                          const Tensor<gpu, 4, Dtype> &data2) {
  cuda::CrossSparseCorrelationForward(out, data1, data2);
}

template<typename Dtype>
inline void CrossSparseCorrelationBackward(const Tensor<gpu, 4, Dtype> &grad_data1,
                                           const Tensor<gpu, 4, Dtype> &grad_data2,
                                           const Tensor<gpu, 4, Dtype> &grad_out,
                                           const Tensor<gpu, 4, Dtype> &data1,
                                           const Tensor<gpu, 4, Dtype> &data2) {
  cuda::CrossSparseCorrelationBackward(grad_data1, grad_data2, grad_out, data1, data2);
}

} // namespace mshadow
namespace mxnet {
namespace op {
  
template<>
Operator* CreateOp<gpu>(CrossSparseCorrelationParam param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, Dtype, {
    op = new CrossSparseCorrelationOp<gpu, Dtype>(param);
  });
  return op;
}
} // namespace op
} // namespace mxnet