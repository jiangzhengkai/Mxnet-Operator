/*!
 * Copyright (c) 2018 by Contributors
 * Copyright (c) 2018 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file cross_weight_propagation-inl.h
 * \brief cross_weight_propagation operator
 * \author Jiang ZhengKai
*/
#include "./cross_weight_propagation-inl.h"
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
__global__ void CrossWeightPropagationForwardKernel(const int count, int N, int Data_C, int Weight_C, int H, int W,
                                                    Dtype* top_data, const Dtype* bottom_data, const Dtype* bottom_weight) {
  CUDA_KERNEL_LOOP(index, count) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % Data_C;
    const int n = (index / (Data_C * H * W));

    for(int i=0; i<Weight_C;i++) {
        if (i<W) {
          top_data[offset(n,c,h,w,N,Data_C,H,W)] += bottom_data[offset(n,c,h,i,N,Data_C,H,W)] * bottom_weight[offset(n,i,h,w,N,Weight_C,H,W)];
        }
        else {
          int weight_h = i -W;
          int weighth = weight_h < h ? weight_h : weight_h + 1;
          top_data[offset(n,c,h,w,N,Data_C,H,W)] += bottom_data[offset(n,c,weighth,w,N,Data_C,H,W)] * bottom_weight[offset(n,i,h,w,N,Weight_C,H,W)];
        }
    }
  } // cuda kernel loop
}

// cross weight propagation forward
template<typename Dtype>
inline void CrossWeightPropagationForward(const Tensor<gpu, 4, Dtype> &out,
                                          const Tensor<gpu, 4, Dtype> &data,
                                          const Tensor<gpu, 4, Dtype> &weight) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_weight = weight.dptr_;
  Dtype *top_data = out.dptr_;
  const int count = out.shape_.Size(); // the number of threads
  int N = data.size(0);
  int Data_C = data.size(1);
  int H = data.size(2);
  int W = data.size(3);
  int Weight_C = H + W - 1;

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Cross Sparse Correlation Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); 
  CrossWeightPropagationForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(count, N, Data_C, Weight_C, H, W, top_data, bottom_data, bottom_weight);
}
 // data backward
template<typename Dtype>
__global__ void CrossWeightPropagationDataBackwardKernel(const int count, int N, int Data_C, int Weight_C, int H, int W,
                                                         const Dtype* bottom_weight, const Dtype* top_grad, Dtype* bottom_grad_data) {
  CUDA_KERNEL_LOOP(index, count) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % Data_C;
    const int n = (index / (Data_C * H * W));

    for(int i=0; i<Weight_C;i++) {
        if (i<W) {
          atomicAdd(bottom_grad_data+offset(n,c,h,i,N,Data_C,H,W),top_grad[offset(n,c,h,w,N,Data_C,H,W)]*bottom_weight[offset(n,i,h,w,N,Weight_C,H,W)]);
        }
        else {
          int weight_h = i -W;
          int weighth = weight_h < h ? weight_h : weight_h + 1;
          atomicAdd(bottom_grad_data+offset(n,c,weighth,w,N,Data_C,H,W),top_grad[offset(n,c,h,w,N,Data_C,H,W)]*bottom_weight[offset(n,i,h,w,N,Weight_C,H,W)]);
        }
    }
  } // cuda kernel loop
}

// weight backward
template<typename Dtype>
__global__ void CrossWeightPropagationWeightBackwardKernel(const int count, int N, int Data_C, int Weight_C, int H, int W,
                                                           const Dtype* bottom_data, const Dtype* top_grad, Dtype* bottom_grad_weight) {
  CUDA_KERNEL_LOOP(index, count) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % Data_C;
    const int n = (index / (Data_C * H * W));

    for(int i=0; i<Weight_C;i++) {
        if (i<W) {
          atomicAdd(bottom_grad_weight+offset(n,i,h,w,N,Weight_C,H,W),top_grad[offset(n,c,h,w,N,Data_C,H,W)]*bottom_data[offset(n,c,h,i,N,Data_C,H,W)]);
        }
        else {
          int weight_h = i -W;
          int weighth = weight_h < h ? weight_h : weight_h + 1;
          atomicAdd(bottom_grad_weight+offset(n,i,h,w,N,Weight_C,H,W),top_grad[offset(n,c,h,w,N,Data_C,H,W)]*bottom_data[offset(n,c,weighth,w,N,Data_C,H,W)]);
        }
    }
  } // cuda kernel loop
}

// cross weight propagation backward
template<typename Dtype>
inline void CrossWeightPropagationBackward(const Tensor<gpu, 4, Dtype> &grad_data,
                                           const Tensor<gpu, 4, Dtype> &grad_weight,
                                           const Tensor<gpu, 4, Dtype> &grad_out,
                                           const Tensor<gpu, 4, Dtype> &data,
                                           const Tensor<gpu, 4, Dtype> &weight) {
  const Dtype *bottom_data = data.dptr_;
  const Dtype *bottom_weight = weight.dptr_;
  const Dtype *top_grad = grad_out.dptr_;
  Dtype *bottom_grad_data = grad_data.dptr_;
  Dtype *bottom_grad_weight = grad_weight.dptr_;

  const int count = data.shape_.Size(); // the number of threads
  int N = data.size(0);
  int Data_C = data.size(1);
  int H = data.size(2);
  int W = data.size(3);
  int Weight_C = weight.size(1);

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Cross Sparse Correlation Forward");
  cudaStream_t stream_data = Stream<gpu>::GetStream(grad_data.stream_);
  CrossWeightPropagationDataBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream_data>>>(count, N, Data_C, Weight_C, H, W, bottom_weight, top_grad, bottom_grad_data);

  cudaStream_t stream_weight = Stream<gpu>::GetStream(grad_weight.stream_);
  CrossWeightPropagationWeightBackwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream_weight>>>(count, N, Data_C, Weight_C, H, W, bottom_data, top_grad, bottom_grad_weight);
}

} // namespace cuda
template<typename Dtype>
inline void CrossWeightPropagationForward(const Tensor<gpu, 4, Dtype> &out,
                                          const Tensor<gpu, 4, Dtype> &data,
                                          const Tensor<gpu, 4, Dtype> &weight) {
  cuda::CrossWeightPropagationForward(out, data, weight);
}

template<typename Dtype>
inline void CrossWeightPropagationBackward(const Tensor<gpu, 4, Dtype> &grad_data,
                                           const Tensor<gpu, 4, Dtype> &grad_weight,
                                           const Tensor<gpu, 4, Dtype> &grad_out,
                                           const Tensor<gpu, 4, Dtype> &data,
                                           const Tensor<gpu, 4, Dtype> &weight) {
  cuda::CrossWeightPropagationBackward(grad_data, grad_weight, grad_out, data, weight);
}
} // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(CrossWeightPropagationParam param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, Dtype, {
    op = new CrossWeightPropagationOp<gpu, Dtype>(param);
  });
  return op;
}
} // namespace op
} // namespace mxnet