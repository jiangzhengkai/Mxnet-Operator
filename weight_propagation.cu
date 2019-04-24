/*!
 * Copyright (c) 2015 by Contributors
 * \file weight propagation weight_propagation-inl.h
 * \brief
 * \author ZhengKai Jiang
*/
#include "./weight_propagation-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"


namespace mshadow {
namespace cuda {
    
inline __device__ int offset(int n, int c, int h, int w, int N, int C, int H, int W) {
    return n*C*H*W + c*H*W + h*W + w;
}

template<typename DType>
__global__ void WeightPropagationForwardKernel(const int count, int N, int C, int size_weights, int H,
                int W, int weight_height_, int weight_width_, int hole_, 
                const DType* bottom_data, const DType* bottom_weights, DType* top_data ) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int n = (index / (C * H * W));

    int p_h = 0,p_w = 0,p_weight = 0;
    for(int i =0; i < weight_height_; i++) {
      for(int j=0; j< weight_width_; j++) {
        p_h = hole_ * (i - weight_height_/2) + h;
        p_w = hole_ * (j - weight_width_/2) + w;
        if(p_h >= 0 && p_w >= 0 && p_h < H && p_w < W) {
          p_weight = i * weight_width_ + j;
          top_data[index] += bottom_data[offset(n,c,p_h,p_w,N,C,H,W)] * bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)];
          }
        }
      }
  } // cuda_kernel_loop

}

template<typename DType>
inline void WeightPropagationForward(const Tensor<gpu, 4, DType> &out,
                                     const Tensor<gpu, 4, DType> &data,
                                     const Tensor<gpu, 4, DType> &weights,
                                     const int weight_height_,
                                     const int weight_width_,
                                     const int hole_) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_weights = weights.dptr_;
  DType *top_data = out.dptr_;
  const int count = out.shape_.Size(); // the number of threads
  int N = data.size(0);
  int C = data.size(1);
  int H = data.size(2);
  int W = data.size(3);

  int size_weights = weight_height_ * weight_width_;

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "WeightPropagation Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); 
  WeightPropagationForwardKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, N, C, size_weights, H, W, weight_height_, weight_width_, hole_, bottom_data, bottom_weights, top_data);
}

template<typename DType>
__global__ void WeightPropagationDataBackwardAccKernel(const int count, int N, int C, int size_weights, int H, int W, int weight_height_,
                                                   int weight_width_, int hole_, const DType* grad_out, const DType* bottom_data, 
                                                   const DType* bottom_weights, DType* grad_data, DType* grad_weights) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int n = (index / (C * H * W));
    int p_h = 0,p_w = 0, p_weight = 0;
    for(int i = 0; i < weight_height_; ++i) {
      for(int j = 0; j < weight_width_; ++j) {
        p_h = hole_ * (i - weight_height_/2) + h;
        p_w = hole_ * (j - weight_width_/2) + w;

        if(p_h >= 0 && p_w >= 0 && p_h < H && p_w < W) {

          p_weight = i * weight_width_ + j;

          atomicAdd(grad_data+offset(n,c,p_h,p_w,N,C,H,W),grad_out[index]*bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)]);
          }
        }
      }
  }

}              


template<typename DType>
__global__ void WeightPropagationWeightsBackwardAccKernel(const int count, int N, int C, int size_weights, int H, int W, int weight_height_,
                                                   int weight_width_, int hole_, const DType* grad_out, const DType* bottom_data, 
                                                   const DType* bottom_weights, DType* grad_data, DType* grad_weights) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int n = (index / (C * H * W));
    int p_h = 0,p_w = 0, p_weight = 0;
    for(int i = 0; i < weight_height_; ++i) {
      for(int j = 0; j < weight_width_; ++j) {
        p_h = hole_ * (i - weight_height_/2) + h;
        p_w = hole_ * (j - weight_width_/2) + w;

        if(p_h >= 0 && p_w >= 0 && p_h < H && p_w < W) {

          p_weight = i * weight_width_ + j;

          atomicAdd(grad_weights+offset(n,p_weight,h,w,N,size_weights,H,W),grad_out[index]*bottom_data[offset(n,c,p_h,p_w,N,C,H,W)]);
          }
        }
      }
  }

}              


template<typename DType>
inline void WeightPropagationBackwardAcc(const Tensor<gpu, 4, DType> &grad_data,
                                         const Tensor<gpu, 4, DType> &grad_weights,
                                         const Tensor<gpu, 4, DType> &grad_out,
                                         const Tensor<gpu, 4, DType> &data,
                                         const Tensor<gpu, 4, DType> &weights,
                                         const int weight_height_,
                                         const int weight_width_,
                                         const int hole_) {
  const DType *top_grad = grad_out.dptr_;
  const DType *bottom_data = data.dptr_;
  const DType *bottom_weights = weights.dptr_;
  DType *bottom_grad_data = grad_data.dptr_;
  DType *bottom_grad_weights = grad_weights.dptr_;
  const int count = data.shape_.Size(); // the number of threads
  int N = data.size(0);
  int C = data.size(1);
  int H = data.size(2);
  int W = data.size(3);
  int size_weights = weight_height_ * weight_width_;
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "WeightPropagation Backward");
  cudaStream_t stream_data = Stream<gpu>::GetStream(grad_data.stream_);
  WeightPropagationDataBackwardAccKernel<DType><<<dimGrid, dimBlock, 0, stream_data>>>(count, N, C, size_weights, 
  H, W, weight_height_, weight_width_, hole_, top_grad, bottom_data, bottom_weights, bottom_grad_data, bottom_grad_weights);

  cudaStream_t stream_weights = Stream<gpu>::GetStream(grad_weights.stream_);
  WeightPropagationWeightsBackwardAccKernel<DType><<<dimGrid, dimBlock, 0, stream_weights>>>(count, N, C, size_weights, 
  H, W, weight_height_, weight_width_, hole_, top_grad, bottom_data, bottom_weights, bottom_grad_data, bottom_grad_weights);

}



} // namespace cuda
template<typename DType>
inline void WeightPropagationForward(const Tensor<gpu, 4, DType> &out,
                                     const Tensor<gpu, 4, DType> &data,
                                     const Tensor<gpu, 4, DType> &weights,
                                     const int weight_height_,
                                     const int weight_width_,
                                     const int hole_) {
  cuda::WeightPropagationForward(out, data, weights, weight_height_, weight_width_, hole_);
}

template<typename DType>
inline void WeightPropagationBackwardAcc(const Tensor<gpu, 4, DType> &grad_data,
                                         const Tensor<gpu, 4, DType> &grad_weights,
                                         const Tensor<gpu, 4, DType> &grad_out,
                                         const Tensor<gpu, 4, DType> &data,
                                         const Tensor<gpu, 4, DType> &weights,
                                         const int weight_height_,
                                         const int weight_width_,
                                         const int hole_) {
  cuda::WeightPropagationBackwardAcc(grad_data, grad_weights, grad_out, data, weights, weight_height_, weight_width_, hole_);
}

} //namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(WeightPropagationParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new WeightPropagationOp<gpu, DType>(param);
  });
  return op;
}


}  // namespace op
} // namespace mxnet
