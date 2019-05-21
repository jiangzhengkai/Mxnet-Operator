/*!
 * Copyright (c) 2015 by Contributors
 * \file weight_generate-inl.h
 * \brief
 * \author ZhengKai Jiang
*/
#include "./align_data_by_index-inl.h"
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
__global__ void AlignDataByIndexForwardKernel(const int count, int N, int C, int H, int W, int index_h, int index_w,
                                              const DType* bottom_data, const DType* bottom_index, DType* top_data ) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {

    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int n = (index / (C * H * W));

    top_data[index] = bottom_data[offset(n,c,h,w,N,C,H,W)+]
 
    int p_h = 0,p_w = 0,n=0;
    for(int i =0; i < 2 * index_w + 1; i++) {
      for(int j=0; j< 2 * index_h + 1; j++) {
        if n == bottom_index[offset(n, 0, h, w, N, 1, H, W)] {
            p_h = (i - index_h) + h;
            p_w = (j - index_w) + w;
            if(p_h >= 0 && p_w >= 0 && p_h < H && p_w < W) {
            top_data[index] += bottom_data[offset(n,c,p_h,p_w,N,C,H,W)]
            }
        }
      }
    }
  } // cuda_kernel_loop
}

template<typename DType>
inline void AlignDataByIndexForward(const Tensor<gpu, 4, DType> &out,
                                    const Tensor<gpu, 4, DType> &data,
                                    const Tensor<gpu, 4, DType> &index) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_index = index.dptr_;
  DType *top_data = out.dptr_;
  const int count = out.shape_.Size(); // the number of threads
  int N = data.size(0);
  int C = data.size(1);
  int H = data.size(2);
  int W = data.size(3);

  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Align Data By Index Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  AlignDataByIndexForwardKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, N, C, H, W, bottom_data, bottom_index, top_data);
}


template<typename DType>
__global__ void AlignDataBackwardKernel(const int count, int N, int K, int C, int H, int W,
                                              const DType* grad_out, const DType* bottom_data,
                                              const DType* bottom_weight, DType* grad_data, DType* grad_weight) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x * gridDim.y) {
    const int w = index % W;
    const int h = (index / W) % H;
    const int c = (index / (H * W)) % C;
    const int n = (index / (C * H * W));

    for (int i=0;i<K;i++) {
      atomicAdd(grad_data+offset5d(n,i,c,h,w,N,K,C,H,W),grad_out[index]*bottom_weight[offset(n,i,h,w,N,K,H,W)]);
      atomicAdd(grad_weight+offset(n,i,h,w,N,K,H,W),grad_out[index]*bottom_data[offset5d(n,i,c,h,w,N,K,C,H,W)]);
      }
    }
}


template<typename DType>
inline void AlignDataBackward(const Tensor<gpu, 5, DType> &grad_data,
                              const Tensor<gpu, 4, DType> &grad_weight,
                              const Tensor<gpu, 5, DType> &data,
                              const Tensor<gpu, 4, DType> &weight,
                              const Tensor<gpu, 4, DType> &grad_out) {
  const DType *top_grad = grad_out.dptr_;
  const DType *bottom_data = data.dptr_;
  const DType *bottom_weight = weight.dptr_;
  DType *bottom_grad_data = grad_data.dptr_;
  DType *bottom_grad_weight = grad_weight.dptr_;

  const int count = grad_out.shape_.Size(); // the number of threads

  int N = data.size(0);
  int K = data.size(1);
  int C = data.size(2);
  int H = data.size(3);
  int W = data.size(4);


  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "Align Data Backward");
  cudaStream_t stream_data = Stream<gpu>::GetStream(grad_data.stream_);

  AlignDataBackwardKernel<DType><<<dimGrid, dimBlock, 0, stream_data>>>(count, N, K, C, H, W,
                                        top_grad, bottom_data, bottom_weight, bottom_grad_data, bottom_grad_weight);

}



} // namespace cuda

template<typename DType>
inline void AlignDataByIndexForward(const Tensor<gpu, 4, DType> &out,
                                    const Tensor<gpu, 4, DType> &data,
                                    const Tensor<gpu, 4, DType> &index) {
  cuda::AlignDataByIndexForward(out, data, weight);
}

template<typename DType>
inline void AlignDataByIndexBackward(const Tensor<gpu, 5, DType> &grad_data,
                                     const Tensor<gpu, 4, DType> &grad_index,
                                     const Tensor<gpu, 5, DType> &data,
                                     const Tensor<gpu, 4, DType> &index,
                                     const Tensor<gpu, 4, DType> &grad_out) {
  cuda::AlignDataByInexBackward(grad_data, grad_index, data, index, grad_out);
}


} //namespace mshadow


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(AlignDataByIndexParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AlignDataByIndexOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
} // namespace mxnet
