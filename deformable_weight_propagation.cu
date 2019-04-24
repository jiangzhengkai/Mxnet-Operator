/*!
 * Copyright (c) 2015 by Contributors
 * \file deformable_weight_propagation-inl.h
 * \brief deformable_weight_propagation operator and symbol
 * \author ZhengKai Jiang
*/
#include "./deformable_weight_propagation-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

namespace mshadow {
namespace cuda {
template <typename Dtype>   
__device__ Dtype offset(Dtype n, Dtype c, Dtype h, Dtype w, Dtype N, Dtype C, Dtype H, Dtype W) {
    return n*C*H*W + c*H*W + h*W + w;
}


template <typename Dtype>
__device__ Dtype roialign_bilinear_interp(const Dtype* data,
                                          Dtype x,
                                          Dtype y,
                                          int width) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  Dtype dist_x = static_cast<Dtype>(x - x1);
  Dtype dist_y = static_cast<Dtype>(y - y1);
  Dtype value11 = data[y1*width + x1];
  Dtype value12 = data[y2*width + x1];
  Dtype value21 = data[y1*width + x2];
  Dtype value22 = data[y2*width + x2];
  Dtype value = (1 - dist_x)*(1 - dist_y)*value11 + (1 - dist_x)*dist_y*value12 + 
                 dist_x*(1 - dist_y)*value21 + dist_x*dist_y*value22;
  return value;
}

template<typename DType>
__global__ void DeformableWeightPropagationForwardKernel(const int count, int N, int C, int size_weights, int size_offsets, int H, int W, 
                                                         int weight_height_, int weight_width_, int hole_, 
                                                         const DType* bottom_data, const DType* bottom_weights, const DType* bottom_offsets, DType* top_data ) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {
      const int w = index % W;
      const int h = (index / W) % H;
      const int c = (index / (H * W)) % C;
      const int n = (index / (C * H * W));

      const DType* data = bottom_data + n * (C * H * W) + c * (H * W);

      int p_h = 0,p_w = 0,p_weight = 0;
      int p_h_offset_height=0, p_w_offset_weight=0; // predict offsets
      DType p_h_position = 0; // true float position
      DType p_w_position = 0; // true float position
      for(int i =0; i < weight_height_; i++) {
        for(int j=0; j< weight_width_; j++) {
            p_h = hole_ * (i - weight_height_/2) + h;
            p_w = hole_ * (j - weight_width_/2) + w;
            p_w_offset_weight = i * weight_width_ + j;
            p_h_offset_height = i * weight_width_ + j + weight_height_ * weight_width_;
            p_w_position = bottom_offsets[offset(n,p_w_offset_weight,h,w,N,size_offsets,H,W)];
            p_h_position = bottom_offsets[offset(n,p_h_offset_height,h,w,N,size_offsets,H,W)];
            p_h_position += p_h;
            p_w_position += p_w;
            p_weight = i * weight_width_ + j;
            int x1 = floor(p_w_position);
            int x2 = ceil(p_w_position);
            int y1 = floor(p_h_position);
            int y2 = ceil(p_h_position);
            if(p_h_position >= 0 && p_w_position >= 0 && p_h_position < H && p_w_position < W && x1 >= 0 && x1 <W &&  x2 >= 0 && x2 < W && y1 >= 0 && y1 <H &&  y2 >= 0 && y2 < H) {
                // according to offsets and weights to calculate top_data
                top_data[index] += roialign_bilinear_interp(data,p_w_position,p_h_position,W) * bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)];
            }
            else {
                int x = ceil(p_w_position);
                int y = ceil(p_h_position);
                int data_w = min(max(0, x), W); 
                int data_h = min(max(0, y), H);
                top_data[index] += bottom_data[offset(n,c,h,w,N,C,H,W)] * bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)];
            }
        }
      }
  } // cuda_kernel_loop

}

template<typename DType>
inline void DeformableWeightPropagationForward(const Tensor<gpu, 4, DType> &out,
                                               const Tensor<gpu, 4, DType> &data,
                                               const Tensor<gpu, 4, DType> &weights,
                                               const Tensor<gpu, 4, DType> &offsets,
                                               const int weight_height_,
                                               const int weight_width_,
                                               const int hole_) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_weights = weights.dptr_;
  const DType *bottom_offsets = offsets.dptr_;
  DType *top_data = out.dptr_;
  
  int N = data.size(0);
  int C = data.size(1);
  int H = data.size(2);
  int W = data.size(3);
  int size_weights = weight_height_ * weight_width_;
  int size_offsets = 2 * weight_height_ * weight_width_;

  const int count = out.shape_.Size(); // the number of threads
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "WeightPropagation Forward");

  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_); 
  DeformableWeightPropagationForwardKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(count, N, C, size_weights, size_offsets, H, W, weight_height_, weight_width_, hole_, 
                                                                                    bottom_data, bottom_weights, bottom_offsets, top_data);
}

template <typename Dtype>
__device__ Dtype bilinear_function(int x,
                                   int y,
                                   Dtype i,
                                   Dtype j) {
  Dtype dist_x = static_cast<Dtype>(i - x);
  Dtype dist_y = static_cast<Dtype>(j - y);
  Dtype value = max(0.0,1-abs(dist_x)) * max(0.0, 1-abs(dist_y));
  return value;
}

template<typename DType>
__global__ void WeightPropagationDataBackwardAccKernel(const int count, int N, int C, int size_weights, int size_offsets, int H, int W, int weight_height_,
                                                       int weight_width_, int hole_, const DType* grad_out, const DType* bottom_data, 
                                                       const DType* bottom_weights, const DType* bottom_offsets, DType* grad_data) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {
      const int w = index % W;
      const int h = (index / W) % H;
      const int c = (index / (H * W)) % C;
      const int n = (index / (C * H * W));
      int p_h = 0,p_w = 0, p_weight = 0;
      int p_h_offset_height=0, p_w_offset_weight=0;
      DType p_h_position = 0;
      DType p_w_position = 0;
    
      for(int i = 0; i < weight_height_; ++i) {
        for(int j = 0; j < weight_width_; ++j) {
            p_h = hole_ * (i - weight_height_/2) + h;
            p_w = hole_ * (j - weight_width_/2) + w;
            p_w_offset_weight = i * weight_width_ + j;
            p_h_offset_height = i * weight_width_ + j + weight_height_ * weight_width_;
            p_w_position = bottom_offsets[offset(n,p_w_offset_weight,h,w,N,size_offsets,H,W)];
            p_h_position = bottom_offsets[offset(n,p_h_offset_height,h,w,N,size_offsets,H,W)];
            p_h_position += p_h;
            p_w_position += p_w;

            int x1 = floor(p_w_position);
            int x2 = ceil(p_w_position);
            int y1 = floor(p_h_position);
            int y2 = ceil(p_h_position);
            if(p_h_position >= 0 && p_w_position >= 0 && p_h_position < H && p_w_position < W ) {
                p_weight = i * weight_width_ + j;
                if(x1 >= 0 && x1 <W &&  y1 >= 0 && y1 <H) {
                    atomicAdd(grad_data+offset(n,c,y1,x1,N,C,H,W),grad_out[index]*bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)]*bilinear_function(x1,y1,p_w_position,p_h_position));
                }
                if(x1 >= 0 && x1 <W &&  y2 >= 0 && y2 <H) {
                    atomicAdd(grad_data+offset(n,c,y2,x1,N,C,H,W),grad_out[index]*bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)]*bilinear_function(x1,y2,p_w_position,p_h_position));
                }
                if(x2 >= 0 && x2 <W &&  y1 >= 0 && y1 <H) {
                    atomicAdd(grad_data+offset(n,c,y1,x2,N,C,H,W),grad_out[index]*bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)]*bilinear_function(x2,y1,p_w_position,p_h_position));
                }
                if(x2 >= 0 && x2 <W &&  y2 >= 0 && y2 <H) {
                    atomicAdd(grad_data+offset(n,c,y2,x2,N,C,H,W),grad_out[index]*bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)]*bilinear_function(x2,y2,p_w_position,p_h_position));
                }
            }
        }
      }
  }

}              


template<typename DType>
__global__ void WeightPropagationWeightsBackwardAccKernel(const int count, int N, int C, int size_weights, int size_offsets, int H, int W, int weight_height_,
                                                          int weight_width_, int hole_, const DType* grad_out, const DType* bottom_data, 
                                                          const DType* bottom_weights, const DType* bottom_offsets, DType* grad_weights) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {
      const int w = index % W;
      const int h = (index / W) % H;
      const int c = (index / (H * W)) % C;
      const int n = (index / (C * H * W));
      int p_h = 0,p_w = 0, p_weight = 0;
      const DType* data = bottom_data + n * (C * H * W) + c * (H * W);
      int p_w_offset_weight=0, p_h_offset_height=0;
      DType p_h_position = 0;
      DType p_w_position = 0;
      for(int i = 0; i < weight_height_; ++i) {
        for(int j = 0; j < weight_width_; ++j) {
            p_h = hole_ * (i - weight_height_/2) + h;
            p_w = hole_ * (j - weight_width_/2) + w;
            p_w_offset_weight = i * weight_width_ + j;
            p_h_offset_height = i * weight_width_ + j + weight_height_ * weight_width_;
            p_w_position = bottom_offsets[offset(n,p_w_offset_weight,h,w,N,size_offsets,H,W)];
            p_h_position = bottom_offsets[offset(n,p_h_offset_height,h,w,N,size_offsets,H,W)];
            p_h_position += p_h;
            p_w_position += p_w;
            int x1 = floor(p_w_position);
            int x2 = ceil(p_w_position);
            int y1 = floor(p_h_position);
            int y2 = ceil(p_h_position);
            if(p_h_position >= 0 && p_w_position >= 0 && p_w_position < H && p_h_position < W && x1 >= 0 && x1 <W &&  x2 >= 0 && x2 < W && y1 >= 0 && y1 < H &&  y2 >= 0 && y2 < H) {
                p_weight = i * weight_width_ + j;
                atomicAdd(grad_weights+offset(n,p_weight,h,w,N,size_weights,H,W),grad_out[index]* roialign_bilinear_interp(data,p_w_position,p_h_position,W));
            }
        }
      }
  }

}        

template<typename DType>
__global__ void WeightPropagationOffsetsBackwardAccKernel(const int count, int N, int C, int size_weights, int size_offsets, int H, int W, int weight_height_,
                                                          int weight_width_, int hole_, const DType* grad_out, const DType* bottom_data, 
                                                          const DType* bottom_weights, const DType* bottom_offsets, DType* grad_offsets) {
  for(int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
      index < count;
      index += blockDim.x * gridDim.x * gridDim.y) {
      const int w = index % W;
      const int h = (index / W) % H;
      const int c = (index / (H * W)) % C;
      const int n = (index / (C * H * W));
      int p_h = 0,p_w = 0, p_weight = 0;
      int p_w_offset_weight=0, p_h_offset_height=0;
      DType p_h_position = 0;
      DType p_w_position = 0;
      for(int i = 0; i < weight_height_; ++i) {
        for(int j = 0; j < weight_width_; ++j) {
            p_h = hole_ * (i - weight_height_/2) + h;
            p_w = hole_ * (j - weight_width_/2) + w;
            p_weight = i * weight_width_ + j;
            p_w_offset_weight = i * weight_width_ + j;
            p_h_offset_height = i * weight_width_ + j + weight_height_ * weight_width_;
            p_w_position = bottom_offsets[offset(n,p_w_offset_weight,h,w,N,size_offsets,H,W)];
            p_h_position = bottom_offsets[offset(n,p_h_offset_height,h,w,N,size_offsets,H,W)];
            p_h_position += p_h;
            p_w_position += p_w;

            int x1 = floor(p_w_position);
            int x2 = ceil(p_w_position);
            int y1 = floor(p_h_position);
            int y2 = ceil(p_h_position);
            if(p_h_position >= 0 && p_w_position >= 0 && p_h_position < H && p_w_position < W ) {
                p_weight = i * weight_width_ + j;
             
                DType dist_x = p_w_position - x1;
                DType dist_y = p_h_position - y1;

                DType grad_offset_u11 =  -1 * (1-dist_y); // x1 y1
                DType grad_offset_u12 = -1 * dist_y; // x1 y2
                DType grad_offset_u21 = 1 - dist_y; // x2 y1
                DType grad_offset_u22 = dist_y; // x2 y2
                DType data_11 = 0;
                DType data_12 = 0;
                DType data_21 = 0;
                DType data_22 = 0;

                if(x1 >= 0 && x1 <W && y1 >= 0 && y1 <H) {
                    data_11 = bottom_data[offset(n,c,y1,x1,N,C,H,W)];
                }
                if(x1 >= 0 && x1 <W && y2 >= 0 && y2 <H) {
                    data_12 = bottom_data[offset(n,c,y2,x1,N,C,H,W)];
                }
                if(x2 >= 0 && x2 <W && y1 >= 0 && y1 <H) {
                    data_21 = bottom_data[offset(n,c,y1,x2,N,C,H,W)];
                }
                if(x1 >= 0 && x1 <W && y1 >= 0 && y1 <H) {
                    data_22 = bottom_data[offset(n,c,y2,x2,N,C,H,W)];
                }

                DType grad_offset_u = grad_offset_u11 * data_11  + grad_offset_u12 * data_12 + grad_offset_u21 * data_21 + grad_offset_u22 * data_22;
               
                atomicAdd(grad_offsets+offset(n,p_w_offset_weight,h,w,N,size_offsets,H,W),grad_out[index]*bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)]*grad_offset_u);
                DType grad_offset_v11 =  -1 * (1-dist_x); // x1 y1
                DType grad_offset_v12 = 1 - dist_x; // x1 y2
                DType grad_offset_v21 = -1 * dist_x; // x2 y1
                DType grad_offset_v22 = dist_x; // x2 y2 
                DType grad_offset_v = grad_offset_v11 * data_11 + grad_offset_v12 * data_12 + grad_offset_v21 * data_21 + grad_offset_v22 * data_22;
                atomicAdd(grad_offsets+offset(n,p_h_offset_height,h,w,N,size_offsets,H,W),grad_out[index]*bottom_weights[offset(n,p_weight,h,w,N,size_weights,H,W)]*grad_offset_v);

                
            }
        }
      }
  }

}                  

template<typename DType>
inline void DeformableWeightPropagationBackwardAcc(const Tensor<gpu, 4, DType> &grad_data,
                                                   const Tensor<gpu, 4, DType> &grad_weights,
                                                   const Tensor<gpu, 4, DType> &grad_offsets,
                                                   const Tensor<gpu, 4, DType> &grad_out,
                                                   const Tensor<gpu, 4, DType> &data,
                                                   const Tensor<gpu, 4, DType> &weights,
                                                   const Tensor<gpu, 4, DType> &offsets,
                                                   const int weight_height_,
                                                   const int weight_width_,
                                                   const int hole_) {
  const DType *top_grad = grad_out.dptr_;
  const DType *bottom_data = data.dptr_;
  const DType *bottom_weights = weights.dptr_;
  const DType *bottom_offsets = offsets.dptr_;
  DType *bottom_grad_data = grad_data.dptr_;
  DType *bottom_grad_weights = grad_weights.dptr_;
  DType *bottom_grad_offsets = grad_offsets.dptr_;
  
  int N = data.size(0);
  int C = data.size(1);
  int H = data.size(2);
  int W = data.size(3);
  int size_weights = weight_height_ * weight_width_;
  int size_offsets = 2 * weight_height_ * weight_width_;

  const int count = grad_out.shape_.Size(); // the number of threads
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "WeightPropagation Backward");
  cudaStream_t stream_data = Stream<gpu>::GetStream(grad_data.stream_);
  WeightPropagationDataBackwardAccKernel<DType><<<dimGrid, dimBlock, 0, stream_data>>>(count, N, C, size_weights, size_offsets, H, W, weight_height_, weight_width_, hole_, 
                                                                                       top_grad, bottom_data, bottom_weights, bottom_offsets, bottom_grad_data);

  cudaStream_t stream_weights = Stream<gpu>::GetStream(grad_weights.stream_);
  WeightPropagationWeightsBackwardAccKernel<DType><<<dimGrid, dimBlock, 0, stream_weights>>>(count, N, C, size_weights, size_offsets, H, W, weight_height_, weight_width_, hole_, 
                                                                                             top_grad, bottom_data, bottom_weights,bottom_offsets, bottom_grad_weights);

  cudaStream_t stream_offsets = Stream<gpu>::GetStream(grad_offsets.stream_);
  WeightPropagationOffsetsBackwardAccKernel<DType><<<dimGrid, dimBlock, 0, stream_offsets>>>(count, N, C, size_weights, size_offsets, H, W, weight_height_, weight_width_, hole_, 
                                                                                             top_grad, bottom_data, bottom_weights, bottom_offsets, bottom_grad_offsets);

}



} // namespace cuda
template<typename DType>
inline void DeformableWeightPropagationForward(const Tensor<gpu, 4, DType> &out,
                                               const Tensor<gpu, 4, DType> &data,
                                               const Tensor<gpu, 4, DType> &weights,
                                               const Tensor<gpu, 4, DType> &offsets,
                                               const int weight_height_,
                                               const int weight_width_,
                                               const int hole_) {
  cuda::DeformableWeightPropagationForward(out, data, weights, offsets, weight_height_, weight_width_, hole_);
}

template<typename DType>
inline void DeformableWeightPropagationBackwardAcc(const Tensor<gpu, 4, DType> &grad_data,
                                                   const Tensor<gpu, 4, DType> &grad_weights,
                                                   const Tensor<gpu, 4, DType> &grad_offsets,
                                                   const Tensor<gpu, 4, DType> &grad_out,
                                                   const Tensor<gpu, 4, DType> &data,
                                                   const Tensor<gpu, 4, DType> &weights,
                                                   const Tensor<gpu, 4, DType> &offsets,
                                                   const int weight_height_,
                                                   const int weight_width_,
                                                   const int hole_) {
  cuda::DeformableWeightPropagationBackwardAcc(grad_data, grad_weights, grad_offsets, grad_out, data, weights, offsets, weight_height_, weight_width_, hole_);
}

} //namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(DeformableWeightPropagationParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new DeformableWeightPropagationOp<gpu, DType>(param);
  });
  return op;
}


}  // namespace op
} // namespace mxnet
