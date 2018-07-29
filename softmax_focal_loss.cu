/*
softmax focal loss
author: ZhengKai Jiang
*/

#include "./softmax_focal_loss-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"

namespace mshadow {
namespace cuda {
template<typename Dtype>
__global__ void SpatialSoftmaxKernel(
    const Dtype* bottom_data,
    Dtype* top_data, const int count, const int spatial_dim,
    const int num_classes) {
  for (int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x * gridDim.y) {

    int s = i % spatial_dim;
    int n = i / (spatial_dim * num_classes);
   
    // subtract max on each cell for numerical reasons
    float max_val = -FLT_MAX;
    for(int c=0; c<num_classes; ++c) {
        int idx = 0;
        idx = n * (spatial_dim * num_classes) + c * spatial_dim + s;
        max_val = max(max_val, bottom_data[idx]);
    }
    // Exponentiate
    float expsum = 0.0f;
    for(int c=0; c<num_classes; ++c) {
        int idx = 0;
        idx = n * (spatial_dim * num_classes) + c * spatial_dim + s;
        float expx = exp(bottom_data[idx] - max_val);
        top_data[idx] = expx;
        expsum += expx;
    }
    // normalization
    for(int c=0; c<num_classes; ++c) {
        int idx = 0;
        idx = n * (spatial_dim * num_classes) + c * spatial_dim + s;
        top_data[idx] /= expsum;
    }
  }
}
template<typename Dtype>
inline void SoftmaxFocalLossForward(const Tensor<gpu, 3, Dtype> &prob,
                                    const Tensor<gpu, 3, Dtype> &in_data,
                                    const Tensor<gpu, 2, Dtype> &label,
                                    const int num_classes) {
  const Dtype* bottom_data = in_data.dptr_;
  Dtype* top_prob = prob.dptr_;
 
  
  const int count = label.shape_.Size();
  const int spatial_dim = in_data.size(2);
  
  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocalLoss Forward");
  
  cudaStream_t stream = Stream<gpu>::GetStream(prob.stream_);
  SpatialSoftmaxKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
    bottom_data, top_prob, count, spatial_dim, num_classes);


}

template<typename Dtype>
__global__ void SoftmaxFocalGradientWeightKernel(
    Dtype* bottom_prob, const Dtype* bottom_label, 
    Dtype* weight, const int count_weight,
    const int num_classes, const int spatial_dim, 
    const float ignore_label, const float alpha, const float gamma) {
    for (int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       i < count_weight; i += blockDim.x * gridDim.x * gridDim.y) {
    
        int s = i % spatial_dim;
        int n = (i / spatial_dim);
        
        const int label = static_cast<int>(bottom_label[i]);
        float z = (label == 0) * (1 - alpha) + (label >= 1) * alpha;
        weight[i] = 0.0;
        if (label != ignore_label) {
            int idx = 0;
            idx = n * spatial_dim * num_classes + label * spatial_dim + s;
            float onemp = 1. - bottom_prob[idx];
            float p = bottom_prob[idx];
            weight[i] = (-pow(onemp, gamma) +
                            gamma * pow(onemp, gamma - 1) * p * log(max(p, FLT_MIN))) * z;
        }
      
    }
}

template<typename Dtype>
__global__ void SoftmaxFocalLossGradientKernel(
    Dtype* bottom_prob, const Dtype* bottom_label, Dtype* bottom_data_diff,
    Dtype* weight,
    const int count, const int num_classes, 
    const int spatial_dim, const float ignore_label,
    const float alpha, const float gamma) {

  for (int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x * gridDim.y) {

      int s = i % spatial_dim;
      int c = (i / spatial_dim) % num_classes; 
      int n = i / (spatial_dim * num_classes);  

      int idx = 0;
      idx = n * spatial_dim + s;
      const int label = static_cast<int>(bottom_label[idx]);

      float c1 = (label != ignore_label) * 1.0;
      float c2 = (label == c) * 1.0;
      
      bottom_data_diff[i] = 0.0;
      bottom_data_diff[i] += c1 * weight[idx] * (c2 - bottom_prob[i]);
  }
}

template<typename Dtype>
inline void SoftmaxFocalLossBackward(const Tensor<gpu, 2, Dtype> &weights,
                                     const Tensor<gpu, 3, Dtype> &prob,
                                     const Tensor<gpu, 3, Dtype> &in_data_grad,
                                     const Tensor<gpu, 2, Dtype> &in_label,
                                     const int num_classes,
                                     const float ignore_label,
                                     const float alpha,
                                     const float gamma) {
  const Dtype* bottom_label = in_label.dptr_;
  Dtype* bottom_data_diff = in_data_grad.dptr_;
  Dtype* bottom_prob = prob.dptr_;
  Dtype* weight = weights.dptr_;
  
  
  const int spatial_dim = in_label.size(1);
  const int count_weight = in_label.shape_.Size();
  
  const int gridSize = (count_weight + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(kMaxGridDim, (gridSize + kMaxGridDim - 1) / kMaxGridDim);
  dim3 dimBlock(kMaxThreadsPerBlock);
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocalLoss Backward Weights");
  
  cudaStream_t stream_prob = Stream<gpu>::GetStream(weights.stream_);
  SoftmaxFocalGradientWeightKernel<Dtype><<<dimGrid, dimBlock, 0, stream_prob>>>(
     bottom_prob, bottom_label, weight, count_weight, num_classes, spatial_dim, ignore_label, alpha, gamma);

  const int count = in_data_grad.shape_.Size();
  dimGrid.y = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocalLoss Backward");

  cudaStream_t stream_grad = Stream<gpu>::GetStream(in_data_grad.stream_);
  SoftmaxFocalLossGradientKernel<Dtype><<<dimGrid, dimBlock, 0, stream_grad>>>(
      bottom_prob, bottom_label, bottom_data_diff, weight, count, num_classes, spatial_dim, 
      ignore_label, alpha, gamma);
}

}  // namespace cuda

template<typename Dtype>
inline void SoftmaxFocalLossForward(const Tensor<gpu, 3, Dtype> &prob,
                                    const Tensor<gpu, 3, Dtype> &data,
                                    const Tensor<gpu, 2, Dtype> &label,
                                    const int num_classes) {
  cuda::SoftmaxFocalLossForward(prob, data, label, num_classes);
}

template<typename Dtype>
inline void SoftmaxFocalLossBackward(const Tensor<gpu, 2, Dtype> &weights,
                                     const Tensor<gpu, 3, Dtype> &prob,
                                     const Tensor<gpu, 3, Dtype> &in_data_grad,
                                     const Tensor<gpu, 2, Dtype> &in_label,
                                     const int num_classes,
                                     const float ignore_label,
                                     const float alpha,
                                     const float gamma) {
  cuda::SoftmaxFocalLossBackward(weights, prob, in_data_grad, in_label, num_classes, ignore_label, alpha, gamma);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(SoftmaxFocalLossParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxFocalLossOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
