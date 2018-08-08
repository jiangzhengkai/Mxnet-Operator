/*
softmax iou score matrix
author: ZhengKai Jiang
*/
#include "./iou_score_matrix-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../common/cuda_utils.h"
#include "mxnet_op.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                               \
for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
     i += blockDim.x * gridDim.x)

constexpr int CAFFE_CUDA_NUM_THREADS = 512;
constexpr int CAFFE_MAXIMUM_NUM_BLOCKS = 4096;

inline int CAFFE_GET_BLOCKS(const int N) {
  return std::min((N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS,
                  CAFFE_MAXIMUM_NUM_BLOCKS);
}

namespace mshadow {
namespace cuda {
template<typename Dtype>
__global__ void SpatialSoftmaxKernel(
    const Dtype* bottom_data,
    Dtype* top_data, 
    const int count,
    const int num_rois,
    const int spatial_num) {
  for (int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
       i < count; i += blockDim.x * gridDim.x * gridDim.y) {
    int s = i % num_rois;
    int n = i / num_rois;
    assert(n > num_rois);
    //LOG(INFO) << " n " << n << "num_rois" << num_rois;
    //printf( "num_rois %d\n",num_rois);
    // ith rois
    float xmin_i = 0.0;
    float ymin_i = 0.0;
    float xmax_i = 0.0;
    float ymax_i = 0.0;
    xmin_i = bottom_data[s * spatial_num];
    ymin_i = bottom_data[s * spatial_num + 1];
    xmax_i = bottom_data[s * spatial_num + 2];
    ymax_i = bottom_data[s * spatial_num + 3];
    // jth rois
    float xmin_j = 0.0;
    float ymin_j = 0.0;
    float xmax_j = 0.0;
    float ymax_j = 0.0;
    xmin_j = bottom_data[n * spatial_num];
    ymin_j = bottom_data[n * spatial_num + 1];
    xmax_j = bottom_data[n * spatial_num + 2];
    ymax_j = bottom_data[n * spatial_num + 3];
    // cal area_i and area_j
    float area_i = 0.0;
    float area_j = 0.0;
    area_i = (xmax_i - xmin_i + 1) * (ymax_i - ymin_i + 1);
    area_j = (xmax_j - xmin_j + 1) * (ymax_j - ymin_j + 1);
    // cal area i U j 
    float xx1 = 0.0;
    float yy1 = 0.0;
    float xx2 = 0.0;
    float yy2 = 0.0;
    float w = 0.0;
    float h = 0.0;
    xx1 = max(xmin_i, xmin_j);
    yy1 = max(ymin_i, ymin_j);
    xx2 = min(xmax_i, xmax_j);
    yy2 = min(ymax_i, ymax_j);
    w = max(0.0, xx2 - xx1 + 1);
    h = max(0.0, yy2 - yy1 + 1);
    // cal iou
    float inter = 0.0;
    float iou = 0.0;
    inter = w * h;
    iou = inter / (area_i + area_j - inter);
    top_data[i] = iou;
  }
}
template<typename Dtype>
inline void IouScoreMatrixForward(const Tensor<gpu, 2, Dtype> &rois,
                                  const Tensor<gpu, 2, Dtype> &matrix) {
  const Dtype* bottom_data = rois.dptr_;
  Dtype* top_matrix = matrix.dptr_;
 
  const int count = matrix.shape_.Size();
  const int num_rois = rois.size(0);
  const int spatial_num = rois.size(1);
  SpatialSoftmaxKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0>>>(
    bottom_data, top_matrix, count, num_rois, spatial_num);

}

}  // namespace cuda

template<typename Dtype>
inline void IouScoreMatrixForward(const Tensor<gpu, 2, Dtype> &rois,
                                  const Tensor<gpu, 2, Dtype> &matrix) {
  cuda::IouScoreMatrixForward(rois, matrix);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(IouScoreMatrixParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new IouScoreMatrixOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
