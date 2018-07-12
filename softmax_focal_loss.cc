#include "./softmax_focal_loss-inl.h"

namespace mshadow {
template<typename Dtype>
inline void SoftmaxFocalLossForward(const Tensor<cpu, 2, Dtype> &out_data,
                                    const Tensor<cpu, 3, Dtype> &prob,
                                    const Tensor<cpu, 3, Dtype> &in_data,
                                    const Tensor<cpu, 2, Dtype> &in_label,
                                    const int num_classes,
                                    const float ignore_label,
                                    const float alpha,
                                    const float gamma) {
  return;
}
template<typename Dtype>
inline void SoftmaxFocalLossBackward(const Tensor<cpu, 2, Dtype> &weight,
                                     const Tensor<cpu, 3, Dtype> &prob,
                                     const Tensor<cpu, 3, Dtype> &in_data_grad,
                                     const Tensor<cpu, 2, Dtype> &in_label,
                                     const int num_classes,
                                     const float ignore_label,
                                     const float alpha,
                                     const float gamma) {
  return;  
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(SoftmaxFocalLossParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxFocalLossOp<cpu, DType>(param);
  });
  return op;
}

Operator *SoftmaxFocalLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(SoftmaxFocalLossParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxFocalLoss, SoftmaxFocalLossProp)
.describe(R"code(SoftmaxFocalLoss)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_argument("label", "NDArray-or-Symbol", "Ground truth label.")
.add_arguments(SoftmaxFocalLossParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
