#include "./iou_score_matrix-inl.h"

namespace mshadow {
template<typename Dtype>
inline void IouScoreMatrixForward(const Tensor<cpu, 2, Dtype> &rois,
                                  const Tensor<cpu, 2, Dtype> &matrix) {
  return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(IouScoreMatrixParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new IouScoreMatrixOp<cpu, DType>(param);
  });
  return op;
}

Operator *IouScoreMatrixProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(IouScoreMatrixParam);

MXNET_REGISTER_OP_PROPERTY(IouScoreMatrix, IouScoreMatrixProp)
.describe(R"code(IouScoreMatrix)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(IouScoreMatrixParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
