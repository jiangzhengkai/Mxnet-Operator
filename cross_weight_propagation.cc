/*!
 * Copyright (c) 2018 by Contributors
 * Copyright (c) 2018 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file cross_weight_propagation-inl.h
 * \brief cross_weight_propagation operator
 * \author Jiang ZhengKai
*/

#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include "./cross_weight_propagation-inl.h"

namespace mshadow {

template<typename Dtype>
inline void CrossWeightPropagationForward(const Tensor<cpu, 4, Dtype> &out,
                                          const Tensor<cpu, 4, Dtype> &data,
                                          const Tensor<cpu, 4, Dtype> &weight) {

  // Not_Implemented
  return;
}

template<typename Dtype>
inline void CrossWeightPropagationBackward(const Tensor<cpu, 4, Dtype> &grad_data,
                                           const Tensor<cpu, 4, Dtype> &grad_weight,
                                           const Tensor<cpu, 4, Dtype> &grad_out,
                                           const Tensor<cpu, 4, Dtype> &data,
                                           const Tensor<cpu, 4, Dtype> &weight) {

  // Not_Implemented
  return;
}

} //namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(CrossWeightPropagationParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new CrossWeightPropagationOp<cpu, DType>(param);
  });
  return op;
}

Operator *CrossWeightPropagationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(CrossWeightPropagationParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_CrossWeightPropagation, CrossWeightPropagationProp)
.describe("cross weight propagation for data and weight")
.add_argument("data", "Symbol", "Input data, a 4D Feature maps")
.add_argument("weight", "Symbol", "Input weight, a 4D Feature maps")
.add_arguments(CrossWeightPropagationParam::__FIELDS__());
} // namespace op
} //namespace mxnet
