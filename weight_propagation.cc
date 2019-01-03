/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file psroi_pooling-inl.h
 * \brief psroi pooling operator and symbol
 * \author Jiang ZhengKai
*/
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include "./weight_propagation-inl.h"


namespace mshadow {

template<typename DType>
inline void WeightPropagationForward(const Tensor<cpu, 4, DType> &out,
                                    const Tensor<cpu, 4, DType> &data,
                                    const Tensor<cpu, 4, DType> &weights,
                                    const int weight_height_,
                                    const int weight_width_,
                                    const int hole_) {
  // NOT_IMPLEMENTED
  return;
}



template<typename DType>
inline void WeightPropagationBackwardAcc(const Tensor<cpu, 4, DType> &grad_data,
                                        const Tensor<cpu, 4, DType> &grad_weights,
                                        const Tensor<cpu, 4, DType> &grad_out,
                                        const Tensor<cpu, 4, DType> &data,
                                        const Tensor<cpu, 4, DType> &weights,
                                        const int weight_height_,
                                        const int weight_width_,
                                        const int hole_) {
  // NOT_IMPLEMENTED
  return;
}

} //namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(WeightPropagationParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new WeightPropagationOp<cpu, DType>(param);
  });
  return op;
}

Operator *WeightPropagationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(WeightPropagationParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_WeightPropagation, WeightPropagationProp)
.describe("weightpropagation to propagation weights with feature")
.add_argument("data", "Symbol", "Input data to the pooling operator, a 4D Feature maps")
.add_argument("weights", "Symbol", "Position sensitive weights, a 4D array ")
.add_arguments(WeightPropagationParam::__FIELDS__());

} //namespace op
} //namespace mxnet

