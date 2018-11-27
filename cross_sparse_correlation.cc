/*!
 * Copyright (c) 2018 by Contributors
 * Copyright (c) 2018 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file cross_sparse_correlation-inl.h
 * \brief cross_sparse_correlation operator
 * \author Jiang ZhengKai
*/

#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include "./cross_sparse_correlation-inl.h"


namespace mshadow {
template<typename Dtype>
inline void CrossSparseCorrelationForward(const Tensor<cpu, 4, Dtype> &out,
                                          const Tensor<cpu, 4, Dtype> &data1,
                                          const Tensor<cpu, 4, Dtype> &data2) {
  // Not_Implemented
  return;
}

template<typename Dtype>
inline void CrossSparseCorrelationBackward(const Tensor<cpu, 4, Dtype> &grad_data1,
                                          const Tensor<cpu, 4, Dtype> &grad_data2,
                                          const Tensor<cpu, 4, Dtype> &grad_out,
                                          const Tensor<cpu, 4, Dtype> &data1,
                                          const Tensor<cpu, 4, Dtype> &data2) {
  // Not_Implemented
  return;
}
} //namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(CrossSparseCorrelationParam param, int dtype) {
  Operator* op = nullptr;
  MSHADOW_REAL_TYPE_SWITCH(dtype, Dtype, {
    op = new CrossSparseCorrelationOp<cpu, Dtype>(param);
  });
  return op;
}

Operator *CrossSparseCorrelationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(CrossSparseCorrelationParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_CrossSparseCorrelation, CrossSparseCorrelationProp)
.describe("cross sparse correlation for data1 and data2")
.add_argument("data1", "Symbol", "Input data1, a 4D Feature maps")
.add_argument("data2", "Symbol", "Input data2, a 4D Feature maps")
.add_arguments(CrossSparseCorrelationParam::__FIELDS__());


} // namespace op

} //namespace mxnet






