/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file dynamic_correlation-inl.h
 * \brief weight generate operator and symbol
 * \author Jiang ZhengKai
*/
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include "./align_data_by_index-inl.h"


namespace mshadow {

template<typename DType>
inline void AlignDataByIndexForward(const Tensor<cpu, 4, DType> &out,
                                    const Tensor<cpu, 4, DType> &data,
                                    const Tensor<cpu, 4, DType> &index) {
  // NOT_IMPLEMENTED
  return;
}


template<typename DType>
inline void AlignDataByIndexBackward(const Tensor<cpu, 4, DType> &grad_data,
                              const Tensor<cpu, 4, DType> &grad_index,
                              const Tensor<cpu, 4, DType> &data,
                              const Tensor<cpu, 4, DType> &index,
                              const Tensor<cpu, 4, DType> &grad_out) {
  // NOT_IMPLEMENTED
  return;
}


} //namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(AlignDataByIndexParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new AlignDataByIndexOp<cpu, DType>(param);
  });
  return op;
}

Operator *AlignDataByIndexProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(AlignDataByIndexParam);
MXNET_REGISTER_OP_PROPERTY(_contrib_AlignDataByIndex, AlignDataByIndexProp)
.describe("weight generate according to the data and data_ref")
.add_argument("data", "Symbol", "data, a 4D Feature maps")
.add_argument("index", "Symbol", "index, a 4D array ")
.add_arguments(AlignDataByIndexParam::__FIELDS__());

} //namespace op
} //namespace mxnet
