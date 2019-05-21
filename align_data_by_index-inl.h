/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \author Jiang ZhengKai
*/
#ifndef MXNET_OPERATOR_CONTRIB_ALIGN_DATA_BY_INDEX_INL_H_
#define MXNET_OPERATOR_CONTRIB_ALIGN_DATA_BY_INDEX_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive
namespace AlignDataByIndex {
enum AlignDataByIndexOpInputs {kData, kIndex};
enum AlignDataByIndexOpOutputs {kOut};
} 

struct AlignDataByIndexParam : public dmlc::Parameter<AlignDataByIndexParam> {
  // Tshape
};

template<typename xpu, typename DType>
class AlignDataByIndexOp : public Operator {
 public:
  explicit AlignDataByIndexOp(AlignDataByIndexParam p) {
    this->param_ = p;
    }
  // forward
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    size_t expected_in = 2;
    size_t expected_out = 1;
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[AlignDataByIndex::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> index = in_data[AlignDataByIndex::kIndex].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> out = out_data[AlignDataByIndex::kOut].get<xpu, 4, DType>(s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(index.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);

    out = 0.0;
    AlignDataByIndexForward(out, data, index);
  }
  // backward
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {

    using namespace mshadow;
    size_t expected_in = 2;
    size_t expected_out = 1;
    CHECK_EQ(in_data.size(), expected_in);
    CHECK_EQ(out_data.size(), expected_out);

    CHECK_NE(req[AlignDataByIndex::kData], kWriteInplace) <<
      "AlignData: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[AlignDataByIndex::kIndex], kWriteInplace) <<
      "AlignData: Backward doesn't support kWriteInplace.";

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[AlignDataByIndex::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[AlignDataByIndex::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> index = in_data[AlignDataByIndex::kIndex].get<xpu, 4, DType>(s);

    Tensor<xpu, 4, DType> grad_data = in_grad[AlignDataByIndex::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_index = in_grad[AlignDataByIndex::kIndex].get<xpu, 4, DType>(s);


    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(index.CheckContiguous(), true);
    CHECK_EQ(grad_data.CheckContiguous(), true);
    CHECK_EQ(grad_index.CheckContiguous(), true);

    grad_data = 0.0;
    grad_index = 0.0;
    AlignDataByIndexBackward(grad_data, grad_index, data, index, grad_out);
  }

 private:
    AlignDataByIndexParam param_;
}; // class dynamic correlation op


// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(AlignDataByIndexParam param, int dtype);

#if DMLC_USE_CXX11
class AlignDataProp : public OperatorProperty {
 public:
    std::vector<std::string> ListArguments() const override {
    return {"data", "index"};
  }
  std::vector<std::string> ListOutputs() const override {
    return {"output"};
  }
  int NumOutputs() const override {
    return 1;
  }
  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }
  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, index]";
    TShape dshape = in_shape->at(AlignDataByIndex::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data1 should be a 4D tensor";

    TShape oshape = in_shape->at(AlignDataByIndex::kIndex);
    CHECK_EQ(oshape.ndim(), 4) << "offset should be a 4D tensor";

    out_shape->clear();
    out_shape->push_back(dshape));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new AlignDataByIndexProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_AlignDataByIndex";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[AlignDataByIndex::kOut], in_data[AlignDataByIndex::kData], in_data[AlignDataByIndex::kIndex]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
 private:
  AlignDataByIndexParam param_;
};
#endif
}  // namespace op
}  // namespace mxnet
#endif
