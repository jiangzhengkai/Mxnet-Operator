/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file psroi_pooling-inl.h
 * \brief psroi pooling operator and symbol
 * \author Jiang ZhengKai
*/
#ifndef MXNET_OPERATOR_CONTRIB_WEIGHT_PROPAGATION_INL_H_
#define MXNET_OPERATOR_CONTRIB_WEIGHT_PROPAGATION_INL_H_

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
namespace weightpropagation {
enum WeightPropagationOpInputs {kData, kWeights};
enum WeightPropagationOpOutputs {kOut};
} // WeightPropogation

struct WeightPropagationParam : public dmlc::Parameter<WeightPropagationParam> {
  // Tshape
  int weight_width;
  int weight_height;
  int hole;
  DMLC_DECLARE_PARAMETER(WeightPropagationParam) {
    DMLC_DECLARE_FIELD(weight_height)
    .describe("weight propagation height");
    DMLC_DECLARE_FIELD(weight_width)
    .describe("weight propagation width");
    DMLC_DECLARE_FIELD(hole).set_default(1)
    .describe("hole params for larger recieve field");
  }
};

template<typename xpu, typename DType>
class WeightPropagationOp : public Operator {
 public:
  explicit WeightPropagationOp(WeightPropagationParam p) {
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
    CHECK_EQ(in_data[weightpropagation::kData].shape_[2], out_data[weightpropagation::kWeights].shape_[2]);
    CHECK_EQ(in_data[weightpropagation::kData].shape_[3], out_data[weightpropagation::kWeights].shape_[3]);
    CHECK_EQ(in_data[weightpropagation::kData].shape_, out_data[weightpropagation::kOut].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[weightpropagation::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> weights = in_data[weightpropagation::kWeights].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[weightpropagation::kOut].get<xpu, 4, DType>(s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(weights.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);

    out = -FLT_MAX;

    WeightPropagationForward(out, data, weights, param_.weight_height, param_.weight_width, param_.hole);

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
    CHECK_EQ(out_grad[weightpropagation::kOut].shape_, in_data[weightpropagation::kData].shape_);

    CHECK_NE(req[weightpropagation::kData], kWriteInplace) <<
      "WeightPropagation: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[weightpropagation::kWeights], kWriteInplace) <<
      "WeightPropagation: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> grad_out = out_grad[weightpropagation::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[weightpropagation::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> weights = in_data[weightpropagation::kWeights].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_data = in_grad[weightpropagation::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> grad_weights = in_grad[weightpropagation::kWeights].get<xpu, 4, DType>(s);

    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(grad_data.CheckContiguous(), true);
    CHECK_EQ(grad_weights.CheckContiguous(), true);
    grad_data = 0;
    grad_weights = 0;

    WeightPropagationBackwardAcc(grad_data, grad_weights, grad_out, data, weights, param_.weight_width, param_.weight_height, param_.hole);
  }

 private:
    WeightPropagationParam param_;
}; // class weightpropagationop


// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(WeightPropagationParam param, int dtype);

#if DMLC_USE_CXX11
class WeightPropagationProp : public OperatorProperty {
 public:
    std::vector<std::string> ListArguments() const override {
    return {"data", "weights"};
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
    CHECK_EQ(in_shape->size(), 2) << "Input:[data, weights]";

    // data: [batch_size, c, h, w]
    TShape dshape = in_shape->at(weightpropagation::kData);
    CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";

    // weights [batch_size, k*k, h, w]
    TShape wshape = in_shape->at(weightpropagation::kWeights);
    CHECK_EQ(wshape.ndim(), 4) << "bbox should be a 4D tensor of shape [batch, k*k, h, w]";

    // out: [batch_size, c, h , w]
    out_shape->clear();
    out_shape->push_back(dshape);
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
    auto ptr = new WeightPropagationProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_WeightPropagation";
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[weightpropagation::kOut], in_data[weightpropagation::kData], in_data[weightpropagation::kWeights]};
  }


  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
 private:
  WeightPropagationParam param_;
};  // class WeightPropogation
#endif
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_WEIGHTPROPAGATION_INL_H_


