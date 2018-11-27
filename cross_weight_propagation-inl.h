/*!
 * Copyright (c) 2018 by Contributors
 * Copyright (c) 2018 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file cross_weight_propagation-inl.h
 * \brief cross_weight_propagation operator
 * \author Jiang ZhengKai
*/
#ifndef MXNET_OPERATOR_CONTRIB_CROSS_WEIGHT_PROPAGATION_INL_H_
#define MXNET_OPERATOR_CONTRIB_CROSS_WEIGHT_PROPAGATION_INL_H_

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
namespace CrossWeightPropagation {
enum CrossWeightPropagationOpInputs {kData, kWeight};
enum CrossWeightPropagationOpOutputs {kOut};
}

struct CrossWeightPropagationParam : public dmlc::Parameter<CrossWeightPropagationParam> {
  // Tshape
  DMLC_DECLARE_PARAMETER(CrossWeightPropagationParam) {
  };

};

template<typename xpu, typename Dtype>
class CrossWeightPropagationOp : public Operator {
 public:
  explicit CrossWeightPropagationOp(CrossWeightPropagationParam p) {
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
    size_t expected_channel = in_data[CrossWeightPropagation::kData].shape_[2] + in_data[CrossWeightPropagation::kData].shape_[3] - 1;
    CHECK_EQ(in_data[CrossWeightPropagation::kWeight].shape_[1], expected_channel);
    // output in first axis is expected to be h + w - 1
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> data = in_data[CrossWeightPropagation::kData].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> weight = in_data[CrossWeightPropagation::kWeight].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> out = out_data[CrossWeightPropagation::kOut].get<xpu, 4, Dtype>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(weight.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    out = 0.0;
    CrossWeightPropagationForward(out, data, weight);
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
    CHECK_NE(req[CrossWeightPropagation::kData], kWriteInplace) <<
      "CrossWeightPropagation: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[CrossWeightPropagation::kWeight], kWriteInplace) <<
      "CrossWeightPropagation: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> grad_out = out_grad[CrossWeightPropagation::kOut].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> data = in_data[CrossWeightPropagation::kData].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> weight = in_data[CrossWeightPropagation::kWeight].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> grad_data = in_grad[CrossWeightPropagation::kData].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> grad_weight = in_grad[CrossWeightPropagation::kWeight].get<xpu, 4, Dtype>(s);
    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(weight.CheckContiguous(), true);
    CHECK_EQ(grad_data.CheckContiguous(), true);
    CHECK_EQ(grad_weight.CheckContiguous(), true);
    grad_data = 0.0;
    grad_weight = 0.0;
    CrossWeightPropagationBackward(grad_data, grad_weight, grad_out, data, weight);
  }

 private:
    CrossWeightPropagationParam param_;
}; // class CrossSparseCorrelationop



// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CrossWeightPropagationParam param, int Dtype);
#if DMLC_USE_CXX11
class CrossWeightPropagationProp : public OperatorProperty {
  public:
    std::vector<std::string> ListArguments() const override {
      return {"data", "weight"};
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
      CHECK_EQ(in_shape->size(), 2) << "Input:[data, weight]";
      // data1: [batch_size, c, h, w]
      TShape dshape = in_shape->at(CrossWeightPropagation::kData);
      CHECK_EQ(dshape.ndim(), 4) << "data should be a 4D tensor";
      // data2 [batch_size, c, h, w]
      TShape wshape = in_shape->at(CrossWeightPropagation::kWeight);
      CHECK_EQ(wshape.ndim(), 4) << "weight should be a 4D tensor";
      CHECK_EQ(wshape[2], dshape[2]) << "data and weight should be the same";
      CHECK_EQ(wshape[3], dshape[3]) << "data and weight should be the same";
      // out: [batch_size, h + w - 1, h , w]
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
      auto ptr = new CrossWeightPropagationProp();
      ptr->param_ = param_;
      return ptr;
    }
    std::string TypeString() const override {
      return "_contrib_CrossWeightPropagation";
    }
    // decalre dependency and inplace optimization options
    std::vector<int> DeclareBackwardDependency(const std::vector<int> &out_grad,
                                               const std::vector<int> &in_data,
                                               const std::vector<int> &out_data) const override {
      return {out_grad[CrossWeightPropagation::kOut], in_data[CrossWeightPropagation::kData], in_data[CrossWeightPropagation::kWeight]};
    }
    Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
    }
    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                               std::vector<int> *in_type) const override;
  private:
    CrossWeightPropagationParam param_;
};  // class CrossWeightPropagation
#endif
} // namespace op
} //namespace mxnet
#endif
