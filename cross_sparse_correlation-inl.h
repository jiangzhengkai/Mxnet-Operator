/*!
 * Copyright (c) 2018 by Contributors
 * Copyright (c) 2018 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file cross_sparse_correlation-inl.h
 * \brief cross_sparse_correlation operator
 * \author Jiang ZhengKai
*/
#ifndef MXNET_OPERATOR_CONTRIB_CROSS_SPARSE_CORRELATION_INL_H_
#define MXNET_OPERATOR_CONTRIB_CROSS_SPARSE_CORRELATION_INL_H_

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
namespace CrossSparseCorrelation {
enum CrossSparseCorrelationOpInputs {kData1, kData2};
enum CrossSparseCorrelationOpOutputs {kOut};
} // CrossSparseCorrelation

struct CrossSparseCorrelationParam : public dmlc::Parameter<CrossSparseCorrelationParam> {
  // Tshape
  DMLC_DECLARE_PARAMETER(CrossSparseCorrelationParam) {
  };
};

template<typename xpu, typename Dtype>
class CrossSparseCorrelationOp : public Operator {
 public:
  explicit CrossSparseCorrelationOp(CrossSparseCorrelationParam p) {
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
    CHECK_EQ(in_data[CrossSparseCorrelation::kData1].shape_, in_data[CrossSparseCorrelation::kData2].shape_);
    CHECK_EQ(in_data[CrossSparseCorrelation::kData1].shape_[0], out_data[CrossSparseCorrelation::kOut].shape_[0]);
    CHECK_EQ(in_data[CrossSparseCorrelation::kData1].shape_[2], out_data[CrossSparseCorrelation::kOut].shape_[2]);
    CHECK_EQ(in_data[CrossSparseCorrelation::kData1].shape_[3], out_data[CrossSparseCorrelation::kOut].shape_[3]);
    size_t expected_channel = in_data[CrossSparseCorrelation::kData1].shape_[2] + in_data[CrossSparseCorrelation::kData1].shape_[3] - 1;
    // output in first axis is expected to be h + w - 1
    CHECK_EQ(out_data[CrossSparseCorrelation::kOut].shape_[1], expected_channel);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> data1 = in_data[CrossSparseCorrelation::kData1].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> data2 = in_data[CrossSparseCorrelation::kData2].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> out = out_data[CrossSparseCorrelation::kOut].get<xpu, 4, Dtype>(s);
    CHECK_EQ(data1.CheckContiguous(), true);
    CHECK_EQ(data2.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    out = 0.0;
    CrossSparseCorrelationForward(out, data1, data2);
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
    CHECK_EQ(in_data[CrossSparseCorrelation::kData1].shape_[0], out_data[CrossSparseCorrelation::kOut].shape_[0]);
    CHECK_EQ(in_data[CrossSparseCorrelation::kData1].shape_[2], out_data[CrossSparseCorrelation::kOut].shape_[2]);
    CHECK_EQ(in_data[CrossSparseCorrelation::kData1].shape_[3], out_data[CrossSparseCorrelation::kOut].shape_[3]);

    CHECK_NE(req[CrossSparseCorrelation::kData1], kWriteInplace) <<
      "CrossSparseCorrelation: Backward doesn't support kWriteInplace.";
    CHECK_NE(req[CrossSparseCorrelation::kData2], kWriteInplace) <<
      "CrossSparseCorrelation: Backward doesn't support kWriteInplace.";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4, Dtype> grad_out = out_grad[CrossSparseCorrelation::kOut].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> data1 = in_data[CrossSparseCorrelation::kData1].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> data2 = in_data[CrossSparseCorrelation::kData2].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> grad_data1 = in_grad[CrossSparseCorrelation::kData1].get<xpu, 4, Dtype>(s);
    Tensor<xpu, 4, Dtype> grad_data2 = in_grad[CrossSparseCorrelation::kData2].get<xpu, 4, Dtype>(s);

    CHECK_EQ(grad_out.CheckContiguous(), true);
    CHECK_EQ(data1.CheckContiguous(), true);
    CHECK_EQ(data2.CheckContiguous(), true);
    CHECK_EQ(grad_data1.CheckContiguous(), true);
    CHECK_EQ(grad_data2.CheckContiguous(), true);
    grad_data1 = 0.0;
    grad_data2 = 0.0;
    CrossSparseCorrelationBackward(grad_data1, grad_data2, grad_out, data1, data2);
  }

 private:
    CrossSparseCorrelationParam param_;
}; // class CrossSparseCorrelationop



// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(CrossSparseCorrelationParam param, int Dtype);

#if DMLC_USE_CXX11
class CrossSparseCorrelationProp : public OperatorProperty {
 public:
    std::vector<std::string> ListArguments() const override {
      return {"data1", "data2"};
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
      CHECK_EQ(in_shape->size(), 2) << "Input:[data1, data2]";
      // data1: [batch_size, c, h, w]
      TShape dshape = in_shape->at(CrossSparseCorrelation::kData1);
      CHECK_EQ(dshape.ndim(), 4) << "data1 should be a 4D tensor";
      // data2 [batch_size, c, h, w]
      TShape wshape = in_shape->at(CrossSparseCorrelation::kData2);
      CHECK_EQ(wshape.ndim(), 4) << "data2 should be a 4D tensor";
      // channel_num = h + w - 1
      size_t channel_num = dshape[2] + dshape[3] - 1;
      // out: [batch_size, h + w - 1, h , w]
      out_shape->clear();
      out_shape->push_back(Shape4(dshape[0], channel_num, dshape[2], dshape[3]));
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
      auto ptr = new CrossSparseCorrelationProp();
      ptr->param_ = param_;
      return ptr;
    }
    std::string TypeString() const override {
      return "_contrib_CrossSparseCorrelation";
    }
    // decalre dependency and inplace optimization options
    std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
      return {out_grad[CrossSparseCorrelation::kOut], in_data[CrossSparseCorrelation::kData1], in_data[CrossSparseCorrelation::kData2]};
    }
    Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
    }
    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;
  private:
    CrossSparseCorrelationParam param_;
};  // class CrossSparseCorrelation
#endif
} // namespace op
} //namespace mxnet
#endif