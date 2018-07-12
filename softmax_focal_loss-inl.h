#ifndef MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_
#define MXNET_OPERATOR_SOFTMAX_FOCAL_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace SoftmaxFocalLoss {
enum SoftmaxFocalLossOpInputs {kData, kLabel};
enum SoftmaxFocalLossOpOutputs {kOut, kProbability, kGradWeight};
enum SoftmaxFocalLossOpResource {kTempSpace};
}  // SoftmaxFocalLoss

struct SoftmaxFocalLossParam : public dmlc::Parameter<SoftmaxFocalLossParam> {
  float grad_scale;
  float ignore_label;
  int num_classes;
  float alpha;
  float gamma;
  DMLC_DECLARE_PARAMETER(SoftmaxFocalLossParam) {
    DMLC_DECLARE_FIELD(grad_scale).set_default(1.0f)
    .describe("Scales the gradient by a float factor.");
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("The instances whose `labels` == `ignore_label` will be ignored "
              "during backward, if `use_ignore` is set to ``true``).");
    DMLC_DECLARE_FIELD(num_classes).set_default(1)
    .describe("number of classes (excluding background)");    
    DMLC_DECLARE_FIELD(alpha).set_default(1.0f)
    .describe("Focal Loss's alpha hyper-parameter.");
    DMLC_DECLARE_FIELD(gamma).set_default(1.0f)
    .describe("Focal Loss's gamma hyper-parameter.");    
  }
};

template<typename xpu, typename DType>
class SoftmaxFocalLossOp : public Operator {
 public:
  explicit SoftmaxFocalLossOp(SoftmaxFocalLossParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, DType> data = in_data[SoftmaxFocalLoss::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> label = in_data[SoftmaxFocalLoss::kLabel].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> loss = out_data[SoftmaxFocalLoss::kOut].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> prob = out_data[SoftmaxFocalLoss::kProbability].get<xpu, 3, DType>(s);

    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(loss.CheckContiguous(), true);

    SoftmaxFocalLossForward(loss, prob, data, label, param_.num_classes, param_.ignore_label, param_.alpha, param_.gamma);

    int num_pos_label = 0;
    Tensor<cpu, 2, DType> workspace = ctx.requested[SoftmaxFocalLoss::kTempSpace].get_host_space_typed<2, DType>(label.shape_);
    Copy(workspace, label, label.stream_);
    for (index_t i = 0; i < workspace.size(0); ++i) {
      for (index_t j = 0; j < workspace.size(1); ++j) {
        if (static_cast<int>(workspace[i][j]) > 0) {
            ++num_pos_label;
        } 
      }
    }
    num_pos_label = num_pos_label == 0 ? 1 : num_pos_label;
    loss *= DType(1.0 / num_pos_label);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> label = in_data[SoftmaxFocalLoss::kLabel].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> in_data_grad = in_grad[SoftmaxFocalLoss::kData].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> prob = out_data[SoftmaxFocalLoss::kProbability].get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> gradweights = out_data[SoftmaxFocalLoss::kGradWeight].get<xpu, 2, DType>(s);
    CHECK_EQ(label.CheckContiguous(), true);
    CHECK_EQ(in_data_grad.CheckContiguous(), true);
    CHECK_EQ(prob.CheckContiguous(), true);

    if (kAddTo == req[SoftmaxFocalLoss::kData] || kWriteTo == req[SoftmaxFocalLoss::kData]) {
      if (kWriteTo == req[SoftmaxFocalLoss::kData]) {
        in_data_grad = 0.0f;
      }

      SoftmaxFocalLossBackward(gradweights, prob, in_data_grad, label, param_.num_classes, param_.ignore_label, param_.alpha, param_.gamma);

      int num_pos_label = 0;
      Tensor<cpu, 2, DType> workspace = ctx.requested[SoftmaxFocalLoss::kTempSpace].get_host_space_typed<2, DType>(label.shape_);
      Copy(workspace, label, label.stream_);
      for (index_t i = 0; i < workspace.size(0); ++i) {
        for (index_t j = 0; j < workspace.size(1); ++j) {
            if (static_cast<int>(workspace[i][j]) > 0) {
              ++num_pos_label;
            }
        }
      }
      num_pos_label = num_pos_label == 0 ? 1 : num_pos_label;
      in_data_grad *= DType(param_.grad_scale / num_pos_label);
    }
    
    if (kWriteTo == req[SoftmaxFocalLoss::kLabel]) {
      Tensor<xpu, 2, DType> in_label_grad = in_grad[SoftmaxFocalLoss::kLabel].get<xpu, 2, DType>(s);
      CHECK_EQ(in_label_grad.CheckContiguous(), true);
      in_label_grad = 0.0f;
    }
  }

 private:
  SoftmaxFocalLossParam param_;
};  // class SoftmaxFocalLossOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(SoftmaxFocalLossParam param, int dtype);
#if DMLC_USE_CXX11
class SoftmaxFocalLossProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"loss", "probability", "gradweights"};
  }
  int NumOutputs() const override {
          return 2;
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
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";

    // data: [batch_size, num_classes, spatial_dim]
    TShape dshape = in_shape->at(SoftmaxFocalLoss::kData);
    CHECK_EQ(dshape.ndim(), 3U) << "data should be a 3D tensor";

    // label: [batch_size, spatial_dim]
    TShape lshape = in_shape->at(SoftmaxFocalLoss::kLabel);
    CHECK_EQ(dshape.ndim(), 2U) << "label should be a 2D tensor";

    CHECK_EQ(dshape[0], lshape[0]) << "data  and label should be same in the first dim";
    CHECK_EQ(dshape[2], lshape[1]) << "data and label should be same in the last dim";
    out_shape->clear();
    out_shape->push_back(lshape);
    out_shape->push_back(dshape);
     out_shape->push_back(lshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    SoftmaxFocalLossProp* Softmax_focal_loss_sym = new SoftmaxFocalLossProp();
    Softmax_focal_loss_sym->param_ = this->param_;
    return Softmax_focal_loss_sym;
  }

  std::string TypeString() const override {
    return "SoftmaxFocalLoss";
  }

  virtual std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[SoftmaxFocalLoss::kData], in_data[SoftmaxFocalLoss::kLabel], out_data[SoftmaxFocalLoss::kProbability]};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SoftmaxFocalLossParam param_;
};  // class SoftmaxFocalLossProp
#endif

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_Softmax_FOCAL_LOSS_INL_H_
