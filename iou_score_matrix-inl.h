/*
author zhengkai.jiang
*/
#ifndef MXNET_OPERATOR_IOU_SCORE_MATRIX_INL_H_
#define MXNET_OPERATOR_IOU_SCORE_MATRIX_INL_H_

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
namespace IouScoreMatrix {
enum IouScoreMatrixOpInputs {kData};
enum IouScoreMatrixOpOutputs {kOut};
}  // iou_score_matrix


struct IouScoreMatrixParam : public dmlc::Parameter<IouScoreMatrixParam> {
  DMLC_DECLARE_PARAMETER(IouScoreMatrixParam) {     
  }
};

template<typename xpu, typename DType>
class IouScoreMatrixOp : public Operator {
 public:
  explicit IouScoreMatrixOp(IouScoreMatrixParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 2, DType> rois = in_data[IouScoreMatrix::kData].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> matrix = out_data[IouScoreMatrix::kOut].get<xpu, 2, DType>(s);


    CHECK_EQ(rois.CheckContiguous(), true);
    CHECK_EQ(matrix.CheckContiguous(), true);

    IouScoreMatrixForward(rois, matrix);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 1);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> rois_grad = in_grad[IouScoreMatrix::kData].get<xpu, 2, DType>(s);
    rois_grad = 0.0;
  }

 private:
  IouScoreMatrixParam param_;
};  // class IouScoreMatrixOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(IouScoreMatrixParam param, int dtype);
#if DMLC_USE_CXX11
class IouScoreMatrixProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"matrix"};
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
    CHECK_EQ(in_shape->size(), 1U) << "Input:[data]";

    // data: [num_rois, 5]
    TShape dshape = in_shape->at(IouScoreMatrix::kData);
    CHECK_EQ(dshape.ndim(), 2U) << "data should be a 2D tensor";
    
    out_shape->clear();
    out_shape->push_back(Shape2(dshape[0], dshape[0]));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {

    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "Input must have specified type";
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    IouScoreMatrixProp* Iou_Score_Matrix_sym = new IouScoreMatrixProp();
    Iou_Score_Matrix_sym->param_ = this->param_;
    return Iou_Score_Matrix_sym;
  }

  std::string TypeString() const override {
    return "IouScoreMatrix";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  IouScoreMatrixParam param_;
};  // class IouScoreMatrixProp
#endif

} // namespace op

} // namespace mxnet

#endif  // MXNET_OPERATOR_IOU_SCORE_MATRIX_INL_H_