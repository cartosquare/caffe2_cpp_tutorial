#ifndef OPERATOR_SVM_LOSS_L1_OP_H
#define OPERATOR_SVM_LOSS_L1_OP_H

#include <caffe2/core/operator.h>

namespace caffe2 {

template <typename T, class Context>
class SVMLossL1Op final : public Operator<Context> {
 public:
  SVMLossL1Op(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override;

 protected:
  int axis_;
  Tensor<Context> correct_class_score_;
};

template<typename T, class Context> 
class SVMLossL1GradientOp final
    : public Operator<Context> {
 public:
  SVMLossL1GradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int axis_;
};

}  // namespace caffe2

#endif  // OPERATOR_SVM_LOSS_L1_OP_H