#include "caffe2/operator/svm_loss_l1_op.h"
#include <iostream>

namespace caffe2 {

template <>
bool SVMLossL1Op<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  const float* Xdata = X.data<float>();

  auto& labels = Input(1);
  const int* labels_data = labels.data<int>();

  auto* Y = Output(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  Y->Resize(N);
  float* Ydata = Y->mutable_data<float>();

  // select correct class score
  if (correct_class_score_.size() != N) {
    correct_class_score_.Resize(N);
  }
  float* correct_class_score_data = correct_class_score_.mutable_data<float>();
  math::Select<float, CPUContext>(N, D, Xdata, labels_data,
                                  correct_class_score_data, &context_);

  // for each sample, calculate svm loss:
  // sum(max(0, result X - correct_class_score(X) + 1))
  for (int i = 0; i < N; ++i) {
    Ydata[i] = 0.0;
    for (int j = 0; j < D; ++j) {
      if (j != labels_data[i]) {
        // sum wrong class margin
        Ydata[i] += std::max<float>(
            0.0, Xdata[i * D + j] - correct_class_score_data[i] + 1);
      }
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(SVMLossL1, SVMLossL1Op<float, CPUContext>);

OPERATOR_SCHEMA(SVMLossL1)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  The operator computes the SVM L1 loss for each layer in the batch of the given input.
  The input is a 2-D tensor(Tensor<float>) of size (batch_size * input_feature_dimensions) and
  a tensor of labels(ground truth). The output tensor has the loss for each sample.
  )DOC")
    .Arg("axis",
         "(int) default to 1; describes the axis of the inputs when coerced "
         "to 2D; defautls to one because the 0th axis most likely describes "
         "the batch size")
    .Input(0, "input",
           "The input tensor that's coerced into a 2D matrix of size (N*D) as "
           "descried above.")
    .Input(1, "labels", "Ground truth")
    .Output(0, "output", "The SVM L1 Loss output values for each sample.");

}  // namespace caffe2