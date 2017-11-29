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

template <>
bool SVMLossL1GradientOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);  // predict scores
  const float* X_data = X.template data<float>();
  int N, D;
  const auto canonical_axis = X.canonical_axis_index(axis_);
  N = X.size_to_dim(canonical_axis);  // batch size
  D = X.size_from_dim(canonical_axis);

  auto& Y = Input(1);  // ground truth labels
  const int* Y_data = Y.template data<int>();
  // check label dimension
  if (Y.ndim() == canonical_axis) {
    CAFFE_ENFORCE_EQ(Y.size(), N);
  } else {
    CAFFE_ENFORCE_EQ(Y.size_to_dim(canonical_axis), N);
    CAFFE_ENFORCE_EQ(Y.size_from_dim(canonical_axis), 1);
  }

  auto& d_avg_loss = Input(2);  // avg_loss grad(gradient from top layer)
  const float* d_avg_loss_data = d_avg_loss.template data<float>();
  CAFFE_ENFORCE(d_avg_loss.ndim() == 1);
  CAFFE_ENFORCE(d_avg_loss.dim32(0) == N);

  auto* dX = Output(0);  // gradient r.s.t predict scores
  dX->ResizeLike(X);
  float* dX_data = dX->template mutable_data<float>();

  // calculate gradient for each sample
  for (int i = 0; i < N; ++i) {
    // for the class(i) that is not target class(j), the gradient is:
    // I(max(0, p_i - p_j + \Delta))
    // for class j, the gradient is:
    // \sum_i I(max(0, p_i - p_j + \Delta))
    int cnt = 0;
    float target_score = X_data[i * D + Y_data[i]];
    for (int j = 0; j < D; ++j) {
      if (j != Y_data[i]) {
        float loss = X_data[i * D + j] - target_score + 1;
        if (loss > 0) {
          dX_data[i * D + j] = 1 * d_avg_loss_data[i];
          ++cnt;
        } else {
          dX_data[i * D + j] = 0;
        }
      }
    }
    dX_data[i * D + Y_data[i]] = -1 * cnt * d_avg_loss_data[i];
  }

  return true;
}

REGISTER_CPU_OPERATOR(SVMLossL1, SVMLossL1Op<float, CPUContext>);
REGISTER_CPU_OPERATOR(SVMLossL1Gradient,
                      SVMLossL1GradientOp<float, CPUContext>);

// Input: X, T(labels); Output: L(loss)
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

// Input: X, T, L, dY; Output: dX
OPERATOR_SCHEMA(SVMLossL1Gradient).NumInputs(3).NumOutputs(1);

class GetSVMLossL1Gradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);
    return SingleGradientDef("SVMLossL1Gradient", "",
                             vector<string>{I(0), I(1), GO(0)},
                             vector<string>{GI(0)});
  }
};
// register gradient for the operator
REGISTER_GRADIENT(SVMLossL1, GetSVMLossL1Gradient);
}  // namespace caffe2