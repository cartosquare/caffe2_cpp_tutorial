#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>

using namespace caffe2;

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();

  Workspace workspace;

  // Setup Input
  const int N = 4;  // batch size
  const int D = 3;  // number of classes
  // ground truth
  auto* labels_blob = workspace.CreateBlob("labels");
  auto* labels_tensor = labels_blob->GetMutable<TensorCPU>();
  labels_tensor->Resize(N);
  int* labels_data = labels_tensor->template mutable_data<int>();
  for (int i = 0; i < N; ++i) {
    labels_data[i] = i % D;
  }
  // batch scores input
  auto* X_blob = workspace.CreateBlob("X");
  auto* X_tensor = X_blob->GetMutable<TensorCPU>();
  std::vector<caffe2::TIndex> shape = {N, D};
  X_tensor->Resize(shape);
  auto* X_mutable_data = X_tensor->template mutable_data<float>();
  std::vector<float> val = {8.4,   -3.9, 7.8,  3.3, 7.6, -2.7,
                            -2.85, 0.86, 0.28, 5.5, 4.7, 6.2};
  memcpy(X_mutable_data, val.data(), N * D * sizeof(float));

  std::cout << "====================== X data ===============\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      std::cout << X_mutable_data[i * D + j] << ", ";
    }
    std::cout << std::endl;
  }

  std::cout << "===================================";
  // create SVMLossL1 operator
  OperatorDef operator_def;
  operator_def.set_type("SVMLossL1");
  operator_def.add_input("X");
  operator_def.add_input("labels");
  operator_def.add_output("Y");
  std::unique_ptr<OperatorBase> op(CreateOperator(operator_def, &workspace));

  // run the operator
  bool status = op->Run();
  if (!status) {
    LOG(ERROR) << "run operator fail!";
    return -1;
  }

  // get output
  auto* Y_blob = workspace.CreateBlob("Y");
  auto* Y_tensor = Y_blob->GetMutable<TensorCPU>();
  float* batch_loss = Y_tensor->template mutable_data<float>();

  std::cout << "++++++++++++++++ Loss ++++++++++++++++++++++++\n";
  for (int i = 0; i < N; ++i) {
    std::cout << batch_loss[i] << ", ";
  }
  std::cout << std::endl;

  // We can verify the result by hand:
  /*
  ====================== X data ===============
  8.4, -3.9, 7.8,
  3.3, 7.6, -2.7,
  -2.85, 0.86, 0.28,
  5.5, 4.7, 6.2,

  ============== Correct class of each sample ================
  0, 1, 2, 1

  ================== svm losses ==============
  max(0, -3.9 - 8.4 + 1) + max(0, 7.8 - 8.4 + 1) = 0.4
  max(0, 3.3 - 7.6 + 1) + max(0, -2.7 - 7.6 + 1) = 0
  max(0, -2.85 - 0.28 + 1) + max(0, 0.86 - 0.28 + 1) = 1.58
  max(0, 4.7 - 5.5 + 1) + max(0, 6.2 - 5.5 + 1) = 1.9
  */
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}