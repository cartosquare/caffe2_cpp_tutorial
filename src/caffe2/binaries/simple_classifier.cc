#include <cmath>
#include <random>

#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include "caffe2/util/plot.h"
#include "caffe2/util/window.h"

// #points
const int N = 100;
// dimensionality
const int D = 2;
// #class
const int K = 3;

namespace caffe2 {

void generate_data(double* X, double* y) {
  superWindow("Three Classes");
  auto name = "scatter";
  setWindowTitle(name, "scatter plots");
  moveWindow(name, 0, 0);
  resizeWindow(name, 600, 600);
  auto& figure = PlotUtil::Shared(name);

  std::vector<PlotUtil::Color> colors = {PlotUtil::Yellow(), PlotUtil::Red(),
                                         PlotUtil::Blue()};
  std::vector<std::string> classes = {"class1", "class2", "class3"};

  std::default_random_engine generator;
  std::normal_distribution<double> distribution(N, 1.0);
  for (int j = 0; j < K; ++j) {
    std::vector<std::pair<float, float>> data;
    for (int i = 0; i < N; ++i) {
      double r = static_cast<double>(i) / N;
      double t = 4 * j + i * 4.0 / N + distribution(generator) * 0.2;
      data.push_back({r * sin(t), r * cos(t)});
      X[(j * N + i) * D + 0] = r * sin(t);
      X[(j * N + i) * D + 1] = r * cos(t);
      y[j * N + i] = j;
    }
    figure.Get(classes[j]).Set(data, PlotUtil::Dots, colors[j]);
    data.clear();
  }

  figure.Show();
  cvWaitKey(0);
}

}  // namespace caffe2

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);

  caffe2::Workspace workspace;
  // Create X and y data storage
  auto* X_blob = workspace.CreateBlob("X");
  auto* X_tensor = X_blob->GetMutable<caffe2::TensorCPU>();
  X_tensor->Resize(N * K, D);
  double* X = X_tensor->template mutable_data<double>();
  auto* y_blob = workspace.CreateBlob("y");
  auto* y_tensor = y_blob->GetMutable<caffe2::TensorCPU>();
  y_tensor->Resize(N * K);
  double* y = y_tensor->template mutable_data<double>();

  caffe2::generate_data(X, y);

  

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}