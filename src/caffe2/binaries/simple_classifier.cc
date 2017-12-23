#include <cmath>
#include <fstream>
#include <random>

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/predictor.h>
#include "caffe2/util/blob.h"
#include "caffe2/util/plot.h"
#include "caffe2/util/window.h"
#include "caffe2/zoo/fc.h"

// #points
const int N = 100;
// dimensionality
const int D = 2;
// #class
const int K = 3;

std::string optimizer = "sgd";
const double lr = 1;
const int iters = 10000;

const std::string db_type = "leveldb";
const std::string db_path = "./train.db";
const std::string model_output = "./reg_fc";

namespace caffe2 {

void create_db(float* X, int* y) {
  std::unique_ptr<db::DB> db(db::CreateDB(db_type, db_path, db::NEW));
  std::unique_ptr<db::Transaction> transaction(db->NewTransaction());
  std::string serialized_protos;
  for (int j = 0; j < K; ++j) {
    for (int i = 0; i < N; ++i) {
      TensorProtos protos;
      TensorProto* data = protos.add_protos();
      TensorProto* label = protos.add_protos();

      data->set_data_type(TensorProto::FLOAT);
      data->add_dims(D);
      label->set_data_type(TensorProto::INT32);

      data->add_float_data(X[(j * N + i) * D + 0]);
      data->add_float_data(X[(j * N + i) * D + 1]);

      label->add_int32_data(y[j * N + i]);

      protos.SerializeToString(&serialized_protos);
      std::ostringstream oss;
      oss << j * N + i;
      transaction->Put(oss.str(), serialized_protos);
    }
  }
}

void generate_data(float* X, int* y) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(N, 1.0);
  for (int j = 0; j < K; ++j) {
    for (int i = 0; i < N; ++i) {
      float r = static_cast<float>(i) / N;
      float t = 4 * j + i * 4.0 / N + distribution(generator) * 0.2;
      X[(j * N + i) * D + 0] = r * sin(t);
      X[(j * N + i) * D + 1] = r * cos(t);
      y[j * N + i] = j;
    }
  }
}

void draw_data(float* X, int* y, int* y_pred) {
  superWindow("Visual Data and Classify Results");
  std::vector<PlotUtil::Color> colors = {PlotUtil::Yellow(), PlotUtil::Red(),
                                         PlotUtil::Blue()};
  std::vector<std::string> classes = {"class1", "class2", "class3"};

  {
    auto name = "ground truth";
    setWindowTitle(name, "ground truth plot");
    moveWindow(name, 0, 0);
    resizeWindow(name, 600, 600);
    auto& figure = PlotUtil::Shared(name);

    for (int j = 0; j < K; ++j) {
      std::vector<std::pair<float, float>> data;

      for (int i = 0; i < N; ++i) {
        data.push_back({X[(j * N + i) * D + 0], X[(j * N + i) * D + 1]});
      }
      figure.Get(classes[j]).Set(data, PlotUtil::Dots, colors[j]);
      data.clear();
    }

    figure.Show();
  }

  {
    auto name = "predict";
    setWindowTitle(name, "predict plot");
    moveWindow(name, 600, 0);
    resizeWindow(name, 600, 600);
    auto& figure = PlotUtil::Shared(name);

    for (int j = 0; j < K; ++j) {
      std::vector<std::pair<float, float>> data;

      for (int i = 0; i < K * N; ++i) {
        if (y_pred[i] == j) {
          data.push_back({X[i * D + 0], X[i * D + 1]});
        }
      }
      figure.Get(classes[j]).Set(data, PlotUtil::Dots, colors[j]);
      data.clear();
    }
    figure.Show();
  }
  cvWaitKey(0);
}

void create_net(FCModel& fc_model, bool deploy = false) {
  if (!deploy) {
    fc_model.AddDatabaseOps("db", "data", db_path, db_type, K * N);
  }

  std::string output_layer = fc_model.Add("data", D, K);
  if (!deploy) {
    fc_model.AddTrainOps(output_layer, lr, optimizer);
  }

  std::cout << fc_model.Short() << std::endl;
}

}  // namespace caffe2

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int>> pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i) result.push_back(pairs[i].second);
  return result;
}

int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();

  caffe2::Workspace workspace;
  /*
  create linear net
  */
  caffe2::NetDef init_net, predict_net;
  caffe2::FCModel fc_model(init_net, predict_net);
  caffe2::create_net(fc_model);

  /*
  create data and label
  */
  float* X = new float[N * K * D];
  int* y = new int[N * K];
  caffe2::generate_data(X, y);

  /*
  save db
  */
  std::ifstream ifs(db_path.c_str());
  if (!ifs.good()) {
    std::cout << "create training db ...\n";
    caffe2::create_db(X, y);
  }

  /*
  train model
  */
  CAFFE_ENFORCE(workspace.RunNetOnce(init_net));
  CAFFE_ENFORCE(workspace.CreateNet(predict_net));
  float accuracy = 0.f;
  float loss = 0.f;
  for (int i = 1; i <= iters; ++i) {
    CAFFE_ENFORCE(workspace.RunNet(predict_net.name()));

    if (i % 1000 == 0) {
      accuracy = caffe2::BlobUtil(*workspace.GetBlob("accuracy"))
                     .Get()
                     .data<float>()[0];
      loss =
          caffe2::BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
      auto iter =
          caffe2::BlobUtil(*workspace.GetBlob("iter")).Get().data<int64_t>()[0];
      auto lr =
          caffe2::BlobUtil(*workspace.GetBlob("lr")).Get().data<float>()[0];
      std::cout << "step: " << iter << "  rate: " << lr << "  loss: " << loss
                << "  accuracy: " << accuracy << std::endl;
    }
  }

  /*
  deploy model
  */
  caffe2::NetDef test_init_net, test_predict_net;
  caffe2::FCModel test_model(test_init_net, test_predict_net);
  std::cout << "create test net ...\n";
  caffe2::create_net(test_model, true);

  caffe2::NetDef deploy_init_net;  // the final initialization model
  caffe2::ModelUtil deploy(deploy_init_net, test_predict_net,
                           "deploy_" + test_model.predict.net.name());
  std::cout << "copy deploy net ...\n";
  test_model.CopyDeploy(deploy, workspace);
  std::cout << "Deploy model dump\n";
  std::cout << deploy.Short() << std::endl;
  deploy.Write(model_output);

  /*
  predict
  */
  int* y_preds = new int[N * K];
  {
    caffe2::Workspace workspace;
    caffe2::NetDef init_net, predict_net;
    caffe2::ModelUtil deploy_model(init_net, predict_net);
    deploy_model.Read(model_output);

    // feed input data to our new workspace
    workspace.CreateBlob("data");
    caffe2::Blob* input_blob = workspace.GetBlob("data");
    caffe2::TensorCPU* input_tensor =
        input_blob->GetMutable<caffe2::TensorCPU>();
    std::vector<int> dims = {N * K, D};
    input_tensor->Resize(dims);
    float* input_data = input_tensor->mutable_data<float>();
    std::copy(X, X + N * D * K, input_data);

    // run predictor
    caffe2::Predictor predictor(init_net, predict_net, &workspace);
    caffe2::Predictor::TensorVector inputs, outputs;
    CAFFE_ENFORCE(predictor.run(inputs, &outputs));

    // get predict label
    const float* output_data = outputs[0]->data<float>();
    for (int i = 0; i < N * K; ++i) {
      std::vector<float> probs(output_data + i * K, output_data + (i + 1) * K);
      std::vector<int> maxs = Argmax(probs, 1);
      y_preds[i] = maxs[0];
    }
  }

  /*
  visualize
  */
  caffe2::draw_data(X, y, y_preds);
  delete[] X;
  delete[] y;
  delete[] y_preds;

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}