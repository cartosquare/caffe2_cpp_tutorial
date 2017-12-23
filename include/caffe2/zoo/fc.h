#ifndef FC_H
#define FC_H

#include "caffe2/util/model.h"

namespace caffe2 {

class FCModel : public ModelUtil {
 public:
  FCModel(NetDef &initnet, NetDef &predictnet)
      : ModelUtil(initnet, predictnet) {}

  OperatorDef *AddFcOps(const std::string &input, const std::string &output,
                        int in_size, int out_size, bool relu = false,
                        float dropout = 0.5) {
    init.AddXavierFillOp({out_size, in_size}, output + "_w");
    predict.AddInput(output + "_w");
    init.AddConstantFillOp({out_size}, output + "_b");
    predict.AddInput(output + "_b");
    auto op = predict.AddFcOp(input, output + "_w", output + "_b", output);
    if (!relu) return op;
    return predict.AddReluOp(output, output);
    // return predict.AddDropoutOp(output, output, dropout);
  }

  std::string Add(const std::string input_name, int in_size, int out_size) {
    predict.SetName("FC");

    predict.AddInput(input_name);
    std::string layer =
        AddFcOps(input_name, "fc1", in_size, 100, true)->output(0);
    layer = AddFcOps(layer, "fc2", 100, out_size, false)->output(0);

    layer = predict.AddSoftmaxOp(layer, "prob")->output(0);
    predict.AddOutput(layer);
    // init.AddConstantFillOp({1}, input_name);

    return layer;
  }
};

}  // namespace caffe2

#endif  // FC_H
