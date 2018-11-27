/*
 * Example tool training feed forward neural network on mnist data
 *
 */


#include "neural/data/mnist_dataloader.h"
#include "neural/layers/linear_layer.h"
#include "neural/layers/relu_layer.h"
#include "neural/layers/softmax_layer.h"
#include "neural/loss/cross_entropy_loss.h"
#include "neural/loss/mean_squared_error_loss.h"

#include <math.h>

#include <glog/logging.h>

using namespace neural;
using namespace std;

float CalcAverage(const vector<float>& vals)
{
    float sum = 0.0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        sum += vals.at(i);
    }
    return sum / ((float)vals.size());
}

float CalcAccuracy(
    LinearLayer& a_firstLayer,
    ReLULayer& a_secondLayer,
    LinearLayer& a_thirdLayer,
    SoftmaxLayer& a_sofmaxLayer,
    MNISTDataloader& a_testDataloader)
{
    float l_numCorrect = 0.0;
    float l_numTotal = 0.0;
    size_t l_numIters = a_testDataloader.DataLength();
    for (size_t i = 0; i < l_numIters; ++i)
    {
        TMutableTensorPtr l_input, l_output;
        a_testDataloader.DataAt(i, l_input, l_output);
        size_t l_targetOutput = l_output->MaxIdx();

        // Forward pass
        TTensorPtr l_output0 = a_firstLayer.Forward(l_input);
        TTensorPtr l_output1 = a_secondLayer.Forward(l_output0);
        TTensorPtr l_output2 = a_thirdLayer.Forward(l_output1);
        TTensorPtr probs = a_sofmaxLayer.Forward(l_output2);
        size_t l_predVal = probs->MaxIdx();

        if (l_predVal == l_targetOutput)
        {
            l_numCorrect += 1.0;
        }

        if (i % 1000 == 0)
        {
            LOG(INFO) << "Processing test set... " << i << endl;
        }
        l_numTotal += 1.0;
    }

    float l_accuracy = (l_numCorrect / l_numTotal) * 100;
    LOG(INFO) << "Accuracy = " << l_numCorrect << "/" << l_numTotal 
              << " = " << l_accuracy << "%" << endl;
    return l_accuracy;
}

int main(int argc, char const *argv[])
{
    // Define data loader
    string l_dataPath = "../data/mnist/";
    MNISTDataloader l_trainDataloader(l_dataPath, true); // second param for isTrain?
    MNISTDataloader l_testDataloader(l_dataPath, false); // second param for isTrain?

    // Define model
    // first linear layer is 784x300
    // 784 inputs, 300 hidden size
    LinearLayer firstLinearLayer(Tensor::Random({784, 300}, -0.01f, 0.01f));

    // Non-linear activation
    ReLULayer activationLayer;
    
    // second linear layer is 300x10
    // 300 hidden units, 10 outputs
    LinearLayer secondLinearLayer(Tensor::Random({300, 10}, -0.01f, 0.01f));

    // Convert outputs to probabilities
    SoftmaxLayer softmaxLayer;

    // Error function
    // MeanSquaredErrorLoss loss;
    CrossEntropyLoss loss;

    // Training loop
    float learningRate = 0.0005;
    size_t numEpochs = 10;
    size_t numIters = l_trainDataloader.DataLength();
    for (size_t i = 0; i < numEpochs; ++i)
    {
        LOG(INFO) << "--EPOCH (" << i << ")--" << endl;
        vector<float> errorAcc;
        size_t numCorrect = 0;
        for (size_t j = 0; j < numIters; ++j)
        {
            // Get training example
            TMutableTensorPtr input, target;
            l_trainDataloader.DataAt(j, input, target);
            size_t targetOutput = target->MaxIdx();

            // Forward pass
            TTensorPtr output0 = firstLinearLayer.Forward(input);
            TTensorPtr output1 = activationLayer.Forward(output0);
            TTensorPtr output2 = secondLinearLayer.Forward(output1);
            TTensorPtr probs = softmaxLayer.Forward(output2);

            size_t yPredVal = probs->MaxIdx();
            if (yPredVal == targetOutput)
            {
                numCorrect += 1;
            }

            // Calc Error
            float error = loss.Forward(probs, target);
            errorAcc.push_back(error);

            // Backward pass
            TTensorPtr errorGrad = loss.Backward(probs, target);
            TTensorPtr probsGrad = softmaxLayer.Backward(output2, errorGrad);
            TTensorPtr grad2 = secondLinearLayer.Backward(output1, probsGrad);
            TTensorPtr grad1 = activationLayer.Backward(output0, grad2);
            TTensorPtr grad0 = firstLinearLayer.Backward(input, grad1);

            // Gradient Descent
            secondLinearLayer.UpdateWeights(learningRate);
            firstLinearLayer.UpdateWeights(learningRate);

            // Only log every 1000 examples
            if (j % 1000 == 0)
            {
                float avgError = CalcAverage(errorAcc);
                LOG(INFO) << "--ITER (" << i << "," << j << ")-- avgError = " << avgError << " lr = " << learningRate << endl;
                for (size_t k = 0; k < probs->Shape().at(1); ++k)
                {
                    LOG(INFO) << "Output: " << probs->At({0, k}) << " Target " << target->At({0, k}) << endl;
                }
                LOG(INFO) << "Got prediction: " << yPredVal << " for target " << targetOutput << endl;
                float accuracy = ((float) numCorrect / (float) errorAcc.size()) * 100;
                LOG(INFO) << "Train accuracy: " << accuracy << endl;
                numCorrect = 0;
                errorAcc.clear();
            }

            if (j % 10000 == 0)
            {
                CalcAccuracy(firstLinearLayer, activationLayer, secondLinearLayer, softmaxLayer, l_testDataloader);
            }
        }
        CalcAccuracy(firstLinearLayer, activationLayer, secondLinearLayer, softmaxLayer, l_testDataloader);
        // learningRate /= 2.0;
    }

    return 0;
}
