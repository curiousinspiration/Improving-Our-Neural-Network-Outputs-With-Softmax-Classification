/*
 * SoftmaxLayer Implementation
 */

#include "neural/layers/softmax_layer.h"
#include "neural/math/tensor_math.h"

#include <glog/logging.h>

#include <sstream>
#include <thread>

#include <math.h>

using namespace std;

namespace neural
{

SoftmaxLayer::SoftmaxLayer()
{

}

TTensorPtr SoftmaxLayer::Forward(const TTensorPtr& a_inputs) const
{
    if (0 == a_inputs->Shape().size())
    {
        return a_inputs;
    }

    if (a_inputs->Shape().size() > 2)
    {
        stringstream l_ss;
        l_ss << "SoftmaxLayer::Forward size > 2 not supported yet size = " 
             << a_inputs->Shape().size() << endl;
        throw(runtime_error(l_ss.str()));
    }

    // For numeric stability we subtract max from input
    // because exp(x) can get very large, but by subtracting the max
    // we guaruntee max == 0
    // see http://cs231n.github.io/linear-classify/#softmax
    TMutableTensorPtr l_inputs = a_inputs->ToMutable();
    *l_inputs -= a_inputs->MaxVal();

    size_t x = l_inputs->Shape().at(0);
    size_t y = l_inputs->Shape().at(1);
    TMutableTensorPtr l_outputs = Tensor::New({x, y});

    vector<float> l_sums;
    for (size_t i = 0; i < x; ++i)
    {
        float l_sum = 0.0;
        for (size_t j = 0; j < y; ++j)
        {
            l_sum += exp(l_inputs->At({i, j}));
        }
        l_sums.push_back(l_sum);
    }

    for (size_t i = 0; i < x; ++i)
    {
        for (size_t j = 0; j < y; ++j)
        {
            float l_val = exp(l_inputs->At({i, j})) / l_sums.at(i);
            l_outputs->SetAt({i, j}, l_val);
        }
    }

    return l_outputs;
}

TTensorPtr SoftmaxLayer::Backward(
    const TTensorPtr& a_origInput, const TTensorPtr& a_gradOutput)
{
    // TODO: cache, not computationally efficient
    TTensorPtr l_outputs = Forward(a_origInput);

    size_t x = l_outputs->Shape().at(0);
    size_t y = l_outputs->Shape().at(1);

    TMutableTensorPtr l_grad = Tensor::New({y, y});

    // References:
    // https://deepnotes.io/softmax-crossentropy
    // https://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
    // https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
    for (size_t i = 0; i < y; ++i)
    {
        for (size_t j = 0; j < y; ++j)
        {
            if (i == j)
            {
                // jacobian_m[i][j] = s[i] * (1-s[i])
                float oi = l_outputs->At({0, i});
                float l_val = oi * (1-oi);
                l_grad->SetAt({i,j}, l_val);
            }
            else
            {
                // jacobian_m[i][j] = -s[i]*s[j]
                float oi = l_outputs->At({0, i});
                float oj = l_outputs->At({0, j});
                float l_val = -oi * oj;
                l_grad->SetAt({i,j}, l_val);
            }
        }
    }

    // Chain rule
    return TensorMath::Multiply(a_gradOutput, l_grad);
}

} // namespace neural
