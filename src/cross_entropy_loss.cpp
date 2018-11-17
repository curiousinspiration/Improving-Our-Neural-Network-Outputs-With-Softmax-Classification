/*
 * CrossEntropyLoss Implementation
 */

#include "neural/loss/cross_entropy_loss.h"

#include <glog/logging.h>

#include <sstream>
#include <math.h>

using namespace std;

namespace neural
{

CrossEntropyLoss::CrossEntropyLoss()
{

}

float CrossEntropyLoss::Forward(
    const TTensorPtr& a_inputs, const TTensorPtr& a_targets) const
{
    if (a_inputs->Shape().size() != 2 || a_targets->Shape().size() != 2)
    {
        stringstream l_ss;
        l_ss << "CrossEntropyLoss::Forward shape size 2 "
             << a_inputs->ShapeStr() << " != " << a_targets->ShapeStr() << endl;
        throw(runtime_error(l_ss.str()));
    }

    /*
    -1/n * sum((y * log(yhat)) + ((1-y) * log(1-yhat)))
    */

    float l_error = 0.0;
    for (size_t i = 0; i < a_inputs->Shape().at(0); ++i)
    {
        for (size_t j = 0; j < a_inputs->Shape().at(1); ++j)
        {
            float y = a_targets->At({i,j}); // target
            float yhat = a_inputs->At({i,j}); // predicted

            // if y == 0 then we take the second half of the equation
            // if y == 1 then we take the first half
            // log(1) = 0
            // log(0.1) = -1
            // log(0.01) = -2
            // log(0.001) = -3 ... etc

            // if y == 0 and yhat == 0.001 then
            // ((1.0 - 0.0) * log(1.0 - 0.001)) = 1.0 * ~0.0 ~= -0.0
            // if y == 0 and yhat == 0.999 then
            // ((1.0 - 0.0) * log(1.0 - 0.999)) = 1.0 * ~-2 = -3.0

            // if y == 1 and yhat == 0.001 then
            // (1 * log(0.001)) = -3.0
            // if y == 1 and yhat == 0.99 then
            // (1 * log(0.99)) ~= 0.0
            l_error += (y * log(yhat)) + ((1.0 - y) * log(1.0 - yhat));
        }
    }

    return -1.0 * (l_error / (float)a_inputs->Shape().at(0));
}

TTensorPtr CrossEntropyLoss::Backward(
    const TTensorPtr& a_origInputs, const TTensorPtr& a_targets)
{
    TMutableTensorPtr l_gradient = Tensor::New(a_targets->Shape());
    for (size_t i = 0; i < a_origInputs->Shape().at(0); ++i)
    {
        for (size_t j = 0; j < a_origInputs->Shape().at(1); ++j)
        {
            float yhat = a_origInputs->At({i,j});
            float y = a_targets->At({i,j});
            
            // derivative of ln(x) is 1/x
            // https://www.wyzant.com/resources/lessons/math/calculus/derivative_proofs/lnx
            float l_gradVal = -1.0 * ((y / yhat) + ((1.0 - yhat) * (1.0 / (1.0 - yhat))));
            // float l_gradVal = yhat - y;
            l_gradient->SetAt({i,j}, l_gradVal);
        }
    }

    return make_shared<Tensor>(*l_gradient /= (float) a_inputs->Shape().at(0));
}

} // namespace neural

