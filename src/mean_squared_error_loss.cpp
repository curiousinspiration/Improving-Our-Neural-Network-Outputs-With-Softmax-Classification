/*
 * MeanSquaredErrorLoss Implementation
 */

#include "neural/loss/mean_squared_error_loss.h"

#include <glog/logging.h>

#include <sstream>
#include <math.h>

using namespace std;

namespace neural
{

MeanSquaredErrorLoss::MeanSquaredErrorLoss()
{

}

float MeanSquaredErrorLoss::Forward(
    const TTensorPtr& a_inputs, const TTensorPtr& a_targets) const
{
    if (!a_inputs->HasSameShape(a_targets))
    {
        string l_errMsg = "MeanSquaredErrorLoss::Forward inputs and targets must have same shape.";
        LOG(ERROR) << l_errMsg << endl;
        throw(l_errMsg);
    }

    float l_error = 0.0;
    for (size_t i = 0; i < a_inputs->Size(); ++i)
    {
        float l_diff = (a_targets->Data().at(i) - a_inputs->Data().at(i));
        l_error += (l_diff * l_diff);
    }

    return (l_error / (float)a_inputs->Size());
}

TTensorPtr MeanSquaredErrorLoss::Backward(
    const TTensorPtr& a_origInputs, const TTensorPtr& a_targets)
{
    if (!a_origInputs->HasSameShape(a_targets))
    {
        string l_errMsg = "MeanSquaredErrorLoss::Backward inputs and targets must have same shape.";
        LOG(ERROR) << l_errMsg << endl;
        throw(l_errMsg);
    }

    size_t l_batchSize = a_origInputs->Shape().at(0);
    size_t l_outputSize = a_origInputs->Shape().at(1);
    TMutableTensorPtr l_grad = Tensor::New({l_batchSize, l_outputSize});

    for (size_t i = 0; i < a_origInputs->Size(); ++i)
    {
        // dedl = -2.0 * (target - input)
        float dedl = -2.0 * (a_targets->Data().at(i) - a_origInputs->Data().at(i));
        l_grad->MutableData().at(i) = dedl;
    }

    return l_grad;
}

} // namespace neural
