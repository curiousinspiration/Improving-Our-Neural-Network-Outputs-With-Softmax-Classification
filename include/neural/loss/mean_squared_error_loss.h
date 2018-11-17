/*
 * Mean Squared Error Loss Definition
 *
 */

#pragma once

#include "neural/math/tensor.h"

namespace neural
{

class MeanSquaredErrorLoss
{
public:
    MeanSquaredErrorLoss();

    float Forward(
        const TTensorPtr& a_inputs, const TTensorPtr& a_targets) const;

    TTensorPtr Backward(
        const TTensorPtr& a_origInputs, const TTensorPtr& a_targets);
};

} // namespace neural
