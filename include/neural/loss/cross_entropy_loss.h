/*
 * CrossEntropyLoss computes error for binary classification
 * 
 * Inputs will look like something out of a softmax function
 *     => [0.1, 0.2, 0.7]
 * Outputs will be the binary classes
 *     => [0.0, 0.0, 1.0]
 * In this case the loss will be
 * (0 * -log(0.1)) + (0 * -log(0.2)) + (1 * -log(0.7)) ~= 0.1549
 */

#pragma once

#include "neural/math/tensor.h"

namespace neural
{

class CrossEntropyLoss
{
public:
    CrossEntropyLoss();

    float Forward(
        const TTensorPtr& a_inputs, const TTensorPtr& a_targets) const;

    TTensorPtr Backward(
        const TTensorPtr& a_origInputs, const TTensorPtr& a_targets);
};

} // namespace neural
