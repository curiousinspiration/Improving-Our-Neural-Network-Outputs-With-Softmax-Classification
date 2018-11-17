/*
 * SoftmaxLayer applies the Softmax function to an n-dimensional input Tensor
 * rescaling them so that the elements of the n-dimensional output Tensor 
 * lie in the range (0,1) and sum to 1
 * y = exp(xi) / sum(exp(xj))
 */

#pragma once

#include "neural/layers/layer.h"

#include <vector>

namespace neural
{

class SoftmaxLayer : public Layer
{
public:
    SoftmaxLayer();

    virtual ~SoftmaxLayer() {};

    // Forward Pass
    virtual TTensorPtr Forward(const TTensorPtr& a_inputs) const override;
    
    // Backward Pass
    virtual TTensorPtr Backward(const TTensorPtr& a_origInput, const TTensorPtr& a_gradOutput) override;

private:

};

} // namespace neural
