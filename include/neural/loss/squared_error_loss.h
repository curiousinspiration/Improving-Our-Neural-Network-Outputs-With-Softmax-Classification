/*
 * Squared Error Loss Definition
 *
 */

#pragma once

#include <vector>

namespace neural
{

class SquaredErrorLoss
{
public:
    SquaredErrorLoss() {};
    float Forward(float output, float target) const;
    float Backward(float input, float target);
private:

};

} // namespace
