/*
 * Squared Error Loss Implementation
 *
 */

#include "neural/loss/squared_error_loss.h"

using namespace std;

namespace neural
{

float SquaredErrorLoss::Forward(float output, float target) const
{
    float difference = target - output;
    return difference * difference; // square the difference
}

float SquaredErrorLoss::Backward(float input, float target)
{
    float grad = -2.0 * (target - input);
    return grad;
}

} // namespace neural
