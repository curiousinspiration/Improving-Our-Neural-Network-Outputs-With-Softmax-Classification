/*
 * Squared Error Test
 *
 */

#include "neural/loss/squared_error_loss.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(SquaredErrorTest, TestForward)
{
    float output = 1.5;
    float target = 2.0;

    SquaredErrorLoss loss;

    // (target - output)^2
    float error = loss.Forward(output, target);
    EXPECT_EQ(0.25, error);
}

TEST(SquaredErrorTest, TestBackward)
{
    float input = 1.5;
    float target = 2.0;

    SquaredErrorLoss loss;

    // -2 * (target - input)
    float grad = loss.Backward(input, target);
    EXPECT_EQ(-1.0, grad);
}
