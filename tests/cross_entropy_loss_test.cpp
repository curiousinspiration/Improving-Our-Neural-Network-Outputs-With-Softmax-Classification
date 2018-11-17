/*
 * Cross Entropy Loss Test
 *
 */

#include "neural/loss/cross_entropy_loss.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(CrossEntropyLossTest, TestForward)
{
    CrossEntropyLoss l_loss;
    TTensorPtr l_inputs = Tensor::New(
        {1,3},
        {
            0.09003057f, 0.24472847f, 0.66524096f
        }
    );

    TTensorPtr l_targets = Tensor::New(
        {1,3},
        {
            0.0, 1.0, 0.0
        }
    );

    float l_error = l_loss.Forward(l_inputs, l_targets);
    EXPECT_NEAR(2.59629f, l_error, 0.00001f);
}
