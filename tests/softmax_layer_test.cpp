/*
 * Softmax Layer Test
 *
 */

#include "neural/layers/softmax_layer.h"

#include <gtest/gtest.h>

using namespace neural;
using namespace std;

// TEST(TestCaseName, IndividualTestName)
TEST(SoftmaxTest, TestForward)
{
    SoftmaxLayer l_layer;
    TTensorPtr l_input = Tensor::New(
        {1,3},
        {
            1.0, 2.0, 3.0
        }
    );

    TTensorPtr l_output = l_layer.Forward(l_input);

    vector<size_t> l_shape = l_output->Shape();
    EXPECT_EQ(1, l_shape.at(0));
    EXPECT_EQ(3, l_shape.at(1));

    /*
    sum = exp(1)+exp(2)+exp(3) ~= 30.19

    exp(1) = 2.718
    exp(2) = 7.389
    exp(3) = 20.08

    x0 = 2.718 / 30.19 = 0.0900
    x1 = 7.389 / 30.19 = 0.2447
    x2 = 20.08 / 30.19 = 0.6652
    */
    EXPECT_NEAR(0.0900, l_output->At({0, 0}), 0.0001f);
    EXPECT_NEAR(0.2447f, l_output->At({0, 1}), 0.0001f);
    EXPECT_NEAR(0.6652f, l_output->At({0, 2}), 0.0001f);
}

TEST(SoftmaxTest, TestNumericStability)
{
    SoftmaxLayer l_layer;

    // math.exp(789) will overflow, so we want to make sure our 
    // Softmax can handle this
    TTensorPtr l_input = Tensor::New(
        {1,3},
        {
            759.0, 760.0, 761.0
        }
    );

    TTensorPtr l_output = l_layer.Forward(l_input);

    vector<size_t> l_shape = l_output->Shape();
    EXPECT_EQ(1, l_shape.at(0));
    EXPECT_EQ(3, l_shape.at(1));

    EXPECT_NEAR(0.09003057f, l_output->At({0, 0}), 0.0001f);
    EXPECT_NEAR(0.24472847f, l_output->At({0, 1}), 0.0001f);
    EXPECT_NEAR(0.66524096f, l_output->At({0, 2}), 0.0001f);
}


// TODO: Test backward with example online
TEST(SoftmaxTest, TestBackward)
{
    SoftmaxLayer l_layer;
    TTensorPtr l_input = Tensor::New(
        {1,2},
        {
            1.0, 2.0
        }
    );

    TTensorPtr l_forward = l_layer.Forward(l_input);
    EXPECT_EQ(1, l_forward->Shape().at(0));
    EXPECT_EQ(2, l_forward->Shape().at(1));

    EXPECT_NEAR(0.26894142, l_forward->At({0, 0}), 0.0001f);
    EXPECT_NEAR(0.73105858, l_forward->At({0, 1}), 0.0001f);

    TTensorPtr l_backward = l_layer.Backward(l_input, l_forward);
    EXPECT_EQ(1, l_backward->Shape().at(0));
    EXPECT_EQ(2, l_backward->Shape().at(1));

    EXPECT_NEAR(-0.0908577, l_backward->At({0, 0}), 0.0001f);
    EXPECT_NEAR(0.0908577, l_backward->At({0, 1}), 0.0001f);
}



