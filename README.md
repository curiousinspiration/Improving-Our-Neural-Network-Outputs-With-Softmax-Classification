# Improving Our Neural Network Outputs With Softmax Classification

How to predict probabilities from a linear layer, to achieve higher accuracy on the MNIST dataset.

Code to go along with blog: [Improving Our Neural Network Outputs With Softmax Classification](http://www.curiousinspiration.com/posts/improving-our-neural-network-outputs-with-softmax-classification)

# Build

`mkdir build`

`cd build`

`cmake -D BLAS_INCLUDE_DIR=/usr/local/opt/openblas/include \
       -D BLAS_LIB_DIR=/usr/local/opt/openblas/lib \
       -D GLOG_INCLUDE_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/include/ \
       -D GLOG_LIB_DIR=~/Code/3rdParty/glog-0.3.5/glog-install/lib/ \
       -D GTEST_INCLUDE_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/include/ \
       -D GTEST_LIB_DIR=~/Code/3rdParty/googletest-release-1.8.0/install/lib/ ..`

`make`

`./tests`

`./feedforward_neural_net`