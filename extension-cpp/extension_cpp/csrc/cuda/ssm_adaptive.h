#pragma once

#include <torch/extension.h>

/*
The following struct contains the common parameters required by all the SSM models
*/
struct SSMaParamsBaseClass{
    torch::PackedTensorAccessor32<float,3> dt;
    torch::PackedTensorAccessor32<float,1> D;
    torch::PackedTensorAccessor32<float,3> a;

    int batch_size, seq_length, d, n;
};

/*
This struct contains the params for manipulating radius
*/
struct SSMaParamsReal{
    torch::PackedTensorAccessor32<float,3> v;
//    torch::PackedTensorAccessor32<float,3> x_rho;   // computed in the host code
    torch::PackedTensorAccessor32<float,3> B;
    torch::PackedTensorAccessor32<float,3> C;
    torch::PackedTensorAccessor32<float,4> G;
};

/*
This struct contains the params for manipulating theta
*/
struct SSMaParamsImag{
    torch::PackedTensorAccessor32<float,3> v_imag;
    torch::PackedTensorAccessor32<float,3> B_theta;
    torch::PackedTensorAccessor32<float,3> C_theta;
    torch::PackedTensorAccessor32<float,4> G_theta;
};
