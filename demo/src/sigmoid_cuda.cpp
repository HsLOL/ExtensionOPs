#include<torch/torch.h>
#include<vector>

// declare cuda forward function
torch::Tensor sigmoid_cuda_forward(torch::Tensor input);

// declare cuda backward function
torch::Tensor sigmoid_cuda_backward(torch::Tensor output);

// macro definition
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), "data must be a CUDA Tensor.")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), "data must be contiguous.")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// define c++ forward function
torch::Tensor sigmoid_forward(torch::Tensor input){
    CHECK_INPUT(input);
    return sigmoid_cuda_forward(input);
}

// define c++ backward function
torch::Tensor sigmoid_backward(torch::Tensor output){
    CHECK_INPUT(output);
    return sigmoid_cuda_backward(output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("forward", &sigmoid_forward, "sigmoid forward (CUDA)");
  m.def("backward", &sigmoid_backward, "sigmoid backward (CUDA)");
}