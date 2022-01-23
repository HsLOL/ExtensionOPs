#include<ATen/ATen.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<vector>

template <typename scalar_t>
__device__ scalar_t sigmoid(scalar_t z){
    return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ scalar_t d_sigmoid(scalar_t z){
    return (1.0 - z) * z;
}

template <typename scalar_t>
__global__ void sigmoid_cuda_forward_kernel(const scalar_t * __restrict__ input, scalar_t * __restrict__ output){
    const int index = blockIdx.x * blockDim.x + blockIdx.y;
    output[index] = sigmoid(input[index]);
}

template <typename scalar_t>
__global__ void sigmoid_cuda_backward_kernel(const scalar_t* __restrict__ output,
                                             scalar_t* __restrict__ new_grad_output){
    const int index = blockIdx.x * blockDim.x + blockIdx.y;
    new_grad_output[index] = d_sigmoid(output[index]);
}

// only using at::Tensor in .cu file
// not using torch::Tensor
at::Tensor sigmoid_cuda_forward(at::Tensor input){
    auto output = at::zeros_like(input);
    dim3 blocks(input.size(0), input.size(1));
    int threads = 1;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "error in sigmoid_cuda_forward", ([&]
        {sigmoid_cuda_forward_kernel<scalar_t> <<<blocks, threads>>> (input.data<scalar_t>(), output.data<scalar_t>());
        }));

    return output;
}

at::Tensor sigmoid_cuda_backward(at::Tensor output){
    auto new_grad_output = at::zeros_like(output);
    dim3 blocks(output.size(0), output.size(1));
    int threads = 1;

    AT_DISPATCH_FLOATING_TYPES(output.type(), "error in sigmoid_cuda_backward", ([&]{
    sigmoid_cuda_backward_kernel<scalar_t> <<<blocks, threads>>> (output.data<scalar_t>(),
                                                                  new_grad_output.data<scalar_t>());}));

    return new_grad_output;
}
