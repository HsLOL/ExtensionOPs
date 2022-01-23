from torch.nn import Module
from torch.autograd import Function
import torch
import sigmoid_cuda


class SigmoidFunction(Function):

    @staticmethod
    def forward(ctx,
                input):
        output = sigmoid_cuda.forward(input)
        print(f'forward result: {output}')
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx,
                 grad_output):
        output = ctx.saved_tensors
        output = output[0]
        grad_sigmoid = sigmoid_cuda.backward(output.contiguous())
        grad_result = grad_sigmoid * grad_output
        print(f'backward result: {grad_result}')
        return grad_result


class Dense(Module):

    def __init__(self):
        super(Dense, self).__init__()

    def forward(self, input):
        return SigmoidFunction.apply(input)


if __name__ == '__main__':
    a = torch.tensor([[1.], [3.]], dtype=torch.float32, requires_grad=True).cuda()
    print(f'input data: {a}')
    m = Dense()
    result = m(a)
    sum_ = result.sum()
    sum_.backward()
