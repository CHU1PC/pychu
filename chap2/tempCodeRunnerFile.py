# class Exp(Function):
#     def forward(self, x):
#         y = np.exp(x)
#         return y

#     def backward(self, gy):
#         x = self.input.data
#         gx = np.exp(x) * gy
#         return gx