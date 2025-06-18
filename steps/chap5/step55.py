
def get_conv_outsize(input_size, filter_size, stride, pad):
    return (input_size + pad * 2 - filter_size) // stride + 1


H, W = 4, 4
FH, FW = 3, 3
SH, SW = 1, 1
PH, PW = 1, 1

OH = get_conv_outsize(H, FH, SH, PH)
OW = get_conv_outsize(W, FW, SW, PW)
print(OH, OW)
