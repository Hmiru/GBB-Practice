import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
print(torch.__version__)

print(torch.cuda.is_available())

print(torch.cuda.get_device_name(0))

print(torch.cuda.device_count())

print("cudnn version:{}".format(torch.backends.cudnn.version()))
print("cuda version: {}".format(torch.version.cuda))