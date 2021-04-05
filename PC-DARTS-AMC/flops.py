import torch
from torchvision.models import resnet50
from thop import profile
from thop import clever_format
from torchsummary import summary
model = resnet50()
model = model.cuda()
#print(model)

input=torch.cuda.FloatTensor(1,3,224,224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
summary(model,(3,224,224))
print(macs,params)