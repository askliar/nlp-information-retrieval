import sys
import torch
import torch.utils
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

try:
    import gpustat
except ImportError:
    raise ImportError("pip install gpustat")


def show_memusage(device=0):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()["gpus"][device]
    print("{}/{}".format(item["memory.used"], item["memory.total"]))


device = 0
show_memusage(device=device)

torch.cuda.set_device(device)
model = models.resnet101(pretrained=False)
model.cuda()
criterion = nn.CrossEntropyLoss()

volatile = False

show_memusage(device=device)
for ii in range(3):
    inputs = torch.randn(20, 3, 224, 224)
    labels = torch.LongTensor(range(20))
    inputs = inputs.cuda()
    labels = labels.cuda()
    inputs = Variable(inputs, volatile=volatile)
    labels = Variable(labels, volatile=volatile)

    print("before run model:")
    show_memusage(device=device)
    outputs = model(inputs)
    print("after run model:")
    show_memusage(device=device)
    loss = criterion(outputs, labels)

    if bool(int(sys.argv[1])):
        print("before backward:")
        show_memusage(device=device)
        loss.backward()
        print("after backward:")
        show_memusage(device=device)
    #     del loss
    #     del outputs
    #     del inputs
    #     del labels
    #     show_memusage(device=device)
    else:
        del loss
        del outputs
    #     # del inputs
    #     # del labels
    #     print "after delete:"
#     show_memusage(device=device)