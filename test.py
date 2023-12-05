import torchvision
from making_dataset import *
import matplotlib.pyplot as plt
import numpy as np
from custom_lenet_model import *
from custom_resnet_model import *
images, labels = next(iter(test_loader))
img_grid = torchvision.utils.make_grid(images)

npimg = img_grid.numpy()
npimg = np.transpose(npimg, (1, 2, 0))

# 이미지 출력
plt.imshow(npimg)
plt.show()

# 학습한 모델로 예측값 뽑아보기
net = MyResNet()
#net = MyLeNet()
net.load_state_dict(torch.load(PATH))
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % label_to_class[predicted[j]]
                              for j in range(4)))
