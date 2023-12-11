import torch.nn as nn
import torchvision.models as models


class MyResNet(nn.Module):  # MnistResNet은 nn.Module 상속
    def __init__(self, in_channels=1):
        super(MyResNet, self).__init__()

        # torchvision.models에서 사전훈련된 resnet 모델 가져오기
        self.model = models.resnet50(pretrained=True)

        # 기본 채널이 3(RGB)이기 때문에 fashion_mnist에 맞게 1(grayscale image)로 바꿔준다.
        # 원래 ResNet의 첫번째 층
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 1000개 클래스 대신 9개 클래스로 바꿔주기
        num_ftrs = self.model.fc.in_features
        # nn.Linear(in_features, out_features ...)
        self.model.fc = nn.Linear(num_ftrs, 9)

    def forward(self, x):  # 모델에 있는 foward 함수 그대로 가져오기
        return self.model(x)
