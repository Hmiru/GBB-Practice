import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import transforms
import time
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import inspect
import torch.nn.functional as F
from making_dataset import *
import matplotlib.pyplot as plt
import numpy as np


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


my_resnet = MyResNet()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def calculate_metric(metric_fn, true_y, pred_y):
    return metric_fn(true_y, pred_y, average="macro")

        # macro : 평균의 평균을 내는 방법
        # micro : 개수 그자체로 평균을 내는 방법



# precision, recall, f1, accuracy를 한번에 보여주기 위한 함수
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")



# 모델 가져와 gpu에 할당
model = MyResNet().to(torch.device("cuda"))

# 에포크, 배치 크기 지정
epochs = 5

# 데이터로더(Dataloaders)
train_loader, valid_loader, test_loader = get_dataloader(32, 0.7, 0.15)
# 기존의 학습 코드

# 손실함수 정의(loss function)
loss_function = nn.CrossEntropyLoss()
# 크로스 엔트로피 : 실제 값과 예측 값의 차이를 줄이기 위한 엔트로피
# 다중 클래스 문제에서 잘 작동

# 옵티마이저 : Adam
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# model(신경망) 파라미터를 optimizer에 전달해줄 때 nn.Module의 parameters() 메소드를 사용
# Karpathy's learning rate 사용 (3e-4)

start_ts = time.time()  # 초단위 시간 반환

losses = []
batches = len(train_loader)
val_batches = len(valid_loader)


for epoch in range(epochs):
    total_loss = 0

    # tqdm : 진행률 프로세스바
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  --------------------
    # training 모델로 설정
    model.train()

    for i, data in progress:
        X, y = data[0].to(device), data[1].to(device)
        true_y = torch.argmax(y, dim=1)

        # 단일 배치마다 training 단계
        model.zero_grad()  # 모든 모델의 파라미터 미분값을 0으로 초기화
        outputs = model(X)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()  # step() : 파라미터를 업데이트함

    progress.set_description("Loss: {:.4f}".format(total_loss / (i + 1)))

    # ----------------- VALIDATION  -----------------
    val_losses = 0
    precision, recall, f1, accuracy = [], [], []

    # set model to evaluating (testing)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            X, y = data[0].to(device), data[1].to(device)

            y = torch.argmax(y, dim=1)

            outputs = model(X)  # 네트워크로부터 예측값 가져오기
            val_losses += loss_function(outputs, y)
            predicted_classes = torch.max(outputs, 1)[1]  # 네트워크의 예측값으로부터 class 값(범주) 가져오기

            # P/R/F1/A metrics for batch 계산
            for acc, metric in zip((precision, recall, f1),
                                   (precision_score, recall_score, f1_score)):
                acc.append(
                    calculate_metric(metric, y.cpu(), predicted_classes.cpu())
                )

    print(
        f"Epoch {epoch + 1}/{epochs}, training loss: {total_loss / batches}, validation loss: {val_losses / val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(total_loss / batches)  # 학습률을 위한 작업
print(f"Training time: {time.time() - start_ts}s")

PATH = './myResNet.pth'
torch.save(my_resnet.state_dict(), PATH)

if __name__=="__main__":
    input = torch.randn((16, 1, 224, 224))
    output = my_resnet(input)
    print(output.shape)

    print(my_resnet)