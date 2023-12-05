import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from cv2 import imshow
import time
from making_dataset import *
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def calculate_metric(metric_fn, true_y, pred_y):
    return metric_fn(true_y, pred_y, average="macro")
    # getfullargspec(func) : 호출 가능한 개체의 매개 변수의 이름과 기본값을 가져옴 (튜플로 반환)
    # kwonlyargs : 모든 parameter 값 확인

    # macro : 평균의 평균을 내는 방법
    # micro : 개수 그자체로 평균을 내는 방법


# precision, recall, f1, accuracy를 한번에 보여주기 위한 함수
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores) / batch_size:.4f}")


my_lenet = MyLeNet().to(torch.device("cuda"))
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 손실함수 정의(loss function)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(my_lenet.parameters(), lr=3e-4)

# 데이터로더(Dataloaders)
train_loader, valid_loader, test_loader = get_dataloader(64, 0.7, 0.15)
# 기존의 학습 코드
start_ts = time.time()  # 초단위 시간 반환

losses = []
batches = len(train_loader)
val_batches = len(valid_loader)
epochs = 2
for epoch in range(epochs):
    training_loss = 0.0

    # tqdm : 진행률 프로세스바
    progress = tqdm(enumerate(train_loader), desc="Loss: ", total=batches)

    # ----------------- TRAINING  --------------------
    my_lenet.train()
    for i, data in progress:
        inputs, labels = data[0].to(device), data[1].to(device)
        labels = torch.argmax(labels, dim=1)
        optimizer.zero_grad()
        outputs = my_lenet(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        current_loss = loss.item()  # item() : 키, 값 반환
        training_loss += current_loss
        predicted_classes = torch.max(outputs, 1)[1]

        if i % 20 == 19:  # print every 2000개마다
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, training_loss / 20))
            training_loss = 0.0


    # ----------------- VALIDATION  -----------------
    val_losses = 0
    precision, recall, f1,accuracy = [], [], [], []
    true_labels = []
    predictions = []
    my_lenet.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            labels = torch.argmax(labels, dim=1)

            outputs = my_lenet(inputs)  # 네트워크로부터 예측값 가져오기
            val_losses += loss_function(outputs, labels)
            predicted_classes = torch.max(outputs, 1)[1]  # 네트워크의 예측값으로부터 class 값(범주) 가져오기
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted_classes.cpu().numpy())
            # P/R/F1/A metrics for batch 계산
            for acc, metric in zip((precision, recall, f1),
                                   (precision_score, recall_score, f1_score)):
                acc.append(
                    calculate_metric(metric, labels.cpu(), predicted_classes.cpu())
                )

    print(
        f"Epoch {epoch + 1}/{epochs}, training loss: {training_loss / batches}, validation loss: {val_losses / val_batches}")
    print_scores(precision, recall, f1, accuracy, val_batches)
    losses.append(training_loss / batches)  # 학습률을 위한 작업

    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
print(f"Training time: {time.time() - start_ts}s")
print('Finished Training')

PATH = './myLeNet.pth'
torch.save(my_lenet.state_dict(), PATH)
if __name__=="__main__":
    pass
