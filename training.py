
from tqdm import tqdm
from evaluation import *
from custom_lenet_model import MyLeNet
from custom_resnet_model import MyResNet
import torch.nn as nn
import torch.optim as optim
from making_dataset import *

class ModelTrainer:
    def __init__(self, learning_rate, batch_size, epochs):
        self.my_model = MyResNet().to(torch.device("cuda"))
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.my_model.parameters(), lr=learning_rate)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloader(batch_size, 0.7, 0.15)
        self.epochs = epochs


    def run_epoch(self, loader, training=False):
        if training:
            self.my_model.train()
        else:
            self.my_model.eval()

        total_loss = 0.0
        all_predictions, all_labels = [], []
        progress_bar = tqdm(loader, desc="Training" if training else "Validation", total=len(loader))

        for data in progress_bar:
            inputs, labels = data[0].to(torch.device("cuda")), data[1].to(torch.device("cuda"))
            if training:
                self.optimizer.zero_grad()

            outputs = self.my_model(inputs)
            loss = self.loss_function(outputs, labels)
            total_loss += loss.item()

            if not training:
                labels = torch.argmax(labels, dim=1)
                _, predicted_classes = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_classes.cpu().numpy())

            if training:
                loss.backward()
                self.optimizer.step()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))

        if not training:
            return total_loss / len(loader), all_labels, all_predictions

        return total_loss / len(loader)

    def train(self):
        train_loss = self.run_epoch(self.train_loader, training=True)
        print(f'Epoch {epoch + 1}, Training loss: {train_loss:.3f}')

    def valid(self):
        val_loss, all_labels, all_predictions = self.run_epoch(self.valid_loader, training=False)
        print(f'Validation loss: {val_loss:.3f}')

        precision, recall, f1 = calculate_metric(all_labels, all_predictions)
        print_scores(precision, recall, f1)

    def test(self):
        test_loss, all_labels, all_predictions = self.run_epoch(self.test_loader, training=False)
        print(f'Test loss: {test_loss:.3f}')

        precision, recall, f1 = calculate_metric(all_labels, all_predictions)
        print_scores(precision, recall, f1)

    def save_model(self):
        model_class_name = type(self.my_model).__name__
        model_save_path = f'./{model_class_name}.pth'
        torch.save(self.my_model.state_dict(), model_save_path)
        print(f"모델 저장됨: {model_save_path}")

if __name__ == "__main__":
    trainer = ModelTrainer(learning_rate=3e-5, batch_size=32, epochs=1)
    for epoch in range(trainer.epochs):
        trainer.train()
        trainer.valid()
    trainer.test()
    trainer.save_model()
