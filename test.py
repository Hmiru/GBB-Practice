import torch
from torchvision import transforms
from preprocess import *
from PIL import Image
from custom_lenet_model import MyLeNet
import yaml
with open('config.yaml') as f:
    file = yaml.full_load(f)
class testMymodel:
    def __init__(self):
        self.transform = make_composed_transform_with_size(224)
        self.model = MyLeNet().to(torch.device("cuda"))
        self.model.eval()
        #self.model.load_state_dict(torch.load(file["Model_path"]))
        ckpt = torch.load(file["Model_path"], map_location=torch.device("cuda"))
        self.model.load_state_dict(ckpt, strict=False)

    def input_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image).unsqueeze(0).to(torch.device("cuda"))

        with torch.no_grad():
            output = self.model(image)

        probabilities = torch.softmax(output, dim=1).squeeze()
        predicted_class = torch.argmax(output, 1).item()
        predicted_prob = probabilities[predicted_class].item() * 100
        return predicted_class, predicted_prob


def print_result(predicted_class, predicted_prob):
    if predicted_class in [0, 4, 8]:
        result = "비겼습니다."
    elif predicted_class in [1, 5, 6]:
        result = "왼쪽 사람이 이겼습니다."
    else:
        result = "오른쪽 사람이 이겼습니다."
    print(f"정확도 {predicted_prob:.1f}%로 {result}")
    print(predicted_class)

if __name__ == "__main__":
    tester = testMymodel()
    image_path = 'C:/Users/mirun/image_dataset/gbb_test/3.jpg'
    predicted_class, predicted_prob = tester.input_image(image_path)
    print_result(predicted_class, predicted_prob)
