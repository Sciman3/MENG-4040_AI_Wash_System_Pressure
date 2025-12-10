import sys
from ultralytics import YOLO
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
classnames = ["Modifications","No Modifications"]
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 3 >> 32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 >> 112

            # Block 2: 32 >> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 >>56

            # Block 3: 64 >> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 >> 28
        )

        # After 3 pools: feature map size = 128 x 28 x 28
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


imageFile = sys.argv[1] if len(sys.argv) > 1 else "testImage.png"

#Show selection Menu
print("Select a Test:")
print("1: Poor class balance YOLO")
print("2: Good class balance YOLO")
print("3: Small CNN")
print("4: ResNet")

#Get user input
choice = input("\nYour selection: ")

#Set acceleration device and print device to be used
device = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print("Using device:", device)

#Run corresponding test
match choice:
    case "1":
        model = YOLO('./models/Train_Poor_GoodBad_Ratio.pt')
        model.to(device)
        results = model(imageFile)
        result = results[0]
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        print("Top-1 class:", top1_idx, "Conf:", top1_conf)
    case "2":
        model = YOLO('./models/Train_More_Good_Image.pt')
        model.to(device)
        results = model(imageFile)
        result = results[0]
        probs = result.probs
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)
        print("Top-1 class:", top1_idx, "Conf:", top1_conf)
    case "3":
        #load model
        model = torch.load("./models/model_cnn.pth", weights_only=False).to(device)
        model.eval()
        # load image
        img = Image.open(imageFile).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # prepare tensor
        x = transform(img).unsqueeze(0).to(device)   # add batch dimension: [1, C, H, W]

        # run inference
        with torch.no_grad():
            output = model(x)
            # convert to probabilities
            probs = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            # probability of top class
            confidence = probs[0][predicted_class].item()
        predicted_class_name = classnames[predicted_class]
        print("Prediction:", predicted_class_name)
        print("Confidence:", confidence)
    case "4":
        #load model
        modelRes = torch.load("./models/model_resnet.pth", weights_only=False).to(device)
        modelRes.eval()
        # load image
        img = Image.open(imageFile).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # prepare tensor
        x = transform(img).unsqueeze(0).to(device)   # add batch dimension: [1, C, H, W]

        # run inference
        with torch.no_grad():
            output = modelRes(x)
            # convert to probabilities
            probs = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            # probability of top class
            confidence = probs[0][predicted_class].item()
        predicted_class_name = classnames[predicted_class]
        print("Prediction:", predicted_class_name)
        print("Confidence:", confidence)