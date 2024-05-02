import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.models import inception_v3

img = read_image("dataset/food/00000.jpg")

# Step 1: Initialize model with the best available weights
model = inception_v3(pretrained=True)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 3: Apply inference preprocessing transforms
img = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
with torch.no_grad():
    prediction = model(img)
prediction = torch.nn.functional.softmax(prediction[0], dim=0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
print(f"Class ID: {class_id}, Score: {100 * score}%")
