from torchvision.io import read_image
#from torchvision.models import resnet50, ResNet50_Weights
import torchvision
from torchvision.models import Inception_V3_Weights

img = read_image("dataset/food/00001.jpg")

# Step 1: Initialize model with the best available weights
#weights = Inception_V3_Weights
model = torchvision.models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
#model_init = torchvision.models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
#model = NoFinalLayerInception(model_init)
model.eval()

# Step 2: Initialize the inference transforms
#preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
#batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(img).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score}%")