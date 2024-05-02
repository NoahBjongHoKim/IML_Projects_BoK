import torch
import torchvision.models as models

# Load the pre-trained InceptionV3 model
model = models.inception_v3(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Dummy input tensor to determine the shape of the output
dummy_input = torch.randn(1, 3, 299, 299)  # assuming input image size is (299, 299)

# Pass the dummy input through the model to get the output
output = model(dummy_input)

# Print the shape of the output
print("Output shape:", output.shape)

# Extract the embedding size from the output shape
embedding_size = output.size(1)
print("Embedding size:", embedding_size)
