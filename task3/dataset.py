import os
import torchvision.datasets as datasets

def calculate_average_image_size(folder_path, num_images=10000):
    total_width = 0
    total_height = 0

    # Define a dataset object to load images from the folder
    dataset = datasets.ImageFolder(root=folder_path)

    # Get the total number of images in the dataset
    num_files = len(dataset)

    # Ensure we don't exceed the number of images available
    num_images = min(num_images, num_files)

    # Iterate over a subset of the images
    for i in range(num_images):
        # Load the image and its label
        img, _ = dataset[i]

        # Get the dimensions of the image
        width, height = img.size

        # Accumulate the width and height
        total_width += width
        total_height += height

    # Calculate the average width and height
    average_width = total_width / num_images
    average_height = total_height / num_images

    return average_width, average_height

# Example usage:
folder_path = "dataset/"
average_width, average_height = calculate_average_image_size(folder_path)
print("Average image width:", average_width)
print("Average image height:", average_height)
