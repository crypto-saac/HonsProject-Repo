import cv2
import os
import numpy as np
import time  # Import the time module

# Define the paths to the three image folders
folder1 = r"C:\Users\User\Downloads\CNNBasedSegmentation\originalvideos\originalvideos\20220322175132\mask\a"
folder2 = r"C:\Users\User\Downloads\CNNBasedSegmentation\testfolder\20220322175132\mask"
folder3 = r"C:\Users\User\Downloads\CNNBasedSegmentation\AllFramesMaskC\AllFrames\20220322175132\raw"

# Define the output folder
output_folder = r"C:\Users\User\Downloads\CNNBasedSegmentation\output"

# Create the output folders if they don't exist
intersection_folder = os.path.join(output_folder, "intersection")
union_folder = os.path.join(output_folder, "union")
majority_folder = os.path.join(output_folder, "majority")
os.makedirs(intersection_folder, exist_ok=True)
os.makedirs(union_folder, exist_ok=True)
os.makedirs(majority_folder, exist_ok=True)

# Record the start time
start_time = time.time()

# Loop through each image filename
for filename in os.listdir(folder3):
    # Print the image file paths
    #print("Image 1 path:", os.path.join(folder1, filename))
    #print("Image 2 path:", os.path.join(folder2, filename))
    #print("Image 3 path:", os.path.join(folder3, filename))

    # Load the corresponding images from each folder
    image1 = cv2.imread(os.path.join(folder1, filename), cv2.IMREAD_GRAYSCALE)
    #print("image dimensions image1:",image1.shape)
    image2 = cv2.imread(os.path.join(folder2, filename), cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(os.path.join(folder3, filename), cv2.IMREAD_GRAYSCALE)

    # Check if any of the images is None (i.e., couldn't be read)
    if image1 is None or image2 is None or image3 is None:
        print(f"Skipping {filename} because one or more images couldn't be read.")
        continue

    # Initialize empty result images for each aggregation function
    intersection_result = np.zeros_like(image1)
    union_result = np.zeros_like(image1)
    majority_result = np.zeros_like(image1)

    # Iterate over pixels
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            # Apply intersection operation
            intersection_result[i, j] = min(image1[i, j], image2[i, j], image3[i, j])

            # Apply union operation
            union_result[i, j] = max(image1[i, j], image2[i, j], image3[i, j])

            # Apply majority operation
            count_on = sum([image1[i, j] > 0, image2[i, j] > 0, image3[i, j] > 0])
            majority_result[i, j] = 255 if count_on >= 2 else 0

    # Save the result images to their respective output folders
    cv2.imwrite(os.path.join(intersection_folder, f"intersection_{filename}"), intersection_result)
    cv2.imwrite(os.path.join(union_folder, f"union_{filename}"), union_result)
    cv2.imwrite(os.path.join(majority_folder, f"majority_{filename}"), majority_result)

# Calculate and print the total runtime
end_time = time.time()
total_runtime = end_time - start_time
print("Total runtime:", total_runtime, "seconds")

print("Result images saved in the output folders:")
print("Intersection:", intersection_folder)
print("Union:", union_folder)
print("Majority:", majority_folder)
