import cv2
import os
import numpy as np
import time
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the paths to the three image folders
folder1 = r"C:\Users\User\Downloads\CNNBasedSegmentation\originalvideos\originalvideos\20220322172908\mask\a"
folder2 = r"C:\Users\User\Downloads\CNNBasedSegmentation\testfolder\20220322172908\mask"
folder3 = r"C:\Users\User\Downloads\CNNBasedSegmentation\AllFramesMaskC\AllFrames\20220322172908\raw"

# Define the output folder
output_folder = r"C:\Users\User\Downloads\CNNBasedSegmentation\outputmpi"

# Create the output folders if they don't exist
intersection_folder = os.path.join(output_folder, "intersection")
union_folder = os.path.join(output_folder, "union")
majority_folder = os.path.join(output_folder, "majority")
os.makedirs(intersection_folder, exist_ok=True)
os.makedirs(union_folder, exist_ok=True)
os.makedirs(majority_folder, exist_ok=True)

# Record the start time
start_time = time.time()

# Divide the filenames among processes
all_filenames = os.listdir(folder3)
chunk_size = len(all_filenames) // size
start_index = rank * chunk_size
end_index = (rank + 1) * chunk_size if rank < size - 1 else len(all_filenames)
process_filenames = all_filenames[start_index:end_index]

# Initialize lists to store results
intersection_results = []
union_results = []
majority_results = []

# Loop through each process's subset of image filenames
for filename in process_filenames:
    try:
        # Load the corresponding images from each folder
        image1 = cv2.imread(os.path.join(folder1, filename), cv2.IMREAD_GRAYSCALE)
        #print("image dimensions image1:",image1.shape)
        image2 = cv2.imread(os.path.join(folder2, filename), cv2.IMREAD_GRAYSCALE)
        #print("image dimensions image2:",image2.shape)
        image3 = cv2.imread(os.path.join(folder3, filename), cv2.IMREAD_GRAYSCALE)
        #print("image dimensions image3:",image3.shape)

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
                #print("intersect, union, maj dims:", intersection_result.shape, union_result.shape, majority_result.shape)
                #print(intersection_result, union_result, majority_result)

        # Append the results to the lists
        intersection_results.append(intersection_result)
        union_results.append(union_result)
        majority_results.append(majority_result)
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# Gather results from all processes
gathered_intersection = comm.gather(intersection_results, root=0)
gathered_union = comm.gather(union_results, root=0)
gathered_majority = comm.gather(majority_results, root=0)

# The root process will save the results
if rank == 0:
    # Concatenate gathered results
    all_intersection = np.concatenate(gathered_intersection)
    all_union = np.concatenate(gathered_union)
    all_majority = np.concatenate(gathered_majority)
   # print("intersect, union, maj dims:", all_intersection.shape, all_union.shape, all_majority.shape)
    # Save the result images to their respective output folders
    
    """"
    for filename in all_filenames:
        cv2.imwrite(os.path.join(intersection_folder, f"intersection_{filename}"), all_intersection)
        cv2.imwrite(os.path.join(union_folder, f"union_{filename}"), all_union)
        cv2.imwrite(os.path.join(majority_folder, f"majority_{filename}"), all_majority)
    """
   # print("len(all_filenames)=",len(all_filenames))
    for i in range(len(all_filenames)):
        filename = all_filenames[i]
        cv2.imwrite(os.path.join(intersection_folder, f"intersection_{filename}"), all_intersection[i])
        cv2.imwrite(os.path.join(union_folder, f"union_{filename}"), all_union[i])
        cv2.imwrite(os.path.join(majority_folder, f"majority_{filename}"), all_majority[i])
        
    # Calculate and print the total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    print("Experiment for folder:", folder2)
    print("Total runtime:", total_runtime, "seconds")

    print("Result images saved in the output folders:")
    print("Intersection:", intersection_folder)
    print("Union:", union_folder)
    print("Majority:", majority_folder)
