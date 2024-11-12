import argparse
import time
from classifier import classifier

# Define command-line arguments
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='pet_images/', help='Path to pet images directory')
    parser.add_argument('--arch', type=str, default='vgg', help='Model architecture (e.g., vgg, alexnet, resnet)')
    parser.add_argument('--dogfile', type=str, default='dognames.txt', help='File containing dog names')
    return parser.parse_args()

# Define a function to get pet labels from image filenames
def get_pet_labels(images_dir):
    import os
    pet_labels = {}
    files = os.listdir(images_dir)
    
    for file in files:
        if file.startswith("."):  # Skip hidden files
            continue
        label = " ".join([word.lower() for word in file.split("_") if word.isalpha()])
        pet_labels[file] = [label]
    
    return pet_labels

# Load dog names from a file to use for classification
def load_dog_names(dogfile):
    dog_names = set()
    with open(dogfile, "r") as f:
        for line in f:
            dog_names.add(line.strip().lower())
    return dog_names

# Run image classification and store results in a dictionary
def classify_images(images_dir, pet_labels, model):
    results = {}
    for filename, label_list in pet_labels.items():
        pet_label = label_list[0]
        classifier_label = classifier(images_dir + filename, model).lower().strip()
        match = 1 if pet_label == classifier_label else 0
        results[filename] = [pet_label, classifier_label, match]
    return results

# Classify each image as a "dog" or "not dog" using dog names file
def classify_labels_as_dogs(results, dog_names):
    for filename, result in results.items():
        pet_label, classifier_label, match = result
        pet_is_dog = 1 if pet_label in dog_names else 0
        classifier_is_dog = 1 if classifier_label in dog_names else 0
        results[filename].extend([pet_is_dog, classifier_is_dog])
    return results

# Calculate and display accuracy scores
def calculate_results_stats(results):
    total_images = len(results)
    correct_matches = sum([1 for result in results.values() if result[2] == 1])
    correct_dog_matches = sum([1 for result in results.values() if result[3] == 1 and result[4] == 1])
    
    # Calculate Accuracy
    accuracy = (correct_matches / total_images) * 100
    dog_match_accuracy = (correct_dog_matches / total_images) * 100
    
    # Display Results
    print(f"Total Images: {total_images}")
    print(f"Model Accuracy: {accuracy:.2f}%")
    print(f"Dog Match Accuracy: {dog_match_accuracy:.2f}%")

def main():
    # Start timing the code execution
    start_time = time.time()
    
    # Get input arguments
    in_args = get_input_args()
    
    # Load pet labels and dog names
    pet_labels = get_pet_labels(in_args.dir)
    dog_names = load_dog_names(in_args.dogfile)
    
    # Classify images
    results = classify_images(in_args.dir, pet_labels, in_args.arch)
    
    # Classify labels as dogs or not
    results = classify_labels_as_dogs(results, dog_names)
    
    # Calculate and display stats
    calculate_results_stats(results)
    
    # Print total runtime
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Runtime: {total_time:.2f} seconds")

# Run the program
if __name__ == "__main__":
    main()
