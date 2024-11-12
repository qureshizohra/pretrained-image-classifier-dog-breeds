# Dog Breed Classifier Project

## Overview
This project uses a pre-trained image classifier to identify dog breeds from images. The classifier leverages popular deep learning models and provides detailed results on the classification accuracy, runtime, and comparison across multiple models.

## Project Structure
- `classify_dogs.py`: Main script to run the image classifier on dog breed images.
- `classifier.py`: Contains functions for running the pre-trained model (e.g., `vgg`).
- `dognames.txt`: Text file with a list of recognized dog breed names.
- `pet_images/`: Directory with sample images for testing.
  
## Key Features
- **Command-Line Interface**: Supports arguments for image directory, model architecture, and dog breed file.
- **Classification Accuracy**: Calculates and displays model performance metrics.
- **Model Comparisons**: Supports multiple models (e.g., VGG, AlexNet, ResNet).
- **Execution Time Measurement**: Measures the time taken to classify the images.

## Requirements
- Python 3.7 or higher
- Libraries: `argparse`, `os`, `time`, `torch`, `PIL`, `numpy` (you can install these using `pip install -r requirements.txt` if a `requirements.txt` file is added)

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/dog-breed-classifier.git
   cd dog-breed-classifier
