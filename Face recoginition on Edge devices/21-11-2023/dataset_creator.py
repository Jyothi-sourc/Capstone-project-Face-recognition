import os
import pickle
from tqdm import tqdm
from pytictoc import TicToc

def extract_images_and_labels(directory):
    images = []
    labels = []
    timer = TicToc()  # Initialize TicToc for timing
    timer.tic()  # Start timer

    # Iterate over each entry in the directory with tqdm for progress display
    for label_folder in tqdm(os.listdir(directory), desc="Extracting images and labels"):
        label_path = os.path.join(directory, label_folder)

        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)

                if os.path.isfile(image_path) and image_file.lower().endswith('.jpg'):
                    images.append(image_path)
                    labels.append(label_folder)

    timer.toc()  # Stop timer
    print(f"Image and label extraction completed in {timer.tocvalue()} seconds")

    return images, labels

def save_data_as_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    directory = r"C:\Users\ADMIN\OneDrive\Projects\Jyothi\lfw"
    images, labels = extract_images_and_labels(directory)
    save_data_as_pickle((images, labels), 'images_labels.pkl')
