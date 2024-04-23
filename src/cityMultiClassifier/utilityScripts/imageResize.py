from PIL import Image
import os
import sys

COMMAND_LINE = False

def resize_images(input_folder, output_folder, new_size=(250, 250)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Check if the file is an image
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Open the image file
            with Image.open(os.path.join(input_folder, filename)) as img:
                # print(filename)
                # Resize the image
                resized_img = img.resize(new_size)
                # Save the resized image to the output folder
                resized_img.save(os.path.join(output_folder, filename))


if COMMAND_LINE:
    input_folder = sys.argv[1]
    side_length = int(sys.argv[2])
    os.mkdir('./output_images')

    if __name__ == "__main__":
        output_folder = "./output_images"
        resize_images("./" + input_folder, output_folder, (side_length, side_length))
else:
    side_length = 250

    root = "D:/WPI/Junior Year/ML/CS539_Project/data/cityImages"
    batch = [
        "zurich"
    ]

    for city in batch:
        os.mkdir(root + city + "/output_images")
        resize_images(root + city + "/images", root + city + "/output_images", (side_length, side_length))