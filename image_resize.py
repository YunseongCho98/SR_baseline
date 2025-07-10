# Code for image 1/4 resizing and saving
import os
from PIL import Image
import argparse

def resize_and_save_images(input_dir, output_dir, scale):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            new_size = (width // scale, height // scale)
            resized_img = img.resize(new_size, Image.BICUBIC)
            resized_img.save(os.path.join(output_dir, filename))
            print(f"Resized and saved: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images in a directory.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save resized images.")
    parser.add_argument('--scale', type=int, default=4, help="Scale factor for resizing images.")

    args = parser.parse_args()

    resize_and_save_images(args.input_dir, args.output_dir, args.scale)
    print("Image resizing completed.")