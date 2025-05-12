import cv2
import os
from pathlib import Path

def images_to_video(folder_path, output_video_path, fps=1):
    folder = Path(folder_path)
    images = sorted([img for img in folder.iterdir() if img.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get dimensions
    first_image = cv2.imread(str(images[0]))
    height, width, layers = first_image.shape

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)  # white
    thickness = 2

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Couldn't read image {img_path}")
            continue
        resized_img = cv2.resize(img, (width, height))

        # Add image name text (without extension)
        img_name = img_path.stem
        text_size = cv2.getTextSize(img_name, font, font_scale, thickness)[0]
        text_x = 10
        text_y = height - 10

        cv2.putText(resized_img, img_name, (text_x, text_y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

        out.write(resized_img)

    out.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a folder of images into a video with image names overlayed.")
    parser.add_argument("folder", help="Path to the folder containing images")
    parser.add_argument("output", help="Output video file path, e.g. output.mp4")
    args = parser.parse_args()

    images_to_video(args.folder, args.output)
