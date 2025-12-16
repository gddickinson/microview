import cv2
from PIL import Image
import os

def avi_to_tiff_stack(input_avi, output_tiff):
    # Open the AVI file
    video = cv2.VideoCapture(input_avi)

    frames = []
    frame_count = 0

    while True:
        # Read a frame from the video
        ret, frame = video.read()

        # If we've reached the end of the video, break the loop
        if not ret:
            break

        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a PIL Image from the frame
        image = Image.fromarray(frame_rgb)

        # Append the image to our list of frames
        frames.append(image)

        frame_count += 1

    # Release the video capture object
    video.release()

    # Save all frames as a single multi-page TIFF file
    if frames:
        frames[0].save(output_tiff, format="TIFF", save_all=True, append_images=frames[1:])
        print(f"Conversion complete. {frame_count} frames saved as a single TIFF stack.")
    else:
        print("No frames were extracted from the video.")

# Example usage
input_avi = "/Users/george/Downloads/3.avi"
output_tiff = "/Users/george/Downloads/tiff_stack.tiff"

avi_to_tiff_stack(input_avi, output_tiff)
