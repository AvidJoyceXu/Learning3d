from PIL import Image, ImageSequence
import sys

def reduce_gif_frames(input_gif, output_gif):
    # Open the input GIF
    with Image.open(input_gif) as img:
        # Extract frames and keep every alternate frame
        frames = [frame.copy() for i, frame in enumerate(ImageSequence.Iterator(img)) if i % 16 == 0]
        
        # Save the new GIF with reduced frames
        if frames:
            frames[0].save(output_gif, save_all=True, append_images=frames[1:], loop=0, duration=img.info['duration'])
            print(f"Reduced GIF saved as {output_gif}")
        else:
            print("No frames to save.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python reduce_gif.py <input_gif> <output_gif>")
    else:
        reduce_gif_frames(sys.argv[1], sys.argv[2])