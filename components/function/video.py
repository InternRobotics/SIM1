import os
import cv2
import re
from pathlib import Path

def natural_sort_key(path):
    """Natural sort that handles numeric order correctly"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(path))]

def find_image_folders(root_folder):
    """Find folders that contain images"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    image_folders = []
    
    print(f"Scanning root folder: {root_folder}")
    
    for root, dirs, files in os.walk(root_folder):
        # Check whether current folder contains image files
        has_images = any(
            any(file.lower().endswith(ext) for ext in image_extensions)
            for file in files
        )
        
        if has_images:
            image_folders.append(root)
            print(f"Found image folder: {root} ({len(files)} files)")
    
    return image_folders

def folder_to_video(input_folder, fps=30):
    """Convert images in a single folder to video"""
    # Supported image formats
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    # Collect all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    # Use natural sort to preserve numeric order
    image_files.sort(key=natural_sort_key)
    
    if not image_files:
        print(f"  Warning: folder {input_folder} contains no image files")
        return False
    
    # Output video path (named after the folder)
    folder_name = Path(input_folder).name
    output_video = Path(input_folder) / f"{folder_name}.mp4"
    
    print(f"  Processing folder: {input_folder}")
    print(f"  Found {len(image_files)} images")
    print(f"  Output video: {output_video}")
    
    try:
        # Read first image to get frame size
        first_image = cv2.imread(str(image_files[0]))
        if first_image is None:
            print(f"  Error: cannot read first image {image_files[0]}")
            return False
        
        height, width = first_image.shape[:2]
        print(f"  Image size: {width} x {height}")
        
        # Open video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        # Process each image
        success_count = 0
        prev_img = None
        
        for i, image_path in enumerate(image_files):
            img = cv2.imread(str(image_path))
            if img is not None:
                # Ensure consistent frame size
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                
                # Check whether image is almost same as previous frame
                if prev_img is not None:
                    # Convert to grayscale for diff
                    gray1 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Compute difference
                    diff = cv2.absdiff(gray1, gray2)
                    non_zero_count = cv2.countNonZero(diff)
                    
                    # Treat as duplicate frame if difference is small
                    if non_zero_count < 100:
                        print(f"    Skip near-duplicate frame: {image_path.name} (diff pixels: {non_zero_count})")
                        continue
                
                prev_img = img.copy()
                out.write(img)
                success_count += 1
                
                if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                    progress = (i + 1) / len(image_files) * 100
                    print(f"    Progress: {i+1}/{len(image_files)} ({progress:.1f}%)")
            else:
                print(f"    Warning: failed to read image {image_path.name}")
        
        out.release()
        print(f"  Done! Successfully processed {success_count}/{len(image_files)} images")
        print(f"  Video length: {success_count/fps:.2f} seconds")
        print(f"  Video saved: {output_video}\n")
        return True
        
    except Exception as e:
        print(f"  Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_images_to_video(root_folder, fps=30):
    """
    Batch mode: scan all subfolders under root and convert each image folder to video.

    Args:
        root_folder: Root directory to scan for image subfolders.
        fps: Frames per second for output video.
    """
    if not os.path.exists(root_folder):
        print(f"Error: root folder {root_folder} does not exist")
        return

    print("=" * 60)
    print("Batch image-to-video tool")
    print("=" * 60)
    print(f"Root folder: {root_folder}")
    print(f"FPS: {fps} FPS\n")

    image_folders = find_image_folders(root_folder)

    if not image_folders:
        print("No image folders found.")
        return

    print(f"\nFound {len(image_folders)} folders with images")
    print("Start batch processing...\n")

    success_count = 0
    for i, folder in enumerate(image_folders, 1):
        print(f"[{i}/{len(image_folders)}] ", end="")
        if folder_to_video(folder, fps):
            success_count += 1

    print("=" * 60)
    print("Batch processing finished!")
    print(f"Succeeded: {success_count}/{len(image_folders)} folders")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert image folders to video")
    parser.add_argument("root_folder", type=str, help="Root folder to scan for image subfolders")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    args = parser.parse_args()
    batch_images_to_video(args.root_folder, args.fps)