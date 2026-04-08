#!/bin/bash

# Source root (each subfolder may contain demo.mp4)
SRC_DIR="../mimicgennew_dp_3.10/out"
# Destination for renamed mp4 files
DEST_DIR="$SRC_DIR/videos"

mkdir -p "$DEST_DIR"

# Walk immediate subdirectories
for subdir in "$SRC_DIR"/*/; do
    # Subfolder name
    folder_name=$(basename "$subdir")
    demo_path="$subdir/demo.mp4"

    # Rename and move demo.mp4 if present
    if [[ -f "$demo_path" ]]; then
        new_name="${folder_name}.mp4"
        mv "$demo_path" "$DEST_DIR/$new_name"
        echo "Moved: $demo_path → $DEST_DIR/$new_name"
    else
        echo "Warning: $demo_path not found"
    fi
done

echo " All done! MP4 files are in: $DEST_DIR"