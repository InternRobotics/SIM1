import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_npz_file(npz_file_path):
    """
    Analyze frame rate and content of NPZ file
    
    Parameters:
    npz_file_path: Path to NPZ file
    """
    try:
        # Check if file exists
        if not os.path.exists(npz_file_path):
            print(f"Error: File '{npz_file_path}' does not exist")
            return
        
        # Load NPZ file
        print(f"Analyzing file: {npz_file_path}")
        print(f"File size: {os.path.getsize(npz_file_path) / 1024 / 1024:.2f} MB")
        print(f"Modified time: {datetime.fromtimestamp(os.path.getmtime(npz_file_path))}")
        print("-" * 50)
        
        npz_data = np.load(npz_file_path, allow_pickle=True)
        
        # Display array information in the file
        print("NPZ file contains the following arrays:")
        print("=" * 50)
        
        for key in npz_data.files:
            data = npz_data[key]
            print(f"\nArray name: '{key}'")
            print(f"  Data type: {data.dtype}")
            print(f"  Array shape: {data.shape}")
            print(f"  Array dimensions: {data.ndim}D")
            print(f"  Total elements: {data.size:,}")
            
            # Display sample data
            if data.ndim > 0 and data.size > 0:
                print(f"  Data samples:")
                if data.ndim == 1:
                    print(f"    First 5: {data[:5]}")
                    print(f"    Last 5: {data[-5:]}")
                elif data.ndim == 2:
                    print(f"    First 3 rows:\n{data[:3]}")
                else:
                    print(f"    First element shape: {data[0].shape}")
            
            # Numerical statistics (if numeric data)
            if np.issubdtype(data.dtype, np.number):
                print(f"  Value range: [{data.min():.4f}, {data.max():.4f}]")
                print(f"  Mean: {data.mean():.4f}")
                print(f"  Standard deviation: {data.std():.4f}")
        
        # Analyze frame rate (assuming timestamps or frame indices are included)
        print("\n" + "=" * 50)
        print("Frame Rate Analysis:")
        print("-" * 50)
        
        frame_rate_info, time_array = analyze_frame_rate(npz_data)
        for key, info in frame_rate_info.items():
            print(f"\nBased on array '{key}':")
            for item, value in info.items():
                print(f"  {item}: {value}")
        
        # Analyze constant joint_q indices
        if 'joint_q' in npz_data.files:
            print("\n" + "=" * 50)
            print("Constant Joint_q Analysis:")
            print("-" * 50)
            analyze_constant_joint_q(npz_data['joint_q'])
        
        # Display file structure summary
        print("\n" + "=" * 50)
        print("File Structure Summary:")
        print("-" * 50)
        print(f"Total number of arrays: {len(npz_data.files)}")
        for key in npz_data.files:
            data = npz_data[key]
            print(f"  - {key}: {data.shape} ({data.dtype})")
        
        # Plot joint_q and openness curves
        if 'joint_q' in npz_data.files or 'openness' in npz_data.files:
            print("\n" + "=" * 50)
            print("Plotting Curves:")
            print("-" * 50)
            plot_joint_q_and_openness(npz_data, time_array, npz_file_path)
        
        # Close file
        npz_data.close()
        
    except Exception as e:
        print(f"Error during analysis: {e}")

def analyze_constant_joint_q(joint_q_data, tolerance=1e-6):
    """
    Analyze and print indices of joint_q that remain constant
    
    Parameters:
    joint_q_data: joint_q array data
    tolerance: tolerance for considering values as constant
    """
    if joint_q_data.ndim != 2:
        print(f"joint_q data is {joint_q_data.ndim}D, expected 2D array")
        return
    
    num_joints = joint_q_data.shape[1]
    num_frames = joint_q_data.shape[0]
    
    print(f"Analyzing {num_joints} joints over {num_frames} frames")
    print(f"Tolerance for constant detection: {tolerance}")
    print("-" * 40)
    
    constant_joints = []
    varying_joints = []
    
    for joint_idx in range(num_joints):
        joint_values = joint_q_data[:, joint_idx]
        
        # Check if all values are the same within tolerance
        if np.all(np.abs(joint_values - joint_values[0]) < tolerance):
            constant_joints.append(joint_idx)
            print(f"Joint {joint_idx}: CONSTANT - Value = {joint_values[0]:.6f}")
        else:
            varying_joints.append(joint_idx)
            min_val = np.min(joint_values)
            max_val = np.max(joint_values)
            variation = max_val - min_val
            print(f"Joint {joint_idx}: VARYING - Range = [{min_val:.6f}, {max_val:.6f}], Variation = {variation:.6f}")
    
    # Summary
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"Constant joints: {len(constant_joints)}")
    if constant_joints:
        print(f"Constant joint indices: {constant_joints}")
        print(f"Constant joint values: {[joint_q_data[0, idx] for idx in constant_joints]}")
    
    print(f"Varying joints: {len(varying_joints)}")
    if varying_joints:
        print(f"Varying joint indices: {varying_joints}")
    
    # Additional analysis: detect joints that become constant after some point
    print("\n" + "-" * 40)
    print("CONSTANT AFTER ANALYSIS:")
    analyze_constant_after_joints(joint_q_data, tolerance)
    
    return constant_joints, varying_joints

def analyze_constant_after_joints(joint_q_data, tolerance=1e-6):
    """
    Analyze joints that become constant after a certain frame
    
    Parameters:
    joint_q_data: joint_q array data
    tolerance: tolerance for considering values as constant
    """
    num_joints = joint_q_data.shape[1]
    num_frames = joint_q_data.shape[0]
    
    constant_after_info = []
    
    for joint_idx in range(num_joints):
        joint_values = joint_q_data[:, joint_idx]
        
        # Find the first frame where the joint becomes constant till the end
        for frame_idx in range(1, num_frames):
            # Check if from this frame to the end, values are constant
            remaining_values = joint_values[frame_idx:]
            if len(remaining_values) > 1 and np.all(np.abs(remaining_values - remaining_values[0]) < tolerance):
                # Check if this is different from the initial value
                initial_value = joint_values[0]
                constant_value = remaining_values[0]
                
                if np.abs(initial_value - constant_value) > tolerance:
                    constant_after_info.append({
                        'joint_idx': joint_idx,
                        'constant_after_frame': frame_idx,
                        'initial_value': initial_value,
                        'constant_value': constant_value,
                        'change_magnitude': abs(constant_value - initial_value)
                    })
                break
    
    if constant_after_info:
        print("Joints that become constant after certain frames:")
        for info in constant_after_info:
            print(f"  Joint {info['joint_idx']}: becomes constant after frame {info['constant_after_frame']} "
                  f"(value changes from {info['initial_value']:.6f} to {info['constant_value']:.6f}, "
                  f"Δ = {info['change_magnitude']:.6f})")
    else:
        print("No joints found that become constant after certain frames")

def analyze_frame_rate(npz_data):
    """
    Analyze frame rate information in NPZ data
    
    Parameters:
    npz_data: Loaded NPZ data
    
    Returns:
    tuple: (frame rate analysis results, time array)
    """
    frame_rate_info = {}
    time_array = None
    
    for key in npz_data.files:
        data = npz_data[key]
        info = {}
        
        # Check if it's a timestamp array
        if (data.ndim == 1 and data.size > 1 and 
            (np.issubdtype(data.dtype, np.number) or 
             np.issubdtype(data.dtype, np.datetime64))):
            
            # Calculate time intervals
            if np.issubdtype(data.dtype, np.datetime64):
                # Handle datetime64 type
                time_diffs = np.diff(data).astype('timedelta64[ms]').astype(float) / 1000.0
                time_array = data
            else:
                # Handle numeric types (assumed to be seconds or milliseconds)
                time_diffs = np.diff(data)
                # If values are large, likely milliseconds
                if data.max() > 1e10:  # Likely millisecond timestamps
                    time_diffs = time_diffs / 1000.0
                    time_array = data / 1000.0  # Convert to seconds
                else:
                    time_array = data
            
            if len(time_diffs) > 0:
                avg_interval = np.mean(time_diffs)
                std_interval = np.std(time_diffs)
                fps = 1.0 / avg_interval if avg_interval > 0 else 0
                
                info["Total frames"] = len(data)
                info["Average time interval"] = f"{avg_interval:.4f} seconds"
                info["Time interval std"] = f"{std_interval:.4f} seconds"
                info["Estimated frame rate"] = f"{fps:.2f} FPS"
                info["Min time interval"] = f"{np.min(time_diffs):.4f} seconds"
                info["Max time interval"] = f"{np.max(time_diffs):.4f} seconds"
                
                frame_rate_info[key] = info
        
        # Check if it's video frame data
        elif data.ndim >= 3 and 'frame' in key.lower():
            info["Total frames"] = data.shape[0]
            info["Frame size"] = f"{data.shape[1]} x {data.shape[2]}"
            if data.ndim == 4:
                info["Channels"] = data.shape[3]
            info["Data type"] = "Video frames"
            frame_rate_info[key] = info
    
    return frame_rate_info, time_array

def plot_joint_q_and_openness(npz_data, time_array, file_path):
    """
    Plot joint_q and openness curves
    
    Parameters:
    npz_data: Loaded NPZ data
    time_array: Time array
    file_path: Original file path
    """
    has_joint_q = 'joint_q' in npz_data.files
    has_openness = 'openness' in npz_data.files
    
    if not has_joint_q and not has_openness:
        print("Arrays 'joint_q' or 'openness' not found")
        return
    
    # Create time axis
    if time_array is not None:
        time_axis = time_array
        xlabel = "Time (seconds)"
    else:
        # If no timestamps, use frame indices
        if has_joint_q:
            time_axis = np.arange(len(npz_data['joint_q']))
        elif has_openness:
            time_axis = np.arange(len(npz_data['openness']))
        xlabel = "Frame Index"
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'NPZ File Data Analysis: {os.path.basename(file_path)}', fontsize=14, fontweight='bold')
    
    # Plot joint_q curve
    if has_joint_q:
        joint_q_data = npz_data['joint_q']
        ax1 = axes[0]
        
        if joint_q_data.ndim == 1:
            # 1D data
            ax1.plot(time_axis[:len(joint_q_data)], joint_q_data, 'b-', linewidth=2, label='joint_q')
            ax1.set_ylabel('joint_q Value', fontsize=12)
        elif joint_q_data.ndim == 2:
            # 2D data, plot all dimensions
            for i in range(joint_q_data.shape[1]):
                ax1.plot(time_axis[:len(joint_q_data)], joint_q_data[:, i], 
                        linewidth=1.5, label=f'joint_q[{i}]')
            ax1.set_ylabel('joint_q Values by Dimension', fontsize=12)
        
        ax1.set_title('joint_q Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlabel(xlabel, fontsize=10)
    
    # Plot openness curve
    if has_openness:
        openness_data = npz_data['openness']
        ax2 = axes[1] if has_joint_q else axes[0]
        
        if openness_data.ndim == 1:
            # 1D data
            ax2.plot(time_axis[:len(openness_data)], openness_data, 'r-', linewidth=2, label='openness')
            ax2.set_ylabel('openness Value', fontsize=12)
        elif openness_data.ndim == 2:
            # 2D data, plot all dimensions
            for i in range(openness_data.shape[1]):
                ax2.plot(time_axis[:len(openness_data)], openness_data[:, i], 
                        linewidth=1.5, label=f'openness[{i}]')
            ax2.set_ylabel('openness Values by Dimension', fontsize=12)
        
        title = 'openness Over Time' if has_joint_q else 'openness Over Time'
        ax2.set_title(title, fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlabel(xlabel, fontsize=10)
    
    # Adjust layout if only one data array exists
    if not has_joint_q or not has_openness:
        fig.delaxes(axes[1])
    
    plt.tight_layout()
    
    # Save image
    output_filename = os.path.splitext(file_path)[0] + '_analysis_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    
    # Display image
    plt.show()
    
    # Print data statistics
    print("\nData Statistics:")
    if has_joint_q:
        joint_q_data = npz_data['joint_q']
        print(f"joint_q - Shape: {joint_q_data.shape}, Range: [{joint_q_data.min():.4f}, {joint_q_data.max():.4f}]")
    if has_openness:
        openness_data = npz_data['openness']
        print(f"openness - Shape: {openness_data.shape}, Range: [{openness_data.min():.4f}, {openness_data.max():.4f}]")

def batch_analyze_npz_files(directory_path):
    """
    Batch analyze all NPZ files in directory
    
    Parameters:
    directory_path: Directory path
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist")
        return
    
    npz_files = [f for f in os.listdir(directory_path) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"No NPZ files found in directory '{directory_path}'")
        return
    
    print(f"Found {len(npz_files)} NPZ files:")
    for i, file in enumerate(npz_files, 1):
        print(f"{i}. {file}")
    
    print("\n" + "=" * 80)
    
    for file in npz_files:
        file_path = os.path.join(directory_path, file)
        analyze_npz_file(file_path)
        print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze NPZ file contents and frame rate")
    parser.add_argument("input", type=str, help="Path to a single .npz file or a directory of .npz files")
    parser.add_argument("--batch", action="store_true", help="Batch analyze all .npz files in a directory")
    args = parser.parse_args()

    if args.batch or os.path.isdir(args.input):
        batch_analyze_npz_files(args.input)
    else:
        analyze_npz_file(args.input)