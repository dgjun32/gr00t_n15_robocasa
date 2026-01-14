import cv2
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import shutil


def augment_subgoal_in_video(video_path, subgoal_list, output_path=None):
    """
    Augment video with subgoal text overlays at specified time intervals.
    
    Args:
        video_path (str): Path to input video file
        subgoal_list (List): Dictionary mapping subgoal text to [start_frame, end_frame]
                            e.g., [("subgoal 1", [0, 200]), ("subgoal 2", [200, 400]), ...]
        output_path (str, optional): Path to save output video. If None, saves as {video_path}_annotated.mp4
    
    Returns:
        str: Path to the output video file
    """
    video_path = Path(video_path)
    
    # Set output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_annotated{video_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer - try multiple codecs for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("⚠️ Warning: VideoWriter failed to open, trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.with_suffix('.avi')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
    if not out.isOpened():
        raise ValueError("Failed to create video writer with any codec")
    
    # Create frame-to-subgoal mapping for faster lookup
    frame_to_subgoal = {}
    for subgoal_text, (start_frame, end_frame) in subgoal_list:
        for frame_idx in range(start_frame, min(end_frame, total_frames)):
            frame_to_subgoal[frame_idx] = subgoal_text
    
    print(f"Created subgoal mapping for {len(frame_to_subgoal)} frames")
    
    # Process each frame
    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add subgoal text if exists for this frame
            if frame_idx in frame_to_subgoal:
                subgoal_text = frame_to_subgoal[frame_idx]
                
                # Text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6  # Smaller font for 240x240 resolution
                font_thickness = 2
                text_color = (0, 0, 255)  # Red (BGR format)
                bg_color = (0, 0, 0)  # Black background
                padding = 10
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    subgoal_text, font, font_scale, font_thickness
                )
                
                # Calculate position (top center)
                text_x = (width - text_width) // 2
                text_y = padding + text_height
                
                # Draw background rectangle
                cv2.rectangle(
                    frame,
                    (text_x - padding, text_y - text_height - padding),
                    (text_x + text_width + padding, text_y + baseline + padding),
                    bg_color,
                    -1  # Filled rectangle
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    subgoal_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    # Re-encode with ffmpeg for better compatibility
    temp_path = output_path
    final_path = output_path.parent / f"{output_path.stem}_final{output_path.suffix}"
    
    print(f"Re-encoding with ffmpeg for better compatibility...")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', str(temp_path),
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
            '-pix_fmt', 'yuv420p', str(final_path)
        ], check=True, capture_output=True)
        
        # Replace temp file with final file
        shutil.move(str(final_path), str(temp_path))
        print(f"✅ Annotated video saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ ffmpeg re-encoding failed, using original output")
        print(f"✅ Annotated video saved to: {output_path}")
    except FileNotFoundError:
        print(f"⚠️ ffmpeg not found, using original output")
        print(f"✅ Annotated video saved to: {output_path}")
    
    return str(output_path)


if __name__ == "__main__":
    subgoal_list = [
        ('subgoal 1', [0, 200]),
        ('subgoal 2', [200, 400]),
        ('subgoal 3', [400, 600]),
        ('subgoal 4', [600, 800]),
        ('subgoal 5', [800, 1000]),
        ('subgoal 6', [1000, 1200]),
    ]

    video_path = "/home/dongjun/Isaac-GR00T-robocasa/eval_composite_tasks/MicrowaveThawing/fail_Pick_the_corn_from_the_counter_and_place_it_in_the_microwave._Then_turn_on_the_microwave._3.mp4"
    augment_subgoal_in_video(video_path, subgoal_list)