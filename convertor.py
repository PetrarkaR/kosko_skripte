import subprocess

def resize_video_with_ffmpeg(input_path, output_path, width, height):
    """
    Resizes a video to the specified resolution using ffmpeg.

    Parameters:
    input_path (str): Path to the input video file.
    output_path (str): Path to save the resized video.
    width (int): The target width of the resized video.
    height (int): The target height of the resized video.
    """
    try:
        # ffmpeg command to resize the video
        command = [
            "ffmpeg", 
            "-i", input_path, 
            "-vf", f"scale={width}:{height}", 
            "-c:v", "libx264", 
            "-preset", "slow", 
            "-crf", "18", 
            output_path
        ]
        
        # Run the ffmpeg command
        subprocess.run(command, check=True)
        print(f"Resized video saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Input and output video paths
input_video_path = r"C:\Users\swagg\OneDrive\Desktop\kosko\Video snimci\2024-03-01-113629.webm"
output_video_path = r"C:\Users\swagg\OneDrive\Desktop\kosko\Video snimci\2024-03-01-113629_resized.mp4"

# Resize dimensions
target_width = 320
target_height = 320

# Call the resizing function
resize_video_with_ffmpeg(input_video_path, output_video_path, target_width, target_height)
