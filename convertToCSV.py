import pandas
import cv2
from moviepy import VideoFileClip

def videoTrim(input_video_path, output_video_path, sequence_multiple):
    video = VideoFileClip(input_video_path)
    fps = video.fps
    duration = video.duration
    total_num_frames = int(fps * duration)
    desired_num_frames = (total_num_frames // sequence_multiple) * sequence_multiple
    new_duration = desired_num_frames / fps
    trimmed_video = video.subclip(0, new_duration)
    trimmed_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    video.close()
    trimmed_video.close()

videoTrim("truth/output_video_20250505_143329.mp4", "truth_new/test.mp4", 30)
