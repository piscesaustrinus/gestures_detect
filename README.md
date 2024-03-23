# gestures_detect

# Gesture Detection using MobileNetV2 and TensorFlow

This project is a Streamlit web application for detecting gestures in video sequences using MobileNetV2 as the feature extraction model.

## Overview

The application allows users to upload two videos:
1. Gesture Video: Contains the gesture that needs to be detected in the test video.
2. Test Video: The video in which the gesture needs to be detected.

After uploading the videos, the user can click the "Process Videos" button to analyze the videos. The application extracts features from both videos using MobileNetV2 and compares the features to determine if the gesture from the gesture video is present in the test video. It then annotates frames in the test video where the gesture is detected with "DETECTED" in bright green.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gesture-detection.git
