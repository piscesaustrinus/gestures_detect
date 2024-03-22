import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import tempfile
import os

# Load the I3D model from TensorFlow Hub
model_url = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"
model = hub.load(model_url)

# Function to preprocess frames for the I3D model
def preprocess_frame_for_i3d(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return frame

# Function to extract features from a video
def extract_features(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame_for_i3d(frame)
        frames.append(preprocessed_frame)

    video.release()

    frames_np = np.array(frames)
    frames_np = np.expand_dims(frames_np, axis=0)

    logits = model.signatures['default'](tf.constant(frames_np, dtype=tf.float32))
    features = tf.nn.softmax(logits['default'])

    return features.numpy()

# Function to check if gesture is present in the test video
def is_gesture_present(gesture_features, test_features, threshold=0.5):
    similarity = np.dot(gesture_features, test_features.T) / (np.linalg.norm(gesture_features) * np.linalg.norm(test_features))
    return similarity > threshold

# Streamlit app
def main():
    st.title("Gesture Detection in Video Sequences")

    gesture_video_file = st.file_uploader("Upload your gesture video", type=['mp4'])
    test_video_file = st.file_uploader("Upload the test video", type=['mp4'])

    if st.button("Process Videos"):
        if gesture_video_file is not None and test_video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_gesture_video:
                temp_gesture_video.write(gesture_video_file.getvalue())
                gesture_video_path = temp_gesture_video.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_test_video:
                temp_test_video.write(test_video_file.getvalue())
                test_video_path = temp_test_video.name

            gesture_features = extract_features(gesture_video_path)
            test_features = extract_features(test_video_path)

            # Check if the gesture is present in the test video
            if is_gesture_present(gesture_features, test_features):
                st.success("Gesture is DETECTED in the test video.")
            else:
                st.warning("Gesture is NOT DETECTED in the test video.")

            # Delete temporary files
            os.remove(gesture_video_path)
            os.remove(test_video_path)

        else:
            st.warning("Please upload both a gesture video and a test video.")

if __name__ == "__main__":
    main()
