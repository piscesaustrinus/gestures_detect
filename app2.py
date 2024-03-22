import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import tempfile
import os

# Load MobileNetV2 model from TensorFlow Hub
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
model = hub.load(model_url)

def extract_features(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess frame
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0 # Ensure the frame is normalized to [0, 1]
        frame = np.expand_dims(frame, axis=0) # Add batch dimension
        frame = tf.convert_to_tensor(frame, dtype=tf.float32) # Convert to tf.float32 tensor

        # Extract features
        features = model(frame)
        frames.append(features)

    video.release()

    # Aggregate features (e.g., average)
    features_np = np.array(frames)
    aggregated_features = np.mean(features_np, axis=0)

    return aggregated_features

def compare_features(features1, features2):
    # Normalize the feature vectors
    features1_norm = features1 / tf.norm(features1)
    features2_norm = features2 / tf.norm(features2)

    # Compute the dot product (cosine similarity)
    similarity = tf.reduce_sum(features1_norm * features2_norm)

    return similarity.numpy()

def annotate_frames(test_video_path, gesture_features, threshold=0.5):
    video = cv2.VideoCapture(test_video_path)
    annotated_frames = []
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess frame
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        frame = tf.convert_to_tensor(frame, dtype=tf.float32) # Convert to tf.float32 tensor

        # Extract features
        test_features = model(frame)

        # Compare features
        similarity = compare_features(gesture_features, test_features)
        if similarity > threshold:
            # Convert frame back to a NumPy array with the correct data type and number of channels
            frame = frame[0].numpy() * 255 # Remove batch dimension and convert back to uint8
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert from RGB to BGR

            # Annotate frame
            cv2.putText(frame, "DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            annotated_frames.append(frame)
        else:
            # Convert frame back to a NumPy array with the correct data type and number of channels
            frame = frame[0].numpy() * 255 # Remove batch dimension and convert back to uint8
            frame = frame.astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert from RGB to BGR

            # Annotate frame
            cv2.putText(frame, "NOT DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            annotated_frames.append(frame)

        frame_count += 1
        if len(annotated_frames) >= 10: # Stop after finding 10 frames
            break

    video.release()
    return annotated_frames


    video.release()
    return annotated_frames



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
            annotated_frames = annotate_frames(test_video_path, gesture_features)

            # Display annotated frames
            for i, frame in enumerate(annotated_frames):
                st.image(frame, caption=f"Annotated Frame {i+1}", channels="BGR", use_column_width=True)

            # Delete temporary files
            os.remove(gesture_video_path)
            os.remove(test_video_path)

        else:
            st.warning("Please upload both a gesture video and a test video.")

if __name__ == "__main__":
    main()
