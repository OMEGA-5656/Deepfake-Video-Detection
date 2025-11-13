"""
Deepfake Video Detection System - Facial Embedding Analyzer
Frame-Level Facial Embedding Analysis Module (CUDA ENABLED)

This module implements the core deepfake detection algorithm using:
1. MTCNN (Multi-Task Cascaded Convolutional Networks) for face detection and alignment
2. InceptionResnetV1 (pre-trained on VGGFace2) for facial feature extraction
3. Cosine Similarity for comparing facial embeddings across frames

Methodology (as per presentation):
- Video Input: Extract frames using OpenCV
- Face Detection: MTCNN detects faces and aligns them
- Feature Extraction: InceptionResnetV1 generates 512-dimensional embeddings
- Similarity Check: Cosine similarity compares embeddings of consecutive frames
- Temporal Analysis: Identifies sudden drops or unstable similarity values
- Classification: Real videos have stable similarity; Deepfakes show fluctuations

Algorithm:
- MTCNN: Face Detection & Alignment
- InceptionResnetV1: Face Embedding / Feature Extraction (512-dimensional vector)
- Cosine Similarity: Similarity Metric (score of 1 = high similarity, same person)
"""

import cv2  # OpenCV for video processing and frame extraction
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.transforms import functional as F
import time
import traceback
import os
import torch  # Import PyTorch

# Global variables to cache models (loaded once, reused for all analyses)
_mtcnn_model = None
_inception_resnet_model = None
_device = None  # Global variable for the device (CPU or CUDA)


def get_models():
    """
    Load and cache deep learning models for face detection and feature extraction.
    Models are moved to the GPU (cuda) if available.
    
    Models:
    - MTCNN: Multi-Task Cascaded Convolutional Networks for face detection
    - InceptionResnetV1: Pre-trained on VGGFace2 for 512-dimensional embedding generation
    
    Returns:
        tuple: (MTCNN model, InceptionResnetV1 model, device)
    """
    global _mtcnn_model, _inception_resnet_model, _device
    
    if _device is None:
        # Determine if CUDA (GPU) is available
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[Facial Embedding Analyzer] Setting device to: {_device}")
        if not torch.cuda.is_available():
            print("[Facial Embedding Analyzer] WARNING: CUDA not available. Running on CPU (this will be slow).")

    if _mtcnn_model is None or _inception_resnet_model is None:
        print("[Facial Embedding Analyzer] Loading deep learning models...")
        print("[Facial Embedding Analyzer] - MTCNN: Face Detection & Alignment")
        print("[Facial Embedding Analyzer] - InceptionResnetV1: Feature Extraction (VGGFace2)")
        try:
            # Load models and move them to the selected device (GPU or CPU)
            _mtcnn_model = MTCNN(device=_device)
            _inception_resnet_model = InceptionResnetV1(pretrained='vggface2').eval().to(_device)
            print("[Facial Embedding Analyzer] Models loaded successfully")
        except Exception as e:
            print(f"[Facial Embedding Analyzer] Error loading models: {str(e)}")
            traceback.print_exc()
            raise
            
    return _mtcnn_model, _inception_resnet_model, _device


def analyze_video_frames(input_video_path, output_video_path):
    """
    Perform frame-level facial embedding analysis on input video.
    
    Process:
    1. Extract frames from video using OpenCV
    2. Detect faces in each frame using MTCNN
    3. Extract 512-dimensional facial embeddings using InceptionResnetV1
    4. Calculate cosine similarity between consecutive frame embeddings
    5. Identify deepfake patterns through temporal analysis
    6. Annotate frames with detection results (green for real, red for deepfake)
    
    Args:
        input_video_path: Path to input video file
        output_video_path: Path to save processed video with annotations
        
    Returns:
        int: Deepfake detection rate (0-100, where >50 indicates deepfake)
    """
    try:
        start_time = time.time()

        # Detection thresholds (as per methodology)
        COSINE_SIMILARITY_THRESHOLD = 0.99  # Threshold for face similarity (cosine similarity)
        DEEPFAKE_FRAME_THRESHOLD = 15  # Number of consecutive frames with low similarity to classify as deepfake

        print(f"[Facial Embedding Analyzer] Initializing models...")
        # Get models and the device (CPU or CUDA)
        mtcnn, inception_resnet, device = get_models()
        
        # Step 1: Video Input - Extract frames using OpenCV
        print(f"[Facial Embedding Analyzer] Opening video: {input_video_path}")
        video_capture = cv2.VideoCapture(input_video_path)
        
        if not video_capture.isOpened():
            print(f"[Facial Embedding Analyzer] ERROR: Could not open video file {input_video_path}")
            return 50  # Return default detection rate
        
        # Get video properties
        frame_count = 0
        fps = int(video_capture.get(cv2.CAP_PROP_FPS)) or 30  # Frames per second (default: 30)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width == 0 or height == 0:
            print("[Facial Embedding Analyzer] ERROR: Invalid video dimensions")
            video_capture.release()
            return 50
        
        print(f"[Facial Embedding Analyzer] Video properties: {width}x{height} @ {fps} fps")
        
        # Initialize video writer for output (with annotations)
        # Try multiple codecs for compatibility
        codecs_to_try = [
            ('XVID', 'XVID'),       # Good compatibility, works on Windows
            ('MJPG', 'Motion JPEG'), # Widely supported
            ('mp4v', 'MPEG-4'),     # Fallback
            ('avc1', 'H.264/AVC'),  # Best for HTML5 but may not work on Windows
        ]
        
        video_writer = None
        used_codec = None
        for codec_name, codec_desc in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_name)
                # Use .avi extension for XVID/MJPG, .mp4 for others
                if codec_name in ['XVID', 'MJPG']:
                    output_path_avi = output_video_path.replace('.mp4', '.avi')
                    video_writer = cv2.VideoWriter(output_path_avi, fourcc, fps, (width, height))
                else:
                    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    
                if video_writer.isOpened():
                    used_codec = codec_desc
                    print(f"[Facial Embedding Analyzer] Using codec: {codec_desc} ({codec_name})")
                    if codec_name in ['XVID', 'MJPG']:
                        output_video_path = output_path_avi  # Update path to .avi file
                    break
                else:
                    if video_writer:
                        video_writer.release()
            except Exception as e:
                print(f"[Facial Embedding Analyzer] Failed to use codec {codec_name}: {str(e)}")
                if video_writer:
                    video_writer.release()
                continue
        
        if video_writer is None or not video_writer.isOpened():
            print(f"[Facial Embedding Analyzer] ERROR: Could not create output video file")
            # Try to copy original video as fallback
            import shutil
            try:
                shutil.copy2(input_video_path, output_video_path)
                print(f"[Facial Embedding Analyzer] Copied original video as fallback")
                video_capture.release()
                return 50
            except Exception as e:
                print(f"[Facial Embedding Analyzer] Could not copy original video: {str(e)}")
            video_capture.release()
            return 50

        # Initialize tracking variables for temporal analysis
        deepfake_frame_count = 0  # Count of frames with low similarity
        consecutive_deepfake_frames = 0  # Consecutive frames with similarity below threshold
        previous_face_embedding = None  # Store embedding from previous frame for comparison
        frames_between_processing = max(1, int(fps / 7))  # Process every 7th frame (adjustable)
        face_resize_dimensions = (80, 80)  # Resize face to fixed dimensions for embedding

        print("[Facial Embedding Analyzer] Processing video frames...")
        print("[Facial Embedding Analyzer] - Step 2: Face Detection (MTCNN)")
        print("[Facial Embedding Analyzer] - Step 3: Feature Extraction (InceptionResnetV1)")
        print("[Facial Embedding Analyzer] - Step 4: Cosine Similarity Calculation")
        print("[Facial Embedding Analyzer] - Step 5: Temporal Analysis")
        
        # Process video frame by frame
        while video_capture.isOpened():
            ret, frame = video_capture.read()  # Read next frame
            if not ret:  # End of video
                break

            try:
                # Process frames at specified intervals (not every frame for efficiency)
                if frame_count % frames_between_processing == 0:
                    # Step 2: Face Detection using MTCNN
                    face_boxes, _ = mtcnn.detect(frame)

                    if face_boxes is not None and len(face_boxes) > 0:
                        # Get first detected face
                        face_box = face_boxes[0].astype(int)
                        
                        # Ensure box coordinates are within frame bounds
                        face_box[0] = max(0, min(face_box[0], width))
                        face_box[1] = max(0, min(face_box[1], height))
                        face_box[2] = max(0, min(face_box[2], width))
                        face_box[3] = max(0, min(face_box[3], height))
                        
                        if face_box[2] > face_box[0] and face_box[3] > face_box[1]:
                            # Extract face region from frame
                            face_region = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]

                            if face_region.size > 0 and face_region.shape[0] > 0 and face_region.shape[1] > 0:
                                # Resize face to fixed dimensions
                                face_region = cv2.resize(face_region, face_resize_dimensions)
                                
                                # Convert to tensor format and MOVE TENSOR TO GPU
                                face_tensor = F.to_tensor(face_region).unsqueeze(0).to(device)
                                
                                # Step 3: Feature Extraction - Generate 512-dimensional embedding
                                # Use no_grad to speed up inference and save memory
                                with torch.no_grad():
                                    current_face_embedding = inception_resnet(face_tensor).cpu().detach().numpy().flatten()

                                # Step 4: Similarity Check - Compare with previous frame embedding
                                if previous_face_embedding is not None:
                                    # Calculate Cosine Similarity between embeddings
                                    cosine_similarity = np.dot(current_face_embedding, previous_face_embedding) / (
                                                np.linalg.norm(current_face_embedding) * np.linalg.norm(previous_face_embedding))

                                    # Step 5: Temporal Analysis - Check for similarity drops
                                    if cosine_similarity < COSINE_SIMILARITY_THRESHOLD:
                                        consecutive_deepfake_frames += 1
                                    else:
                                        consecutive_deepfake_frames = 0

                                    # Step 6: Classification - Annotate frame based on analysis
                                    if consecutive_deepfake_frames > DEEPFAKE_FRAME_THRESHOLD:
                                        # Deepfake Detected: Draw red border and label
                                        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 0, 255), 2)
                                        cv2.putText(frame, f'Deepfake Detected - Frame {frame_count}', (10, 30),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                        deepfake_frame_count += 1
                                    else:
                                        # Real Frame: Draw green border and label
                                        cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0, 255, 0), 2)
                                        cv2.putText(frame, 'Real Frame', (face_box[0], face_box[1] - 10), 
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

                                # Store current embedding for next frame comparison
                                previous_face_embedding = current_face_embedding
            except Exception as e:
                print(f"[Facial Embedding Analyzer] Error processing frame {frame_count}: {str(e)}")
                traceback.print_exc()
                # Continue processing other frames

            frame_count += 1
            video_writer.write(frame)  # Write annotated frame to output video

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"[Facial Embedding Analyzer] Analysis complete")
        print(f"[Facial Embedding Analyzer] Total execution time: {execution_time:.2f} seconds")
        print(f"[Facial Embedding Analyzer] Processed {frame_count} frames")

        # Release video resources
        video_capture.release()
        video_writer.release()

        if frame_count == 0:
            print("[Facial Embedding Analyzer] ERROR: No frames processed")
            return 50

        # Calculate deepfake detection rate
        # Formula: (deepfake_frames / total_frames) * 1000, capped at 95%
        deepfake_detection_rate = (deepfake_frame_count / frame_count) * 1000

        if deepfake_detection_rate > 100:
            deepfake_detection_rate = 95  # Cap at 95%

        print(f"[Facial Embedding Analyzer] Deepfake detection rate: {int(deepfake_detection_rate)}%")
        print(f"[Facial Embedding Analyzer] Output video saved to: {output_video_path}")
        return int(deepfake_detection_rate)
        
    except Exception as e:
        print(f"[Facial Embedding Analyzer] ERROR: {str(e)}")
        traceback.print_exc()
        # Try to copy original video as fallback
        try:
            import shutil
            if os.path.exists(input_video_path):
                shutil.copy2(input_video_path, output_video_path)
                print(f"[Facial Embedding Analyzer] Copied original video as fallback due to error")
        except:
            pass
        return 50  # Return default detection rate on error