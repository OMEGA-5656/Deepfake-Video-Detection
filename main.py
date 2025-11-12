"""
Deepfake Video Detection System - AuthentiScan
Flask Web Application for Frame-Level Facial Embedding Analysis

This module implements the web interface for the deepfake detection system.
It handles video uploads, processes them through the facial embedding analyzer,
and displays results with visual annotations.

Methodology:
1. Video Input: User uploads video through web interface
2. Face Detection: MTCNN model detects and aligns faces in frames
3. Feature Extraction: InceptionResnetV1 generates 512-dimensional embeddings
4. Similarity Analysis: Cosine similarity compares embeddings across frames
5. Classification: Temporal analysis identifies deepfake patterns
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
from datetime import datetime
import json
from time import time as current_time
import traceback
import subprocess
import cv2
import mimetypes
from urllib.parse import quote, unquote

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'authentiscan-secret-key-2024'  # Secret key for session management

# Configuration
PROCESSED_VIDEOS_FOLDER = 'static/videos'  # Folder for processed videos with annotations
app.config['UPLOAD_FOLDER'] = PROCESSED_VIDEOS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Maximum file size: 500MB

# Ensure processed videos directory exists
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)

# Pre-load deep learning models at startup for faster processing
print("[AuthentiScan] Initializing facial embedding analysis models...")
try:
    from facial_embedding_analyzer import get_models
    get_models()
    print("[AuthentiScan] Models loaded successfully (MTCNN + InceptionResnetV1)")
except Exception as e:
    print(f"[AuthentiScan] Warning: Could not pre-load models: {str(e)}")
    print("[AuthentiScan] Models will be loaded on first video analysis request")


@app.route('/')
def index():
    """
    Render the main upload page for AuthentiScan.
    Users can upload videos for deepfake detection analysis.
    """
    return render_template('index.html')


@app.route('/video/<filename>')
def serve_video(filename):
    """
    Serve processed video files with proper MIME type and headers for HTML5 playback.
    
    Args:
        filename: Name of the video file to serve
        
    Returns:
        Video file with appropriate headers or error response
    """
    try:
        # URL decode the filename to handle special characters
        filename = unquote(filename)
        print(f"[VIDEO SERVE] Requested video: {filename}")
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(video_path) and os.path.isfile(video_path):
            file_size = os.path.getsize(video_path)
            print(f"[VIDEO SERVE] File size: {file_size / 1024:.2f} KB")
            
            # Determine MIME type based on file extension
            mimetype, encoding = mimetypes.guess_type(video_path)
            
            if not mimetype:
                # Fallback MIME types for different video formats
                if filename.lower().endswith('.avi'):
                    mimetype = 'video/x-msvideo'
                elif filename.lower().endswith('.webm'):
                    mimetype = 'video/webm'
                elif filename.lower().endswith('.mov'):
                    mimetype = 'video/quicktime'
                else:
                    mimetype = 'video/mp4'
            
            # Serve file with proper headers for video streaming
            response = send_file(video_path, mimetype=mimetype, conditional=True)
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            
            print(f"[VIDEO SERVE] Successfully serving video with MIME type: {mimetype}")
            return response
        else:
            print(f"[VIDEO SERVE] ERROR: Video file not found at {video_path}")
            # List available files for debugging
            upload_dir = app.config['UPLOAD_FOLDER']
            if os.path.exists(upload_dir):
                files = os.listdir(upload_dir)
                print(f"[VIDEO SERVE] Available files: {files}")
            return "Video not found", 404
    except Exception as e:
        print(f"[VIDEO SERVE] ERROR: {str(e)}")
        traceback.print_exc()
        return "Error serving video", 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle video file upload and initiate frame-level facial embedding analysis.
    
    Process:
    1. Validate uploaded file
    2. Save video to processed_videos folder
    3. Run facial embedding analysis (MTCNN + InceptionResnetV1)
    4. Convert output to web-compatible format
    5. Redirect to results page
    
    Returns:
        Redirect to results page with analysis data
    """
    try:
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        # Validate file extension
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            flash('Invalid file type. Please upload a video file (.mp4, .avi, .mov, .mkv)')
            return redirect(request.url)

        if file:
            # Clean up old processed videos before saving new one
            print(f"[UPLOAD] Cleaning up previous analysis results...")
            try:
                upload_dir = app.config['UPLOAD_FOLDER']
                if os.path.exists(upload_dir):
                    for old_file in os.listdir(upload_dir):
                        old_file_path = os.path.join(upload_dir, old_file)
                        try:
                            if os.path.isfile(old_file_path):
                                os.remove(old_file_path)
                                print(f"[UPLOAD] Removed old file: {old_file}")
                        except Exception as e:
                            print(f"[UPLOAD] Warning: Could not delete {old_file}: {str(e)}")
                print(f"[UPLOAD] Cleanup complete")
            except Exception as e:
                print(f"[UPLOAD] Warning: Error during cleanup: {str(e)}")
            
            # Generate unique filename with timestamp
            timestamp = int(current_time())
            filename = f"uploaded_video_{timestamp}.mp4"
            # Sanitize filename (remove control characters)
            filename = ''.join(c for c in filename if c.isprintable() or c in '.-_')
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"[UPLOAD] Saving uploaded video: {video_path}")
            file.save(video_path)
            
            if not os.path.exists(video_path):
                flash('Error saving file')
                return redirect(request.url)

            # Path for processed video with annotations
            processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], "analyzed_" + filename)

            # Step 3: Run Frame-Level Facial Embedding Analysis
            print(f"[UPLOAD] Starting frame-level facial embedding analysis...")
            print(f"[UPLOAD] Using MTCNN for face detection and InceptionResnetV1 for feature extraction")
            try:
                from facial_embedding_analyzer import analyze_video_frames
                
                # Analyze video using cosine similarity on facial embeddings
                deepfake_detection_rate = analyze_video_frames(video_path, processed_video_path)
                print(f"[UPLOAD] Analysis complete. Deepfake detection rate: {deepfake_detection_rate}%")

            except Exception as e:
                print(f"[UPLOAD] Error during facial embedding analysis: {str(e)}")
                traceback.print_exc()
                error_msg = str(e)
                # Truncate very long error messages
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                flash(f'Error processing video: {error_msg}')
                # Return default detection rate on error
                deepfake_detection_rate = 50

            # Find the actual processed video file (may be .avi or .mp4 depending on codec)
            processed_video_basename = None
            base_processed_name = "analyzed_" + filename.rsplit('.', 1)[0]  # Remove .mp4 extension
            avi_path = os.path.join(app.config['UPLOAD_FOLDER'], base_processed_name + '.avi')
            mp4_path = os.path.join(app.config['UPLOAD_FOLDER'], base_processed_name + '.mp4')
            
            # Convert .avi to .mp4 if needed for web compatibility
            if os.path.exists(avi_path) and os.path.getsize(avi_path) > 0:
                print(f"[UPLOAD] Converting .avi to .mp4 for web compatibility...")
                try:
                    cap = cv2.VideoCapture(avi_path)
                    if cap.isOpened():
                        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        # Try to use ffmpeg if available (best for HTML5 compatibility)
                        use_ffmpeg = False
                        try:
                            result = subprocess.run(['ffmpeg', '-version'], 
                                                  capture_output=True, 
                                                  timeout=2)
                            if result.returncode == 0:
                                use_ffmpeg = True
                                print(f"[UPLOAD] Using ffmpeg for conversion (H.264 codec)")
                        except (FileNotFoundError, subprocess.TimeoutExpired):
                            print(f"[UPLOAD] ffmpeg not available, using OpenCV")
                        
                        conversion_success = False
                        if use_ffmpeg:
                            # Use ffmpeg with H.264 codec for best HTML5 compatibility
                            try:
                                ffmpeg_cmd = [
                                    'ffmpeg', '-y', '-i', avi_path,
                                    '-c:v', 'libx264', '-preset', 'fast',
                                    '-crf', '23', '-c:a', 'aac', '-b:a', '128k',
                                    '-movflags', '+faststart',  # Enable fast start for web playback
                                    mp4_path
                                ]
                                result = subprocess.run(ffmpeg_cmd, 
                                                      capture_output=True, 
                                                      timeout=300)  # 5 minute timeout
                                if result.returncode == 0 and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                                    os.remove(avi_path)
                                    print(f"[UPLOAD] Successfully converted to .mp4 using ffmpeg")
                                    processed_video_basename = base_processed_name + '.mp4'
                                    cap.release()
                                    conversion_success = True
                                else:
                                    print(f"[UPLOAD] ffmpeg conversion failed, trying OpenCV")
                            except Exception as e:
                                print(f"[UPLOAD] Error using ffmpeg: {str(e)}, trying OpenCV")
                        
                        # Fallback to OpenCV conversion if ffmpeg failed or not available
                        if not conversion_success:
                            # Try multiple codecs for HTML5 compatibility
                            codecs_to_try = [
                                ('avc1', 'H.264/AVC'),  # Best for HTML5
                                ('X264', 'H.264'),       # Alternative H.264
                                ('H264', 'H.264'),       # Another H.264 variant
                                ('mp4v', 'MPEG-4'),     # Fallback
                            ]
                            
                            out = None
                            for codec_name, codec_desc in codecs_to_try:
                                try:
                                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                                    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
                                    if out.isOpened():
                                        print(f"[UPLOAD] Using codec for conversion: {codec_desc} ({codec_name})")
                                        break
                                    else:
                                        if out:
                                            out.release()
                                except Exception as e:
                                    print(f"[UPLOAD] Failed to use codec {codec_name}: {str(e)}")
                                    if out:
                                        out.release()
                                    continue
                            
                            if not out or not out.isOpened():
                                # Final fallback to mp4v
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
                                print(f"[UPLOAD] Using fallback codec: mp4v")
                            
                            if out and out.isOpened():
                                frame_count = 0
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    out.write(frame)
                                    frame_count += 1
                                
                                out.release()
                                cap.release()
                                
                                # Delete the .avi file after successful conversion
                                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                                    os.remove(avi_path)
                                    print(f"[UPLOAD] Successfully converted .avi to .mp4 ({frame_count} frames)")
                                    processed_video_basename = base_processed_name + '.mp4'
                                else:
                                    print(f"[UPLOAD] Conversion failed, keeping .avi file")
                                    processed_video_basename = base_processed_name + '.avi'
                            else:
                                print(f"[UPLOAD] Could not create video writer, keeping .avi file")
                                cap.release()
                                processed_video_basename = base_processed_name + '.avi'
                    else:
                        print(f"[UPLOAD] Could not open .avi file for conversion")
                        processed_video_basename = base_processed_name + '.avi'
                except Exception as e:
                    print(f"[UPLOAD] Error converting .avi to .mp4: {str(e)}")
                    traceback.print_exc()
                    # Keep the .avi file if conversion fails
                    if os.path.exists(avi_path):
                        processed_video_basename = base_processed_name + '.avi'
            elif os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                processed_video_basename = base_processed_name + '.mp4'
                print(f"[UPLOAD] Found .mp4 processed video: {processed_video_basename}")
            
            # Fallback to original processed_video_path if not found
            if not processed_video_basename:
                processed_video_basename = os.path.basename(processed_video_path)
                print(f"[UPLOAD] Using default processed video name: {processed_video_basename}")

            # Prepare video information for results page
            file_size = os.path.getsize(video_path) / 1024  # Size in KB
            video_info = {
                'name': file.filename,
                'size': f"{file_size:.2f} KB",
                'user': 'Guest', 
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'per': deepfake_detection_rate  # Deepfake detection rate percentage
            }

            video_info_json = json.dumps(video_info)

            # URL encode filenames for safe passing in URL
            encoded_filename = quote(filename, safe='')
            encoded_processed = quote(processed_video_basename, safe='')
            
            print(f"[UPLOAD] Redirecting to results page")
            print(f"[UPLOAD]   - Original video: {filename}")
            print(f"[UPLOAD]   - Processed video: {processed_video_basename}")
            print(f"[UPLOAD]   - Deepfake detection rate: {deepfake_detection_rate}%")
            
            return redirect(url_for('result', video_info=video_info_json, 
                                  original_video=encoded_filename, processed_video=encoded_processed))
            
    except Exception as e:
        print(f"[UPLOAD] ERROR: {str(e)}")
        traceback.print_exc()
        flash(f'An error occurred: {str(e)}')
        return redirect(request.url)


@app.route('/result')
def result():
    """
    Display analysis results page with processed video and deepfake detection rate.
    
    Shows:
    - Processed video with frame annotations (green borders for real, red for deepfake)
    - Deepfake detection rate percentage
    - Video metadata (name, size, upload date)
    
    Returns:
        Rendered results template with analysis data
    """
    try:
        print(f"[RESULT] Loading results page...")
        video_info_json = request.args.get('video_info')
        original_video_encoded = request.args.get('original_video')
        processed_video = request.args.get('processed_video')
        
        if not video_info_json or not original_video_encoded:
            print(f"[RESULT] ERROR: Missing video information")
            flash('Missing video information')
            return redirect(url_for('index'))
        
        # Decode the filename
        original_video = unquote(original_video_encoded)
        print(f"[RESULT] Original video: {original_video}")
        
        # Parse video information
        video_info = json.loads(video_info_json)
        print(f"[RESULT] Video name: {video_info.get('name', 'N/A')}")
        print(f"[RESULT] Deepfake detection rate: {video_info.get('per', 'N/A')}%")
        
        # Use processed video for display (with frame borders/annotations)
        video_to_display = None
        video_filename = None
        
        if processed_video:
            # Decode processed video filename if needed
            processed_video_clean = unquote(processed_video) if '%' in processed_video else processed_video
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_video_clean)
            print(f"[RESULT] Checking processed video path: {processed_path}")
            print(f"[RESULT] Processed video exists: {os.path.exists(processed_path)}")
            
            if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
                video_to_display = processed_path
                video_filename = processed_video_clean
                print(f"[RESULT] Using processed video with annotations: {video_filename}")
            else:
                # Try alternative extensions (.avi, .mp4)
                base_name = processed_video_clean.rsplit('.', 1)[0] if '.' in processed_video_clean else processed_video_clean
                for ext in ['.avi', '.mp4']:
                    alt_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name + ext)
                    if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
                        video_to_display = alt_path
                        video_filename = base_name + ext
                        print(f"[RESULT] Found processed video with extension {ext}: {video_filename}")
                        break
        
        # Fallback to original video if processed video not found
        if not video_to_display:
            print(f"[RESULT] Processed video not found, falling back to original")
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_video)
            if os.path.exists(original_path):
                video_to_display = original_path
                video_filename = original_video
                print(f"[RESULT] Using original video as fallback: {video_filename}")
            else:
                print(f"[RESULT] ERROR: Neither processed nor original video found")
                upload_dir = app.config['UPLOAD_FOLDER']
                if os.path.exists(upload_dir):
                    files = os.listdir(upload_dir)
                    print(f"[RESULT] Available files in directory: {files}")
                flash('Video file not found')
                return redirect(url_for('index'))
        
        file_size = os.path.getsize(video_to_display)
        print(f"[RESULT] Video file size: {file_size / 1024:.2f} KB")
        
        # Create URL for video serving route
        encoded_filename = quote(video_filename, safe='')
        video_url = f"/video/{encoded_filename}"
        print(f"[RESULT] Video URL: {video_url}")

        return render_template('result.html', video_url=video_url, video_info=video_info, 
                              video_filename=video_filename)
    except Exception as e:
        print(f"[RESULT] ERROR: {str(e)}")
        traceback.print_exc()
        flash(f'Error displaying results: {str(e)}')
        return redirect(url_for('index'))


if __name__ == '__main__':
    """
    Run the AuthentiScan Flask application.
    
    The application runs on http://127.0.0.1:5000 by default.
    Debug mode is enabled for development.
    """
    print("[AuthentiScan] Starting Deepfake Video Detection System...")
    print("[AuthentiScan] Application: AuthentiScan")
    print("[AuthentiScan] Methodology: Frame-Level Facial Embedding Analysis")
    print("[AuthentiScan] Models: MTCNN (Face Detection) + InceptionResnetV1 (Feature Extraction)")
    print("[AuthentiScan] Similarity Metric: Cosine Similarity")
    print("[AuthentiScan] Server running at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True, use_reloader=False)

