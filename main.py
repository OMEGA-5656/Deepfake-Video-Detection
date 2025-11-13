"""
Deepfake Video Detection System
Flask Web Application for Frame-Level Facial Embedding Analysis

This module implements the web interface for the deepfake detection system.
It handles video uploads (single and batch) processes them through 
the facial embedding analyzer, and displays results.

Methodology:
1. Video Input: User uploads video(s) through web interface
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
app.secret_key = 'deepfake-detection-secret-key-2024'  # Secret key for session management

# Configuration
PROCESSED_VIDEOS_FOLDER = 'static/videos'  # Folder for processed videos with annotations
app.config['UPLOAD_FOLDER'] = PROCESSED_VIDEOS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # Maximum file size: 2GB (for batch)

# Ensure processed videos directory exists
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)

# Ensure result.txt exists
if not os.path.exists('result.txt'):
    try:
        with open('result.txt', 'w') as f:
            f.write("# Deepfake Analysis Results (filename:is_fake)\n")
    except IOError as e:
        print(f"[Deepfake Detection] CRITICAL: Could not create result.txt: {e}")


# Pre-load deep learning models at startup for faster processing
print("[Deepfake Detection] Initializing facial embedding analysis models...")
try:
    from facial_embedding_analyzer import get_models
    get_models()
    print("[Deepfake Detection] Models loaded successfully (MTCNN + InceptionResnetV1)")
except Exception as e:
    print(f"[Deepfake Detection] Warning: Could not pre-load models: {str(e)}")
    print("[Deepfake Detection] Models will be loaded on first video analysis request")


def calculate_accuracy():
    """
    Calculates the deepfake detection accuracy based on result.txt.
    Assumes all videos in the log are deepfakes (Ground Truth = True).
    """
    try:
        total_predictions = 0
        correct_predictions = 0
        
        # Ensure the log file exists
        if not os.path.exists('result.txt'):
            return "N/A (result.txt not found)"

        with open('result.txt', 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for valid log lines (e.g., "filename.mp4:True" or "filename.mp4:False")
                # Skip comments, empty lines, or lines without a colon
                if not line or line.startswith('#') or ':' not in line:
                    continue
                    
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                    
                # Get the last part as the prediction
                prediction_str = parts[-1].strip()
                
                # Ensure it's a valid boolean prediction
                if prediction_str not in ['True', 'False']:
                    continue

                total_predictions += 1
                
                # Ground truth is always True (all are deepfakes)
                # A correct prediction is when the model also said "True"
                if prediction_str == 'True':
                    correct_predictions += 1
        
        if total_predictions == 0:
            return "N/A (No results)"
        
        # Calculate accuracy and format to one decimal place
        accuracy = (correct_predictions / total_predictions) * 100
        return f"{accuracy:.1f}" # Return as formatted string "92.7"
        
    except Exception as e:
        print(f"[Accuracy Calc] Error: {e}")
        traceback.print_exc()
        return "N/A (Error)"

def calculate_accuracy():
    """
    Calculates the deepfake detection accuracy based on result.txt.
    Assumes all videos in the log are deepfakes (Ground Truth = True).
    """
    try:
        total_predictions = 0
        correct_predictions = 0
        
        # Ensure the log file exists
        if not os.path.exists('result.txt'):
            return "N/A (result.txt not found)"

        with open('result.txt', 'r') as f:
            for line in f:
                line = line.strip()
                
                # Check for valid log lines (e.g., "filename.mp4:True" or "filename.mp4:False")
                # Skip comments, empty lines, or lines without a colon
                if not line or line.startswith('#') or ':' not in line:
                    continue
                    
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                    
                # Get the last part as the prediction
                prediction_str = parts[-1].strip()
                
                # Ensure it's a valid boolean prediction
                if prediction_str not in ['True', 'False']:
                    continue

                total_predictions += 1
                
                # Ground truth is always True (all are deepfakes)
                # A correct prediction is when the model also said "True"
                if prediction_str == 'True':
                    correct_predictions += 1
        
        if total_predictions == 0:
            return "N/A (No results)"
        
        # Calculate accuracy and format to one decimal place
        accuracy = (correct_predictions / total_predictions) * 100
        return f"{accuracy:.1f}" # Return as formatted string "92.7"
        
    except Exception as e:
        print(f"[Accuracy Calc] Error: {e}")
        traceback.print_exc()
        return "N/A (Error)"

@app.route('/')
def index():
    """
    Render the main upload page for Deepfake Detection.
    Users can upload a single video or a folder for batch analysis.
    
    NOW INCLUDES ACCURACY CALCULATION.
    """
    print("[Deepfake Detection] Calculating accuracy for homepage...")
    current_accuracy = calculate_accuracy()
    print(f"[Deepfake Detection] Current accuracy: {current_accuracy}%")
    
    # Pass the calculated accuracy to the template
    return render_template('index.html', accuracy=current_accuracy)


@app.route('/video/<filename>')
def serve_video(filename):
    """
    Serve processed video files (from single upload) with proper MIME type.
    
    Args:
        filename: Name of the video file to serve from the root UPLOAD_FOLDER
        
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


@app.route('/video/<batch_folder>/<filename>')
def serve_video_batch(batch_folder, filename):
    """
    Serve processed video files from a specific batch folder.
    
    Args:
        batch_folder: The sub-folder for the analysis batch
        filename: Name of the video file to serve
        
    Returns:
        Video file with appropriate headers or error response
    """
    try:
        filename = unquote(filename)
        batch_folder = unquote(batch_folder)
        print(f"[VIDEO SERVE BATCH] Requested video: {batch_folder}/{filename}")
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], batch_folder, filename)
        
        if os.path.exists(video_path) and os.path.isfile(video_path):
            file_size = os.path.getsize(video_path)
            print(f"[VIDEO SERVE BATCH] File size: {file_size / 1024:.2f} KB")

            mimetype, encoding = mimetypes.guess_type(video_path)
            
            if not mimetype:
                if filename.lower().endswith('.avi'): mimetype = 'video/x-msvideo'
                elif filename.lower().endswith('.webm'): mimetype = 'video/webm'
                elif filename.lower().endswith('.mov'): mimetype = 'video/quicktime'
                else: mimetype = 'video/mp4'
            
            response = send_file(video_path, mimetype=mimetype, conditional=True)
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            
            print(f"[VIDEO SERVE BATCH] Successfully serving video with MIME type: {mimetype}")
            return response
        else:
            print(f"[VIDEO SERVE BATCH] ERROR: Video file not found at {video_path}")
            return "Video not found", 404
    except Exception as e:
        print(f"[VIDEO SERVE BATCH] ERROR: {str(e)}")
        traceback.print_exc()
        return "Error serving video", 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle SINGLE video file upload and initiate analysis.
    
    Returns:
        Redirect to results page with analysis data
    """
    try:
        if 'file' not in request.files:
            flash('No file part in the request', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        # Validate file extension
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            flash('Invalid file type. Please upload a video file (.mp4, .avi, .mov, .mkv)', 'danger')
            return redirect(request.url)

        if file:
            # Generate unique filename with timestamp
            timestamp = int(current_time())
            filename = f"uploaded_video_{timestamp}.mp4"
            # Sanitize filename (remove control characters)
            filename = ''.join(c for c in filename if c.isprintable() or c in '.-_')
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"[UPLOAD] Saving uploaded video: {video_path}")
            file.save(video_path)
            
            if not os.path.exists(video_path):
                flash('Error saving file', 'danger')
                return redirect(request.url)

            # Path for processed video with annotations
            processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], "analyzed_" + filename)

            # Step 3: Run Frame-Level Facial Embedding Analysis
            print(f"[UPLOAD] Starting frame-level facial embedding analysis...")
            try:
                from facial_embedding_analyzer import analyze_video_frames
                
                deepfake_detection_rate = analyze_video_frames(video_path, processed_video_path)
                print(f"[UPLOAD] Analysis complete. Deepfake detection rate: {deepfake_detection_rate}%")

                # Write result to result.txt
                is_fake = deepfake_detection_rate >= 50
                try:
                    with open('result.txt', 'a') as f:
                        f.write(f"{file.filename} (saved as {filename}):{is_fake}\n")
                except IOError as e:
                     print(f"[UPLOAD] ERROR: Could not write to result.txt: {e}")

            except Exception as e:
                print(f"[UPLOAD] Error during facial embedding analysis: {str(e)}")
                traceback.print_exc()
                error_msg = str(e)
                if len(error_msg) > 200: error_msg = error_msg[:200] + "..."
                flash(f'Error processing video: {error_msg}', 'danger')
                deepfake_detection_rate = 50 # Default on error

            # Find the actual processed video file (may be .avi or .mp4)
            processed_video_basename = None
            base_processed_name = "analyzed_" + filename.rsplit('.', 1)[0]  # Remove .mp4 extension
            avi_path = os.path.join(app.config['UPLOAD_FOLDER'], base_processed_name + '.avi')
            mp4_path = os.path.join(app.config['UPLOAD_FOLDER'], base_processed_name + '.mp4')
            
            # --- This logic attempts to convert .avi to .mp4 for web compatibility ---
            if os.path.exists(avi_path) and os.path.getsize(avi_path) > 0:
                print(f"[UPLOAD] Converting .avi to .mp4 for web compatibility...")
                try:
                    cap = cv2.VideoCapture(avi_path)
                    if cap.isOpened():
                        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        
                        use_ffmpeg = False
                        try:
                            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
                            if result.returncode == 0: use_ffmpeg = True
                        except (FileNotFoundError, subprocess.TimeoutExpired):
                            pass # ffmpeg not available
                        
                        conversion_success = False
                        if use_ffmpeg:
                            try:
                                ffmpeg_cmd = [
                                    'ffmpeg', '-y', '-i', avi_path,
                                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', 
                                    '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
                                    mp4_path
                                ]
                                result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=300)
                                if result.returncode == 0 and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                                    os.remove(avi_path)
                                    processed_video_basename = base_processed_name + '.mp4'
                                    cap.release()
                                    conversion_success = True
                            except Exception as e:
                                print(f"[UPLOAD] Error using ffmpeg: {str(e)}, trying OpenCV")
                        
                        if not conversion_success:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))
                            if out and out.isOpened():
                                while True:
                                    ret, frame = cap.read()
                                    if not ret: break
                                    out.write(frame)
                                out.release()
                                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                                    os.remove(avi_path)
                                    processed_video_basename = base_processed_name + '.mp4'
                                else:
                                    processed_video_basename = base_processed_name + '.avi'
                            else:
                                processed_video_basename = base_processed_name + '.avi'
                            cap.release()
                    else:
                        processed_video_basename = base_processed_name + '.avi'
                except Exception as e:
                    print(f"[UPLOAD] Error converting .avi to .mp4: {str(e)}")
                    if os.path.exists(avi_path):
                        processed_video_basename = base_processed_name + '.avi'
            elif os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                processed_video_basename = base_processed_name + '.mp4'
            
            if not processed_video_basename:
                processed_video_basename = os.path.basename(processed_video_path)

            # Prepare video information for results page
            file_size = os.path.getsize(video_path) / 1024  # Size in KB
            video_info = {
                'name': file.filename,
                'size': f"{file_size:.2f} KB",
                'user': 'Guest', 
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'per': deepfake_detection_rate
            }
            video_info_json = json.dumps(video_info)

            encoded_filename = quote(filename, safe='')
            encoded_processed = quote(processed_video_basename, safe='')
            
            return redirect(url_for('result', video_info=video_info_json, 
                                  original_video=encoded_filename, processed_video=encoded_processed))
            
    except Exception as e:
        print(f"[UPLOAD] ERROR: {str(e)}")
        traceback.print_exc()
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(request.url)


@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    """
    Handle FOLDER (batch) video upload and initiate analysis for each file.
    
    Returns:
        Redirect to a new results list page
    """
    print("[Deepfake Detection] Serving '/upload_folder' route (batch folder).")
    try:
        files = request.files.getlist('files[]')
        if not files or len(files) == 0 or (len(files) == 1 and files[0].filename == ''):
            flash('No files selected in the folder', 'danger')
            return redirect(url_for('index'))

        results_list = []
        
        # Create a unique subfolder for this batch
        batch_folder_name = f"batch_{int(current_time())}"
        batch_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], batch_folder_name)
        os.makedirs(batch_folder_path, exist_ok=True)
        
        print(f"[UPLOAD_FOLDER] Processing batch in: {batch_folder_path}")

        video_files_found = 0
        
        for file in files:
            # Get original filename (e.g., "data/example.mp4")
            original_filename = file.filename
            
            # Check for valid video files
            if original_filename == '' or not original_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"[UPLOAD_FOLDER] Skipping non-video file: {original_filename}")
                continue
            
            video_files_found += 1
            
            # --- FIX: Handle relative paths ---
            # Get just the base filename (e.g., "example.mp4" from "data/example.mp4")
            base_filename = os.path.basename(original_filename)
            
            # Sanitize the base filename
            safe_filename = ''.join(c for c in base_filename if c.isprintable() or c in '.-_')
            
            if not safe_filename:
                print(f"[UPLOAD_FOLDER] Skipping file with invalid name: {original_filename}")
                continue
            # --- END FIX ---

            try:
                # Save the file using the SAFE base name
                video_path = os.path.join(batch_folder_path, safe_filename) 
                print(f"[UPLOAD_FOLDER] Saving: {original_filename} as {video_path}")
                file.save(video_path)
                
                if not os.path.exists(video_path):
                    print(f"[UPLOAD_FOLDER] CRITICAL: File.save() failed for: {original_filename}")
                    raise Exception("File not saved correctly") # Trigger the except block
                    
                # Path for processed video (target .mp4)
                base_processed_name = "analyzed_" + safe_filename.rsplit('.', 1)[0]
                processed_video_path = os.path.join(batch_folder_path, base_processed_name + '.mp4')
                
                print(f"[UPLOAD_FOLDER] Analyzing: {safe_filename}")
                from facial_embedding_analyzer import analyze_video_frames
                deepfake_detection_rate = analyze_video_frames(video_path, processed_video_path)
                print(f"[UPLOAD_FOLDER] Analysis complete for {safe_filename}. Rate: {deepfake_detection_rate}%")
                
                is_fake = deepfake_detection_rate >= 50
                
                # Write to result.txt (append mode)
                try:
                    with open('result.txt', 'a') as f:
                        f.write(f"{original_filename}:{is_fake}\n")
                except IOError as e:
                     print(f"[UPLOAD_FOLDER] ERROR: Could not write to result.txt: {e}")
                    
                # Find the actual processed file (mp4 or avi)
                processed_video_url = None
                avi_path = os.path.join(batch_folder_path, base_processed_name + '.avi')
                mp4_path = processed_video_path # This was our target

                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                    processed_video_basename = os.path.basename(mp4_path)
                    processed_video_url = url_for('serve_video_batch', batch_folder=batch_folder_name, filename=quote(processed_video_basename, safe=''))
                elif os.path.exists(avi_path) and os.path.getsize(avi_path) > 0:
                    # Conversion to mp4 might have failed or not run, link to AVI
                    processed_video_basename = os.path.basename(avi_path) # <-- FIX 1: Was avi_.path
                    processed_video_url = url_for('serve_video_batch', batch_folder=batch_folder_name, filename=quote(processed_video_basename, safe=''))
                    print(f"[UPLOAD_FOLDER] Warning: using .avi for {safe_filename}, web playback may fail.")
                
                # Append result using the ORIGINAL filename for display
                results_list.append({
                    'filename': original_filename,
                    'rate': deepfake_detection_rate,
                    'is_fake': is_fake,
                    'processed_video_url': processed_video_url, # Link to annotated video
                    'original_video_url': url_for('serve_video_batch', batch_folder=batch_folder_name, filename=quote(safe_filename, safe='')) # Link to original
                })
            
            except Exception as e:
                print(f"[UPLOAD_FOLDER] ERROR processing file {original_filename}: {str(e)}")
                traceback.print_exc() # Print full error
                results_list.append({
                    'filename': original_filename, # Show the original name the user uploaded
                    'rate': 'Error',
                    'is_fake': None,
                    'processed_video_url': None,
                    'original_video_url': None # <-- FIX 2: Was original_s
                })
        
        if video_files_found == 0:
            flash('No valid video files (.mp4, .avi, .mov, .mkv) found in the selected folder.', 'danger')
            return redirect(url_for('index'))

        # This line correctly renders 'results_list.html' for batch uploads
        print("[Deepfake Detection] Rendering 'results_list.html' with batch data.")
        return render_template('results_list.html', results=results_list)

    except Exception as e:
        print(f"[UPLOAD_FOLDER] FATAL ERROR: {str(e)}")
        traceback.print_exc()
        flash(f'An error occurred during batch processing: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/result')
def result():
    """
    Display analysis results page for a SINGLE video.
    """
    try:
        print(f"[RESULT] Loading results page...")
        video_info_json = request.args.get('video_info')
        original_video_encoded = request.args.get('original_video')
        processed_video = request.args.get('processed_video')
        
        if not video_info_json or not original_video_encoded:
            print(f"[RESULT] ERROR: Missing video information")
            flash('Missing video information', 'danger')
            return redirect(url_for('index'))
        
        original_video = unquote(original_video_encoded)
        video_info = json.loads(video_info_json)
        
        video_to_display = None
        video_filename = None
        
        if processed_video:
            processed_video_clean = unquote(processed_video) if '%' in processed_video else processed_video
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_video_clean)
            
            if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
                video_to_display = processed_path
                video_filename = processed_video_clean
            else:
                # Try alternative extensions
                base_name = processed_video_clean.rsplit('.', 1)[0] if '.' in processed_video_clean else processed_video_clean
                for ext in ['.avi', '.mp4']:
                    alt_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name + ext)
                    if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
                        video_to_display = alt_path
                        video_filename = base_name + ext
                        break
        
        # Fallback to original video
        if not video_to_display:
            print(f"[RESULT] Processed video not found, falling back to original")
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_video)
            if os.path.exists(original_path):
                video_to_display = original_path
                video_filename = original_video
            else:
                print(f"[RESULT] ERROR: Neither processed nor original video found")
                flash('Video file not found', 'danger')
                return redirect(url_for('index'))
        
        # Create URL for the single-file video serving route
        encoded_filename = quote(video_filename, safe='')
        video_url = f"/video/{encoded_filename}"
        print(f"[RESULT] Video URL: {video_url}")

        # This line correctly renders 'result.html' for single videos
        return render_template('result.html', video_url=video_url, video_info=video_info, 
                              video_filename=video_filename)
    except Exception as e:
        print(f"[RESULT] ERROR: {str(e)}")
        traceback.print_exc()
        flash(f'Error displaying results: {str(e)}', 'danger')
        return redirect(url_for('index'))


if __name__ == '__main__':
    """
    Run the Deepfake Detection Flask application.
    """
    print("[Deepfake Detection] Starting Deepfake Video Detection System...")
    print("[Deepfake Detection] Application: Deepfake Detection")
    print("[Deepfake Detection] Methodology: Frame-Level Facial Embedding Analysis")
    print("[Deepfake Detection] Models: MTCNN (Face Detection) + InceptionResnetV1 (Feature Extraction)")
    print("[Deepfake Detection] Server running at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True, use_reloader=False)