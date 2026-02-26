"""
Deepfake Video Detection System
Flask Web Application for Frame-Level Facial Embedding Analysis (API Backend)
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
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
app.secret_key = 'deepfake-detection-secret-key-2024'
CORS(app) # Enable CORS for React frontend communication

# Configuration
PROCESSED_VIDEOS_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = PROCESSED_VIDEOS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max

# Ensure processed videos directory exists
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)

# Ensure result.txt exists
if not os.path.exists('result.txt'):
    try:
        with open('result.txt', 'w') as f:
            f.write("# Deepfake Analysis Results (filename:is_fake)\n")
    except IOError as e:
        print(f"[Deepfake Detection] CRITICAL: Could not create result.txt: {e}")

# Pre-load deep learning models at startup
print("[Deepfake Detection] Initializing facial embedding analysis models...")
try:
    from facial_embedding_analyzer import get_models
    get_models()
    print("[Deepfake Detection] Models loaded successfully")
except Exception as e:
    print(f"[Deepfake Detection] Warning: Could not pre-load models: {str(e)}")

def calculate_accuracy():
    """Calculates the deepfake detection accuracy based on result.txt."""
    try:
        total_predictions = 0
        correct_predictions = 0
        
        if not os.path.exists('result.txt'):
            return "N/A"

        with open('result.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or ':' not in line:
                    continue
                    
                parts = line.split(':')
                if len(parts) < 2:
                    continue
                    
                prediction_str = parts[-1].strip()
                if prediction_str not in ['True', 'False']:
                    continue

                total_predictions += 1
                if prediction_str == 'True':
                    correct_predictions += 1
        
        if total_predictions == 0:
            return "N/A"
        
        accuracy = (correct_predictions / total_predictions) * 100
        return f"{accuracy:.1f}"
        
    except Exception as e:
        print(f"[Accuracy Calc] Error: {e}")
        return "N/A"

@app.route('/api/accuracy', methods=['GET'])
def get_accuracy():
    """API Endpoint to get current model accuracy."""
    accuracy = calculate_accuracy()
    return jsonify({"accuracy": accuracy})

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve processed video files."""
    try:
        filename = unquote(filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if os.path.exists(video_path) and os.path.isfile(video_path):
            mimetype, _ = mimetypes.guess_type(video_path)
            if not mimetype:
                if filename.lower().endswith('.avi'): mimetype = 'video/x-msvideo'
                elif filename.lower().endswith('.webm'): mimetype = 'video/webm'
                elif filename.lower().endswith('.mov'): mimetype = 'video/quicktime'
                else: mimetype = 'video/mp4'
            
            response = send_file(video_path, mimetype=mimetype, conditional=True)
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
        else:
            return jsonify({"error": "Video not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video/<batch_folder>/<filename>')
def serve_video_batch(batch_folder, filename):
    """Serve batch processed video files."""
    try:
        filename = unquote(filename)
        batch_folder = unquote(batch_folder)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], batch_folder, filename)
        
        if os.path.exists(video_path) and os.path.isfile(video_path):
            mimetype, _ = mimetypes.guess_type(video_path)
            if not mimetype:
                if filename.lower().endswith('.avi'): mimetype = 'video/x-msvideo'
                elif filename.lower().endswith('.webm'): mimetype = 'video/webm'
                elif filename.lower().endswith('.mov'): mimetype = 'video/quicktime'
                else: mimetype = 'video/mp4'
            
            response = send_file(video_path, mimetype=mimetype, conditional=True)
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'
            return response
        else:
            return jsonify({"error": "Video not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """API Endpoint for SINGLE video analysis."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return jsonify({"error": "Invalid file type. Please upload a video file (.mp4, .avi, .mov, .mkv)"}), 400

        if file:
            timestamp = int(current_time())
            filename = f"uploaded_video_{timestamp}.mp4"
            filename = ''.join(c for c in filename if c.isprintable() or c in '.-_')
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(video_path)
            
            if not os.path.exists(video_path):
                return jsonify({"error": "Error saving file"}), 500

            processed_video_path = os.path.join(app.config['UPLOAD_FOLDER'], "analyzed_" + filename)

            try:
                from facial_embedding_analyzer import analyze_video_frames
                deepfake_detection_rate = analyze_video_frames(video_path, processed_video_path)
                
                is_fake = deepfake_detection_rate >= 50
                try:
                    with open('result.txt', 'a') as f:
                        f.write(f"{file.filename} (saved as {filename}):{is_fake}\n")
                except IOError as e:
                    pass
            except Exception as e:
                traceback.print_exc()
                return jsonify({"error": f"Error processing video: {str(e)}"}), 500

            processed_video_basename = None
            base_processed_name = "analyzed_" + filename.rsplit('.', 1)[0]
            avi_path = os.path.join(app.config['UPLOAD_FOLDER'], base_processed_name + '.avi')
            mp4_path = os.path.join(app.config['UPLOAD_FOLDER'], base_processed_name + '.mp4')
            
            if os.path.exists(avi_path) and os.path.getsize(avi_path) > 0:
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
                        except: pass
                        
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
                            except: pass
                        
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
                except:
                    if os.path.exists(avi_path):
                        processed_video_basename = base_processed_name + '.avi'
            elif os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                processed_video_basename = base_processed_name + '.mp4'
            
            if not processed_video_basename:
                processed_video_basename = os.path.basename(processed_video_path)

            file_size = os.path.getsize(video_path) / 1024
            video_info = {
                'name': file.filename,
                'size': f"{file_size:.2f} KB",
                'user': 'Guest', 
                'source': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'per': deepfake_detection_rate
            }

            return jsonify({
                "success": True,
                "video_info": video_info,
                "original_video": quote(filename, safe=''),
                "processed_video": quote(processed_video_basename, safe=''),
                "video_url": f"/video/{quote(processed_video_basename, safe='')}"
            })
            
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload_folder', methods=['POST'])
def upload_folder():
    """API Endpoint for BATCH folder analysis."""
    try:
        files = request.files.getlist('files[]')
        if not files or len(files) == 0 or (len(files) == 1 and files[0].filename == ''):
            return jsonify({"error": "No files selected"}), 400

        results_list = []
        batch_folder_name = f"batch_{int(current_time())}"
        batch_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], batch_folder_name)
        os.makedirs(batch_folder_path, exist_ok=True)
        
        video_files_found = 0
        
        for file in files:
            original_filename = file.filename
            if original_filename == '' or not original_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            
            video_files_found += 1
            base_filename = os.path.basename(original_filename)
            safe_filename = ''.join(c for c in base_filename if c.isprintable() or c in '.-_')
            
            if not safe_filename: continue

            try:
                video_path = os.path.join(batch_folder_path, safe_filename) 
                file.save(video_path)
                
                if not os.path.exists(video_path):
                    raise Exception("File not saved correctly")
                    
                base_processed_name = "analyzed_" + safe_filename.rsplit('.', 1)[0]
                processed_video_path = os.path.join(batch_folder_path, base_processed_name + '.mp4')
                
                from facial_embedding_analyzer import analyze_video_frames
                deepfake_detection_rate = analyze_video_frames(video_path, processed_video_path)
                
                is_fake = deepfake_detection_rate >= 50
                try:
                    with open('result.txt', 'a') as f:
                        f.write(f"{original_filename}:{is_fake}\n")
                except: pass
                    
                processed_video_url = None
                avi_path = os.path.join(batch_folder_path, base_processed_name + '.avi')
                mp4_path = processed_video_path

                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                    processed_video_basename = os.path.basename(mp4_path)
                    processed_video_url = f"/video/{batch_folder_name}/{quote(processed_video_basename, safe='')}"
                elif os.path.exists(avi_path) and os.path.getsize(avi_path) > 0:
                    processed_video_basename = os.path.basename(avi_path)
                    processed_video_url = f"/video/{batch_folder_name}/{quote(processed_video_basename, safe='')}"
                
                results_list.append({
                    'filename': original_filename,
                    'rate': deepfake_detection_rate,
                    'is_fake': is_fake,
                    'processed_video_url': processed_video_url,
                    'original_video_url': f"/video/{batch_folder_name}/{quote(safe_filename, safe='')}"
                })
            except Exception as e:
                results_list.append({
                    'filename': original_filename,
                    'rate': 'Error',
                    'is_fake': None,
                    'processed_video_url': None,
                    'original_video_url': None
                })
        
        if video_files_found == 0:
            return jsonify({"error": "No valid video files found"}), 400

        return jsonify({
            "success": True,
            "results": results_list
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("[Deepfake Detection] Starting Deepfake Video Detection System API...")
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True, use_reloader=False)