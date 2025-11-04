"""
People Counter - Memory-Optimized for Free Tier Deployment
YOLOv8 + DeepSORT
Production-ready with garbage collection and memory management
"""

from flask import Flask, request, render_template_string, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import os
from werkzeug.utils import secure_filename
import tempfile
import shutil
import numpy as np
import logging
from datetime import datetime
import gc
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# HTML Template - Modern UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>People Counter API - YOLOv8 + DeepSORT</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .badge {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            margin: 10px 5px 20px 5px;
            font-weight: bold;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 20px;
            transition: all 0.3s;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f8f9ff;
        }
        .upload-area.dragover {
            background: #e8ebff;
            border-color: #764ba2;
        }
        input[type="file"] {
            display: none;
        }
        .upload-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }
        .upload-text {
            color: #666;
            font-size: 1.1em;
        }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 30px;
            font-size: 1.1em;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
            font-weight: bold;
            margin-top: 15px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            margin-top: 30px;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            text-align: center;
            color: white;
            display: none;
        }
        .result.show {
            display: block;
            animation: slideIn 0.5s;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        .count {
            font-size: 4em;
            font-weight: bold;
            margin: 10px 0;
        }
        .result-label {
            font-size: 1.3em;
            opacity: 0.9;
        }
        .stats {
            margin-top: 20px;
            display: flex;
            justify-content: space-around;
            font-size: 0.9em;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 5px;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .file-name {
            margin-top: 15px;
            color: #667eea;
            font-weight: bold;
            display: none;
        }
        .file-name.show {
            display: block;
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 15px;
            display: none;
            padding: 15px;
            background: #ffe8e8;
            border-radius: 10px;
        }
        .error.show {
            display: block;
        }
        .info {
            background: #e8f4fd;
            color: #0066cc;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 0.9em;
            text-align: left;
        }
        .info-title {
            font-weight: bold;
            margin-bottom: 8px;
        }
        .api-info {
            background: #f0e8ff;
            color: #6e4ba2;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 0.85em;
            text-align: left;
        }
        .api-endpoint {
            background: #333;
            color: #0f0;
            padding: 10px;
            border-radius: 5px;
            margin-top: 8px;
            font-family: monospace;
            overflow-x: auto;
            word-break: break-all;
        }
        .limits {
            background: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 People Counter</h1>
        <p class="subtitle">Accurate counting with YOLOv8 + DeepSORT</p>
        <div style="text-align: center;">
            <span class="badge">✓ Unique ID Tracking</span>
            <span class="badge">✓ No Duplicates</span>
        </div>
        
        <form id="uploadForm" enctype="multipart/form-data">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📹</div>
                <div class="upload-text">Click to select or drag & drop video</div>
                <div class="upload-text" style="font-size: 0.9em; margin-top: 10px; color: #999;">
                    Supported: MP4, AVI, MOV
                </div>
            </div>
            <input type="file" id="videoFile" name="video" accept="video/*" required>
            <div class="file-name" id="fileName"></div>
            <button type="submit" class="btn" id="submitBtn">Count People</button>
            <div class="error" id="error"></div>
        </form>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing video with DeepSORT...</p>
            <p style="font-size: 0.9em; margin-top: 10px; color: #666;">This may take 1-2 minutes for optimization</p>
        </div>

        <div class="result" id="result">
            <div class="result-label">Total Unique People</div>
            <div class="count" id="count">0</div>
            <div class="result-label">detected in your video</div>
            <div class="stats">
                <div class="stat-item">
                    <span>Frames Analyzed</span>
                    <span class="stat-value" id="frames">0</span>
                </div>
                <div class="stat-item">
                    <span>Processing Time</span>
                    <span class="stat-value" id="time">0s</span>
                </div>
            </div>
        </div>

        <div class="info">
            <div class="info-title">🔍 How it works:</div>
            <ul style="margin-left: 20px; margin-top: 8px;">
                <li>Uses YOLOv8 Nano for accurate person detection</li>
                <li>DeepSORT assigns unique IDs to each person</li>
                <li>Tracks people across frames to prevent duplicates</li>
                <li>High confidence threshold ensures accuracy</li>
            </ul>
        </div>

        <div class="limits">
            <strong>⚠️ Free Tier Limits:</strong>
            <ul style="margin-left: 20px; margin-top: 8px;">
                <li>Maximum video size: 50MB</li>
                <li>Maximum duration: 60 seconds</li>
                <li>Processing time: 1-2 minutes (be patient!)</li>
            </ul>
        </div>

        <div class="api-info">
            <div class="info-title">📡 API Usage:</div>
            <div class="api-endpoint">POST /api/count<br>Form data: "video" (file)</div>
            <p style="margin-top: 8px;">Response: {"count": number, "frames_processed": number, "processing_time": seconds}</p>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('videoFile');
        const fileName = document.getElementById('fileName');
        const form = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');
        const countElement = document.getElementById('count');
        const framesElement = document.getElementById('frames');
        const timeElement = document.getElementById('time');
        const submitBtn = document.getElementById('submitBtn');
        const errorElement = document.getElementById('error');

        let startTime = 0;

        uploadArea.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                fileName.textContent = `Selected: ${e.target.files[0].name}`;
                fileName.classList.add('show');
                result.classList.remove('show');
                errorElement.classList.remove('show');
            }
        });

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileName.textContent = `Selected: ${e.dataTransfer.files[0].name}`;
                fileName.classList.add('show');
                result.classList.remove('show');
                errorElement.classList.remove('show');
            }
        });

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (!fileInput.files.length) {
                errorElement.textContent = 'Please select a video file';
                errorElement.classList.add('show');
                return;
            }

            const formData = new FormData();
            formData.append('video', fileInput.files[0]);

            startTime = Date.now();
            loading.classList.add('show');
            result.classList.remove('show');
            errorElement.classList.remove('show');
            submitBtn.disabled = true;

            try {
                const response = await fetch('/api/count', {
                    method: 'POST',
                    body: formData,
                    timeout: 300000  // 5 minute timeout
                });

                const data = await response.json();
                const processingTime = ((Date.now() - startTime) / 1000).toFixed(2);

                loading.classList.remove('show');
                submitBtn.disabled = false;

                if (data.error) {
                    errorElement.textContent = data.error;
                    errorElement.classList.add('show');
                } else {
                    countElement.textContent = data.count;
                    framesElement.textContent = data.frames_processed;
                    timeElement.textContent = processingTime + 's';
                    result.classList.add('show');
                }
            } catch (error) {
                loading.classList.remove('show');
                submitBtn.disabled = false;
                errorElement.textContent = 'Error processing video. Server may be slow. Please try again.';
                errorElement.classList.add('show');
                console.error('Error:', error);
            }
        });
    </script>
</body>
</html>
"""

# DeepSORT Tracker Implementation
class DeepSORTTracker:
    """
    Simplified DeepSORT tracker for accurate unique person counting
    Uses appearance features and IoU matching to maintain consistent IDs
    Optimized for memory efficiency
    """
    def __init__(self, max_age=20, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        self.confirmed_ids = set()
        
    def update(self, detections, frame):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Extract person detections only (class 0)
        person_detections = [d for d in detections if int(d[5]) == 0]
        
        if len(person_detections) == 0:
            self._age_tracks()
            return []
        
        if len(self.tracks) == 0:
            for det in person_detections:
                self._create_track(det, frame)
            return list(self.tracks.keys())
        
        # Match detections to existing tracks
        matches, unmatched_dets, unmatched_tracks = self._match_detections(
            person_detections, frame
        )
        
        # Update matched tracks
        for track_id, det_idx in matches:
            self.tracks[track_id]['bbox'] = person_detections[det_idx][:4]
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['hits'] += 1
            self.tracks[track_id]['appearance'] = self._extract_appearance(
                frame, person_detections[det_idx][:4]
            )
            
            if self.tracks[track_id]['hits'] >= self.min_hits:
                self.confirmed_ids.add(track_id)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(person_detections[det_idx], frame)
        
        # Remove old tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        return list(self.confirmed_ids)
    
    def _create_track(self, detection, frame):
        """Create new track"""
        track_id = self.next_id
        self.next_id += 1
        
        self.tracks[track_id] = {
            'bbox': detection[:4],
            'age': 0,
            'hits': 1,
            'appearance': self._extract_appearance(frame, detection[:4])
        }
    
    def _extract_appearance(self, frame, bbox):
        """Extract appearance features from detection"""
        x1, y1, x2, y2 = map(int, bbox)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(96)
        
        # Resize to smaller size for memory efficiency
        roi_resized = cv2.resize(roi, (48, 96))
        
        # Compute color histogram
        hist_b = cv2.calcHist([roi_resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([roi_resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([roi_resized], [2], None, [32], [0, 256])
        
        feature = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        feature = feature / (np.linalg.norm(feature) + 1e-6)
        
        return feature
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def _compute_appearance_distance(self, feat1, feat2):
        """Compute cosine distance between appearance features"""
        return 1 - np.dot(feat1, feat2)
    
    def _match_detections(self, detections, frame):
        """Match detections to existing tracks"""
        if len(detections) == 0 or len(self.tracks) == 0:
            return [], list(range(len(detections))), list(self.tracks.keys())
        
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())
        
        for i, track_id in enumerate(track_ids):
            track_bbox = self.tracks[track_id]['bbox']
            track_appearance = self.tracks[track_id]['appearance']
            
            for j, det in enumerate(detections):
                det_bbox = det[:4]
                det_appearance = self._extract_appearance(frame, det_bbox)
                
                iou = self._compute_iou(track_bbox, det_bbox)
                app_dist = self._compute_appearance_distance(
                    track_appearance, det_appearance
                )
                
                # Combined cost
                cost = 0.7 * (1 - iou) + 0.3 * app_dist
                cost_matrix[i, j] = cost
        
        matches = []
        unmatched_tracks = set(range(len(track_ids)))
        unmatched_dets = set(range(len(detections)))
        
        flat_indices = np.argsort(cost_matrix.flatten())
        
        for flat_idx in flat_indices:
            track_idx = flat_idx // len(detections)
            det_idx = flat_idx % len(detections)
            
            if track_idx in unmatched_tracks and det_idx in unmatched_dets:
                cost = cost_matrix[track_idx, det_idx]
                
                if cost < 0.7:
                    matches.append((track_ids[track_idx], det_idx))
                    unmatched_tracks.remove(track_idx)
                    unmatched_dets.remove(det_idx)
        
        unmatched_track_ids = [track_ids[i] for i in unmatched_tracks]
        
        return matches, list(unmatched_dets), unmatched_track_ids
    
    def _age_tracks(self):
        """Age all tracks when no detections"""
        tracks_to_delete = []
        for track_id in self.tracks:
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                tracks_to_delete.append(track_id)
        
        for track_id in tracks_to_delete:
            del self.tracks[track_id]
    
    def get_unique_count(self):
        """Get count of confirmed unique people"""
        return len(self.confirmed_ids)


# Load YOLO model - Nano (optimized for free tier)
try:
    logger.info("Loading YOLOv8 Nano model (memory optimized)...")
    model = YOLO('yolov8n.pt')  # Smallest model
    model.fuse()  # Optimize for inference
    logger.info("✓ Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.route('/')
def index():
    """Render the main UI page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/count', methods=['POST'])
def count_people():
    """API endpoint to count people in uploaded video - Memory optimized"""
    start_time = datetime.now()
    temp_dir = None
    
    try:
        logger.info("=" * 70)
        logger.info("NEW REQUEST: Video upload received")
        logger.info("=" * 70)
        
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        
        filename = secure_filename(video_file.filename)
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, filename)
        video_file.save(video_path)
        
        # Check file size
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb > 50:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.warning(f"Video too large: {file_size_mb:.2f}MB > 50MB")
            return jsonify({'error': f'Video too large ({file_size_mb:.2f}MB). Max 50MB on free tier'}), 400
        
        logger.info(f"Processing video: {filename} ({file_size_mb:.2f}MB)")
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
            
            # Get video info
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video info: {total_frames} frames, {duration:.1f}s, {fps:.2f} fps")
            
            if duration > 60:  # Max 1 minute
                cap.release()
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.warning(f"Video too long: {duration:.1f}s > 60s")
                return jsonify({'error': f'Video too long ({duration:.1f}s). Max 60 seconds on free tier'}), 400
            
            # Initialize tracker with optimized settings
            tracker = DeepSORTTracker(max_age=20, min_hits=2, iou_threshold=0.3)
            
            frame_count = 0
            processed_frames = 0
            
            logger.info("Starting video processing...")
            
            # Process video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 5th frame (reduce memory and speed up)
                if frame_count % 5 != 0:
                    continue
                
                processed_frames += 1
                
                # Resize frame to 640x480 (memory optimization)
                frame = cv2.resize(frame, (640, 480))
                
                # Run detection with lower resource usage
                results = model(frame, verbose=False, conf=0.55, imgsz=640)
                
                # Extract detections
                detections = []
                if len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    confidences = results[0].boxes.conf.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confidences, classes):
                        detections.append([
                            box[0], box[1], box[2], box[3], conf, cls
                        ])
                
                # Update tracker
                tracker.update(detections, frame)
                
                # Force garbage collection every 25 processed frames
                if processed_frames % 25 == 0:
                    gc.collect()
                    current_count = tracker.get_unique_count()
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(f"Progress: {frame_count}/{total_frames} frames | "
                              f"Processed: {processed_frames} | Count: {current_count} | "
                              f"Time: {elapsed:.1f}s")
            
            cap.release()
            gc.collect()  # Final cleanup
            
            # Clean up temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
            temp_dir = None
            
            people_count = tracker.get_unique_count()
            elapsed_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("=" * 70)
            logger.info(f"✓ RESULT: {people_count} unique people detected")
            logger.info(f"✓ Frames processed: {processed_frames}/{frame_count}")
            logger.info(f"✓ Total time: {elapsed_time:.2f}s")
            logger.info("=" * 70)
            
            return jsonify({
                'count': people_count,
                'frames_processed': processed_frames,
                'processing_time': round(elapsed_time, 2)
            })
        
        except Exception as e:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            logger.error(f"Error during processing: {str(e)}", exc_info=True)
            raise e
    
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Error: {str(e)}'}), 500
    
    finally:
        gc.collect()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'People Counter API',
        'tier': 'free',
        'limits': {
            'max_video_size_mb': 50,
            'max_duration_seconds': 60
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info("\n" + "="*70)
    logger.info("🎥 People Counter API - YOLOv8 + DeepSORT")
    logger.info("="*70)
    logger.info(f"\n✓ Mode: {'Development' if debug else 'Production'}")
    logger.info(f"✓ Port: {port}")
    logger.info(f"✓ Model: YOLOv8 Nano (optimized)")
    logger.info(f"✓ Max video: 50MB / 60 seconds")
    logger.info(f"✓ Free tier deployment ready")
    logger.info("\n" + "="*70 + "\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port, threaded=True)
