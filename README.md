# Deepfake Video Detection System

**Frame-Level Facial Embedding Analysis for Deepfake Detection**

A web-based application that detects deepfake videos using artificial intelligence through frame-level facial embedding analysis.

## Project Information

**Project Title:** Deepfake Video Detection System  
**Application Name:** AuthentiScan  
**Institution:** Sapthagiri College of Engineering, Department of Computer Science Engineering  
**Affiliation:** VTU, Belagavi | AICTE Approved | NAAC & NBA Accredited

**Authors:**
- Abhishek Saini (1SG23CS004)
- Arnav Singh Thapa (1SG23CS016)
- Bhavana Hegde (1SG23CS026)
- Chaithanya D S (1SG23CS028)

**Guide:** Assistant Prof. Hemalatha K, Department of Computer Science Engineering

## Technology Stack

- **Backend:** Python, Flask
- **Computer Vision:** OpenCV
- **Deep Learning:** PyTorch, facenet-pytorch
- **Models:** MTCNN (Face Detection), InceptionResnetV1 (Feature Extraction)
- **Similarity Metric:** Cosine Similarity

## Setup Environment

```bash
# Make sure your PIP is up to date
pip install -U pip wheel setuptools

# Install required dependencies
pip install -r requirements.txt
```

## Running the Application

Run the Flask application:

```bash
python app.py
```

The application will be available at: **http://127.0.0.1:5000**

## Methodology

The system implements **Frame-Level Facial Embedding Analysis** using the following process:

1. **Video Input:** User uploads video through web interface. OpenCV extracts individual frames.

2. **Face Detection:** MTCNN (Multi-Task Cascaded Convolutional Networks) detects faces in each frame and aligns them for accurate comparison.

3. **Feature Extraction:** The InceptionResnetV1 model (pre-trained on VGGFace2) processes each aligned face and converts it into a 512-dimensional numerical feature vector (embedding).

4. **Similarity Check:** Cosine similarity is used to compare the embeddings of consecutive frames to measure how similar they are.

5. **Temporal Analysis:** The system looks for sudden drops or unstable similarity values over time, as these suggest unnatural changes typical of deepfakes.

6. **Final Classification:** 
   - If similarity scores remain stable (above threshold), the video is classified as **Real**.
   - If similarity scores fluctuate beyond the threshold, it's classified as **Deepfake**.

## Architecture

### Algorithms Used

1. **MTCNN (Multi-Task Cascaded Convolutional Networks)**
   - Purpose: Face Detection & Alignment
   - Process: Uses a cascade of three neural networks (P-Net, R-Net, O-Net) to progressively refine face detection

2. **InceptionResnetV1 (Trained on VGGFace2)**
   - Purpose: Face Embedding / Feature Extraction
   - Output: 512-dimensional vector that uniquely represents facial identity

3. **Cosine Similarity**
   - Purpose: Similarity Metric
   - Range: 0 to 1 (1 = high similarity, likely same person)

## Features

- **Frame-by-Frame Analysis:** Ensures consistent and thorough analysis
- **Facial Embeddings:** Detects subtle manipulations using advanced feature extraction
- **Visual Annotations:** Processed videos show green borders for real frames and red borders for deepfake frames
- **Web Interface:** Simple, user-friendly interface for video upload and analysis
- **Real-time Processing:** Efficient processing with model caching

## Output

The system provides:
- **Deepfake Detection Rate:** Percentage indicating likelihood of manipulation
- **Annotated Video:** Processed video with visual indicators (green = real, red = deepfake)
- **Video Metadata:** File information and analysis timestamp

## Advantages

- Mitigates Information Warfare
- Safeguards Reputations and Assets
- Restores Digital Trust
- Strengthens Cyber Resilience
- Supports Legal and Forensic Integrity

## Real World Applications

- Social Media Monitoring
- Content Moderation
- Identity Verification
- Forensic Analysis
- Financial Transaction Security
- Broadcast and News Integrity

## Contributing

If you want to contribute to this project, please follow these steps:

- **Fork:** Fork this repository to your GitHub account.
- **Create a Branch:** Create a new branch to add a new feature or fix a bug.
- **Commit:** Add clear commit messages explaining your changes.
- **Push:** Push your changes to the repository you forked.
- **Pull Request:** Create a pull request on GitHub.
