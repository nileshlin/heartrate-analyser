# Real-Time Heart Rate Estimation Using Face Video

This project is a web-based application built with Flask that estimates a person's heart rate in real time using a webcam feed. It implements remote photoplethysmography (rPPG) by applying signal processing to the green channel of facial regions detected in video frames via MediaPipe FaceMesh.

## Key Features

- Real-time face detection and tracking using MediaPipe FaceMesh
- Automatic region of interest (ROI) masking based on facial landmarks
- Dynamic skin-pixel thresholding to improve signal quality
- Temporal RGB signal extraction with moving window approach
- Heart rate (BPM) estimation using Welch's method for spectral analysis
- Live visualization of heart rate measurements overlaid on video frames
- Signal quality indicators with confidence metrics

## Technologies Used

- **Backend**: Python 3.8+
- **Web Framework**: Flask
- **Computer Vision**: OpenCV
- **Facial Analysis**: MediaPipe
- **Signal Processing**: SciPy, NumPy
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Transmission**: Base64 encoding for frame streaming

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/nileshlin/heartrate-analyzer.git
   cd heartrate-analyzer
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # On macOS/Linux:
   source venv/bin/activate
   
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. **Access the application**:
   Open your browser and navigate to: `http://127.0.0.1:5000`

## How It Works

1. **Video Acquisition**: Browser captures webcam frames and sends them to the Flask backend
2. **Face Detection**: MediaPipe FaceMesh identifies 468 facial landmarks
3. **ROI Extraction**: Facial region is isolated and a skin-pixel mask is applied
4. **Signal Processing**:
   - RGB values are averaged from the masked facial region
   - Values are stored in a time-series buffer (typically 10-30 seconds)
   - Signals are detrended and filtered to remove noise
5. **Frequency Analysis**: 
   - Welch's method estimates the power spectral density
   - Dominant frequency in the 0.75-3.33 Hz range (45-200 BPM) is identified
6. **Heart Rate Calculation**: Frequency is converted to beats per minute
7. **Visualization**: BPM and confidence metrics are overlaid on the returned video frame

## Limitations & Considerations

- **Environmental Factors**: Performance depends on consistent, adequate lighting
- **Motion Artifacts**: Excessive movement can degrade signal quality
- **Hardware Dependencies**: Camera quality impacts measurement accuracy

## Privacy Note

This application processes all video data locally in your browser and server. No video data is stored permanently or transmitted to external services.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
