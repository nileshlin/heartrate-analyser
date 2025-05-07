*****Real-Time Heart Rate Estimation Using Face Video*****
This project is a web-based application built with Flask that estimates a person's heart rate in real time using a webcam feed. It uses MediaPipe FaceMesh for facial landmark detection and remote photoplethysmography (rPPG) via signal processing on the green channel of the video feed.

🧠 Key Features
1. Real-time face detection using MediaPipe

2. Automatic masking of the face region

3. Dynamic thresholding to isolate facial pixels

4. Extraction of average RGB signals over time

5. Heart rate (BPM) estimation using Welch’s method

6. Visualization of heart rate on video frames

🛠️ Technologies Used
1. Python 3

2. Flask (web framework)

3. OpenCV (image processing)

4. MediaPipe (face mesh landmarks)

5. SciPy (signal processing)

6. NumPy

7. HTML5, JavaScript (frontend)

8. Base64 for image frame transmission



📦 Setup Instructions

1. Clone the repository:

git clone https://github.com/yourusername/heartrate-analyser.git
cd heartrate-analyser

2. Create a virtual environment and activate it:

python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Run the Flask app:

python app.py

Open your browser and navigate to: http://127.0.0.1:5000



📈 How It Works-

1. Video frames are captured in the browser and sent to the Flask backend.

2. MediaPipe detects facial landmarks and extracts the facial region.

3. Dynamic thresholding removes background noise.

4. Average RGB values are stored in a buffer over time.

5. Welch’s method is applied to the green channel to estimate dominant pulse frequency.

6. BPM is computed and overlaid on the returned video frame.



🔐 Notes
1. Ensure your environment supports webcam access via browser.

2. Performance and accuracy may vary with lighting and camera quality.


