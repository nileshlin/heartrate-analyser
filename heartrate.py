from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
import mediapipe as mp
from scipy.signal import welch

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)

# Buffer to store RGB values over time
rgb_buffer = {'r': [], 'g': [], 'b': []}
FPS = 30  # Adjust according to your video frame rate
WINDOW_SIZE = 150  # Use last 150 frames (~5 sec at 30 FPS)



@app.route('/')
def index():
    return render_template('index.html')

def blackout_outside_dynamic_threshold(frame, lower_factor=0.48, upper_factor=1.74):
    """
    Blackouts pixels that fall outside a dynamically set threshold range based on the average intensity.
    """
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    non_zero_pixels = gray_image[gray_image != 0]
    average_value = np.median(non_zero_pixels)
 
    lower_threshold = max(0, int(average_value * lower_factor))
    upper_threshold = min(255, int(average_value * upper_factor))

    mask = (gray_image >= lower_threshold) & (gray_image <= upper_threshold)
    updated_frame = np.zeros_like(frame)
    updated_frame[mask] = frame[mask]

    return updated_frame

def create_face_mask_with_colors(image):
    """
    Detects the face using MediaPipe and extracts the facial region.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            points = np.array(
                [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                 for landmark in face_landmarks.landmark], dtype=np.int32
            )
            hull = cv2.convexHull(points)
            cv2.fillPoly(mask, [hull], 255)
    
    masked_face = cv2.bitwise_and(image, image, mask=mask)
    return masked_face

def extract_rgb_signals(frame):
    """
    Extracts the mean RGB values from the face region.
    """
    face_pixels = frame[frame.sum(axis=2) > 0]  # Exclude black pixels

    if len(face_pixels) == 0:
        return (0, 0, 0)  # Return zeros if no valid pixels found

    mean_r = int(np.mean(face_pixels[:, 2]))  # Red channel
    mean_g = int(np.mean(face_pixels[:, 1]))  # Green channel
    mean_b = int(np.mean(face_pixels[:, 0]))  # Blue channel

    return (mean_r, mean_g, mean_b)

def estimate_heart_rate(r_signal, g_signal, b_signal, fps=30):
    """
    Estimate heart rate (BPM) from R, G, B signals using Welch’s method.
    """
    if len(g_signal) < 30:  # Need at least 1 second of data
        return 0

    # Use Green Channel (best for rPPG)
    signal = np.array(g_signal) - np.mean(g_signal)  # Detrend signal

    # Apply Welch's method for frequency analysis
    freqs, psd = welch(signal, fs=fps, nperseg=len(signal)//2)

    # Define valid heart rate range (0.7 Hz to 3 Hz → 42-180 BPM)
    min_hr_hz, max_hr_hz = 0.7, 3.0
    valid_range = (freqs >= min_hr_hz) & (freqs <= max_hr_hz)

    if np.any(valid_range):
        peak_freq = freqs[valid_range][np.argmax(psd[valid_range])]
        heart_rate = peak_freq * 60  # Convert Hz to BPM
    else:
        heart_rate = 0  # No valid heart rate detected

    return round(heart_rate, 2)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get base64 encoded frame
        data = request.get_json()
        base64_image = data['frame'].split(',')[1]  

        # Decode base64 to numpy array
        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Create the face mask with original colors
        face_mask = create_face_mask_with_colors(frame)

        # Apply blackout process
        face = blackout_outside_dynamic_threshold(face_mask)

        # Extract RGB signal
        mean_r, mean_g, mean_b = extract_rgb_signals(face)

        # Store RGB values in buffer
        rgb_buffer['r'].append(mean_r)
        rgb_buffer['g'].append(mean_g)
        rgb_buffer['b'].append(mean_b)

        # Keep only the last WINDOW_SIZE frames
        rgb_buffer['r'] = rgb_buffer['r'][-WINDOW_SIZE:]
        rgb_buffer['g'] = rgb_buffer['g'][-WINDOW_SIZE:]
        rgb_buffer['b'] = rgb_buffer['b'][-WINDOW_SIZE:]

        # Calculate heart rate
        heart_rate = estimate_heart_rate(rgb_buffer['r'], rgb_buffer['g'], rgb_buffer['b'], FPS)

        # Overlay RGB values and HR onto the frame
        cv2.putText(face, f"HR: {heart_rate} BPM", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode processed frame back to base64
        _, buffer = cv2.imencode('.jpg', face)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'frame': f'data:image/jpeg;base64,{processed_base64}',
            'heart_rate': heart_rate
        })

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)






# from flask import Flask, request, jsonify, render_template
# import cv2
# import numpy as np
# import base64
# import mediapipe as mp
# from collections import deque

# app = Flask(__name__)

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True)


# def blackout_outside_dynamic_threshold(frame, lower_factor=0.48, upper_factor=1.74):
#     """ Blackouts pixels that fall outside a dynamically set threshold range. """
#     gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     non_zero_pixels = gray_image[gray_image != 0]
#     average_value = np.median(non_zero_pixels)
#     lower_threshold = max(0, int(average_value * lower_factor))
#     upper_threshold = min(255, int(average_value * upper_factor))
#     mask = (gray_image >= lower_threshold) & (gray_image <= upper_threshold)
#     updated_frame = np.zeros_like(frame)
#     updated_frame[mask] = frame[mask]
#     return updated_frame

# def create_face_mask_with_colors(image):
#     """ Detects the face using MediaPipe and extracts the facial region. """
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_image)
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             points = np.array([(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
#                                for landmark in face_landmarks.landmark], dtype=np.int32)
#             hull = cv2.convexHull(points)
#             cv2.fillPoly(mask, [hull], 255)

#     masked_face = cv2.bitwise_and(image, image, mask=mask)
#     return masked_face

# def extract_rgb_signals(frame):
#     """ Extracts the mean RGB values from the face region. """
#     face_pixels = frame[frame.sum(axis=2) > 0]  # Exclude black pixels

#     if len(face_pixels) == 0:
#         return (0, 0, 0)

#     mean_r = int(np.mean(face_pixels[:, 2]))  # Red channel
#     mean_g = int(np.mean(face_pixels[:, 1]))  # Green channel
#     mean_b = int(np.mean(face_pixels[:, 0]))  # Blue channel

#     return (mean_r, mean_g, mean_b)

# def draw_rgb_waveform(frame):
#     """ Draws real-time RGB signal waveforms in the top-right corner of the frame. """
#     graph_width, graph_height = 200, 100  # Graph dimensions
#     x_offset, y_offset = frame.shape[1] - graph_width - 10, 10  # Top-right position

#     # Create an overlay area for the graph
#     graph_bg = (50, 50, 50)  # Dark background for contrast
#     cv2.rectangle(frame, (x_offset, y_offset), 
#                   (x_offset + graph_width, y_offset + graph_height), 
#                   graph_bg, -1)

#     # Convert RGB values to y-coordinates for plotting
#     def normalize(val, max_val=255):
#         return int(y_offset + graph_height - (val / max_val) * graph_height)

#     # Create point lists
#     points_r = np.array([[x_offset + i * (graph_width // N), normalize(val)] 
#                           for i, val in enumerate(rgb_buffer["r"])], np.int32)
#     points_g = np.array([[x_offset + i * (graph_width // N), normalize(val)] 
#                           for i, val in enumerate(rgb_buffer["g"])], np.int32)
#     points_b = np.array([[x_offset + i * (graph_width // N), normalize(val)] 
#                           for i, val in enumerate(rgb_buffer["b"])], np.int32)

#     # Draw the RGB signals
#     if len(points_r) > 1:
#         cv2.polylines(frame, [points_r], isClosed=False, color=(0, 0, 255), thickness=2)  # Red
#         cv2.polylines(frame, [points_g], isClosed=False, color=(0, 255, 0), thickness=2)  # Green
#         cv2.polylines(frame, [points_b], isClosed=False, color=(255, 0, 0), thickness=2)  # Blue

#     return frame


# # Store last N frames' RGB values for plotting
# N = 100  # Number of points in the graph
# rgb_buffer = {
#     "r": deque(maxlen=N),
#     "g": deque(maxlen=N),
#     "b": deque(maxlen=N),
# }

# # Store the variable sent once
# my_variable = None

# @app.route('/')
# def index():
#     return render_template('index1.html')

# @app.route('/set_variable', methods=['POST'])
# def set_variable():
#     global my_variable
#     data = request.get_json()
#     my_variable = data.get('variable', None)
#     print(f"Variable received: {my_variable}")
#     rgb_buffer['r'].clear()
#     rgb_buffer['g'].clear()
#     rgb_buffer['b'].clear()
#     return jsonify({'status': 'Variable received successfully'})






# import numpy as np
# import scipy.signal

# # ... [rest of your existing imports and code]

# def extract_bvp_signal(rgb_values):
#     """ Converts RGB values into a Blood Volume Pulse (BVP) signal. """
#     # Simple approach: Use the green channel for BVP estimation
#     bvp_signal = np.array(rgb_values["g"])  # Assuming you're using green channel
#     return bvp_signal

# def estimate_heart_rate(bvp_signal, fps=30):
#     """ Estimates heart rate from the BVP signal using peak detection. """
#     # Bandpass filter the BVP signal
#     nyquist = 0.5 * fps
#     low = 0.5 / nyquist
#     high = 4 / nyquist
#     b, a = scipy.signal.butter(1, [low, high], btype='band')
#     filtered_bvp = scipy.signal.filtfilt(b, a, bvp_signal)

#     # Find peaks in the filtered BVP signal
#     peaks, _ = scipy.signal.find_peaks(filtered_bvp, distance=fps/2.5)  # Minimum distance between peaks
#     heart_rate = len(peaks) * (60 / fps)  # Convert to BPM
#     return heart_rate

# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     try:
#         # Get base64 encoded frame
#         data = request.get_json()
#         base64_image = data['frame'].split(',')[1]  
#         image_bytes = base64.b64decode(base64_image)
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Create the face mask with original colors
#         face_mask = create_face_mask_with_colors(frame)

#         if face_mask is None or not np.any(face_mask):
#             frame_with_graph = draw_rgb_waveform(frame)
#             cv2.putText(frame_with_graph, "No face detected", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         else:
#             face = blackout_outside_dynamic_threshold(face_mask)
#             mean_r, mean_g, mean_b = extract_rgb_signals(face)

#             rgb_buffer["r"].append(mean_r)
#             rgb_buffer["g"].append(mean_g)
#             rgb_buffer["b"].append(mean_b)

#             frame_with_graph = draw_rgb_waveform(frame)

#             # Extract BVP signal from RGB values
#             bvp_signal = extract_bvp_signal(rgb_buffer)

#             # Estimate heart rate from BVP signal
#             heart_rate = estimate_heart_rate(np.array(bvp_signal))

#             cv2.putText(frame_with_graph, f"R: {mean_r}, G: {mean_g}, B: {mean_b}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#             cv2.putText(frame_with_graph, f"Heart Rate: {int(heart_rate)} bpm", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         _, buffer = cv2.imencode('.jpg', frame_with_graph)
#         processed_base64 = base64.b64encode(buffer).decode('utf-8')

#         return jsonify({'frame': f'data:image/jpeg;base64,{processed_base64}'})
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # ... [rest of your existing code]



# if __name__ == '__main__':
#     app.run(debug=True)
