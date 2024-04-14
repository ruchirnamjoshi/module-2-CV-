from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Response
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from measure import measure_dimensions  # Your real-time measurement module
from panorama import create_panorama  # Your panorama creation module

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PANORAMA_FOLDER'] = 'static/panoramas'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PANORAMA_FOLDER'], exist_ok=True)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # Now create the panorama
            panorama_image_path = create_panorama(video_path)  # This needs to be implemented
            return redirect(url_for('get_panorama', filename=panorama_image_path))
    else:
        return render_template('panorama.html')

def compute_integral_image(image):
    # Initialize the integral image with an extra row and column filled with zeros
    integral_image = np.zeros((image.shape[0] + 1, image.shape[1] + 1), dtype=np.uint64)

    # Compute the integral image
    for y in range(1, integral_image.shape[0]):
        for x in range(1, integral_image.shape[1]):
            integral_image[y, x] = image[y-1, x-1] + integral_image[y-1, x] + integral_image[y, x-1] - integral_image[y-1, x-1]

    # Remove the first row and column to get the final integral image
    return integral_image[1:, 1:]
def gen_frames2():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute the integral image for the grayscale frame
            integral_image = compute_integral_image(gray_frame)

            # Normalize and convert to uint8 for visualization
            integral_image_normalized = cv2.normalize(integral_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Resize original and integral images to the same height for horizontal stacking
            height = max(frame.shape[0], integral_image_normalized.shape[0])
            frame_resized = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))

            # Convert the grayscale integral image to a 3-channel image
            integral_image_3_channel = cv2.cvtColor(integral_image_normalized, cv2.COLOR_GRAY2BGR)

            integral_image_resized = cv2.resize(integral_image_3_channel, (
            int(integral_image_normalized.shape[1] * height / integral_image_normalized.shape[0]), height))

            # Combine the original frame and the integral image horizontally
            combined_image = np.hstack((frame_resized, integral_image_resized))

            ret, buffer = cv2.imencode('.jpg', combined_image)
            combined_frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + combined_frame + b'\r\n')

@app.route('/video_feed2')
def video_feed2():
    # This route now returns the combined regular and integral feed
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/panorama/<filename>')
def get_panorama(filename):
    return send_from_directory(app.config['PANORAMA_FOLDER'], filename)


@app.route('/measure')
def measure():
    return render_template('measure.html')

camera_mtx = np.load('camera_mtx.npy')
dist_coeffs = np.load('dist_coeffs.npy')
distance_to_object = 360

latest_measurements = {'width': 0, 'height': 0}

def gen_frames():
    cap = cv2.VideoCapture(0)  # Use the correct device index
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = measure_dimensions(frame, camera_mtx, dist_coeffs, distance_to_object)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/measurements')
def measurements():
    print(f"Width: {latest_measurements['width']:.2f} mm, Height: {latest_measurements['height']:.2f} mm")
    return jsonify(latest_measurements)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/integral_feed')
def integral_feed():
    return render_template('integral_image.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('action') == 'Start':
            return render_template('index.html', start=True)
    return render_template('index.html', start=False)
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
