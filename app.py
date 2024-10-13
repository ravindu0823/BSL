from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained models
with open('one_hand_model.pkl', 'rb') as f:
    one_hand_model = pickle.load(f)

with open('two_hand_model.pkl', 'rb') as f:
    two_hand_model = pickle.load(f)


def image_processed(hand_img):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,  # Set to False for real-time video processing
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    output = hands.process(img_rgb)
    hands.close()

    landmarks_list = []
    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:
            landmarks = [np.array([landmark.x, landmark.y, landmark.z])
                         for landmark in hand_landmarks.landmark]
            landmarks = np.array(landmarks).flatten()
            landmarks_list.append(landmarks)

    return landmarks_list

# Route to serve the index HTML page


@app.route('/')
def home():
    # Flask automatically looks for index.html in the templates folder
    return render_template('index.html')

# Endpoint to handle frame predictions


@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_data = request.form['image']
        # Log part of the base64 string for debugging
        print(f"Received image data: {img_data[:50]}...")

        # Check if the image data has the base64 prefix (e.g., 'data:image/jpeg;base64,')
        if img_data.startswith('data:image'):
            # Remove the prefix, keep the base64 part
            img_data = img_data.split(',')[1]

        # Decode the base64 image
        img_data = base64.b64decode(img_data)
        npimg = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Process the image
        landmarks_list = image_processed(img)

        if len(landmarks_list) == 0:
            return jsonify({'prediction': 'No hands detected'})

        # Select the model based on the number of hands detected
        if len(landmarks_list) == 1:
            model = one_hand_model
            data = np.array(landmarks_list[0]).reshape(1, -1)
        elif len(landmarks_list) == 2:
            model = two_hand_model
            data = np.concatenate(landmarks_list[:2]).reshape(1, -1)

        # Predict the output
        y_pred = model.predict(data)
        prediction = str(y_pred[0])

        # Extract the meaningful part (number or letter)
        if '-' in prediction:  # If prediction is in the form of "Word - symbol"
            # Get the second part and strip whitespace
            symbol = prediction.split('-')[1].strip()
        else:
            symbol = prediction.strip()  # Just in case there's no '-'

        # Check if symbol is a digit or a letter
        if symbol.isdigit():
            result = symbol  # If it's a number, return the digit
        else:
            # If it's a letter, return the uppercase letter
            result = prediction.split('-')[0].strip()

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
