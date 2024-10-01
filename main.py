import cv2
import mediapipe as mp
import numpy as np
import pickle

def image_processed(hand_img):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,  # Changed to static mode for image processing
                           max_num_hands=2, 
                           min_detection_confidence=0.5, 
                           min_tracking_confidence=0.5)
    
    output = hands.process(img_rgb)
    hands.close()

    landmarks_list = []
    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:
            landmarks = [np.array([landmark.x, landmark.y, landmark.z]) for landmark in hand_landmarks.landmark]
            landmarks = np.array(landmarks).flatten()
            landmarks_list.append(landmarks)

    return landmarks_list

# Load the pre-trained models
with open('one_hand_model.pkl', 'rb') as f:
    one_hand_model = pickle.load(f)
    
with open('two_hand_model.pkl', 'rb') as f:
    two_hand_model = pickle.load(f)

# Load the image
image_path = '10.png'  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print(f"Could not read the image from {image_path}")
    exit()

# Process the image
landmarks_list = image_processed(image)

# Determine the model and make predictions
if len(landmarks_list) == 0:
    output = "No hands detected"
else:
    if len(landmarks_list) == 1:
        model = one_hand_model
        data = np.array(landmarks_list[0]).reshape(1, -1)
    elif len(landmarks_list) == 2:
        model = two_hand_model
        data = np.concatenate(landmarks_list[:2]).reshape(1, -1) if len(landmarks_list) > 1 else np.array(landmarks_list[0]).reshape(1, -1)
        
    y_pred = model.predict(data)
    output = str(y_pred[0])

# Display the result on the image
image = cv2.putText(image, 'OUTPUT: ' + output, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 3, cv2.LINE_AA)

# Show the image with prediction
cv2.imshow('Prediction', image)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()
