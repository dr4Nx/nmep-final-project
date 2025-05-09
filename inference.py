import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import time

class ASLClassifier(nn.Module):
    def __init__(self):
        super(ASLClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 28)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier()
model.load_state_dict(torch.load("asl_classifier_model.pth", map_location=device))
model.to(device)
model.eval()


classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del"] + ["space"]
idx_to_class = {i: c for i, c in enumerate(classes)}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("Webcam ASL translator running. Press 'q' to quit.")

captured_text = ""
last_prediction = None
prediction_start_time = None
capture_flash_time = 0
CONFIRMATION_TIME = 2.0 
FLASH_DURATION = 0.2    

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction = "No hand"
    confidence = 0.0
    bbox = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        input_tensor = torch.tensor(landmarks).view(1, 21, 3).permute(0, 2, 1).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            prediction = idx_to_class[pred_idx.item()]

        h, w, _ = frame.shape
        x_vals = [lm.x * w for lm in hand_landmarks.landmark]
        y_vals = [lm.y * h for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x_vals)), int(max(x_vals))
        y_min, y_max = int(min(y_vals)), int(max(y_vals))
        bbox = (x_min, y_min, x_max, y_max)

        current_time = time.time()
        if prediction == last_prediction:
            if prediction_start_time and current_time - prediction_start_time >= CONFIRMATION_TIME:
                if prediction != "space" and prediction != "del":
                    captured_text += prediction
                elif prediction == "del":
                    captured_text = captured_text[:-1]
                elif prediction == "space":
                    captured_text += " "
                prediction_start_time = current_time 
                capture_flash_time = current_time
        else:
            last_prediction = prediction
            prediction_start_time = current_time

    if bbox:
        x_min, y_min, x_max, y_max = bbox
        if time.time() - capture_flash_time <= FLASH_DURATION:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), -1)
        else:
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.putText(frame, f"Prediction: {prediction} ({confidence:.2f})", (10, 40), cv2.FONT_HERSHEY_DUPLEX,
                1.2, (0, 100, 0), 2)
    cv2.putText(frame, f"Captured: {captured_text}", (10, 80), cv2.FONT_HERSHEY_DUPLEX,
                1.0, (0, 0, 0), 2)

    cv2.imshow("ASL Translator", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        captured_text = ""
    elif key == ord('d'):
        captured_text = captured_text[:-1]

cap.release()
cv2.destroyAllWindows()
