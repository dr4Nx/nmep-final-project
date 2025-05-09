import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np

# === Define the same model class used in training ===
class ASLClassifier(nn.Module):
    def __init__(self):
        super(ASLClassifier, self).__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 28)  # 27 classes (A-Z + space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier()
model.load_state_dict(torch.load("asl_classifier_model.pth", map_location=device))
model.to(device)
model.eval()

# === Class label mapping ===
# Ensure this matches the order used during training
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space"] + ["nothing"]
idx_to_class = {i: c for i, c in enumerate(classes)}

# === MediaPipe setup ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# === Webcam setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

print("Webcam ASL translator running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    prediction = "No hand"
    confidence = 0.0
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

    # Draw prediction with confidence
    cv2.putText(frame, f"Prediction: {prediction} ({confidence:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 255, 0), 3)

    cv2.imshow("ASL Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()