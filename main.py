import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template, app
import cv2
import numpy as np
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torch import nn
import base64
from io import BytesIO

app = Flask(__name__)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 36)  # 36 output classes
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def segment_handwriting(image):
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    digits = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Filter out small contours
            digit = thresh[y:y + h, x:x + w]
            digit = cv2.resize(digit, (28, 28))
            digits.append(digit)

    return digits


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize to 28x28
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

    return resized

#####

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN()
## while loading pre trained model in other devices - you need to map it to new device
state_dict = torch.load('cnn_model.pth', map_location=device, weights_only=True)
model.load_state_dict(state_dict)


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the image file from the request
#     file = request.files['image']
#
#     # Read the image
#     img = Image.open(io.BytesIO(file.read())).convert('L')
#
#     # Preprocess the image
#     img_tensor = transform(img).unsqueeze(0)
#
#     pil_img = Image.fromarray(img_tensor)
#
#     # Make prediction
#     with torch.no_grad():
#         output = model(img_tensor)
#         _, predicted = torch.max(output, 1)
#         digit = predicted.item()
#
#
#     return jsonify({'digit': digit})


@app.route('/predict', methods=['POST'])
def predict_guess():
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    digits = segment_handwriting(img)

    # print(digits)
    predictions = []

    for digit in digits:
        # Convert to PIL Image
        pil_img = Image.fromarray(digit)

        # Preprocess the image
        img_tensor = transform(pil_img).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)
            predictions.append(int(predicted.item()))

    return jsonify({'digits': predictions})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)