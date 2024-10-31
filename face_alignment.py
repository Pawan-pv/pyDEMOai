# import torch
# import os
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# import cv2
# import numpy as np
# import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch'))

# # Import BiSeNet from model.py
# from model import BiSeNet

# # Load the BiSeNet model
# def load_model():
#     n_classes = 19  # For face parsing, typically 19 classes
#     model = BiSeNet(n_classes=n_classes)
#     model_path = '/workspaces/codespaces-blank/face-parsing.PyTorch/res/cp/79999_iter.pth'
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model weights not found at {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
#     model.eval()  # Set model to evaluation mode
#     return model

# # Process the face image
# def parse_face(image_path, model):
#     # Load image and resize
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (512, 512))  # BiSeNet expects 512x512 resolution

#     # Convert image to RGB and normalize
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
#     image = (image - 0.5) / 0.5  # Standardization to [-1, 1]

#     # Convert image to tensor
#     image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)  # Shape (1, 3, 512, 512)

#     # Get parsing result
#     with torch.no_grad():
#         output = model(image)[0]  # Output shape (1, 19, 512, 512)

#     # Get the class with the highest confidence
#     parsing = output.argmax(0).cpu().numpy()
#     return parsing

# # Display the parsed image
# def display_parsing(parsing):
#     # Generate random colors for each class
#     colors = np.random.randint(0, 255, size=(19, 3), dtype=np.uint8)
#     color_parsing = colors[parsing]

#     # Show parsed result
#     cv2.imshow("Parsed Face", color_parsing)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Main function
# def main():
#     model = load_model()
#     image_path = 'avatar.png'  # Replace with your image path
#     parsing = parse_face(image_path, model)
#     display_parsing(parsing)

# if __name__ == "__main__":
#     main()








# import torch
# import os
# import cv2
# import numpy as np
# import sys
# from PIL import Image  # Import Pillow for saving images as PNG

# # Set the environment variable for headless operation
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# # Add the path to the face-parsing.PyTorch module
# sys.path.append(os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch'))

# # Import BiSeNet from model.py   
# from model import BiSeNet

# # Load the BiSeNet model
# def load_model():
#     n_classes = 19  # For face parsing, typically 19 classes
#     model = BiSeNet(n_classes=n_classes)
#     model_path = '/workspaces/codespaces-blank/face-parsing.PyTorch/res/cp/79999_iter.pth'
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model weights not found at {model_path}")
#     model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
#     model.eval()  # Set model to evaluation mode
#     return model

# # Process the face image
# def parse_face(image_path, model):
#     # Load image and resize
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
#     print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
#     image = cv2.resize(image, (512, 512))  # BiSeNet expects 512x512 resolution

#     # Convert image to RGB and normalize
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
#     image = (image - 0.5) / 0.5  # Standardization to [-1, 1]

#     # Convert image to tensor
#     image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)  # Shape (1, 3, 512, 512)

#     # Get parsing result
#     with torch.no_grad():
#         output = model(image)[0]  # Output shape (1, 19, 512, 512)

#     # Get the class with the highest confidence
#     parsing = output.argmax(0).cpu().numpy()
#     return parsing

# # Save the parsed image using Pillow
# def save_parsing(parsing, output_path='parsed_face.png'):
#     # Generate random colors for each class
#     colors = np.random.randint(0, 255, size=(19, 3), dtype=np.uint8)
#     color_parsing = colors[parsing]

#     # Convert OpenCV image format to RGB format for PIL
#     color_parsing_rgb = cv2.cvtColor(color_parsing, cv2.COLOR_BGR2RGB)

#     # Use PIL to save the image as PNG
#     pil_image = Image.fromarray(color_parsing_rgb)
#     pil_image.save(output_path, format='PNG')
#     print(f"Parsed face saved to {output_path}")

# # Main function
# def main():
#     model = load_model()
#     image_path = 'avatar.png'  # Replace with your image path
#     parsing = parse_face(image_path, model)
#     save_parsing(parsing)

# if __name__ == "__main__":
#     main()











import torch
import os
import cv2
import numpy as np
import sys
import dlib  # Ensure you have dlib installed
# or use MediaPipe instead of dlib by commenting the above line and uncommenting below
# import mediapipe as mp

# Set the environment variable for headless operation
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add the path to the face-parsing.PyTorch module
sys.path.append(os.path.join(os.path.dirname(__file__), 'face-parsing.PyTorch'))

# Import BiSeNet from model.py
from model import BiSeNet

# Load the BiSeNet model
def load_model():
    n_classes = 19  # For face parsing, typically 19 classes
    model = BiSeNet(n_classes=n_classes)
    model_path = '/workspaces/codespaces-blank/face-parsing.PyTorch/res/cp/79999_iter.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()  # Set model to evaluation mode
    return model

# Face alignment using Dlib
def align_face(image):
    # Initialize dlib's face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Convert image to RGB for Dlib
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector(rgb_image)

    if len(detections) > 0:
        # Get the first detected face
        face = detections[0]
        landmarks = predictor(rgb_image, face)

        # Get the coordinates for eyes and mouth
        points = np.array([[p.x, p.y] for p in landmarks.parts()])
        # Calculate the center of the face for alignment
        center = np.mean(points, axis=0).astype(int)

        # Crop and align the face around the center point
        aligned_face = image[face.top():face.bottom(), face.left():face.right()]
        return aligned_face
    else:
        raise ValueError("No face detected in the image.")

# Process the face image
def parse_face(image, model):
    # Resize the aligned face
    image = cv2.resize(image, (512, 512))  # BiSeNet expects 512x512 resolution

    # Convert image to RGB and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = (image - 0.5) / 0.5  # Standardization to [-1, 1]

    # Convert image to tensor
    image = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0)  # Shape (1, 3, 512, 512)

    # Get parsing result
    with torch.no_grad():
        output = model(image)[0]  # Output shape (1, 19, 512, 512)

    # Get the class with the highest confidence
    parsing = output.argmax(0).cpu().numpy()
    return parsing

# Save the parsed image
def save_parsing(parsing, output_path='parsed_face.png'):
    # Generate random colors for each class
    colors = np.random.randint(0, 255, size=(19, 3), dtype=np.uint8)
    color_parsing = colors[parsing]

    # Save the parsed result as an image
    cv2.imwrite(output_path, color_parsing)
    print(f"Parsed face saved to {output_path}")

# Main function
def main():
    model = load_model()
    image_path = 'avatar.png'  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")

    # Align the face in the image
    aligned_face = align_face(image)
    
    # Process the aligned face for parsing
    parsing = parse_face(aligned_face, model)
    save_parsing(parsing)

if __name__ == "__main__":
    main()
