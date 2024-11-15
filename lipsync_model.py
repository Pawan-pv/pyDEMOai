import torch
import dlib
import cv2
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip
from Wav2Lip.models import Wav2Lip
from Wav2Lip.inference import load_model as wav2lip_load_model, lip_sync

# Load the pre-trained Wav2Lip model
def load_model():
    model = Wav2Lip()  # Initialize the Wav2Lip model
    model.load_state_dict(torch.load("checkpoints/wav2lip_gan.pth"))
    model.eval()  # Set the model to evaluation mode
    return model

# Load Dlib’s face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/workspaces/codespaces-blank/shape_predictor_68_face_landmarks.dat")

# Function to detect landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        return [(p.x, p.y) for p in landmarks.parts()]
    return None

# Function to convert text to speech
def generate_audio(text, language='en'):
    tts = gTTS(text, lang=language)
    tts.save("audio.mp3")
    return "audio.mp3"

# Generate frames for lip-syncing
def generate_frames(image_path, audio_path, model):
    # Generate lip-synced video frames with Wav2Lip
    frames = lip_sync(model, image_path, audio_path)
    return frames

# Create a video from frames and add audio
def create_video(frames, audio_path, fps=30):
    clip = ImageSequenceClip(frames, fps=fps)
    audio = AudioFileClip(audio_path)
    clip = clip.set_audio(audio)
    output_path = "output.mp4"
    clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    return output_path

# Main function to process and generate the lip-sync video
def generate_lip_sync_video(image_path, text):
    # Load the model
    model = load_model()

    # Generate audio from text
    audio_path = generate_audio(text)

    # Generate lip-synced frames
    frames = generate_frames(image_path, audio_path, model)

    # Create the final video
    video_path = create_video(frames, audio_path)
    return video_path

# Example usage
video_file_path = generate_lip_sync_video("Screenshot.png", "Hello, this is a test.")
print("Generated video saved at:", video_file_path)
