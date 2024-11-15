import os
import uuid
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil

app = FastAPI()

# Serve static files from the `static` directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/generate-video")
async def generate_video(
    text: str = Form(...),
    image: UploadFile = File(...)
):
    # Save the uploaded image file temporarily
    temp_image_path = f"temp_{image.filename}"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Generate a unique filename for the output video
    video_filename = f"{uuid.uuid4()}.mp4"
    video_path = f"static/{video_filename}"

    # Call your lip-sync function here with `text` and `temp_image_path`
    # Replace this with your actual model call to generate the video
    # e.g., lipsync_model.generate_video(temp_image_path, text, video_path)

    # For now, this is just a placeholder; replace with actual video generation logic
    shutil.copy(temp_image_path, video_path)  # Temporarily copying image as mock video

    # Clean up the temporary image file
    os.remove(temp_image_path)

    # Return the URL to the generated video
    video_url = f"http://127.0.0.1:8000/static/{video_filename}"
    response_data = {"video_url": video_url}

    return JSONResponse(content=response_data)
