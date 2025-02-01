# Man Script
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from bson import ObjectId
import bcrypt
import certifi
from typing import List
import uvicorn
import traceback
import shutil
import uuid
from datetime import datetime
from google.cloud import vision
from google.oauth2 import service_account
import pandas as pd
import joblib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
origins = ["http://localhost:3000", "http://localhost:3001"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = "./uploads"
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

# DB Settings
# DB Configurations - NVD DB
MONGODB_CONNECTION_URL = "mongodb+srv://dbuser:111222333@cluster0.3ktcg.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = AsyncIOMotorClient(MONGODB_CONNECTION_URL, tlsCAFile=certifi.where())
db = client["nvd_db"]
user_collection = db["users"]
records_collection = db["records"]
feedback_collection = db["feedback-reinforcement"]
direction_records_collection = db["preposition-game records"]
vocabulary_records_collection = db.vocabulary_records

class UserSignUpRequest(BaseModel):
    guardian_name: str = Field(..., min_length=3, max_length=100)
    guardian_email: str = Field(..., min_length=10)
    guardian_contact: str = Field(..., min_length=10, max_length=15)
    child_name: str = Field(..., min_length=3, max_length=100)
    child_age: int = Field(..., ge=1, le=99)
    child_gender: str
    password: str = Field(..., min_length=8, max_length=100)
    vocabulary: int = Field(0, ge=0, description="Vocabulary score, default is 0")
    identify_difference: int = Field(0, ge=0, description="Identify difference score, default is 0")
    # avatar: Optional[str]

# Endpoint for handling user sign-up
@app.post("/signup")
async def sign_up(data: UserSignUpRequest):
    try:
        # Handle the avatar upload if provided
        avatar_filename = None
        # if avatar:
        #     avatar_filename = f"{uuid.uuid4()}_{avatar.filename}"
        #     avatar_path = UPLOADS_DIR / avatar_filename
        #     with avatar_path.open("wb") as buffer:
        #         shutil.copyfileobj(avatar.file, buffer)

        # Example response - Replace with database logic as required
        user_data = {
            "guardian_name": data.guardian_name,
            "guardian_email": data.guardian_email,
            "guardian_contact": data.guardian_contact,
            "child_name": data.child_name,
            "child_age": data.child_age,
            "child_gender": data.child_gender,
            "password": data.password
            # "avatar": avatar_filename,
        }

        await user_collection.insert_one(user_data)
        # Return success response
        return JSONResponse(status_code=201, content={"message": "Registration successful"})

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=400, detail=str(e))

class SignInData(BaseModel):
    guardian_email: str
    password: str

@app.post("/token")
async def sign_in(data: SignInData):
    user = await user_collection.find_one({"guardian_email": data.guardian_email})
    print(user)
    if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
    if not user or not data.password == user['password']:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "access_token": '',
        "token_type": "bearer",
        "user": {
            "_id": str(user['_id']),
            "full_name": user['child_name'],
            "email": user['guardian_email'],
            "username": user['guardian_email'],
            "contact_number": user['guardian_contact'],
            "vocabulary": user.get('vocabulary', 0),
            "identify_difference": user.get('identify_difference', 0),
            "created_at": ""
        }
    }

@app.get("/users", response_model=List[UserSignUpRequest])
async def get_all_users():
    users = await user_collection.find().to_list(length=None)
    if not users:
        raise HTTPException(status_code=404, detail="No users found")
    return users

class UpdateScoreRequest(BaseModel):
    vocabulary: int = Field(None, ge=0, description="Vocabulary score to update")
    identify_difference: int = Field(None, ge=0, description="Identify difference score to update")

# Endpoint for updating user scores
@app.put("/users/{user_id}/update_score")
async def update_score(user_id: str, data: UpdateScoreRequest):
    try:
        # Ensure at least one field is provided
        if data.vocabulary is None and data.identify_difference is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one score (vocabulary or identify_difference) must be provided"
            )
        
        # Convert user_id to ObjectId
        if not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID format"
            )
        
        # Prepare the update query
        update_fields = {}
        if data.vocabulary is not None:
            update_fields["vocabulary"] = data.vocabulary
        if data.identify_difference is not None:
            update_fields["identify_difference"] = data.identify_difference

        # Update the user document in the database
        result = await user_collection.update_one(
            {"_id": ObjectId(user_id)},  # Match user by ID
            {"$set": update_fields}  # Update the provided fields
        )

        # Check if any document was modified
        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {"message": "User score updated successfully"}

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=400, detail=str(e))


# Feedback
class FeedbackModel(BaseModel):
    user: str
    recorded_date: datetime = Field(default_factory=datetime.utcnow)
    category: str
    answers: List[int]
    score: int
    moves: int
    time_taken: int
    difficulty: int
    observation: Optional[str] = None
    made_by: str

@app.post("/feedback", status_code=201)
async def create_feedback(feedback: FeedbackModel):
    feedback_data = feedback.dict()
    result = await feedback_collection.insert_one(feedback_data)
    if result.inserted_id:
        return {"message": "Feedback record created successfully", "id": str(result.inserted_id)}
    raise HTTPException(status_code=500, detail="Failed to create feedback record")

class DirectionRecordModel(BaseModel):
    user: str
    level: int
    score: float
    time_taken: int
    steps_made: int

# POST endpoint for creating a direction record
@app.post("/direction-records", status_code=201)
async def create_direction_record(record: DirectionRecordModel):
    record_data = record.dict()
    result = await direction_records_collection.insert_one(record_data)
    if result.inserted_id:
        return {"message": "Direction record created successfully", "id": str(result.inserted_id)}
    raise HTTPException(status_code=500, detail="Failed to create direction record")


# Get All Feedback Records
def serialize_document(doc):
    """Convert MongoDB document ObjectId fields to string."""
    doc["_id"] = str(doc["_id"])
    return doc

@app.get("/feedback", status_code=200)
async def get_all_feedback():
    feedback_cursor = feedback_collection.find()
    feedback_list = await feedback_cursor.to_list(100)  # Limit to 100 for safety
    serialized_feedback = [serialize_document(feedback) for feedback in feedback_list]
    return {"feedback_records": serialized_feedback}

# Get Feedback Records by User
@app.get("/feedback/user/{user_id}", status_code=200)
async def get_feedback_by_user(user_id: str):
    feedback_list = await feedback_collection.find({"user": user_id}).to_list(100)  # Limit to 100 for safety
    if feedback_list:
        return {"user_feedback_records": feedback_list}
    raise HTTPException(status_code=404, detail="No feedback records found for the user")

@app.patch("/feedback", status_code=200)
async def update_feedback_opinion(observation: str = Body(..., embed=True)):
    result = await feedback_collection.update_one(
        {"_id": "6749c9c948676f39ff74cc0c"},  # Find by feedback ID
        {"$set": {"observation": observation}}  # Update the observation field
    )
    if result.modified_count > 0:
        return {"message": f"Updated observation for feedback ID."}
    raise HTTPException(status_code=404, detail="No feedback record found for the given ID or no updates made.")


# Difficulty Adjust
class DifficultyRequest(BaseModel):
    answers: list
    score: int
    moves: int
    time_taken: int

MODEL_DIFFICULTY = joblib.load('difficulty_predictor.joblib')
SCALER = joblib.load('scaler.joblib')

# Endpoint to predict difficulty
@app.post("/predict-difficulty")
async def predict_difficulty(data: DifficultyRequest):
    """
    Predicts the difficulty based on the input parameters.

    Args:
    - data (DifficultyRequest): The input data for prediction.

    Returns:
    - dict: The predicted difficulty.
    """
    try:
        # Extract features from answers
        answers = np.array(data.answers)
        answers_mean = np.mean(answers)
        answers_var = np.var(answers)
        answers_min = np.min(answers)
        answers_max = np.max(answers)

        # Prepare input data
        input_data = pd.DataFrame([{
            "score": data.score,
            "moves": data.moves,
            "time_taken": data.time_taken,
            "answers_mean": answers_mean,
            "answers_var": answers_var,
            "answers_min": answers_min,
            "answers_max": answers_max
        }])

        # Scale input data
        input_scaled = SCALER.transform(input_data)

        # Predict difficulty
        predicted_difficulty = MODEL_DIFFICULTY.predict(input_scaled)
        return {"predicted_difficulty": predicted_difficulty[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting difficulty: {str(e)}")

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

class UserQuery(BaseModel):
    query: str

def get_nvld_advice(user_input: str) -> str:
    """Generate advice for NVLD patients based on user input."""
    prompt = f"Advice for someone with NVLD: {user_input}\n\nAdvice:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=150,
            temperature=0.7,  # Controls randomness (lower = more deterministic)
            top_p=0.9,  # Nucleus sampling for better coherence
            repetition_penalty=1.2,  # Reduces repetitive text
            do_sample=False  # Enables sampling for diverse responses
        )
    
    advice = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the generated advice
    advice = advice.replace(prompt, "").strip()
    
    return advice

@app.post("/get_nvld_advice")
async def get_advice(user_query: UserQuery):
    """API endpoint to generate advice for NVLD based on user query."""
    advice = get_nvld_advice(user_query.query)
    return {"advice": advice}

# Model for Vocabulary Record
class VocabularyRecordModel(BaseModel):
    user: str
    activity: str
    type: str
    recorded_date: datetime = Field(default_factory=datetime.utcnow)
    score: float
    time_taken: int
    difficulty: int
    suggestions: Optional[List[str]]

def format_id(record):
    record["id"] = str(record.pop("_id"))
    return record

@app.post("/vocabulary-records", status_code=201)
async def create_vocabulary_record(record: VocabularyRecordModel):
    record_data = record.dict()
    result = await vocabulary_records_collection.insert_one(record_data)
    if result.inserted_id:
        return {"message": "Vocabulary record created successfully", "id": str(result.inserted_id)}
    raise HTTPException(status_code=500, detail="Failed to create vocabulary record")

@app.get("/vocabulary-records", response_model=List[dict])
async def get_all_vocabulary_records():
    records = await vocabulary_records_collection.find().to_list(length=100)
    return [format_id(record) for record in records]

@app.get("/vocabulary-records/user/{user}", response_model=List[dict])
async def get_vocabulary_records_by_user(user: str):
    records = await vocabulary_records_collection.find({"user": user}).to_list(length=100)
    if not records:
        raise HTTPException(status_code=404, detail="No records found for the given user")
    return [format_id(record) for record in records]

@app.delete("/vocabulary-records/{record_id}", status_code=200)
async def delete_vocabulary_record(record_id: str):
    result = await vocabulary_records_collection.delete_one({"_id": ObjectId(record_id)})
    if result.deleted_count:
        return {"message": "Vocabulary record deleted successfully"}
    raise HTTPException(status_code=404, detail="Vocabulary record not found")


# Define the QA model for receiving input
class QA(BaseModel):
    phrase: str

@app.post("/puzzle-endpoint")
async def recognize_im(qa: QA):
    try:
       
        return {"ai_response": ''}

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

@app.post("/im-compare-endpoint")
async def recognize_im(qa: QA):
    try:
       
        return {"ai_response": ''}

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

@app.post("/object-describe-endpoint")
async def recognize_im(qa: QA):
    try:
       
        return {"ai_response": ''}

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")

@app.post("/vocablury-endpoint")
async def recognize_im(qa: QA):
    try:
       
        return {"ai_response": ''}

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error occurred: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}. See server logs for more details.")


# OCR

class RecognizedResult(BaseModel):
    recognized_text: str


@app.post("/api/recognize-word-ocr")
async def recognize_and_parse_prescription(file: UploadFile = File(...)):
    try:
        # Step 1: Save the uploaded file
        file_path = f"{UPLOADS_DIR}/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        # Step 2: Initialize Google Cloud Vision client for OCR
        credentials = service_account.Credentials.from_service_account_file('ocr-key.json')
        client = vision.ImageAnnotatorClient(credentials=credentials)

        # Load the image into memory
        with open(file_path, "rb") as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(f'{response.error.message}')

        # Get the recognized text
        recognized_text = texts[0].description if texts else ""

        print(recognized_text)

        # Step 4: Return the recognized text and parsed prescription details
        return {
            "recognized_text": recognized_text,
        }

    except Exception as e:
        return {"error": f"Error: {str(e)}"}




if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
