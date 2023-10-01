from fastapi import FastAPI, Path
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Interaction(BaseModel):
    user_id: int
    item_id: int

# Load user_id_to_token and item_id_to_token from JSON files
with open("investors/user_id_to_token.json", "r") as user_file:
    user_id_to_token = json.load(user_file)

with open("investors/item_id_to_token.json", "r") as item_file:
    item_id_to_token = json.load(item_file)

with open("innovators/user_id_to_token.json", "r") as user_file:
    user_id_to_token2 = json.load(user_file)

with open("innovators/item_id_to_token.json", "r") as item_file:
    item_id_to_token2 = json.load(item_file)

# Load the saved model
model = tf.keras.models.load_model("investors/recommender_model_investors.h5")
model2 = tf.keras.models.load_model("innovators/recommender_model_innovators.h5")

@app.get("/getInvestors/{user_id}")
def predict(user_id: str = Path(..., title="The user ID to get predictions for")):
    # Check if the user_id is valid
    if user_id in user_id_to_token:
        tokenized_user_id = user_id_to_token[user_id]
        all_item_ids = np.array(list(range(len(item_id_to_token))))  # All possible item IDs
        user_ids_for_prediction = np.array([tokenized_user_id] * len(item_id_to_token))
        predictions = model.predict([user_ids_for_prediction, all_item_ids])
        predicted_ratings = predictions.flatten()
        # Sort item IDs based on predicted ratings in descending order
        relevant_item_ids = np.argsort(predicted_ratings)[::-1]
        # Map tokenized item IDs back to original item IDs
        relevant_item_ids = [key for key, value in item_id_to_token.items() if value in relevant_item_ids]
        top_3_recommendations = relevant_item_ids[:3]
        return {"user_id": user_id, "top_recommendations": top_3_recommendations}
    else:
        return {"error": "Invalid user_id"}

@app.get("/getInnovators/{user_id}")
def predict(user_id: str = Path(..., title="The user ID to get predictions for")):
    # Check if the user_id is valid
    if user_id in user_id_to_token2:
        tokenized_user_id = user_id_to_token2[user_id]
        all_item_ids = np.array(list(range(len(item_id_to_token2))))  # All possible item IDs
        user_ids_for_prediction = np.array([tokenized_user_id] * len(item_id_to_token2))
        predictions = model2.predict([user_ids_for_prediction, all_item_ids])
        predicted_ratings = predictions.flatten()
        # Sort item IDs based on predicted ratings in descending order
        relevant_item_ids = np.argsort(predicted_ratings)[::-1]
        # Map tokenized item IDs back to original item IDs
        relevant_item_ids = [key for key, value in item_id_to_token2.items() if value in relevant_item_ids]
        top_3_recommendations = relevant_item_ids[:3]
        return {"user_id": user_id, "top_recommendations": top_3_recommendations}
    else:
        return {"error": "Invalid user_id"}
