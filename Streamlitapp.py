import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import boto3
import os
import mysql.connector
import pickle

# Load model from S3
s3 = boto3.client("s3")
bucket_name = "testingmys3bucket1991"
model_path = "/home/ec2-user/cnn_sentiment_model.h5"  # Updated path for EC2
s3.download_file(bucket_name, "cnn_sentiment_model.h5", model_path)

model = tf.keras.models.load_model(model_path)

# Load tokenizer
tokenizer_path = "/home/ec2-user/tokenizer.pkl"
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

# MySQL Database Configuration
db_config = {
    "host": "sentiment-db.cbkegyy2203z.ap-south-1.rds.amazonaws.com",
    "user": "admin",
    "password": "your_password",
    "database": "sentiment_analysis"
}

# Function to log user logins
def log_user_login(username):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "INSERT INTO user_logins (username) VALUES (%s)"
        cursor.execute(query, (username,))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ User {username} login recorded.")
    except Exception as e:
        print("❌ Error logging user login:", e)

# Streamlit UI
st.title("Sentiment Analysis Web App")

# Simulate user login (In real scenario, use authentication)
username = "streamlit_user"  # Replace with actual username input
log_user_login(username)

# Sentiment Analysis Input
user_input = st.text_input("Enter a Tweet:")
if st.button("Predict"):
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=100, padding="post")
    prediction = np.argmax(model.predict(padded_seq), axis=1)
    
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Irrelevant"}
    st.write(f"Predicted Sentiment: {sentiment_map[prediction[0]]}")
