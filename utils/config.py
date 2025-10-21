import os
from dotenv import load_dotenv
import joblib
import tensorflow as tf

# load .env file
load_dotenv(override=True)

# Get the variables
APP_NAME = os.getenv("APP_NAME")
VERSION = os.getenv("VERSION")
API_SECRET_KEY = os.getenv("API_SECRET_KEY")


FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load processor and model
preprocessor = joblib.load(os.path.join(FOLDER_PATH, "models", "preprocessor.pkl"))
model = tf.keras.models.load_model(os.path.join(FOLDER_PATH, "models", "best_model.keras"))

