import roboflow
import os
from dotenv import load_dotenv

# dot env
load_dotenv()

# api key
robo_api = os.getenv("ROBO_API")

# data and paths
rf = roboflow.Roboflow(api_key=robo_api)
project = rf.workspace("fracture-pcnbk")
hitter_path = "C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/datasets/roi_frames_2/"

# upload data
project.upload_dataset(
    dataset_path = hitter_path, 
    project_name= "pitch-id-roi-hts2i",
    num_workers=15
)

