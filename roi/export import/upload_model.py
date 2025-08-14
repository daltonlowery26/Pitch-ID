from roboflow import Roboflow
import os
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK']= 'TRUE'

# dot env
load_dotenv()

# api key
robo_api = os.getenv("ROBO_API")

rf = Roboflow(api_key=robo_api)
workspace = rf.workspace("fracture-pcnbk").project("pitch-id-roi-hts2i")
workspace_one = workspace.version(9)

workspace_one.deploy(
  model_type="yolov8",
  model_path="C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/runs/detect/train_colab/",
  filename="C:/Users/dalto/OneDrive/Pictures/Documents/Projects/Coding Projects/Pitch ID Model/runs/detect/train_colab/weights/best.pt"
)