from roboflow import Roboflow
import os
from dotenv import load_dotenv

# dot env
load_dotenv()

# api key
robo_api = os.getenv("ROBO_API")

# load data
rf = Roboflow(api_key=robo_api)
project = rf.workspace("fracture-pcnbk").project("pitch-id-roi-hts2i")
version = project.version(5)
dataset = version.download("YOLOv8")