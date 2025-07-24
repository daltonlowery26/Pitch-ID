from roboflow import Roboflow
import os

# api key
robo_api = os.getenv("ROBO_API")

# load data
rf = Roboflow(api_key=robo_api)
project = rf.workspace("fracture-pcnbk").project("hitter-pose")
version = project.version(5)
dataset = version.download("coco")