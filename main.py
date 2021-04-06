from io import BytesIO
import logging
import numpy as np
import onnxruntime as ort
from PIL import Image
import json
from fastapi import FastAPI, File, Form

# get logger
logging.basicConfig(
    format="[%(levelname)s] %(asctime)s %(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)

# load config files
model_config = json.load(open("./config/modelConfig.json", "r"))
ch_norm = model_config["channelNormalization"]
img_size = model_config["imageSize"]
model_version = model_config["modelVersion"]
class_map = json.load(open("./config/classMap.json", "r"))
logger.info("Config loaded")

# loading a session with the ML model
ort_session = ort.InferenceSession("./model/dog_classifier.onnx")
logger.info("ONNX-Runtime inference session loaded")

# initialize FastAPI
app = FastAPI()
logger.info("FastAPI initialized")
# the route for classifying a dog breed from an image


@app.post("/v1/classify_dog_breed")
async def classify_dog_breed(
        image: bytes = File(...),
        topN: int = Form(...),
):
    # load image from bytes into PIL image
    image_data = BytesIO(image)
    img = Image.open(image_data)
    img.load()
    img = img.resize([img_size, img_size])

    # transform image into numpy array of disired shape
    data = np.asarray(img, dtype="float32") / 255.0
    data = np.moveaxis(data, 2, 0)
    data = np.expand_dims(data, axis=0).copy()

    # normalize image data per chanel
    for i, ch in enumerate(ch_norm.keys()):
        data[:, i, :, :] = (
            (data[:, i, :, :] - ch_norm[ch]["mean"]) / ch_norm[ch]["std"]
        )

    # make predictions for the image
    outputs = ort_session.run(None, {"input.1": data})[0][0]
    # get the top N most likely predictions and map the indices to the dog breed in class_map
    topNIdx = np.argsort(outputs)[:-topN-1:-1]
    result = [
        {"breed": class_map[str(idx)], "probability": outputs.tolist()[idx]} for idx in topNIdx
    ]

    # return predictions
    return {"predictions": result, "modelVersion": model_version}

# health check route


@app.get("/health")
def health():
    return {"status": "ok"}
