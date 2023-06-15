from fastapi import FastAPI, File, Form, UploadFile
import numpy as np
import uvicorn
import os
import cv2
from PIL import Image
from io import BytesIO
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import base64
import json
from pydantic import BaseModel
from typing import Optional, Tuple
from utils.cim_measure import cim_measure, auoDrawBbox
import time

app = FastAPI()

#############################################################################
print(os.getcwd())
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs2", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    For local js, css swagger in AUO
    :return:
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


##############################################################################

@app.get("/")
def HelloWorld():
    return {"Hello": "World"}


class StructureBase(BaseModel):
    match_threshold4: Optional[float] = 0.1
    match_threshold3: Optional[float] = 0.1
    clear_threshold: Optional[float] = 8
    ssim_threshold: Optional[float] = 0.95

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/auoMeasure/")
def auoMeasure(parameter: StructureBase = Form(...), file: UploadFile = File(...),
               template1: UploadFile = File(...), template2: UploadFile = File(...),
               template3: UploadFile = File(...), template4: UploadFile = File(...)):
    t0 = time.time()

    # get image
    cv2_img = bytes_to_cv2image(file.file.read())
    cv2_template1 = bytes_to_cv2image(template1.file.read())
    cv2_template2 = bytes_to_cv2image(template2.file.read())
    cv2_template3 = bytes_to_cv2image(template3.file.read())
    cv2_template4 = bytes_to_cv2image(template4.file.read())

    # input_dict = data
    match_threshold4 = parameter.match_threshold4
    match_threshold3 = parameter.match_threshold3
    clear_threshold = parameter.clear_threshold
    ssim_threshold = parameter.ssim_threshold

    result = cim_measure(cv2_img, cv2_template1, cv2_template2, cv2_template3, cv2_template4, match_threshold4, match_threshold3,
                         clear_threshold, ssim_threshold)

    api_result = {"match_dist4": result[0],
                  "clear_score": result[1],
                  "ssim_score": result[2],
                  "match_dist3": result[3],
                  "object1": [result[4], result[5]],
                  "object2": [result[6], result[7]]}

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    output_dict = {"result": api_result, 'fps': f_fps}
    return output_dict



class recBase(BaseModel):
    x: int
    y: int
    w: int
    h: int
    line_width: Optional[int] = 2
    line_color: Optional[tuple] = (255, 255, 0)
    # line_color: Tuple[Optional[int], Optional[int], Optional[int]] = (255, 255, 0)

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/draw_rectangle/")
def draw_rectangle(parameter: recBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()

    # get image
    cv2_img = bytes_to_cv2image(file.file.read())

    # get info
    bbox_min = (parameter.x, parameter.y)
    bbox_max = (parameter.x + parameter.w, parameter.y + parameter.h)
    line_color = parameter.line_color
    line_width = parameter.line_width

    # draw
    cv2_img = auoDrawBbox(cv2_img, bbox_min, bbox_max, line_color, line_width=line_width)

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
    return output_dict


class lineBase(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    line_width: Optional[int] = 2
    line_color: Optional[tuple] = (255, 255, 0)

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/draw_line/")
def draw_line(parameter: lineBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()

    # get image
    cv2_img = bytes_to_cv2image(file.file.read())

    # get info
    point1 = (parameter.x1, parameter.y1)
    point2 = (parameter.x2, parameter.y2)
    line_color = parameter.line_color
    line_width = parameter.line_width

    # draw
    cv2_img = cv2.line(cv2_img, point1, point2, color=line_color, thickness=line_width)

    t1 = time.time()
    f_fps = 1.0 / (t1 - t0)
    output_dict = {"img_base64": cv2image_to_base64(cv2_img), 'fps': f_fps}
    return output_dict


def bytes_to_cv2image(imgdata):
    cv2img = cv2.cvtColor(np.array(Image.open(BytesIO(imgdata))), cv2.COLOR_RGB2BGR)
    return cv2img


def cv2image_to_base64(cv2img):
    retval, buffer_img = cv2.imencode('.jpg', cv2img)
    base64_str = base64.b64encode(buffer_img)
    str_a = base64_str.decode('utf-8')
    return str_a


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5110))
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)
