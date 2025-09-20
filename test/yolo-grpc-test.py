from io import BytesIO
from PIL import Image
import json, sys, os
import grpc

def image_to_bytes(image):
    # compress and convert the image to bytes
    imgByteArr = BytesIO()
    image.save(imgByteArr, format='WEBP')
    return imgByteArr.getvalue()

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(PARENT_DIR, "proto/generated"))
import hyrch_serving_pb2
import hyrch_serving_pb2_grpc

VISION_SERVICE_IP = os.environ.get("VISION_SERVICE_IP", "localhost")
YOLO_SERVICE_PORT = os.environ.get("YOLO_SERVICE_PORT", "50050").split(",")[0]

channel = grpc.insecure_channel(f'{VISION_SERVICE_IP}:{YOLO_SERVICE_PORT}')
stub = hyrch_serving_pb2_grpc.YoloServiceStub(channel)

detect_request = hyrch_serving_pb2.DetectRequest(image_data=image_to_bytes(Image.open("./images/kitchen.webp")), conf=0.3)
response = stub.DetectStream(detect_request)

class_request = hyrch_serving_pb2.SetClassRequest(class_names=['shoes'])
stub.SetClasses(class_request)

json_results = json.loads(response.json_data)
print(json_results)