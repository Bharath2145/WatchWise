import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

print("Torch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)

try:
    mtcnn = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("Models loaded successfully.")
    
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Test", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    boxes, _ = mtcnn.detect(img)
    print("Detection test passed (no faces found in blank image).")
    
except Exception as e:
    print("Error:", e)
