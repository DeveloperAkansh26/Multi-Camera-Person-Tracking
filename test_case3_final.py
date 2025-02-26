import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import torch
import timm
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_DIR = "/home/akansh_26/Hackathons/VisionX-AI/Person-Re-Id-Dataset/train"
DEVICE = "cuda"

MAIN_ENCODING = None
count = 0
URL1 = "/home/akansh_26/Hackathons/VisionX-AI/CV Test Dataset/test_case_3/frame1.mp4"
URL2 = "/home/akansh_26/Hackathons/VisionX-AI/CV Test Dataset/test_case_3/frame2.mp4"
URL3 = "/home/akansh_26/Hackathons/VisionX-AI/CV Test Dataset/test_case_3/frame3.mp4"


model_yolo1 = YOLO("yolo11s.pt")
cam1 = cv2.VideoCapture(URL1)

model_yolo2 = YOLO("yolo11s.pt")
cam2 = cv2.VideoCapture(URL2)
cam3 = cv2.VideoCapture(URL3)


def main():
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        ret3, frame3 = cam3.read()

        for i in range(2):
            ret1, frame1 = cam1.read()
            ret2, frame2 = cam2.read()
            ret3, frame3 = cam3.read()

        if not ret1 or not ret2:
            break

        annotated_frame1 = process_frame(frame1, model_yolo1, 1)
        annotated_frame2 = process_frame(frame2, model_yolo2, 2)
        annotated_frame3 = process_frame(frame3, model_yolo2, 2)

        cv2.imshow("frame1", cv2.resize(annotated_frame1, (640, 320)))
        cv2.imshow("frame2", cv2.resize(annotated_frame2, (640, 320)))
        cv2.imshow("frame3", cv2.resize(annotated_frame3, (640, 320)))

        cv2.waitKey(1)
    cv2.destroyAllWindows()


def process_frame(frame, model_yolo, camera_number):
    global count
    global MAIN_ENCODING

    dist = 1000000

    results = model_yolo(frame, verbose=False)
    distances = []
    bounding_boxes_list = []

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if cls != 0:
                continue
            
            count += 1

            x1, y1, x2, y2 = box
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            
            if count == 1:
                img_person = frame[int(y1):int(y2), int(x1):int(x2)]
                img_person = cv2.resize(img_person, (128, 64))
                img_person = torch.from_numpy(img_person).permute(2, 0, 1) / 255.0
                model_siamese.eval()
                with torch.no_grad():
                    img_person = img_person.to(DEVICE)
                    MAIN_ENCODING = model_siamese(img_person.unsqueeze(0))
                    MAIN_ENCODING = MAIN_ENCODING.detach().cpu().numpy()
                    print(f"Enc ID-1: {MAIN_ENCODING}")

            if MAIN_ENCODING is not None and MAIN_ENCODING.any():
                img_person = frame[int(y1):int(y2), int(x1):int(x2)]
                img_person = cv2.resize(img_person, (128, 64))
                img_person = torch.from_numpy(img_person).permute(2, 0, 1) / 255.0
                model_siamese.eval()
                with torch.no_grad():
                    img_person = img_person.to(DEVICE)
                    person_enc = model_siamese(img_person.unsqueeze(0))
                    person_enc = person_enc.detach().cpu().numpy()

                dist = euclidean_dist(MAIN_ENCODING, person_enc)

                threshold = 4
                if dist < threshold:
                    distances.append(dist)
                    bounding_boxes_list.append(box)

    if len(distances) == 1:
        x1, y1, x2, y2 = bounding_boxes_list[0]
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, f"ID:{1}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        try:
            idx = np.argmin(distances)
            x1, y1, x2, y2 = bounding_boxes_list[idx]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, f"ID:{1}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except:
            pass

    return frame


class APN_Model(nn.Module):

    def __init__(self, emb_size=512):
        super(APN_Model, self).__init__()

        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features=emb_size)

    def forward(self, images):

        embeddings = self.efficientnet(images)
        return embeddings


model_siamese = APN_Model()
model_siamese.to(DEVICE)
model_siamese.load_state_dict(torch.load("best_model.pt"))


def get_input():
    global user_input
    user_input = input("Enter a number: ")

def euclidean_dist(enc1, enc2):
    dist = np.sqrt(np.dot(enc1 - enc2, (enc1 - enc2).T))
    return dist


if __name__ == "__main__":
    main()
