from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import cv2

import facenet_pytorch as fn
from face_tracker import FaceDetector


class Matcher:
    def __init__(self, path):
        dataset = datasets.ImageFolder(path)
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        self.idx_to_class = dataset.idx_to_class
        self.class_to_idx = dataset.class_to_idx
        self.loader = DataLoader(dataset, collate_fn=self.collate_fn)
        self.resnet = fn.InceptionResnetV1(
            pretrained='vggface2').eval()
        self.mtcnn = fn.MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709,
                post_process=True
            )

        self.matcher_db = self.init_matcher_db()

    @staticmethod
    def collate_fn(x):
        return x[0]

    def init_matcher_db(self):
        aligned = []
        names = []
        matcher_db = defaultdict(list)

        for x, y in self.loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                aligned.append(x_aligned)
                names.append(self.idx_to_class[y])

        aligned = torch.stack(aligned)
        embeddings = self.resnet(aligned)

        for name, emb in zip(names, embeddings):
            matcher_db[name].append(emb)

        return matcher_db

    def recognize(self, img):
        """
        img: takes image as numpy array
        """
        face, prob = self.mtcnn(img, return_prob=True)
        if face is None:
            return (None, None)
        emb = self.resnet(face.unsqueeze(0))

        dists = []
        for name in self.matcher_db:
            for pic_emb in self.matcher_db[name]:
                dist = (pic_emb - emb).norm().item()
                dists.append((name, dist))

        # return the name with least distance
        top_name = sorted(dists, key=lambda x: x[1])[0]
        return top_name


if __name__ == '__main__':
    cap = cv2.VideoCapture('output.avi')
    # detector = FaceDetector('haar', './haarcascade_frontalface_default.xml')
    detector = FaceDetector('mtcnn')

    matcher = Matcher('./db/')
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    while success:
        faces, landmarks = detector.detect(frame)
        name, dist = matcher.recognize(frame)
        print(name, dist)
        # check if faces are present
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2 = face
                cv2.rectangle(frame, (x1, y1),
                              (x2, y2), (0, 0, 255), 2)
        if landmarks is not None:
            for landmark in landmarks:
                for point in landmark:
                    x, y = point
                    cv2.circle(frame, (x, y), radius=2,
                               color=(0, 255, 0), thickness=-1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break

        success, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()
