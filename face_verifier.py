from collections import defaultdict
import pickle
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import cv2

import facenet_pytorch as fn
from face_tracker import FaceDetector


class Matcher:
    def __init__(self, path, recache=False):
        self.resnet = fn.InceptionResnetV1(
            pretrained='vggface2').eval()
        self.mtcnn = fn.MTCNN(
                image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709,
                post_process=True
            )
        cache_path = os.path.join(path, 'faces.db')
        if os.path.exists(cache_path) and not recache:
            print('Loading from: ', cache_path)
            with open(cache_path, 'rb') as f:
                self.matcher_db = pickle.load(f)
        else:
            print('Building Cache...')
            self.matcher_db = self.init_matcher_db(path)

    @staticmethod
    def collate_fn(x):
        return x[0]

    def init_matcher_db(self, path):
        dataset = datasets.ImageFolder(path)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        idx_to_class = dataset.idx_to_class
        class_to_idx = dataset.class_to_idx
        loader = DataLoader(dataset, collate_fn=self.collate_fn)
        aligned = []
        names = []
        matcher_db = defaultdict(list)

        for x, y in loader:
            x_aligned, prob = self.mtcnn(x, return_prob=True)
            if x_aligned is not None:
                aligned.append(x_aligned)
                names.append(idx_to_class[y])

        aligned = torch.stack(aligned)
        embeddings = self.resnet(aligned)

        for name, emb in zip(names, embeddings):
            matcher_db[name].append(emb)

        with open('./db/faces.db', 'wb') as f:
            pickle.dump(matcher_db, f, pickle.HIGHEST_PROTOCOL)
            print('matcher_db is cached!')
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

    def verify(self, image, name, threshold=1.0):
        """
        image: takes image as numpy array
        name: takes the name that is in the db as string
        """
        assert name in self.matcher_db, f"{name} not found in db"
        rec_name, dist = self.recognize(image)

        if rec_name is None and dist is None:
            return None
        elif name == rec_name and dist <= threshold:
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-o', '--output',
            help='The name of the file to save recoded video to',
            default='processed.avi'
            )
    parser.add_argument(
            '-i', '--input',
            help='The file to run verifier on.',
            default='output.avi'
            )
    parser.add_argument(
            'name',
            help='The name of the person to run verification on'
            )
    parser.add_argument(
            '--recache',
            help='Re-cache the table again',
            action='store_true'
            )
    args = parser.parse_args()

    # load vido files
    cap = cv2.VideoCapture(args.input)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (640,480))
    # detector = FaceDetector('haar', './haarcascade_frontalface_default.xml')
    detector = FaceDetector('mtcnn')

    matcher = Matcher('./db/', recache=args.recache)
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    while success:
        faces, landmarks = detector.detect(frame)
        isVerified = matcher.verify(frame, args.name)

        # check if faces are present
        if faces is not None:
            for face in faces:
                x1, y1, x2, y2 = face

                # give green if verified
                if isVerified is True:
                    color = (0, 255, 0)
                elif isVerified is False:
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1),
                              (x2, y2), color, 2)
            # print closest name
            # cv2.putText(frame, f"{name}: {dist:.3f}",
                        # (25, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        # 0.5, (0, 0, 0), 1)
        if landmarks is not None:
            for landmark in landmarks:
                for point in landmark:
                    x, y = point
                    cv2.circle(frame, (x, y), radius=2,
                               color=(255, 255, 255), thickness=-1)


        cv2.imshow('frame', frame)
        out.write(frame)

        if cv2.waitKey(1) &0xFF == ord('q'):
            break

        success, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
