import cv2
from PIL import Image

import facenet_pytorch as fn

class FaceDetector:
    def __init__(self, method='mtcnn', cascPath=None):
        assert method in ['mtcnn', 'haar'], 'Method not supported'
        self.method = method
        if method == 'mtcnn':
            self.mtcnn = fn.MTCNN(select_largest=False,
                                  device='cpu')
        if method == 'haar':
            assert cascPath is not None, 'Cascad not provided'
            self.faceCascade = cv2.CascadeClassifier(cascPath)

    def detect(self, image):
        if self.method == 'mtcnn':
            faces, probs, landmarks = self.mtcnn.detect(
                    Image.fromarray(image),
                    landmarks=True
                    )
            return faces, landmarks

        elif self.method == 'haar':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = self.faceCascade.detectMultiScale(
                        image,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

            new_faces = []
            for face in faces:
                x, y, h, w = face
                new_faces.append([x, y, x+w, y+h])
            return new_faces, None


if __name__ == '__main__':
    cap = cv2.VideoCapture('output.avi')
    # detector = FaceDetector('haar', './haarcascade_frontalface_default.xml')
    detector = FaceDetector('mtcnn')
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    while success:
        faces, landmarks = detector.detect(frame)
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
