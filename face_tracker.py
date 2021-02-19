import cv2
from PIL import Image

import facenet_pytorch as fn

class FaceDetector:
    def __init__(self, method='mtcnn'):
        assert method in ['mtcnn', 'haar'], 'Method not supported'
        self.method = method
        if method == 'mtcnn':
            self.mtcnn = fn.MTCNN(select_largest=False,
                                  device='cpu')

    def detect(self, image):
        if self.method == 'mtcnn':
            faces, probs, landmarks = self.mtcnn.detect(
                    Image.fromarray(image),
                    landmarks=True
                    )

        elif self.method == 'haar':
            pass
        return faces, landmarks


cap = cv2.VideoCapture('output.avi')
detector = FaceDetector('mtcnn')
success, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
while success:
    faces, landmarks = detector.detect(frame)
    # check if faces are present
    if faces is not None:
        for face in faces:
            x1, y1, x2, y2 = face.squeeze()
            cv2.rectangle(frame, (x1, y1),
                          (x2, y2), (255, 0, 0), 2)
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
