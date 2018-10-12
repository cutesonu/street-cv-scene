import os
import cv2
import time
import sys
import numpy as np

LABELS = ["background", "aeroplane", "bicycle", "bird", "boat",
          "bottle", "bus", "car", "cat", "chair",
          "cow", "diningtable", "dog", "horse", "motorbike",
          "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
PERSON = [15]
TRAFFIC_OBJs = [6, 7, 19]


_cur_dir = os.path.dirname(os.path.realpath(__file__))
print("cur dir", _cur_dir)


class SsdUtils:
    def __init__(self, targets=None):
        if targets is None:
            targets = PERSON

        model_dir = os.path.join(_cur_dir, "SSD_MOBILE")
        prototxt = model_dir + "/MobileNetSSD_deploy.prototxt"
        model = model_dir + "/MobileNetSSD_deploy.caffemodel"
        self.scale = 0.00784  # 2/256
        self.mean_subtraction = (127.5, 127.5, 127.5)

        if not os.path.exists(prototxt) or not os.path.exists(model):
            sys.stderr.write("no exist SSD camera_models\n")
            return
        else:
            sys.stdout.write("loading SSD cnn_model...\n")
            self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

        if len(targets) == 0:
            self.targets = range(1000)
        else:
            self.targets = targets

        self.confidence = 0.8
        self.ssd_sz = (500, 300)

    def detect(self, img, ssd_sz=None):
        h, w = img.shape[:2]
        if ssd_sz is None:
            ssd_sz = (int(w * self.ssd_sz[1] / h), self.ssd_sz[1])

        # blob = cv2.dnn.blobFromImage(cv2.resize(img, ssd_sz), 0.007843, ssd_sz, 127.5)
        blob = cv2.dnn.blobFromImage(cv2.resize(img, ssd_sz), self.scale, ssd_sz, self.mean_subtraction)

        self.net.setInput(blob)
        detections = self.net.forward()

        persons = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            idx = int(detections[0, 0, i, 1])

            if confidence > self.confidence and idx in self.targets:
                box = detections[0, 0, i, 3:7]

                persons.append({'label': LABELS[idx],
                                'confidence': confidence,
                                'rect': box})
        return persons

    def draw_dets(self, img, objects):
        show_img = img.copy()
        rect_box = np.zeros_like(show_img)

        img_h, img_w = img.shape[:2]
        for obj in objects:
            (x, y, x2, y2) = (obj['rect'] * np.array([img_w, img_h, img_w, img_h])).astype(dtype=np.int)

            if 0 < x < x2 < img_w and 0 < y < y2 < img_h:
                cv2.rectangle(rect_box, (x, y), (x2, y2), (255, 255, 0), 2)

        show_img = cv2.addWeighted(show_img, 1.0, rect_box, 1.0, 0)
        return show_img


if __name__ == '__main__':
    ssd = SsdUtils(targets=PERSON)

    video_path = os.path.join(_cur_dir, "../",
                              "/home/suno/Desktop/Chop-Photo/street-cv-scene/data/StillCamLife - 16 Minutes in Budapest Hungary beautiful people walking and cars driving by.mp4")
    print(video_path)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    saver = cv2.VideoWriter('output.mp4', fourcc, 25.0, (int(width / 2), int(height / 2)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()

        _persons = ssd.detect(img=frame)
        ret = ssd.draw_dets(img=frame, objects=_persons)

        cv2.imshow("frame", cv2.resize(ret, None, fx=0.5, fy=0.5))
        saver.write(cv2.resize(ret, None, fx=0.5, fy=0.5))

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    saver.release()

