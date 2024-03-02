import argparse
import cv2
from main import fuse, init

parser = argparse.ArgumentParser()
parser.add_argument('target', help="Path of target image.")
args = parser.parse_args()

cap = cv2.VideoCapture(0)

detector, predictor, target_im, target_landmarks, target_mask = init(args.target)

while(1):
    ret, origin_im = cap.read()
    cv2.imshow("capture", fuse(origin_im, detector, predictor, target_im, target_landmarks, target_mask))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 