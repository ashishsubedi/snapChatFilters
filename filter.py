import cv2
import numpy as np
import dlib
import time

faceToIndex = dict(
    left_eye_left=36,
    left_eye_right=39,
    left_eye_top_1=37,
    left_eye_top_2=38,
    left_eye_bottom_1=41,
    left_eye_bottom_2=40,
    right_eye_left=42,
    right_eye_right=45,
    right_eye_top_1=43,
    right_eye_top_2=44,
    right_eye_bottom_1=47,
    right_eye_bottom_2=46,
    center_eye=27
)


def main():
    global faceToIndex
    filter = 'glass.png'
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    filterImgOrigi = cv2.imread('filters/'+filter)
    filterImg = filterImgOrigi.copy()
    # filterImg_gray = cv2.cvtColor(filterImg, cv2.COLOR_BGR2GRAY)
    # _, filterImg_thresh = cv2.threshold(
    #     filterImg_gray, 25, 255, cv2.THRESH_BINARY_INV)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for i, rect in enumerate(rects):
            try:
                shape = predictor(gray, rect)

                center = (shape.part(faceToIndex['center_eye']).x, shape.part(
                    faceToIndex['center_eye']).y)
                width = int(((shape.part(
                    faceToIndex['right_eye_right']).x-shape.part(faceToIndex['left_eye_left']).x)+10)*2.5)

                height = int(0.52*width)
                topLeft = (center[0]-width//2, center[1]-height//2)
                bottomRight = (center[0]+width//2, center[1]+height//2)

                filterImg = filterImgOrigi.copy()
                filterImg_gray = cv2.cvtColor(filterImg, cv2.COLOR_BGR2GRAY)
                _, filterImg_thresh = cv2.threshold(
                    filterImg_gray, 25, 255, cv2.THRESH_BINARY_INV)
                filterImg = cv2.resize(filterImg, (width, height))
                filterImg_thresh = cv2.resize(
                    filterImg_thresh, (width, height))

                roi = frame[topLeft[1]:topLeft[1] +
                            height, topLeft[0]: topLeft[0]+width]

                roi_masked = cv2.bitwise_and(
                    roi, roi, mask=filterImg_thresh)
                frame[topLeft[1]:topLeft[1] +
                      height, topLeft[0]:topLeft[0]+width] = cv2.add(roi_masked, filterImg)
            except(Exception):
                pass

            # cv2.imshow('roi', roi)
        total = time.time()-start
        cv2.putText(frame, f'FPS: {1/total}', (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
        cv2.imshow('frame', frame)
        cv2.imshow('roi_masked', roi_masked)
        cv2.imshow('filterImg_thresh', filterImg_thresh)
        cv2.imshow('filter', filterImg)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
