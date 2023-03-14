import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt


def dog_filter(face: np.array) -> np.array:
    '''
    Function which replaces parts of the frame with the dog.png picture. 
    Resizes the dog mask to the size of the input array and then replaces
    the pixels of the frame

    Input:
        face: np.array - array of the frame which contains the face

    Output:
        face_with_mask: np.array - modified array of face input with dog ears and nose
    '''

    mask = cv.imread("./assets/dog.png")

    mask_h, mask_w, _ = mask.shape
    face_h, face_w, _ = face.shape

    # resize dog mask to scale with face
    factor = min(face_h / mask_h, face_w / mask_w)
    new_mask_w = int(factor * mask_w)
    new_mask_h = int(factor * mask_h)
    new_mask_shape = (new_mask_w, new_mask_h)
    resized_mask = cv.resize(mask, new_mask_shape)

    face_with_mask = face.copy()
    non_white_pixels = (resized_mask < 250).all(axis=2)

    # finding offset for height and width to correctly place mask on center of the face
    off_h = int((face_h - new_mask_h) / 2)
    off_w = int((face_w - new_mask_w) / 2)
    face_with_mask[off_h: off_h+new_mask_h, off_w: off_w +
                   new_mask_w][non_white_pixels] = resized_mask[non_white_pixels]

    return face_with_mask


def japanese(face: np.array) -> np.array:

    blur_factor = 3
    face_h, face_w, _ = face.shape
    kernel_h, kernel_w = face_h // blur_factor, face_w//blur_factor

    # makes kernel sizes odd to ensure kernel has a center pixel
    if kernel_h % 2 == 0:
        kernel_h += 1
    if kernel_w % 2 == 0:
        kernel_w += 1

    japan_size = 10  # number of 'box partitions' to achieve the pixelated effect
    blurred_face = cv.GaussianBlur(face, (kernel_h, kernel_w), 0)
    temp = cv.resize(blurred_face, (japan_size, japan_size),
                     interpolation=cv.INTER_LINEAR)

    output = cv.resize(temp, (face_h, face_w), interpolation=cv.INTER_NEAREST)

    return output


def find_face(frame, path):

    frontal_face = './assets/haarcascade_frontalface_default.xml'

    # initialize front face classifier
    cascade = cv.CascadeClassifier(frontal_face)

    frame_h, frame_w, _ = frame.shape
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    black_white = cv.equalizeHist(frame_gray)

    rects = cascade.detectMultiScale(
        black_white, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    for [x, y, w, h] in rects:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        match path:
            case 'dog':
                y0, y1 = int(y - 0.25*h), int(y+0.70*h)
                x0, x1 = x, x+w

                if x0 < 0 or y0 < 0 or x1 > frame_w or y1 > frame_h:
                    continue
                frame[y0:y1, x0:x1] = dog_filter(frame[y0:y1, x0:x1])
            case 'contour':
                y0, y1 = int(y - 0.25*h), int(y+1.25*h)
                x0, x1 = int(x - 0.25*w), int(x+w*1.25)

                # _, thresh = cv.threshold(
                #     frame_gray, 140, 255, cv.THRESH_TRUNC)

                # get the position of the whole head
                face = frame_gray[y0:y1, x0:x1]
                canny = cv.Canny(face, threshold1=100,
                                 threshold2=180, apertureSize=3, L2gradient=True)
                frame_gray[y0:y1, x0:x1] = canny
                contours, hierarchy = cv.findContours(
                    image=canny, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
                sorted_contours = sorted(
                    contours, key=cv.contourArea, reverse=True)[:10]
                max_area = max(contours, key=cv.contourArea)

                cv.drawContours(image=frame[y0:y1, x0:x1], contours=sorted_contours, contourIdx=0,
                                color=(0, 255, 0), thickness=4, lineType=cv.LINE_AA)
            case 'japanese':
                y0, y1 = int(y - 0.1*h), int(y+1.1*h)
                x0, x1 = int(x - 0.1*w), int(x+w*1.1)
                if x0 < 0 or y0 < 0 or x1 > frame_w or y1 > frame_h:
                    continue
                frame[y0:y1, x0:x1] = japanese(frame[y0:y1, x0:x1])
            case 'subtract':
                frame_copy = frame.copy()
                frame[y:y+h, x:x+w] = (0, 0, 0)
                return frame_copy

    return frame


def main():
    path = 'dog'
    vid = cv.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        # Display each frame
        cv.imshow("frame", find_face(frame, path))

        # Terminate capturing when the 'Q' button is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
            # When everything done, release the capture
        elif cv.waitKey(1) & 0xFF == ord("s"):
            cv.imwrite("./output/test.jpg", frame)
            continue
    vid.release()


if __name__ == "__main__":
    main()
    # img = cv.imread('./assets/dog.png')
    # cv.imshow('filter', img)

    # cv.waitKey(0)

    # cv.destroyAllWindows()
