import cv2
import numpy as np


def rectify(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


def process(image: cv2.Mat):
    # add image here.
    # We can also use laptop's webcam if the resolution is good enough to capture
    # readable document content

    # resize image so it can be processed
    # choose optimal dimensions such that important content is not lost

    # creating copy of original image
    image = cv2.normalize(image, None, alpha=0,beta=200, norm_type=cv2.NORM_MINMAX)
    original = image.copy()

    # apply Canny Edge Detection
    image = cv2.medianBlur(image, 3)
    for _ in range(2):
        image = cv2.GaussianBlur(image, (3, 3), 0)
    blurred = image.copy()

    hsv_img = image
    # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)

    def blur_and_canny(image, canny_value):
        return cv2.Canny(image, *canny_value)

    H_canny = blur_and_canny(H, (50, 100))
    S_canny = blur_and_canny(S, (0, 100))
    V_canny = blur_and_canny(V, (100, 200))
    edged = cv2.add(H_canny, V_canny)
    edged = cv2.add(edged, S_canny)

    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 30, 100)
    orig_edged = edged.copy()
    """

    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # x,y,w,h = cv2.boundingRect(contours[0])
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),0)

    # get approximate contour
    temp = True
    target = None
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        cv2.drawContours(image, [approx], -1, (0, 0, 255), 1)

        if len(approx) == 4 and temp:
            target = approx
            temp = False
    
    if target is None:
        return

    # mapping target points to 800x800 quadrilateral
    approx = rectify(target)

    pts2 = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])
    M = cv2.getPerspectiveTransform(approx, pts2)
    dst = cv2.warpPerspective(original, M, (800, 800))

    cv2.drawContours(image, [target], -1, (0, 255, 0), 2)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # imshow
    cv2.imshow('Original', original)
    cv2.imshow('blurred', blurred)
    cv2.imshow('Edge', edged)
    cv2.imshow('Outline', image)
    cv2.imshow('dst.jpg', dst)