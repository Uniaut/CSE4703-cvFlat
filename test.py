import cv2


KEYCODE_ESC = 27


def process(img: cv2.Mat):
    img = cv2.resize(img, dsize=(0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    for _ in range(7):
        img = cv2.blur(img, (3, 3))
    else:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow('asdf', img)


def main():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    while True:
        ret, frame = capture.read()
        process(frame)
        if cv2.waitKey(25) == 27:
            break


if __name__ == '__main__':
    print(cv2.__version__)
    main()