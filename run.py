import cv2
import flat
import tk_test


def process(iamge: cv2.Mat):
    try:
        tk_test.process(iamge)
    except:
        pass

def main():
    capture = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cv2.destroyAllWindows()
    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
        process(frame)
        if cv2.waitKey(25) == 27:
            break


if __name__ == '__main__':
    main()