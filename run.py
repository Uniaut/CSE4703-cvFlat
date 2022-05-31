import cv2
import flat
import tk_test


def process(image: cv2.Mat):
    try:
        # flat.process(image)
        tk_test.process(image)
    except Exception as e:
        print(e)
        pass

def main():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.destroyAllWindows()
    while True:
        ret, frame = capture.read()
        frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
        process(frame)
        if cv2.waitKey(25) == 27:
            break


if __name__ == '__main__':
    print(cv2.__version__)
    tk_test.process(cv2.imread('../img/test.jpg', cv2.IMREAD_ANYCOLOR))
    cv2.waitKey(0)
    main()