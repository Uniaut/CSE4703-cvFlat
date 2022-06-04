import cv2
import flat
import change


def process(original: cv2.Mat, image: cv2.Mat):
    try:
        # flat.process(image)
        change.process(original, image)
    except Exception as e:
        print(e)
        pass

def main():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.destroyAllWindows()
    _, original = capture.read()
    while True:
        _, frame = capture.read()
        frame = cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)
        process(original, frame)
        keycode = cv2.waitKey(25)
        if keycode == 27:
            break
        elif keycode == ord('d'):
            original = frame
        

if __name__ == '__main__':
    print(cv2.__version__)
    cv2.waitKey(0)
    main()