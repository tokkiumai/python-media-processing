import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    cam.set(3, 800)
    cam.set(4, 600)

    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # cv2.circle(hsv, (300, 500), 1, (255, 0, 0), -1)

        pixel = hsv[400, 300]
        if (pixel[0] in range(0, 30) and pixel[1] >= 50 and pixel[2] >= 50):
            print("RED")
            cv2.putText(img, "RED", (400, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif (pixel[0] in range(40, 80) and pixel[1] >= 50 and pixel[2] >= 50):
            print("GREEN")
            cv2.putText(img, "GREEN", (400, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        elif (pixel[0] in range(100, 130) and pixel[1] >= 50 and pixel[2] >= 50):
            print("BLUE")
            cv2.putText(img, "BLUE", (400, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('my webcam', img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def checkCenterPixel(img):
    # cv2.circle(img, (800/2, 600/2), 5, (255, 255, 255), -1)
    pass


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
