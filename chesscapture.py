import cv2

def capture(cap=None):

    cap = cap or cv2.VideoCapture(0)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if cap.isOpened():

        pic = 0

        try:
            window = cv2.namedWindow('ChessboardCapture', cv2.WINDOW_AUTOSIZE)
            while True:
                ret, frame = cap.read()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

                if ret:
                    refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    cv2.drawChessboardCorners(gray, (9,6), refined, ret)

                cv2.imshow('ChessboardCapture', gray)
                k = cv2.waitKey(1)

                if k == 32 and ret:
                    fname = f'imgs/calib-{pic:02}.jpg'
                    cv2.imwrite(fname, frame)
                    print(f'[INFO]: Image {pic:02} saved to {fname}.')
                    pic += 1

                if k == 27:
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f'{pic} images saved.')

    else:
        print('[ERR]: Unable to open camera')

if __name__ == '__main__':
    cap = build_pipeline()
    capture(cap)