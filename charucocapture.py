import cv2
import os, sys
import numpy as np

class CharucoCapture():
    def __init__(self, filename='params', pipeline=False, aruco_dict=cv2.aruco.DICT_4X4_250):
        
        self.cap = self._build_pipeline() if pipeline else cv2.VideoCapture(0)

        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.board = cv2.aruco.CharucoBoard_create(5, 7, .04, .02, self.dictionary)  

        self.image_size = None

        self.mat = None
        self.dist = None
        self.tvecs = None
        self.rvecs = None

        self.filename = filename  


    def detect_charuco(self, gray, draw=True):
        charuco_corners = []
        charuco_ids = []

        self.image_size = gray.shape[0:2][::-1]

        # find corners and display
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if corners:
            ids = ids.flatten()

            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners, markerIds=ids, image=gray, board=self.board)

            if ret > 20:                                
                return charuco_corners, charuco_ids
        
        cv2.imshow('Charuco', gray)

        return [], []
    
    def calculate(self, corners, ids):
        print(f'Saving parameters to {self.filename}.npz... ', end='')
        calib, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=corners, charucoIds=ids, board=self.board,
            imageSize=self.image_size, cameraMatrix=None, distCoeffs=None)
        
        np.savez(f'{self.filename}.npz', mtx=self.mtx, dist=self.dist,
                 rvecs=self.rvecs, tvecs=self.tvecs)
        
        print('done.')
        
    def _save(self, filename=None):
        np.savez(f'params/{filename or self.filename}.npz', mtx=self.mtx,
                 dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
    
    def _build_pipeline(self, SENSOR_ID = 0, CAP_WIDTH = 1920, CAP_HEIGHT = 1080, DISP_WIDTH = 960, DISP_HEIGHT = 540, FRAMERATE = 30, FLIP = 0):
        PIPELINE = f'nvarguscamerasrc sensor-id={SENSOR_ID} ! video/x-raw(memory:NVMM), width=(int){CAP_WIDTH}, height=(int){CAP_HEIGHT}, framerate=(fraction){FRAMERATE}/1 ! nvvidconv flip-method={FLIP} ! video/x-raw, width=(int){DISP_WIDTH}, height=(int){DISP_HEIGHT}, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

        # open video stream and wait for camera to start
        print("[INFO] starting video stream...")
        return cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)

    def draw_charuco(self):
        cv2.imwrite('markers/charuco.jpg', self.board.draw((800,1000)))

    def capture(self):
        if self.cap.isOpened():
            pic = 0
            i = 0

            try:
                while True:
                    # get new frame
                    ret, frame = self.cap.read()

                    # detect aruco markers
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids = self.detect_charuco(gray)
                    if len(corners) > 0:
                        print(f'Good image {i}')
                        i += 1

                    # save image
                    k = cv2.waitKey(1)
                    if k == 32 and len(corners) > 0:
                            fname = f'imgs/aruco-calib-{pic:02}.jpg'
                            cv2.imwrite(fname, gray)
                            print(f'[INFO]: Image {pic:02} saved to {fname}.')
                            pic += 1

                    # close window if ESC or Q pressed
                    if k == 27 or k == ord('q'):
                        break
            
            finally:
                self.cap.release()
                cv2.destroyAllWindows()

        else:
            print('[ERROR]: Unable to open camera')

    def get_images(self):
        n = 0
        for file in os.listdir('imgs/'):
            if 'aruco' in file: n += 1

        # reads in all images
        return [cv2.imread(f'imgs/aruco-calib-{i:02}.jpg') for i in range(n)]


if __name__ == '__main__':
    char = CharucoCapture(filename='params/xps-params')

    charuco_corners = []
    charuco_ids = []
    for img in char.get_images():
        corners, ids = char.detect_charuco(img)
        if len(corners) > 0 and len(ids) > 0:
            charuco_corners.append(corners)
            charuco_ids.append(ids)

    char.calculate(charuco_corners, charuco_ids)