import numpy as np
import cv2

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
MARKER_SIZE = 3.75

def load_params(fname='params'):
    """
    Load intrinsic camera parameters from npz file.
    Params:
        fname : optional, 'params'
            npz file name
    Returns:
        mtx, dist
            Intrinsic camera properties
    """

    with np.load(f'params/{fname}.npz') as f:
        return f['mtx'], f['dist']
    
def calc_distance(frame, mtx, dist):
    """
    Find aruco in frame and print its distance from the camera.
    Params:
        frame : img
            Image in which to look for chessboard
    """
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    if corners:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners)
    
        data = cv2.aruco.estimatePoseSingleMarkers(corners, 1, mtx, dist)
        rvecs, tvecs, objpoints = data[0], data[1], data[2]

        print(f'{np.linalg.norm(tvecs)*MARKER_SIZE:.2f} in')
                    
    cv2.imshow('img',frame)


if __name__ == '__main__':
    mtx, dist = load_params('xps-params')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            calc_distance(frame, mtx, dist)

            k = cv2.waitKey(1)
            if k == 27:
                break

            

    cap.release()
    cv2.destroyAllWindows()