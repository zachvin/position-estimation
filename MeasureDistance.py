import numpy as np
import cv2

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

    with np.load(f'{fname}.npz') as f:
        return f['mtx'], f['dist']
    
def calc_distance(frame):
    """
    Find chessboard in frame and print its distance from the camera.
    Params:
        frame : img
            Image in which to look for chessboard
    """
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
    if ret == True:
        refined = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    
        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv2.solvePnP(objp, refined, mtx, dist)

        cv2.drawChessboardCorners(frame, (9,6), refined, ret)
        print(f'{np.linalg.norm(tvecs):.2f}')
                    
    cv2.imshow('img',frame)


if __name__ == '__main__':

    mtx, dist = load_params()

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) * 2/3

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if ret:
            calc_distance(frame)

            k = cv2.waitKey(1)
            if k == 27:
                break

            

    cap.release()
    cv2.destroyAllWindows()