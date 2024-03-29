import cv2
import numpy as np
import os

class Calculator():
    def __init__(self, shape=(9,6), filename='params'):
        """
        Create Calculator object for calculating intrinsic parameters of camera.
        Params:
            shape : optional, (6,7)
                Shape of checkerboard pattern (rows, cols)
            filename : optional, 'params'
                Name of intrinsic parameter file to be saved
        """
        self.square_size = 21.8 # in mm

        self.filename = filename
        self.shape = shape

        self.mtx    = None
        self.dist   = None
        self.rvecs  = None
        self.tvecs  = None

    def save(self, filename=None):
        """
        Save calculated camera parameters.
        Params:
            filename : optional, Calculator.filename
                Can be used to specify filename different from that passed in
                __init__().
        """

        assert self.mtx is not None
        assert self.dist is not None
        assert self.rvecs is not None
        assert self.tvecs is not None

        np.savez(f'params/{filename or self.filename}.npz', mtx=self.mtx,
                 dist=self.dist, rvecs=self.rvecs, tvecs=self.tvecs)
        
    def _preprocess_images(self,images):
        """
        Convert images to grayscale.
        Params:
            images
                List of images to be converted
        """

        return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    
    def calculate(self, images, show=False):
        """
        Calculate camera's intrinsic parameters from list of images.
        Params:
            images : list of images
                Input images from single camera with checkerboard in view.
        """

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

        # prepare object points
        objp = np.zeros((self.shape[0] * self.shape[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.shape[1],0:self.shape[0]].T.reshape(-1,2)

        # arrays to store object points and image points
        objpoints = []
        imgpoints = []

        images = self._preprocess_images(images)

        for i,img in enumerate(images):
            # find chessboard corners
            ret, corners = cv2.findChessboardCorners(img, self.shape, None) # (cols, rows)

            # refine points and add (should run on all photos, i.e. all photos should have successful recognition)
            if ret:
                objpoints.append(objp)

                subcorners = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(subcorners)

                # draw and display corners
                cv2.drawChessboardCorners(img, self.shape, subcorners, ret)
            else:
                print(f'[WARN] no pattern found in image {i}')

            if show: cv2.imshow(f'Image {i}', img)

            k = cv2.waitKey(500)
            cv2.destroyAllWindows()

            if k == 27:
                break

        cv2.destroyAllWindows()

        # calibrate
        ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
        
        if not ret:
            print('[ERROR] Parameters not calculated')
            return
        
        print('[INFO] Parameters saved')


if __name__ == '__main__':

    # find number of files in image directory
    n = 0
    for file in os.listdir('/imgs'):
        if os.path.isfile(file): n += 1

    # reads in all images
    images = [cv2.imread(f'imgs/calib-{i:02}.jpg') for i in range(n)]

    # uses images to calibrate camera and save parameters
    calc = Calculator()
    calc.calculate(images)
    calc.save(filename='xps-params')