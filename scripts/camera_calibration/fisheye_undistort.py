import numpy as np
import cv2

import sys

# You should replace these 3 lines with the output in calibration step
DIM=(1280, 720)
K=np.array([[603.2367723329799, 0.0, 631.040490386724], [0.0, 606.6503954431822, 394.50828189919275], [0.0, 0.0, 1.0]])
D=np.array([[-0.06211310762499229], [0.11678409244618092], [-0.20084647516958823], [0.10142080878217873]])

def undistort(img_path):
    img_name, img_extension = img_path.split('.')

    img = cv2.imread(img_path)
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite('%s_undistorted.%s' % (img_name, img_extension), undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def undistort_balanced(img_path, balance=0.0, dim2=None, dim3=None):
    img_name, img_extension = img_path.split('.')

    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite('%s_undistorted_balanced.%s' % (img_name, img_extension), undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for p in sys.argv[1:]:
        undistort_balanced(p, balance=0.75)