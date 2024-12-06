import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_bbox(filepath):
    file = open(filepath, 'r')
    bboxes = file.readlines()
    bboxes = [bbox.strip().split() for bbox in bboxes]
    bboxes = [list(map(float, bbox)) for bbox in bboxes]
    bboxes = np.array(bboxes, dtype=np.float64)
    bboxes = bboxes[:,1:]
    file.close()
    return bboxes


def read_image(filepath):
    image = cv2.imread(filepath)
    return image


def read_K(filepath):
    file = open(filepath, 'r')
    K = file.readlines()
    K = K[0].split(',')
    K = [float(k) for k in K if k != '']
    K = np.array(K, dtype=np.float64).reshape((3,3))
    return K


def project_points(bboxes, K, rvec, tvec):
    all_transformed_points_2d = []
    for i in range(0,bboxes.shape[1],3):
        points_3d = bboxes[:,i:i+3]
        points_2d, _ = cv2.projectPoints(points_3d.T, 
                                         rvec, tvec,
                                         K, 
                                         None)
        points_2d = np.array(points_2d).squeeze().flatten()
        all_transformed_points_2d.append(points_2d.tolist())
    return np.array(all_transformed_points_2d, dtype=np.float64)


def draw_boxes(transformed_points_2d, image):
    for transformed_point in transformed_points_2d:
        x = []
        y = []
        for i in range(0, len(transformed_point), 2):
            x.append(transformed_point[i])
            y.append(transformed_point[i+1])
        x_min = int(min(x))
        y_min = int(min(y))
        x_max = int(max(x))
        y_max = int(max(y))
        cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
        print((x_min,y_min), (x_max,y_max))
    return image


def main():
    # Reading files
    bboxes = read_bbox('data/bbox/002_1652711597.7408926.txt')
    image  = read_image('data/images/002_1652711597.73561988.png')
    K = read_K('data/K/002_1652711597.7356198.txt')
    
    # Preparing transformation
    R = [  [0.6663243,  0.0046953,  0.7456473], 
           [-0.7456621,  0.0041490,  0.6663113],
           [0.0000349, -0.9999804,  0.0062657] ]
    R = np.array(R, dtype=np.float64)
    R = Rotation.from_matrix(R.T)
    # R = Rotation.from_euler('xyz',[89.641, -0.002, 138.216], degrees=True)
    # print(R.as_matrix().tolist())
    rvec = np.float64(R.as_rotvec().reshape((3,1)))
    # tvec = np.array([0.381, 0.064, -0.215], dtype=np.float64).reshape((3,1))
    tvec = np.array([0.224, 0.566, -0.225], dtype=np.float64).reshape((3,1))

    # Projecting points
    transformed_bboxes = project_points(bboxes, K, rvec, tvec)
    # print(transformed_bboxes)
    # for transformed_bbox in transformed_bboxes:
    #     print(transformed_bbox)

    # Draw bboxes
    image = draw_boxes(transformed_bboxes, image)
    print(type(image))
    cv2.imshow('Image', image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()