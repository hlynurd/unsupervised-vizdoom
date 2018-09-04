import numpy as np
import cv2, os

from sklearn import metrics
from PIL import Image
from skimage import color
from sklearn import datasets
def load_ini_data2():
    X = []
    Y = []
    new_dim = [32, 32]
    root_dir = "ini_data/"
    for class_dir in os.listdir(root_dir):    
        for image_filename in os.listdir(root_dir+class_dir):    
            if image_filename.endswith(".ppm"): 
                img = Image.open(root_dir + class_dir + "/" + image_filename)
                img = color.rgb2gray(np.array(img))
                # Throw away too large images, less risk of resizing artifacts
                if img.shape[0] > 35 or img.shape[1]>35:
                    continue
                # Ensure that all images have the same dimension
                img = cv2.resize(img, (new_dim[0], new_dim[1])) 
                X.append(img)
                Y.append(int(class_dir))
    X = np.array(X)
    X = np.resize(X, (X.shape[0], X.shape[1] * X.shape[2]))
    return X, np.array(Y)

def load_ini_data():
    X = []
    Y = []
    new_dim = [32, 32]
    classes = [3, 11, 35]
    root_dir = "ini_data/"
    for class_dir in os.listdir(root_dir):
        for npy_file in os.listdir(root_dir+class_dir):
            X.append(np.load(root_dir + class_dir + "/" + npy_file))
            Y.append(int(class_dir))
    X = np.array(X)
    X = np.resize(X, (X.shape[0], X.shape[1] * X.shape[2]))
    Y = np.array(Y)
    return X, Y

def load_mnist_data():
    digits = datasets.load_digits()
    XX = digits.data
    YY = digits.target
    return XX, YY

# One of the functions they should implement
# got this from https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)