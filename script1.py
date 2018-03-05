import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import glob
import warnings
from sklearn.model_selection import train_test_split

from app import app
app.run(debug = True)

warnings.filterwarnings('ignore')

PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)
ORIENTATIONS = 8

def resize_frame(frame, ratio):
    new_h = int(frame.shape[0] / ratio)
    new_w = int(frame.shape[1] / ratio)
    # print(new_w, new_h)
    return cv2.resize(frame, (new_w, new_h))

def save_frames(x_s, y_s, window):
    cap = cv2.VideoCapture('vid.ogv')
    i = 0
    while True:
        ret, frame = cap.read()

        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame.copy()
        gray = resize_frame(gray, 4)
        cv2.imshow('original', gray)
        # Size: 720 1280

        # Size: 360, 640,
        # 180 320

        #cv2.imshow('cut', gray[22:150, 172:300])
        #cv2.imshow('cut', gray[22:150, 10:138])
        cv2.imshow('cut', gray[y_s:(y_s+window), x_s:(x_s+window)])
        # _, sas = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
        # cv2.imwrite('train\\train{}.png'.format(i), gray[22:150, 172:300])
        # cv2.imwrite('train\\train{}.png'.format(i), gray[22:150, 40:168])
        cv2.imwrite('train\\train{}.png'.format(i), gray[y_s:(y_s+window), x_s:(x_s+window)])
        i += 1
        # cv2.imshow('HOGImage', sas)
        k = cv2.waitKey(27) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def load_train_images():
    Mugs = glob.glob('train/mug/*.png')
    Non_mug = glob.glob('train/no_mug/*.png')

    mug_images = []
    non_mug_images = []

    for mug in Mugs:
        img = cv2.imread(mug)
        mug_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        #mug_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
        # mug_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    print('Loaded Mugs')
    for non_mug in Non_mug:
        img = cv2.imread(non_mug)
        non_mug_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        #non_mug_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2YUV))
        # non_mug_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    print('Loaded non mugs')
    # plt.imshow(mug_images[0], cmap='hsv')
    plt.imshow(mug_images[0])
    return mug_images, non_mug_images

def images2HOGfeatures(mugs_array, non_mug_array):
    mug_features = []
    non_mug_features = []

    for mug in mugs_array:

        mug_features.append(hog(mug, pixels_per_cell=PIXELS_PER_CELL,
                                cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS))
    for non_mug in non_mug_array:

        non_mug_features.append(hog(non_mug, pixels_per_cell=PIXELS_PER_CELL,
                                    cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS))

    return mug_features, non_mug_features

def img2HOG3Channel(img):
    hog_features = []
    for channel in range(img.shape[2]):
        #hog_features.append(get_hog_features(mug[:,:,channel], orient, pix_per_cell, cell_per_block))
        features = hog(img[:, :, channel], orientations=ORIENTATIONS, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, transform_sqrt=False)
        hog_features.append(features)
    return np.ravel(hog_features)


def find_mugg(image, classif):
    WINDOW_SIZE = 128
    x_step, y_step = 32, 32
    window = 128
    scale = 1.0
    j = 0

    image = cv2.resize(image, (np.int(image.shape[1] / scale), np.int(image.shape[0] / scale)))
    saving = image.copy()
    # print(image.shape)
    x_windows = (image.shape[1] - window) // x_step
    y_windows = (image.shape[0] - window) // y_step
    # print(x_windows, y_windows)
    for x in range(x_windows + 1):
        for y in range(y_windows + 1):
            patch = saving[(y_step * y):(y_step * y + window), (x_step * x):(x_step * x + window)]
            patch_hog = hog(patch, pixels_per_cell=PIXELS_PER_CELL, cells_per_block=CELLS_PER_BLOCK, orientations=ORIENTATIONS).ravel()
            #patch_hog = img2HOG3Channel(patch)
            prediction = classif.predict([patch_hog])[0]
            if prediction == 1.0:
                cv2.rectangle(image, (x_step * x, y_step * y), (x_step * x + window, y_step * y + window), (0, 0, 255),
                              2)
                #sav = cv2.cvtColor(saving[(y_step * y):(y_step * y + window), (x_step * x):(x_step * x + window)], cv2.COLOR_YUV2BGR)
                cv2.imwrite('wah\\{}.png'.format(j),
                            img=saving[(y_step * y):(y_step * y + window), (x_step * x):(x_step * x + window)])
                j += 1
            # print(classif.predict([patch_hog]))
    return image
    # print(classif.intercept_)

if __name__ == '__main__':
    # save_frames(150, 0, 128)

    mugs, non_mugs = load_train_images()
    mug_features, non_mug_features = images2HOGfeatures(mugs, non_mugs)
    print('Transformed to hogs...')

    X = np.vstack((mug_features, non_mug_features)).astype(np.float64)

    Y = np.hstack((np.ones(len(mug_features)), np.zeros(len(non_mug_features)))).astype(np.float64)


    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC()

    svc.fit(X_train, y_train)
    # Check the accuracy of the SVC
    print('Test Accuracy =', round(svc.score(X_test, y_test), 4))

    cap = cv2.VideoCapture('vid.ogv')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        gray = resize_frame(gray, 4)
        found = find_mugg(gray, svc)
        # cv2.imshow('original', cv2.cvtColor(found, cv2.COLOR_YUV2BGR))
        cv2.imshow('original', cv2.cvtColor(found, cv2.COLOR_GRAY2BGR))
        # Size: 720 1280

        # 180 320


        k = cv2.waitKey(27) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

