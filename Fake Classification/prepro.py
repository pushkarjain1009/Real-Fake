import numpy as np
import cv2
from sklearn.utils import shuffle

x_train = []
y_train = []

for i in range(1,771):
    img = cv2.imread("/home/webhead/Downloads/Real and Fake/real_and_fake_face/training_fake/{}.jpg".format(i), 1)

    x_train.append(img)
    y_train.append(1)

for i in range(1, 867):
    img = cv2.imread("/home/webhead/Downloads/Real and Fake/real_and_fake_face/training_real/{}.jpg".format(i), 1)

    x_train.append(img)
    y_train.append(0)

x_train, y_train = shuffle(x_train, y_train)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []

for i in range(771, 960):
    img = cv2.imread("/home/webhead/Downloads/Real and Fake/real_and_fake_face/training_fake/{}.jpg".format(i), 1)

    x_test.append(img)
    y_test.append(1)

for i in range(867, 1082):
    img = cv2.imread("/home/webhead/Downloads/Real and Fake/real_and_fake_face/training_real/{}.jpg".format(i), 1)

    x_test.append(img)
    y_test.append(0)

x_test, y_test = shuffle(x_test, y_test)

x_test = np.array(x_test)
y_test = np.array(y_test)

np.save("x_train.npy", x_train)
np.save("x_test.npy", x_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
