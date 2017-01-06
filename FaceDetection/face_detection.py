import numpy
from math import floor
from skimage.transform import pyramid_gaussian
from skimage import exposure, img_as_float
from skimage import io
from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

io.use_plugin('matplotlib')


class Window:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image


class NeuralNetwork:
    def __init__(self): pass

    def base_train(self):
        data = fetch_olivetti_faces()
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(78,), random_state=1)
        X_train = data.data
        y_train = np.ones((len(X_train),), dtype=np.int)
        model = clf.fit(X_train, y_train)
        pickle.dump(model, open('neural_model.sav', 'wb'))


class FaceDetection:
    def __init__(self, base_image, window_size):
        self.base_image = base_image
        self.step_size = self._calculate_step_size()
        self.window_size = window_size
        self.downscale = 2

    def detect_faces(self):
        for resized_image in pyramid_gaussian(self.base_image, self.downscale):
            for window in self._sliding_window(resized_image):
                if window.image.shape[0] != self.window_size[1] or window.image.shape[1] != self.window_size[0]:
                    continue

                X = self._hist_equalization(window.image).ravel()
                clf = pickle.load(open("neural_model.sav", "rb"))
                result = clf.predict([X])

        return self.base_image

    def _sliding_window(self, image):
        for y in range(0, image.shape[0], self.step_size):
            for x in range(0, image.shape[1], self.step_size):
                yield (Window(x, y, image[y: y + self.window_size[1], x: x + self.window_size[0]]))

    def _hist_equalization(self, image):
        float_window = img_as_float(image)
        p2, p98 = numpy.percentile(float_window, (2, 98))
        window_rescale = exposure.rescale_intensity(float_window, in_range=(p2, p98))

        return exposure.equalize_hist(window_rescale)

    def _calculate_step_size(self):
        size = max(self.base_image.shape)

        return floor(size/40)
