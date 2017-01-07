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
from PIL.ImageDraw import Image as Img, Draw
from skimage import color

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
        detection_points = []
        image_array = numpy.asarray(self.base_image)
        grayscale = color.rgb2gray(image_array)

        for idx, resized_image in enumerate(pyramid_gaussian(grayscale, self.downscale)):
            for window in self._sliding_window(resized_image):
                if window.image.shape[0] != self.window_size[1] or window.image.shape[1] != self.window_size[0]:
                    continue

                X = self._hist_equalization(window.image).ravel()
                clf = pickle.load(open("neural_model.sav", "rb"))
                result = clf.predict([X])

                if result[0] == 1:
                    detection_points.append([window.x, window.y, idx])

        self._mark_faces(detection_points)

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
        size = max(np.asarray(self.base_image).shape)

        return floor(size / 40)

    def _mark_faces(self, points=list):
        if points:
            for point in points:
                self._draw_rectangle(point[0], point[1], point[2])

    def _draw_rectangle(self, x, y, scale):
        resize = scale * self.downscale
        x *= resize
        y *= resize
        end_x = x + self.window_size[0] * resize
        end_y = y + self.window_size[1] * resize

        draw = Draw(self.base_image)
        draw.rectangle([x, y, end_x, end_y], outline='green')
        del draw
