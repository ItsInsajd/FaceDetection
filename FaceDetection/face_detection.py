import pickle
from math import floor

import numpy
import numpy as np
from PIL.ImageDraw import Draw
from skimage import color
from skimage import exposure, img_as_float
from skimage import io
from skimage.transform import pyramid_gaussian

io.use_plugin('matplotlib')


class Window:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image


class FaceDetection:
    def __init__(self, base_image, window_size):
        self.base_image = base_image
        self.step_size = int(self._calculate_step_size())
        self.window_size = window_size
        self.downscale = 10

    def detect_faces(self):
        detection_points = []
        image_array = numpy.asarray(self.base_image)
        grayscale = color.rgb2gray(image_array)

        face = 0
        nonface = 0
        clf = pickle.load(open("neural_model.sav", "rb"))
        for idx, resized_image in enumerate(pyramid_gaussian(grayscale, self.downscale)):
            for window in self._sliding_window(resized_image):
                if window.image.shape[0] != self.window_size[1] or window.image.shape[1] != self.window_size[0]:
                    continue

                X = self._hist_equalization(window.image).ravel()
                result = clf.predict([X])

                if result[0] == 1:
                    face+=1
                    detection_points.append([window.x, window.y, idx])
                else:
                    nonface+=1
            print(idx)
            
        self._mark_faces(detection_points)
        print(str(face) + ' ' + str(nonface))

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

    def _mark_faces(self, points):
        if points:
            for point in points:
                self._draw_rectangle(point[0], point[1], point[2])

    def _draw_rectangle(self, x, y, scale):
        if scale == 0:
            scale = 1/self.downscale

        resize = scale * self.downscale
        end_x = x + self.window_size[0] * resize
        end_y = y + self.window_size[1] * resize
        # print('x = ' + str(x) + ' y = ' + str(y) + ' end_x = ' + str(end_x) + ' end_y = ' + str(end_y))
        draw = Draw(self.base_image)
        draw.rectangle([x, y, end_x, end_y], outline='green')
        del draw
