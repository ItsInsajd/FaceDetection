import numpy
from skimage.transform import pyramid_gaussian
from skimage import exposure, img_as_float
from skimage import io

io.use_plugin('matplotlib')


class Window:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image


class FaceDetection:
    def __init__(self, base_image, step_size, window_size):
        self.base_image = base_image
        self.step_size = step_size
        self.window_size = window_size
        self.downscale = 2

    def detect_faces(self):
        for resized_image in pyramid_gaussian(self.base_image, self.downscale):
            for window in self._sliding_window(resized_image):
                if window.image.shape[0] != self.window_size or window.image.shape[1] != self.window_size:
                    continue

                window_eq = self._hist_equalization(window.image)
                # call neural network here

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
