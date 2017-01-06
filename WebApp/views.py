import numpy
from django.core.files import File
from django.template import loader
from django.http import HttpResponse
from django.shortcuts import render
from WebApp.forms import ImageForm
from WebApp.models import Image
from FaceDetection.face_detection import FaceDetection
from PIL import Image as Img
from skimage import data
from scipy.misc import toimage
import matplotlib.pyplot as plt


def index(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            file = Image(docfile=request.FILES.get('docfile'))
            # file.save()
            image = Img.open(file.docfile)

            detector = FaceDetection(image, [64, 64])
            detected = detector.detect_faces()
            image = toimage(detected)
            image.save('detected.png') # for debugging purposes
            # file.docfile = File(image)
            # file.save()
            return render(request, 'FaceDetection/index.html', {'image': file})
        else:
            return HttpResponse(404)

    else:
        form = ImageForm()
        return render(request, 'FaceDetection/index.html', {'form': form})

# def upload_image(request):
