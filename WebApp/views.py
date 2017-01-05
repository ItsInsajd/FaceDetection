import numpy
from django.template import loader
from django.http import HttpResponse
from django.shortcuts import render
from WebApp.forms import ImageForm
from WebApp.models import Image
from FaceDetection.face_detection import FaceDetection
from PIL import Image as Img
from skimage import data
from scipy.misc import toimage


def index(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            # Image(docfile=request.FILES.get('docfile')).save()

            file = request.FILES.get('docfile')
            raw_image = Img.open(file)
            detector = FaceDetection(raw_image, 4, [20, 20])
            detected = detector.detect_faces()
            image = toimage(detected)

            return render(request, 'FaceDetection/index.html', {'image': image})
        else:
            return HttpResponse(404)

    else:
        form = ImageForm()
        return render(request, 'FaceDetection/index.html', {'form': form})

# def upload_image(request):
