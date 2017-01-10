from io import BytesIO
import numpy
from django.core.files import File
from django.core.files.base import ContentFile
from django.core.files.images import ImageFile
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
            img = Image(docfile=request.FILES.get('docfile'))
            image = Img.open(img.docfile)

            detector = FaceDetection(image, [64, 64])
            detected = detector.detect_faces()

            img_io = BytesIO()
            detected.save(img_io, format='PNG')
            img.docfile.save(img.docfile.name, ContentFile(img_io.getvalue()))

            return render(request, 'FaceDetection/index.html', {'image': img})
        else:
            return HttpResponse(404)

    else:
        model = Image.objects.all()
        images = model[:3]
        form = ImageForm()
        return render(request, 'FaceDetection/index.html', {'form': form})

# def upload_image(request):
