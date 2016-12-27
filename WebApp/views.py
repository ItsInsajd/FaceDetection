from django.template import loader
from django.http import HttpResponse
from django.shortcuts import render
from WebApp.forms import ImageForm
from WebApp.models import Image


def index(request):
    form = ImageForm()

    return render(request, 'FaceDetection/index.html', {'form': form})


def upload_image(request):
    form = ImageForm(request.POST, request.FILES)

    if form.is_valid():
        image = Image(docfile=request.FILES['docfile'])
        image.save()
        return HttpResponse('done')
    else:
        return HttpResponse(404)
