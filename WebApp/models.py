from django.db import models


class Image(models.Model):
    docfile = models.FileField(upload_to='images')
