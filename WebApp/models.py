from django.db import models


class Image(models.Model):
    docfile = models.ImageField(upload_to='images')
