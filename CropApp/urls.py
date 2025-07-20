from django.urls import path

from . import views
# Crop/urls.py
from django.http import HttpResponse

urlpatterns = [path("index.html", views.index, name="index"),
	       path('LoadModel', views.LoadModel, name="LoadModel"),
	       path("CropRecommend.html", views.CropRecommend, name="CropRecommend"),
	       path('CropRecommendAction', views.CropRecommendAction, name="CropRecommendAction"),
           path('health', lambda request: HttpResponse("OK")),
]