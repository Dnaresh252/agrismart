from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="index"),
    path('index', views.index, name="index"),
    path('LoadModel', views.LoadModel, name="LoadModel"),
    path('CropRecommend', views.CropRecommend, name="CropRecommend"),
    path('CropRecommendAction', views.CropRecommendAction, name="CropRecommendAction"),
]