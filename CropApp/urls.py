from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path('LoadModel', views.LoadModel, name="LoadModel"),
	       path("CropRecommend.html", views.CropRecommend, name="CropRecommend"),
	       path('CropRecommendAction', views.CropRecommendAction, name="CropRecommendAction"),
]