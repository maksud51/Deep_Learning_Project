from django.urls import path
from .views import DetectProduct
urlpatterns = [
    path('detection_api/', DetectProduct.as_view(),),
    
]