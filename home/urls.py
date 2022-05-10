from django.urls import path
from .views import *
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth import views as auth_views
from django.urls import include, path
from . import views
urlpatterns = [
    path('upload/', UploadImage.as_view()),
    path('', IndexView.as_view()),
]
