from django.urls import path
from . import views

urlpatterns = [
    path('capture_person/', views.capture_person, name='capture_person'),
    path('', views.home, name='home'),
    path('selfie-success/', views.selfie_success, name='selfie_success'),
    path('capture-and-recognize/', views.capture_and_recognize, name='capture_and_recognize'),
    path('tracking/records/', views.tracking_record_list, name='tracking_record_list'),
    path('ai-status/', views.ai_status, name='ai_status'),
    path('recognition-status/', views.recognition_status, name='recognition_status'),
    path('persons/', views.person_list, name='person-list'),
    path('persons/<int:pk>/', views.person_detail, name='person-detail'),
    path('persons/<int:pk>/authorize/', views.person_authorize, name='person-authorize'),
    path('persons/<int:pk>/authorize/toggle/', views.authorize_toggle, name='authorize-toggle'),
    path('authorization/', views.authorization_list, name='authorization-list'),
    path('persons/<int:pk>/delete/', views.person_delete, name='person-delete'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('camera-config/', views.camera_config_create, name='camera_config_create'),
    path('camera-config/list/', views.camera_config_list, name='camera_config_list'),
    path('camera-config/update/<int:pk>/', views.camera_config_update, name='camera_config_update'),
    path('camera-config/delete/<int:pk>/', views.camera_config_delete, name='camera_config_delete'),
    path('video_feed/<int:pk>/', views.video_feed, name='video_feed'),
    path('monitor/', views.monitor, name='monitor'),
    path('stop-tracking/', views.stop_tracking, name='stop_tracking'),
    path('single_frame/<int:pk>/', views.single_frame, name='single_frame'),
]
    

