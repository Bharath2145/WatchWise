from django.apps import AppConfig


class App1Config(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app1'

    def ready(self):
        from django.db.models.signals import post_migrate
        post_migrate.connect(self.create_default_camera, sender=self)

    def create_default_camera(self, sender, **kwargs):
        from .models import CameraConfiguration
        
        try:
            # Create the system camera if it doesn't exist by source
            if not CameraConfiguration.objects.filter(camera_source='0').exists():
                CameraConfiguration.objects.get_or_create(
                    camera_source='0',
                    defaults={
                        'name': "Local Laptop Camera",
                        'threshold': 0.6
                    }
                )
        except:
            pass
