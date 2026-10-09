from django.contrib import admin
from .models import Person, TrackingRecord, CameraConfiguration

@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'phone_number', 'person_class', 'authorized']
    list_filter = ['person_class', 'authorized']
    search_fields = ['name', 'email']

@admin.register(TrackingRecord)
class TrackingRecordAdmin(admin.ModelAdmin):
    list_display = ['person', 'date', 'check_in_time', 'check_out_time']
    list_filter = ['date']
    search_fields = ['person__name']

    def get_readonly_fields(self, request, obj=None):
        if obj:  # Editing an existing object
            return ['person', 'date', 'check_in_time', 'check_out_time']
        else:  # Adding a new object
            return ['date', 'check_in_time', 'check_out_time']

    def save_model(self, request, obj, form, change):
        if change:  # Editing an existing object
            # Ensure check-in and check-out times cannot be modified via admin
            obj.check_in_time = TrackingRecord.objects.get(id=obj.id).check_in_time
            obj.check_out_time = TrackingRecord.objects.get(id=obj.id).check_out_time
        super().save_model(request, obj, form, change)


@admin.register(CameraConfiguration)
class CameraConfigurationAdmin(admin.ModelAdmin):
    list_display = ['name', 'camera_source', 'threshold']
    search_fields = ['name']