# forms.py
from django import forms
from .models import Person, CameraConfiguration

class UploadImageForm(forms.ModelForm):
    name = forms.CharField(label='Name', max_length=100, widget=forms.TextInput(attrs={'class': 'form-control'}))
    image = forms.ImageField(label='Image', widget=forms.FileInput(attrs={'class': 'form-control-file'}))

    class Meta:
        model = Person
        fields = ['name', 'image']

class CameraConfigurationForm(forms.ModelForm):
    class Meta:
        model = CameraConfiguration
        fields = ['name', 'camera_source', 'threshold', 'rotation']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Camera Room/Location'}),
            'camera_source': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '0 or IP RTSP/HTTP URL'}),
            'threshold': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.01', 'min': '0.0', 'max': '1.0'}),
            'rotation': forms.Select(attrs={'class': 'form-control'}),
        }
    
    def clean_threshold(self):
        threshold = self.cleaned_data.get('threshold')
        if threshold < 0 or threshold > 1:
            raise forms.ValidationError("Threshold must be between 0.0 and 1.0")
        return threshold
