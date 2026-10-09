from django.test import TestCase
from .models import Person, TrackingRecord, CameraConfiguration
from django.utils import timezone
from django.core.files.uploadedfile import SimpleUploadedFile

class ModelTests(TestCase):
    def setUp(self):
        self.person = Person.objects.create(
            name="John Doe",
            email="john@example.com",
            phone_number="1234567890",
            person_class="Staff",
            authorized=True
        )

    def test_person_creation(self):
        self.assertEqual(self.person.name, "John Doe")
        self.assertEqual(str(self.person), "John Doe")

    def test_tracking_record_creation(self):
        record = TrackingRecord.objects.create(person=self.person)
        self.assertEqual(record.person, self.person)
        self.assertIsNotNone(record.date)
        self.assertEqual(str(record), f"John Doe - {record.date}")

    def test_mark_checked_in_out(self):
        record = TrackingRecord.objects.create(person=self.person)
        record.mark_checked_in()
        self.assertIsNotNone(record.check_in_time)
        
        record.mark_checked_out()
        self.assertIsNotNone(record.check_out_time)
        
        duration = record.calculate_duration()
        self.assertIn("0h 0m", duration)

    def test_camera_configuration_creation(self):
        config = CameraConfiguration.objects.create(
            name="Main Entrance",
            camera_source="0",
            threshold=0.6,
            rotation=0
        )
        self.assertEqual(config.name, "Main Entrance")
        self.assertEqual(str(config), "Main Entrance")

