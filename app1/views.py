import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from django.shortcuts import render, redirect, get_object_or_404
from .forms import UploadImageForm, CameraConfigurationForm
from django.conf import settings
from .models import Person, TrackingRecord, CameraConfiguration
from django.core.files.base import ContentFile
from datetime import datetime, timedelta
from django.utils import timezone
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import StreamingHttpResponse, HttpResponse, JsonResponse
import threading
import time
import base64
from django.db import IntegrityError, transaction
import traceback

# Global storage for latest frames
last_frames = {}
last_frames_raw = {}
frame_lock = threading.Lock()
db_lock = threading.Lock() # Lock for SQLite database operations in threads

# Initialize MTCNN and InceptionResnetV1
# Using more sensitive thresholds for better detection in low light [P, R, O]
mtcnn = MTCNN(keep_all=True, min_face_size=20, thresholds=[0.5, 0.6, 0.6])
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Tracking status flag
ai_engine_active = False

# Global list to manage tracking threads
global_stop_events = []

# Store last recognized person for display
last_recognized = {'name': None, 'timestamp': None, 'recognized': False}

# Cache for known face encodings
known_faces_cache = {
    'encodings': np.array([]),
    'names': [],
    'last_updated': 0
}

# Function to detect and encode faces
def detect_and_encode(image, fast_mode=False, return_landmarks=False):
    def run_detection(img):
        try:
            # Validate input image
            if img is None or img.size == 0:
                return [], [], []
            if len(img.shape) != 3 or img.shape[2] != 3:
                return [], [], []
                
            with torch.no_grad():
                boxes, _, landmarks = mtcnn.detect(img, landmarks=True)
                if boxes is not None:
                    faces = []
                    valid_boxes = []
                    valid_landmarks = []
                    h, w = img.shape[:2]
                    for i, box in enumerate(boxes):
                        # Clip boxes to image boundaries
                        x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))
                        # Skip faces that are too small for reliable encoding (40x40 min)
                        if (x2 - x1) < 40 or (y2 - y1) < 40:
                            continue
                        
                        if not return_landmarks:
                            face = img[y1:y2, x1:x2]
                            if face.size == 0:
                                continue
                            face = cv2.resize(face, (160, 160))
                            # Standard Facenet normalization: (x - 127.5) / 128.0
                            face = (face.astype(np.float32) - 127.5) / 128.0
                            face = np.transpose(face, (2, 0, 1))
                            face_tensor = torch.tensor(face).unsqueeze(0)
                            encoding = resnet(face_tensor).detach().numpy().flatten()
                            faces.append(encoding)
                        
                        valid_boxes.append((x1, y1, x2, y2))
                        if landmarks is not None:
                            valid_landmarks.append(landmarks[i])
                    
                    return faces, valid_boxes, valid_landmarks
            return [], [], []
        except Exception as e:
            err_msg = str(e)
            if "does not support image input" in err_msg or "Cannot read" in err_msg:
                print(f"Skipping invalid image in detection", flush=True)
            else:
                print(f"MTCNN detection error: {e}", flush=True)
            return [], [], []

    # Stage 1: Original image
    encodings, boxes, marks = run_detection(image)
    if encodings or boxes or fast_mode:
        return (encodings, boxes, marks) if return_landmarks else (encodings, boxes)

    # Stage 2: Brighten image
    avg_brightness = np.mean(image)
    if avg_brightness < 100:
        enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
        encodings, boxes, marks = run_detection(enhanced)
        if encodings or boxes:
            return (encodings, boxes, marks) if return_landmarks else (encodings, boxes)

    # Stage 3: Equalize Histogram
    try:
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        enhanced_hist = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        encodings, boxes, marks = run_detection(enhanced_hist)
        if encodings or boxes:
            return (encodings, boxes, marks) if return_landmarks else (encodings, boxes)
    except Exception as e:
        print(f"Enhancement error: {e}", flush=True)

    # NEW Stage 4: Sharpening for blurry streams
    try:
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        encodings, boxes, marks = run_detection(sharpened)
        return (encodings, boxes, marks) if return_landmarks else (encodings, boxes)
    except: pass

    return [], []

def determine_best_rotation(img_rgb):
    """Analyzes facial landmarks to determine if the head is upright in 4 possible rotations."""
    # Optimization: Downscale to a standard processing size (e.g., max 640px) 
    # This prevents high-res IP cams from hanging the thread during 4-way rotation checks
    h, w = img_rgb.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)

    rotations = [0, 90, 180, 270]
    best_rot = 0
    max_score = -1
    
    for rot in rotations:
        test_img = img_rgb.copy()
        if rot == 90:
            test_img = cv2.rotate(test_img, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            test_img = cv2.rotate(test_img, cv2.ROTATE_180)
        elif rot == 270:
            test_img = cv2.rotate(test_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        # We only need landmarks, not encodings, for orientation check
        _, boxes, landmarks = detect_and_encode(test_img, fast_mode=True, return_landmarks=True)
        
        if boxes:
            total_upright_score = 0
            for marks in landmarks:
                if len(marks) < 5: continue
                # Landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
                le, re, n, lm, rm = marks
                # Vertical check: Eyes should be above the nose, and nose above the mouth
                # Added 5px tolerance to handle slight head tilts better
                if (le[1] < n[1] + 5) and (re[1] < n[1] + 5) and (n[1] < lm[1] + 5) and (n[1] < rm[1] + 5):
                    total_upright_score += 1
            
            if total_upright_score > max_score:
                max_score = total_upright_score
                best_rot = rot
                
    return best_rot if max_score > 0 else None

# Function to encode uploaded images with caching
def encode_uploaded_images(force_reload=False):
    global known_faces_cache
    
    current_time = time.time()
    # Return cache if not forced and less than 5 minutes old
    if not force_reload and (current_time - known_faces_cache['last_updated'] < 300) and len(known_faces_cache['names']) > 0:
        return known_faces_cache['encodings'], known_faces_cache['names']

    known_face_encodings = []
    known_face_names = []

    # Fetch only authorized people
    uploaded_images = Person.objects.filter(authorized=True)

    for person in uploaded_images:
        image_path = os.path.join(settings.MEDIA_ROOT, str(person.image))
        
        if not os.path.exists(image_path):
            print(f"DEBUG: Image not found for {person.name}: {image_path}", flush=True)
            continue
        
        # Check file extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"DEBUG: Skipping unsupported image format for {person.name}: {ext}", flush=True)
            continue
            
        try:
            known_image = cv2.imread(image_path)
            if known_image is None:
                print(f"DEBUG: Failed to read image for {person.name}: {image_path}", flush=True)
                continue
            if known_image.size == 0:
                print(f"DEBUG: Empty image for {person.name}", flush=True)
                continue
            known_image_rgb = cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB)
            encodings, _ = detect_and_encode(known_image_rgb, fast_mode=True)
            if encodings:
                print(f"DEBUG: Found {len(encodings)} face(s) for {person.name}", flush=True)
                for encoding in encodings:
                    known_face_encodings.append(encoding)
                    known_face_names.append(person.name)
            else:
                print(f"DEBUG: No face found in image for {person.name}", flush=True)
        except Exception as e:
            print(f"DEBUG: Error encoding image for {person.name}: {e}", flush=True)
            continue

    known_faces_cache['encodings'] = np.array(known_face_encodings)
    known_faces_cache['names'] = known_face_names
    known_faces_cache['last_updated'] = current_time

    return known_faces_cache['encodings'], known_faces_cache['names']

# Function to recognize faces
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    recognized_names = []
    if len(known_encodings) == 0:
        return ['Not Recognized'] * len(test_encodings)
        
    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_distance_idx = np.argmin(distances)
        min_dist = distances[min_distance_idx]
        print(f"DEBUG: Closest match distance: {min_dist:.4f} (Threshold: {threshold})", flush=True)
        if min_dist < threshold:
            recognized_names.append(known_names[min_distance_idx])
        else:
            recognized_names.append('Not Recognized')
    return recognized_names

# View for capturing person information and image
def capture_person(request):
    # Don't start tracking threads here - registration uses browser's webcam directly
    # Starting tracking would conflict with the browser's webcam access
            
    configs = CameraConfiguration.objects.all()
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone_number = request.POST.get('phone_number')
        person_class = request.POST.get('person_class')
        image_data = request.POST.get('image_data')

        if image_data and ',' in image_data:
            try:
                header, encoded = image_data.split(',', 1)
                image_file = ContentFile(base64.b64decode(encoded), name=f"{name}.jpg")

                person = Person(
                    name=name,
                    email=email,
                    phone_number=phone_number,
                    person_class=person_class,
                    image=image_file,
                    authorized=False
                )
                person.save()
                messages.success(request, f"Person {name} registered successfully. Waiting for authorization.")
                return redirect('selfie_success')
            except Exception as e:
                messages.error(request, f"Error saving person: {e}")

    return render(request, 'capture_person.html', {'cameras': configs})

def selfie_success(request):
    return render(request, 'selfie_success.html')


# Helper function to start tracking threads
def start_tracking_internal():
    global global_stop_events, ai_engine_active

    # Validate that at least one camera config exists
    cam_configs = list(CameraConfiguration.objects.all())
    if not cam_configs:
        raise ValueError("No camera configurations found. Please add a camera configuration first.")

    # Validate each camera source before spawning threads
    for cfg in cam_configs:
        src = cfg.camera_source.strip() if cfg.camera_source else ''
        if not src:
            raise ValueError(f"Camera '{cfg.name}' has an empty camera source. Please update its configuration.")

    # Signal all existing tracking threads to stop
    for event in global_stop_events:
        event.set()
    global_stop_events = []

    # Give Windows time to release camera device handles before re-opening
    time.sleep(0.5)

    # Resolve the sound file path once relative to the project base
    _sound_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app1', 'suc.wav')

    def process_frame(cam_config, stop_event):
        cap = None
        try:
            source = cam_config.camera_source.strip()
            if source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                print(f"Unable to access camera {cam_config.name}.", flush=True)
                return

            # Optimization: Force buffer size to 1 to prevent queueing old frames (reduces latency on IP Cameras)
            if not source.isdigit():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            threshold = cam_config.threshold

            # Initialize Known Faces
            last_reload_time = time.time()
            try:
                known_face_encodings, known_face_names = encode_uploaded_images()
                print(f"Loaded {len(known_face_names)} reference faces for {cam_config.name}.", flush=True)
            except Exception as e:
                print(f"Error loading known faces for {cam_config.name}: {e}", flush=True)
                known_face_encodings, known_face_names = np.array([]), []

            # Initialize pygame mixer for sound playback (use resolved absolute path)
            success_sound = None
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                if os.path.exists(_sound_path):
                    success_sound = pygame.mixer.Sound(_sound_path)
            except Exception as e:
                print(f"Audio init failed for {cam_config.name}: {e}", flush=True)

            frame_count = 0
            process_every_n_frames = 15  # Optimization: Balanced for real-time response
            current_faces = [] # Cache for boxes and names
            
            # Runtime state for rotation
            instance_rotation = cam_config.rotation
            rotation_locked = False

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print(f"DEBUG: Camera {cam_config.name} failed to return frame", flush=True)
                    time.sleep(0.1)
                    continue

                # User requested local camera to act as a mirror image
                if source.isdigit():
                    frame = cv2.flip(frame, 1)
                
                # Auto-Rotation Logic: Detect orientation once if set to -1
                if instance_rotation == -1:
                    if not rotation_locked:
                        # Try to determine rotation every 30 frames until successful
                        if frame_count % 30 == 0:
                            rgb_for_rot = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            detected_rot = determine_best_rotation(rgb_for_rot)
                            if detected_rot is not None:
                                instance_rotation = detected_rot
                                rotation_locked = True
                                print(f"AUTO-ROTATE: Locked {cam_config.name} at {detected_rot} degrees.", flush=True)
                
                # Apply the current rotation (either from config or auto-detected)
                if instance_rotation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif instance_rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif instance_rotation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                if frame_count % 100 == 0:
                    print(f"DEBUG: Camera {cam_config.name} received 100 frames total", flush=True)
                
                frame_count += 1
                
                # Periodic reload of authorized faces (every 2 minutes)
                if time.time() - last_reload_time > 120:
                    try:
                        known_face_encodings, known_face_names = encode_uploaded_images()
                        last_reload_time = time.time()
                    except: pass

                # Performance Optimization: Skip AI processing for some frames
                # Run on first frame (frame_count=1) and then every N frames
                if frame_count == 1 or frame_count % process_every_n_frames == 0:
                    try:
                        # Sharpen before processing for better accuracy
                        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                        proc_frame = cv2.filter2D(frame, -1, kernel)
                        
                        # Downscale for speed (Optimization: 0.5 is a safe middle ground)
                        scale_factor = 0.5
                        small_frame = cv2.resize(proc_frame, (0, 0), fx=scale_factor, fy=scale_factor)
                        small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        
                        test_face_encodings, face_boxes = detect_and_encode(small_frame_rgb, fast_mode=True)
                        if face_boxes:
                            print(f"DEBUG: MTCNN detected {len(face_boxes)} faces on {cam_config.name}", flush=True)
                        
                        if test_face_encodings:
                            names = recognize_faces(known_face_encodings, known_face_names, test_face_encodings, threshold)
                            # Upscale boxes back to original size
                            current_faces = []
                            inv_scale = 1.0 / scale_factor
                            for (x1, y1, x2, y2), name in zip(face_boxes, names):
                                current_faces.append(((int(x1*inv_scale), int(y1*inv_scale), int(x2*inv_scale), int(y2*inv_scale)), name))
                        else:
                            current_faces = []

                    except Exception as e:
                        print(f"Detection error on {cam_config.name}: {e}")
                
                # Store RAW frame for registration (clean)
                try:
                    ret_raw, jpeg_raw = cv2.imencode('.jpg', frame.copy(), [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret_raw:
                        with frame_lock:
                            last_frames_raw[cam_config.id] = jpeg_raw.tobytes()
                except Exception as e:
                    print(f"Raw frame encoding error on {cam_config.name}: {e}")

                # Draw all detected faces for MONITOR ONLY (Recognized = Green, Unknown = Red)
                for (x1, y1, x2, y2), name in current_faces:
                    color = (0, 255, 0) if name != 'Not Recognized' else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Update last_recognized for display
                    global last_recognized
                    last_recognized = {
                        'name': name,
                        'timestamp': time.time(),
                        'recognized': name != 'Not Recognized'
                    }

                    # Update database only on processed frames for efficiency
                    if (frame_count == 1 or frame_count % process_every_n_frames == 0) and name != 'Not Recognized':
                        with db_lock:
                            persons = Person.objects.filter(name=name)
                            if persons.exists():
                                person = persons.first()
                                today = timezone.now().date()
                                tracking_record, created = TrackingRecord.objects.get_or_create(person=person, date=today)
                                if created:
                                    tracking_record.mark_checked_in()
                                    if success_sound: success_sound.play()
                                    print(f"CHECK-IN: {name} on {cam_config.name}")
                                else:
                                    if tracking_record.check_in_time and not tracking_record.check_out_time:
                                        if timezone.now() >= tracking_record.check_in_time + timedelta(seconds=60):
                                            tracking_record.mark_checked_out()
                                            if success_sound: success_sound.play()
                                            print(f"CHECK-OUT: {name} on {cam_config.name}")

                # Store PROCESSED frame for monitor (with boxes)
                try:
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        with frame_lock:
                            last_frames[cam_config.id] = jpeg.tobytes()
                except Exception as e:
                    print(f"Frame encoding error on {cam_config.name}: {e}")
                
        except Exception as e:
            print(f"Critical error in thread for {cam_config.name}: {e}", flush=True)
            traceback.print_exc()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    for cam_config in cam_configs:
        stop_event = threading.Event()
        global_stop_events.append(stop_event)
        camera_thread = threading.Thread(target=process_frame, args=(cam_config, stop_event), daemon=True)
        camera_thread.start()
        print(f"Started tracking thread for {cam_config.name}", flush=True)

    ai_engine_active = True

def capture_and_recognize(request):
    try:
        start_tracking_internal()
        messages.success(request, "Tracking started successfully.")
    except Exception as e:
        traceback.print_exc()
        return render(request, 'error.html', {'error_message': str(e)})
    
    return redirect('monitor')

def stop_tracking(request):
    global global_stop_events, ai_engine_active, last_frames, last_frames_raw
    
    # 1. Signal all threads to stop
    for event in global_stop_events:
        event.set()
    
    # 2. Clear state
    global_stop_events = []
    ai_engine_active = False
    
    with frame_lock:
        last_frames.clear()
        last_frames_raw.clear()
        
    messages.success(request, "AI Tracking Engine stopped. Camera hardware released.")
    return redirect('home')

def gen_frames(camera_id, raw=False):
    cid = int(camera_id)
    while True:
        with frame_lock:
            if raw:
                frame = last_frames_raw.get(cid)
            else:
                frame = last_frames.get(cid)
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            time.sleep(0.1)

def video_feed(request, pk):
    raw = request.GET.get('raw') == '1'
    return StreamingHttpResponse(gen_frames(pk, raw=raw),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def single_frame(request, pk):
    """Returns a single JPEG frame for capture to avoid Canvas CORS/streaming issues."""
    cid = int(pk)
    frame = None
    with frame_lock:
        frame = last_frames_raw.get(cid)
    
    if frame:
        return HttpResponse(frame, content_type="image/jpeg")
    return HttpResponse(status=404)

@login_required
def monitor(request):
    # Auto-start tracking if not already running
    if not global_stop_events:
        try:
            start_tracking_internal()
        except Exception as e:
            print(f"Auto-start tracking error: {e}")
    
    cam_configs = CameraConfiguration.objects.all()
    return render(request, 'monitor.html', {
        'cam_configs': cam_configs,
        'ai_engine_active': ai_engine_active
    })

@login_required
def ai_status(request):
    """AJAX endpoint to get AI tracking status"""
    global ai_engine_active
    return JsonResponse({'active': ai_engine_active, 'threads': len(global_stop_events)})

@login_required
def recognition_status(request):
    """AJAX endpoint to get recognition status"""
    global last_recognized, ai_engine_active
    import time
    current_time = time.time()
    
    # Check if last_recognized is still valid (within 5 seconds)
    if last_recognized.get('timestamp') and (current_time - last_recognized['timestamp']) < 5:
        return JsonResponse({
            'name': last_recognized['name'],
            'recognized': last_recognized['recognized'],
            'active': ai_engine_active
        })
    else:
        return JsonResponse({
            'name': None,
            'recognized': False,
            'active': ai_engine_active
        })

def tracking_record_list(request):
    search_query = request.GET.get('search', '')
    date_filter = request.GET.get('tracking_date', '')
    
    # Fetch records directly for a flat list with optimization
    tracking_records = TrackingRecord.objects.select_related('person').all()
    
    if search_query:
        tracking_records = tracking_records.filter(person__name__icontains=search_query)
        
    if date_filter:
        tracking_records = tracking_records.filter(date=date_filter)
        
    tracking_records = tracking_records.order_by('-date', '-check_in_time')

    context = {
        'tracking_records': tracking_records,
        'search_query': search_query,
        'date_filter': date_filter
    }
    return render(request, 'tracking_record_list.html', context)

def home(request):
    # REMOVED: Auto-start removed for privacy
    context = {
        'total_persons': Person.objects.count(),
        'total_tracking_records': TrackingRecord.objects.count(),
        'total_check_ins': TrackingRecord.objects.filter(check_in_time__isnull=False).count(),
        'total_check_outs': TrackingRecord.objects.filter(check_out_time__isnull=False).count(),
        'total_cameras': CameraConfiguration.objects.count(),
        'ai_engine_active': ai_engine_active,
    }
    return render(request, 'home.html', context)

def is_admin(user):
    return user.is_superuser

@login_required
# @user_passes_test(is_admin)
def person_list(request):
    persons = Person.objects.all()
    return render(request, 'person_list.html', {'persons': persons})

@login_required
@user_passes_test(is_admin)
def person_detail(request, pk):
    person = get_object_or_404(Person, pk=pk)
    return render(request, 'person_detail.html', {'person': person})

@login_required
@user_passes_test(is_admin)
def person_authorize(request, pk):
    person = get_object_or_404(Person, pk=pk)
    if request.method == 'POST':
        authorized = request.POST.get('authorized', False)
        person.authorized = bool(authorized)
        person.save()
        messages.success(request, f"Person {person.name} authorization status updated.")
        return redirect('person-detail', pk=pk)
    return render(request, 'person_authorize.html', {'person': person})

@login_required
def authorize_toggle(request, pk):
    """AJAX view to toggle authorization status"""
    person = get_object_or_404(Person, pk=pk)
    person.authorized = not person.authorized
    person.save()
    return JsonResponse({'authorized': person.authorized})

@login_required
def authorization_list(request):
    """Show all persons with authorization toggle buttons"""
    persons = Person.objects.all().order_by('-authorized', 'name')
    return render(request, 'authorization_list.html', {'persons': persons})

@login_required
@user_passes_test(is_admin)
def person_delete(request, pk):
    person = get_object_or_404(Person, pk=pk)
    if request.method == 'POST':
        person.delete()
        messages.success(request, 'Person deleted successfully.')
        return redirect('person-list')
    return render(request, 'person_delete_confirm.html', {'person': person})

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('login')

@login_required
@user_passes_test(is_admin)
def camera_config_create(request):
    if request.method == "POST":
        form = CameraConfigurationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Camera configuration created successfully.")
            return redirect('camera_config_list')
        else:
            for error in form.errors.values():
                messages.error(request, error)
    else:
        form = CameraConfigurationForm()
    return render(request, 'camera_config_form.html', {'form': form})

@login_required
@user_passes_test(is_admin)
def camera_config_list(request):
    configs = CameraConfiguration.objects.all()
    return render(request, 'camera_config_list.html', {'configs': configs})

@login_required
@user_passes_test(is_admin)
def camera_config_update(request, pk):
    config = get_object_or_404(CameraConfiguration, pk=pk)
    if request.method == "POST":
        form = CameraConfigurationForm(request.POST, instance=config)
        if form.is_valid():
            form.save()
            messages.success(request, "Camera configuration updated successfully.")
            return redirect('camera_config_list')
        else:
            for error in form.errors.values():
                messages.error(request, error)
    else:
        form = CameraConfigurationForm(instance=config)
    return render(request, 'camera_config_form.html', {'form': form, 'config': config})

@login_required
@user_passes_test(is_admin)
def camera_config_delete(request, pk):
    config = get_object_or_404(CameraConfiguration, pk=pk)
    if request.method == "POST":
        config.delete()
        messages.success(request, "Camera configuration deleted.")
        return redirect('camera_config_list')
    return render(request, 'camera_config_delete.html', {'config': config})
