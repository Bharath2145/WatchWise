# WatchWise

Real-time multi-camera face recognition and tracking, built with Django.

People register with a webcam selfie, an admin authorizes them, and a background
tracking engine recognizes them live on webcam or IP-camera feeds — logging
check-in/check-out records on a monitoring dashboard with annotated video.

## Features

- Selfie enrollment from the browser webcam (base64 capture)
- Admin authorization workflow (only authorized faces are recognized)
- One tracking thread per camera (webcam index or RTSP/HTTP URL)
- Face detection (MTCNN) + recognition (InceptionResnetV1 embeddings, Euclidean match)
- Auto camera-rotation detection via facial landmarks
- Live MJPEG monitor with green (known) / red (unknown) boxes
- Daily check-in / check-out records with durations + alert sounds
- Login + admin-only camera/person management, AJAX status endpoints

## Built with

| Tech | Used for |
|---|---|
| Python 3.11 | App language |
| Django 5.0 | Web framework: routing, ORM, auth, templates, MJPEG streaming |
| SQLite | Database (Django ORM, migrations included) |
| facenet-pytorch (MTCNN) | Face detection + 5-point facial landmarks |
| facenet-pytorch (InceptionResnetV1, VGGFace2 weights) | 512-dim face embeddings |
| PyTorch (CPU) | Model inference (`torch.no_grad()`) |
| OpenCV | Video capture, flip/rotate/resize, lighting enhancement, JPEG encoding |
| NumPy | Normalization, Euclidean-distance matching, image kernels |
| pygame (`mixer` only) | Check-in/check-out alert sound |
| Pillow | Image upload handling |
| HTML + JS (fetch polling) | Monitor dashboard, registration pages |

No Docker, cloud, CI/CD, or REST framework — this is a local monolith by design.

## Run it

```bat
REM create env and install (CPU-only torch, ~1 GB)
uv venv .venv --python 3.11
.venv\Scripts\activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 torchvision==0.17.2
uv pip install -r requirements_utf8.txt

REM start
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 127.0.0.1:8000
```

Open http://127.0.0.1:8000 — log in, add a camera (`0` = webcam), register a
person, authorize them, then open **Monitor** to start tracking.

## Layout

```
app1/            models, face-recognition + tracking engine (views.py), forms, urls
WatchWise/       Django settings / root urls / wsgi
templates/       HTML pages (excluded from GitHub language stats)
app1/migrations/ DB migrations
media/           uploaded photos (local only, git-ignored)
```

## Notes

- Dev-only settings: `SECRET_KEY` is hardcoded and `DEBUG=True`. Rotate the key
  and harden settings before any real deployment.
- Heavy first run: facenet-pytorch downloads model weights on first import.
- Tested on Windows + Python 3.11, CPU only.
