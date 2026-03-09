# Easy IMG Studio

Web-based version of Easy IMG Studio using FastAPI + Docker.

## Features

- Convert image formats (single or bulk)
- HEIC/HEIF input support (with `pillow-heif`)
- Enhance images using Real-ESRGAN models
- Browser UI with job polling and output download links
- Docker deployment support

## Folder

- Backend/API: `web_app/app/main.py`
- Frontend UI: `web_app/app/static/index.html`
- Docker image: `web_app/Dockerfile`
- Compose file: `docker-compose.yml`

## Model Weights

Put Real-ESRGAN `.pth` files in either:

- `Model/`
- `weights/`

Supported model names:

- `RealESRGAN_x4plus`
- `RealESRGAN_x2plus`
- `RealESRGAN_x4plus_anime_6B`

The API auto-detects matching weight file by model name.

## Local Run (without Docker)

```bash
pip install -r web_app/requirements-web.txt
uvicorn web_app.app.main:app --reload --port 8000
```

Open:

- http://localhost:8000

## Docker Run

From project root:

```bash
docker compose up --build
```

Open:

- http://localhost:8000

## API Endpoints

- `GET /api/models`
- `POST /api/jobs/convert`
- `POST /api/jobs/enhance`
- `GET /api/jobs/{job_id}`
- `GET /api/download/{job_id}/{filename}`
