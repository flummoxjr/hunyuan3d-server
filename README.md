# Hunyuan3D Cloud Server

Text to 3D model generation server powered by OpenAI DALL-E.

## Quick Deploy to Render

1. Fork or clone this repository
2. Create account on [render.com](https://render.com)
3. New Web Service → Connect this repo
4. Add environment variable: `OPENAI_API_KEY`
5. Deploy!

## Features

- Generate images with DALL-E 3
- Simulated 3D model generation
- REST API for mobile/web apps
- Simple web interface

## Endpoints

- `POST /generate` - Generate 3D model from text
- `GET /status/{job_id}` - Check generation status
- `GET /viewer/{job_id}` - View generated model
- `GET /health` - Health check

## Local Development

```bash
pip install -r requirements_cloud.txt
python cloud_server.py
```

Server runs on http://localhost:8000