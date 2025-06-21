"""
Cloud-optimized server for Hunyuan3D with OpenAI DALL-E integration
Lightweight version without heavy ML dependencies
"""

import os
import json
import asyncio
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("3dModels")
STATIC_DIR = Path("static")
PORT = int(os.getenv("PORT", 8000))

# Create directories
for dir_path in [UPLOAD_DIR, MODELS_DIR, STATIC_DIR]:
    dir_path.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hunyuan3D Cloud Server", 
    version="1.0.1",
    description="Text to 3D model generation using OpenAI DALL-E - Public API"
)

# CORS configuration - allow all origins for public access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")

# Initialize OpenAI
openai_client = None

def init_openai():
    global openai_client
    if OPENAI_API_KEY:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            openai_client = None
            return False
    else:
        logger.error("OpenAI API key not configured! Set OPENAI_API_KEY environment variable.")
        return False

# Try to initialize on startup
init_openai()

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    use_openai: bool = True
    model_variant: str = "standard"
    export_format: str = "glb"
    image_size: str = "1024x1024"

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    result_url: Optional[str] = None
    viewer_url: Optional[str] = None
    created_at: str
    updated_at: str
    image_url: Optional[str] = None

# In-memory job tracking (use Redis in production)
jobs: Dict[str, JobStatus] = {}

# Clean up old jobs periodically
async def cleanup_old_jobs():
    """Remove jobs older than 1 hour"""
    while True:
        await asyncio.sleep(3600)  # Check every hour
        current_time = datetime.now()
        jobs_to_remove = []
        
        for job_id, job in jobs.items():
            job_time = datetime.fromisoformat(job.created_at)
            if (current_time - job_time).total_seconds() > 3600:
                jobs_to_remove.append(job_id)
                # Clean up files
                try:
                    shutil.rmtree(MODELS_DIR / job_id, ignore_errors=True)
                    for f in UPLOAD_DIR.glob(f"{job_id}*"):
                        f.unlink(missing_ok=True)
                except:
                    pass
        
        for job_id in jobs_to_remove:
            jobs.pop(job_id, None)

# Start cleanup task on startup
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_jobs())

async def generate_image_with_dalle(prompt: str, output_path: Path, size: str = "1024x1024") -> str:
    """Generate image using OpenAI DALL-E API"""
    try:
        if not openai_client:
            raise Exception("OpenAI API key not configured")
        
        # Enhanced prompt for 3D model generation
        enhanced_prompt = f"""Create a single, centered 3D-model-ready image of: {prompt}
        Requirements: plain white background, clear lighting, single object focus, no shadows, centered composition."""
        
        logger.info(f"Generating image with DALL-E for: {prompt}")
        
        # Generate image with DALL-E 3
        response = openai_client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            size=size,
            quality="standard",
            n=1,
        )
        
        # Download the generated image
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        # Save the image
        img = Image.open(BytesIO(image_response.content))
        img.save(output_path)
        
        logger.info(f"DALL-E image saved to: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"DALL-E generation failed: {e}")
        raise

async def simulate_3d_generation(
    job_id: str,
    image_path: str,
    model_variant: str,
    export_format: str,
    prompt: str = "3D model"
):
    """Simulate 3D generation (replace with actual Hunyuan3D when available)"""
    try:
        # Update job status
        jobs[job_id].status = "processing"
        jobs[job_id].progress = 0.1
        jobs[job_id].message = "Processing image..."
        
        # Simulate processing with delays
        await asyncio.sleep(3)
        jobs[job_id].progress = 0.3
        jobs[job_id].message = "Generating 3D mesh..."
        
        await asyncio.sleep(3)
        jobs[job_id].progress = 0.6
        jobs[job_id].message = "Applying textures..."
        
        await asyncio.sleep(3)
        jobs[job_id].progress = 0.9
        jobs[job_id].message = "Finalizing model..."
        
        # Create output directory
        output_dir = MODELS_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        
        # Copy the generated image as preview
        preview_path = output_dir / "preview.png"
        shutil.copy(image_path, preview_path)
        
        # Create a sample 3D model file (cube in OBJ format)
        model_file = output_dir / f"model.{export_format}"
        
        if export_format == "obj":
            # Simple cube OBJ
            obj_content = """# Simple Cube
v -1.0 -1.0 -1.0
v -1.0 -1.0 1.0
v -1.0 1.0 -1.0
v -1.0 1.0 1.0
v 1.0 -1.0 -1.0
v 1.0 -1.0 1.0
v 1.0 1.0 -1.0
v 1.0 1.0 1.0

f 1 2 4 3
f 5 7 8 6
f 1 5 6 2
f 3 4 8 7
f 1 3 7 5
f 2 6 8 4
"""
            model_file.write_text(obj_content)
        else:
            # Placeholder for other formats
            model_file.write_text(f"Placeholder {export_format.upper()} model\nGenerated from: {prompt}")
        
        # Update job completion
        jobs[job_id].status = "completed"
        jobs[job_id].progress = 1.0
        jobs[job_id].message = "3D model ready!"
        jobs[job_id].result_url = f"/models/{job_id}/model.{export_format}"
        jobs[job_id].viewer_url = f"/viewer/{job_id}"
        jobs[job_id].image_url = f"/models/{job_id}/preview.png"
        jobs[job_id].updated_at = datetime.now().isoformat()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"3D generation failed for job {job_id}: {e}")
        jobs[job_id].status = "failed"
        jobs[job_id].message = f"Generation failed: {str(e)}"
        jobs[job_id].updated_at = datetime.now().isoformat()

@app.get("/")
async def root():
    """Serve the main web interface"""
    if (STATIC_DIR / "index.html").exists():
        return FileResponse(STATIC_DIR / "index.html")
    else:
        return {
            "message": "Hunyuan3D Cloud Server", 
            "version": "1.0.1", 
            "status": "running",
            "openai_configured": bool(openai_client),
            "endpoints": {
                "health": "/health",
                "generate": "/generate (POST)",
                "docs": "/docs"
            }
        }

@app.post("/generate", response_model=JobStatus)
async def generate_3d_model(request: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate 3D model from text prompt"""
    if not openai_client:
        raise HTTPException(500, "OpenAI API key not configured on server")
    
    job_id = str(uuid.uuid4())
    prompt = request.prompt[:200]  # Limit prompt length
    
    # Initialize job
    job = JobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Starting generation...",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )
    jobs[job_id] = job
    
    try:
        # Generate image using DALL-E
        job.message = "Generating image with DALL-E..."
        image_path = UPLOAD_DIR / f"{job_id}_dalle.png"
        await generate_image_with_dalle(prompt, image_path, request.image_size)
        
        # Queue 3D generation
        background_tasks.add_task(
            simulate_3d_generation,
            job_id,
            str(image_path),
            request.model_variant,
            request.export_format,
            prompt
        )
        
        return job
        
    except Exception as e:
        job.status = "failed"
        job.message = str(e)
        logger.error(f"Generation failed: {e}")
        raise HTTPException(500, str(e))

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]

@app.get("/viewer/{job_id}")
async def get_viewer(job_id: str):
    """Get 3D viewer page"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    
    job = jobs[job_id]
    if job.status != "completed":
        return HTMLResponse(f"""
        <html>
        <head>
            <title>Processing...</title>
            <meta http-equiv="refresh" content="2">
        </head>
        <body style="font-family: Arial; text-align: center; padding: 50px;">
            <h2>Processing your 3D model...</h2>
            <p>{job.message}</p>
            <p>Progress: {int(job.progress * 100)}%</p>
        </body>
        </html>
        """)
    
    # Simple viewer for completed models
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Model - {job_id}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
                background: #f0f0f0;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            img {{
                width: 100%;
                max-width: 600px;
                display: block;
                margin: 20px auto;
                border-radius: 8px;
            }}
            .controls {{
                text-align: center;
                margin-top: 20px;
            }}
            .btn {{
                display: inline-block;
                padding: 10px 20px;
                margin: 0 10px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                transition: background 0.3s;
            }}
            .btn:hover {{
                background: #45a049;
            }}
            .info {{
                background: #f9f9f9;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Your 3D Model is Ready!</h1>
            <img src="{job.image_url}" alt="Generated Image">
            
            <div class="controls">
                <a href="{job.result_url}" class="btn" download>Download 3D Model</a>
                <a href="/" class="btn" style="background: #2196F3;">Generate Another</a>
            </div>
            
            <div class="info">
                <p><strong>Note:</strong> This is a demo server. Full 3D generation requires GPU resources.</p>
                <p>The generated image shows what your 3D model would look like.</p>
                <p>Job ID: {job_id}</p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "active_jobs": len([j for j in jobs.values() if j.status == "processing"]),
        "total_jobs": len(jobs)
    }

@app.get("/list-jobs")
async def list_jobs(limit: int = 10):
    """List recent jobs"""
    sorted_jobs = sorted(jobs.values(), key=lambda x: x.created_at, reverse=True)
    return sorted_jobs[:limit]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)