from typing import List
from click import File

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from PIL import Image

from sqlalchemy import create_engine, Column, Integer, String, MetaData, Table, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import os

from database import metadata, engine, SessionLocal
from models import images, animal, human, nature, food, place, etc


from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()


# ----------------------------------------

images_folder = "static/images/upload"
if not os.path.exists(images_folder):
    os.makedirs(images_folder)

# Mount the static folder to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Use Jinja2Templates for HTML templates
templates = Jinja2Templates(directory="templates")
def get_image_files():
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        images = [file for file in os.listdir(images_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return images
    return []

def get_detail_filenames(category):
    db = engine.connect()
    try:
        query = text(f"SELECT filename FROM {category}")
        result = db.execute(query).fetchall()
        file_names = [row[0] for row in result]
        return file_names
    finally:
        db.close()
        
def get_all_images():
    if os.path.exists(images_folder) and os.path.isdir(images_folder):
        images = [file for file in os.listdir(images_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return images
    return []

def create_table(category):
    table_name = f"{category}"
    
    # Check if the table already exists in the metadata
    existing_table = metadata.tables.get(table_name)
    
    if existing_table is not None:
        return existing_table
    
    return Table(
        table_name, metadata,
        Column('id', Integer, primary_key=True),
        Column('filename', String(255), nullable=False),
        extend_existing=True  # Allow redefinition
    )

def seperate_category():
    db = SessionLocal()
    try:
        # Query all data from the images table
        result = db.execute(images.select()).fetchall()
        # image_data = [{'id': row.id, 'filename': row.filename} for row in result]
        image_data = [row.filename for row in result]
    finally:
        db.close()

    # Get a list of all image files in the folder
    # image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    image_files = [filename for filename in image_data if filename in os.listdir(images_folder)]

    images_obj = {
            "human": [],
            "animal": [],
            "food": [],
            "nature": [],
            "place": [],
            "etc": []
        }
    
    # Use the processor to prepare the input
    class_name =[
            "a photo of a human",
            "a photo of people",
            "a photo of an animal",
            "a photo of animals",
            "a photo of food",
            "a photo of nature",
            "photo of places and strudtures",
            "a photo of documents"
        ]
    class_dict = {
        "a photo of a human": "human",
        "a photo of people": "human",
        "a photo of an animal": "animal",
        "a photo of animals": "animal",
        "a photo of food": "food",
        "a photo of nature": "nature",
        "photo of places and strudtures": "place",
        "a photo of documents": "documents",
    }
    # Loop through each image file
    for image_file in image_files:
        # Create the full path to the image
        image_path = os.path.join(images_folder, image_file)

        # Open the image
        image = Image.open(image_path)

        inputs = processor(
            text=class_name,
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Assuming 'model' is your pre-trained model
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        categoryStr = class_name[probs.argmax()]
        # max_index = arr.index(max(arr))

        category = class_dict.get(categoryStr)
        if (max(probs.tolist()[0]) >= 0.45):
            category = class_dict.get(categoryStr)
        else:
            category = "etc"

        if category in images_obj:
            images_obj[category].append(image_file)

            # Check if the table exists; if not, create it
            table = create_table(category)
            table.create(engine, checkfirst=True)

            # Store the filename in the corresponding database table
            db = SessionLocal()
            try:
                db.execute(table.insert().values(filename=image_file))
                db.commit()
            finally:
                db.close()
    return []

def get_table_names():
    # table_names = metadata.tables.keys()
    table_names = [
        table_name
        for table_name, table in metadata.tables.items()
        if table is not None and SessionLocal().execute(table.select().limit(1)).first() is not None
    ]
    return list(table_names)


@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/upload")
async def create_upload_file(request: Request, files: List[UploadFile] = File(...)):
    # Delete all image files in the 'images' folder
    for filename in os.listdir(images_folder):
        file_path = os.path.join(images_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    db = SessionLocal()
    db.execute(images.delete())
    db.execute(animal.delete())
    db.execute(etc.delete())
    db.execute(food.delete())
    db.execute(human.delete())
    db.execute(nature.delete())
    db.execute(place.delete())
            
    for file in files:
        contents = await file.read()
        file_path = os.path.join(images_folder, file.filename)

        with open(file_path, "wb") as f:
            f.write(contents)

        try:
            db.execute(images.insert().values(filename=file.filename))
            db.commit()
        finally:
            db.close()

    # return JSONResponse(content={"files_uploaded": [file.filename for file in files]})
    context = {"request": request, "message": "Files uploaded successfully"}
    return templates.TemplateResponse("gallery_main.html", context)

@app.get("/gallery/", response_class=HTMLResponse)
async def read_images(request: Request):
    images = get_image_files()
    return templates.TemplateResponse("gallery_main.html", {"request": request, "images": images})

@app.get("/gallery/all", response_class=HTMLResponse)
async def read_all_images(request: Request):
    images = get_all_images()
    return templates.TemplateResponse("gallery_all.html", {"request": request, "images": images})


@app.get("/gallery/seperate", response_class=HTMLResponse)
async def seperate_images(request: Request):
    seperate_category()
    table_names = get_table_names()
    table_names.remove('images')
    return templates.TemplateResponse("gallery_seperate.html", {"request": request, "table_names": table_names})

@app.get("/gallery/detail", response_class=HTMLResponse)
async def read_detail_images(request: Request, category: str = None):
    images = get_detail_filenames(category)
    if category == 'human':
        category = '사람'
    elif category == 'animal':
        category = '동물'
    elif category == 'food':
        category = '음식'
    elif category == 'nature':
        category = '자연'
    elif category == 'place':
        category = '장소'
    elif category == 'etc':
        category = '기타'
    return templates.TemplateResponse("gallery_detail.html", {"request": request, "images": images, "category": category})
