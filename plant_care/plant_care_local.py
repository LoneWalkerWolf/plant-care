OPENAI_API_KEY = APIKEYHERE

import os
import base64
import json
import re
import json
from io import BytesIO
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
import markdown
from openai import OpenAI
import logging
from colorama import Fore, Style, init as colorama_init
from werkzeug.utils import secure_filename

# --- Initialize color logging ---
colorama_init()

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class EmojiFormatter(logging.Formatter):
    def format(self, record):
        prefix = {
            logging.DEBUG: f"{Fore.BLUE}🐛 DEBUG",
            logging.INFO: f"{Fore.GREEN}✅ INFO",
            logging.WARNING: f"{Fore.YELLOW}⚠️  WARNING",
            logging.ERROR: f"{Fore.RED}❌ ERROR",
            logging.CRITICAL: f"{Fore.RED + Style.BRIGHT}🔥 CRITICAL"
        }.get(record.levelno, "")
        message = super().format(record)
        return f"{prefix}{Style.RESET_ALL} {message}"

logger = logging.getLogger("plantcare")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(EmojiFormatter("%(message)s"))
logger.addHandler(handler)

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = "super-secret-key"
client = OpenAI(api_key=OPENAI_API_KEY)

PLANT_DIR = "static/plants"
LOG_PATH = "plant_log.json"
ALL_PLANTS_IMAGE = "all_plants.jpg"

os.makedirs(PLANT_DIR, exist_ok=True)

def slugify(name):
    return re.sub(r'\W+', '_', name.strip().lower())

def load_log():
    if not os.path.exists(LOG_PATH):
        logger.info(f"⚠️ {LOG_PATH} does not exist. Creating a new log file.")
        # Create the log file with an empty dictionary
        with open(LOG_PATH, "w") as f:
            json.dump({}, f, indent=2)
        return {}

    try:
        with open(LOG_PATH, "r") as f:
            content = f.read().strip()
            if not content:
                logger.warning(f"⚠️ {LOG_PATH} is empty. Starting fresh.")
                return {}
            return json.loads(content)
    except Exception as e:
        logger.error(f"❌ Failed to load plant log: {e}")
        return {}

def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

def clean_markdown_to_html(markdown_text):
    lines = markdown_text.strip().splitlines()
    fixed_lines = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[-•*]\s+", stripped):
            fixed_lines.append(stripped)
        elif re.match(r"^[A-Za-z].*:\s*$", stripped):
            fixed_lines.append(f"### {stripped[:-1]}")
        else:
            fixed_lines.append(stripped)
    formatted = "\n".join(fixed_lines)
    return markdown.markdown(formatted)

def get_care_tips(image_path):
    logger.info(f"🌱 Generating care tips for {image_path}")
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Provide structured care tips for this plant:\n"
                    "- Health check\n- Watering advice\n- Maintenance\n- Urgent actions if any\n"
                    "Format in markdown. No hashtags or code blocks. Do not bullet-point nor number."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]
        }],
        max_tokens=500
    )
    raw = response.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError("❌ GPT returned empty content for plant extraction.")
    raw = raw.strip()
    return clean_markdown_to_html(raw)

def identify_plant_name(image_path):
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Please identify the plant in this image. Return only the name of the plant." },
                        { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{base64_image}" } }
                    ]
                }
            ],
            max_tokens=20
        )
    except Exception as e:
        logger.error(f"❌ GPT failed to identify plant: {e}")
        return None

    raw = response.choices[0].message.content.strip()
    name = raw.split("\n")[0].strip()
    return name

def process_diagnosis_text(diagnosis_text):
    # Replace ordered list (ol) with unordered list (ul)
    diagnosis_text = re.sub(r'<ol>', '<ul>', diagnosis_text)
    diagnosis_text = re.sub(r'</ol>', '</ul>', diagnosis_text)

    # Optionally, replace <li> items to ensure no numbering in the list
    # e.g. in case some numbers were left
    diagnosis_text = re.sub(r'<li>(\d+)\. ', '<li>', diagnosis_text)

    return diagnosis_text

def get_diagnosis(image_path):
    logger.info(f"🩺 Diagnosing plant in {image_path}")
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Diagnose the plant. Mention signs of overwatering, underwatering, pests, etc. Format in markdown."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]
        }],
        max_tokens=400
    )
    raw = response.choices[0].message.content.strip()
    
    return clean_markdown_to_html(raw)

from PIL import Image

def resize_image(image_path, target_width=1920, target_height=1080):
    """Resize the image to fit within target resolution (1920x1080) and pad it if necessary."""
    img = Image.open(image_path)
    img = img.convert("RGB")  # Convert to RGB to ensure no alpha channel issues
    img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)  # Resize while maintaining aspect ratio

    # Create a new image with a black background to pad the image to 1920x1080
    new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    new_img.paste(img, ((target_width - img.width) // 2, (target_height - img.height) // 2))
    return new_img

def extract_plants_from_group_photo():
    if not os.path.exists(ALL_PLANTS_IMAGE):
        logger.warning("❌ all_plants.jpg not found. Cannot extract plants.")
        return []

    try:
        img = Image.open(ALL_PLANTS_IMAGE)
        img.verify()  # check it's a real image
    except Exception as e:
        logger.error(f"❌ all_plants.jpg is not a valid image: {e}")
        return []

    logger.info("📸 Processing all_plants.jpg to extract individual crops")
    
    # Resize the image to 1920x1080 before encoding
    resized_image = resize_image(ALL_PLANTS_IMAGE)
    resized_image_path = "resized_all_plants.jpg"
    resized_image.save(resized_image_path)
    
    # Open the resized image and encode it to base64
    with open(resized_image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    # Send resized image to GPT for plant detection and bounding boxes
    prompt = (
        "This image shows some houseplants in a 1920 x 1080 image. Return a JSON array with plant name and large padded bounding box around each plant. Ignore relections from a mirror or window. You need to make sure the bounding box encapsulates the whole plant, with a minimum size of 300x300, ensure it's pot and surrounding area are included in the box. Here's an example:\n"
        '[{"name": "Pothos", "x": 100, "y": 120, "width": 300, "height": 400}]'
    )
    
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
            ]
        }],
        max_tokens=800
    )
    
    # Get the raw response from GPT
    raw = response.choices[0].message.content.strip()
    logger.debug(f"🔎 GPT returned:\n{repr(raw)}")
    
    # Handle empty or invalid responses
    if "no visible houseplants" in raw.lower():
        logger.warning("🌵 GPT detected no houseplants in the image.")
        return []

    # Try to extract the JSON block containing plant data
    try:
        match = re.search(r'\[\s*{.*?}\s*\]', raw, re.DOTALL)
        if match:
            raw_json = match.group(0)
            plant_data = json.loads(raw_json)
        else:
            logger.warning("⚠️ No JSON array found in GPT response. Returning empty list.")
            return []
    except Exception as e:
        logger.error(f"❌ Failed to parse GPT response as JSON: {e}")
        logger.debug(f"📤 Full GPT response:\n{repr(raw)}")
        return []

    # Open the resized full image
    img = resized_image
    width, height = img.size
    full_img = img.copy()
    
    saved = []
    
    # Adjust bounding boxes based on padding and crop the image for each plant
    for plant in plant_data:
        name = plant.get("name", f"plant_{len(saved)+1}")
        slug = slugify(name)
        
        # Adjust bounding box with some padding
        padding_factor = 0.1  # Padding around bounding box (10%)
        
        # Adjust bounding box with padding
        x = max(0, int(plant["x"] - plant["width"] * padding_factor))  # Apply padding left
        y = max(0, int(plant["y"] - plant["height"] * padding_factor))  # Apply padding top
        w = int(plant["width"] * (1 + padding_factor * 2))  # Increase width by 20% for padding on both sides
        h = int(plant["height"] * (1 + padding_factor * 2))  # Increase height by 20% for padding on both sides
        
        # Ensure coordinates stay within image bounds
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        
        # Crop the image based on the adjusted bounding box
        crop = full_img.crop((x, y, x2, y2))
        filename = f"{slug}.jpg"
        
        # Save the cropped image
        crop.save(os.path.join(PLANT_DIR, filename))
        saved.append((slug, name))
        logger.info(f"🖼️ Saved {filename} for plant '{name}'")
    
    return saved

@app.route('/reidentify/<plant_id>')
def reidentify(plant_id):
    log = load_log()
    file_path = os.path.join(PLANT_DIR, plant_id)

    if not os.path.exists(file_path):
        flash(f"❌ Image not found for {plant_id}")
        return redirect(url_for('index'))

    new_name = identify_plant_name(file_path)
    if not new_name:
        flash("⚠️ Could not re-identify the plant.")
        return redirect(url_for('index'))

    current_entry = log.get(plant_id)
    old_name = current_entry["name"]
    
    if new_name.lower() != old_name.lower():
        new_id = f"{slugify(new_name)}.jpg"
        new_path = os.path.join(PLANT_DIR, new_id)
        os.rename(file_path, new_path)
        log[new_id] = log.pop(plant_id)
        log[new_id]["name"] = new_name
        logger.info(f"🔄 Renamed '{old_name}' to '{new_name}'")
        flash(f"🔄 Renamed '{old_name}' to '{new_name}'")
    else:
        flash("✅ Name is still correct — no changes made.")
    
    save_log(log)
    return redirect(url_for('index'))

@app.route('/')
def index():
    log = load_log()
    plant_cards = []

    if os.path.exists(ALL_PLANTS_IMAGE) and not os.listdir(PLANT_DIR):
        named_plants = extract_plants_from_group_photo()
        for slug, name in named_plants:
            log[f"{slug}.jpg"] = {
                "name": name,
                "last_watered": "Never",
                "care_tips": "<em>Not cared for yet.</em>",
                "diagnosis": "<em>Not diagnosed yet.</em>"
            }
        save_log(log)

    # Check if there are no plants in the log
    if not log or not any(plant.get("name") for plant in log.values()):
        logger.info("⚠️ No plants found. Prompting for plant upload.")
        return render_template("index.html", plants=plant_cards, no_plants=(len(plant_cards) == 0))

    for filename in sorted(os.listdir(PLANT_DIR)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        plant_id = filename
        entry = log.get(plant_id, {})
        last_watered = entry.get("last_watered", "Never")
        care_tips = entry.get("care_tips", "<em>Not cared for yet.</em>")
        diagnosis = entry.get("diagnosis", "<em>Not diagnosed yet.</em>")
        name = entry.get("name", plant_id.split(".")[0].replace("_", " ").title())
        plant_cards.append({
            "id": plant_id,
            "name": name,
            "image_url": f"/static/plants/{filename}",
            "last_watered": last_watered,
            "care_tips": care_tips,
            "diagnosis": diagnosis,
            "care_history": entry.get("care_history", []),
            "diagnosis_history": entry.get("diagnosis_history", [])
        })
    return render_template("index.html", plants=plant_cards)

@app.route('/schedule')
def schedule():
    log = load_log()
    events = []

    for plant_id, data in log.items():
        name = data.get("name", plant_id.split(".")[0].replace("_", " ").title())
        last_watered = data.get("last_watered", "Never")
        care_tips = data.get("care_tips", "")

        if last_watered == "Never":
            next_dt = datetime.now() + timedelta(days=1)
        else:
            try:
                last_dt = datetime.strptime(last_watered, "%Y-%m-%d %H:%M")
                freq_days = 7
                tips = care_tips.lower()
                if "2-3 days" in tips:
                    freq_days = 3
                elif "5-7 days" in tips:
                    freq_days = 6
                elif "10-14 days" in tips:
                    freq_days = 12
                elif "weekly" in tips:
                    freq_days = 7
                elif "daily" in tips:
                    freq_days = 1
                next_dt = last_dt + timedelta(days=freq_days)
            except Exception as e:
                logger.warning(f"Could not parse last watered for {name}: {e}")
                next_dt = datetime.now() + timedelta(days=3)

        events.append({
            "title": f"💧 Water {name}",
            "start": next_dt.strftime("%Y-%m-%d")
        })

    logger.info(f"📅 Loaded {len(events)} watering events")
    return render_template("schedule.html", events=json.dumps(events))

@app.route('/sync')
def sync_log():
    log = load_log()
    updated = False

    for filename in os.listdir(PLANT_DIR):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        if filename not in log:
            name = filename.rsplit('.', 1)[0].replace("_", " ").title()
            log[filename] = {
                "name": name,
                "last_watered": "Never",
                "care_tips": "<em>Not generated yet.</em>",
                "diagnosis": "<em>Not diagnosed yet.</em>"
            }
            logger.info(f"🆕 Added missing log entry for: {filename}")
            updated = True

    if updated:
        save_log(log)
        logger.info("✅ Synced plant log successfully.")
    else:
        logger.info("🔍 Sync ran — no missing entries found.")

    return redirect(url_for('index'))


@app.route('/refresh_all')
def refresh_all():
    log = load_log()
    for plant_id in log:
        img_path = os.path.join(PLANT_DIR, plant_id)
        if os.path.exists(img_path):
            try:
                care_tips = get_care_tips(img_path)
                log[plant_id].setdefault("care_history", []).append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "content": care_tips
                })
                log[plant_id]["care_tips"] = care_tips
                save_log(log)
            except Exception as e:
                log[plant_id]["care_tips"] = f"<em>Error: {e}</em>"
    save_log(log)
    logger.info("🔄 Refreshed care tips for all plants")
    return redirect(url_for('index'))

@app.route('/diagnose_all')
def diagnose_all():
    log = load_log()
    for plant_id in log:
        img_path = os.path.join(PLANT_DIR, plant_id)
        if os.path.exists(img_path):
            try:
                diagnosis = get_diagnosis(img_path)
                log[plant_id].setdefault("diagnosis_history", []).append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "content": diagnosis
                })
                log[plant_id]["diagnosis"] = diagnosis
                save_log(log)
            except Exception as e:
                log[plant_id]["diagnosis"] = f"<em>Error: {e}</em>"
    save_log(log)
    logger.info("🩺 Diagnosed all plants")
    return redirect(url_for('index'))

@app.route('/refresh/<plant_id>')
def refresh_tips(plant_id):
    log = load_log()
    img_path = os.path.join(PLANT_DIR, plant_id)
    if os.path.exists(img_path):
        try:
            care_tips = get_care_tips(img_path)
            log[plant_id].setdefault("care_history", []).append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "content": care_tips
            })
            log[plant_id]["care_tips"] = care_tips
            save_log(log)
        except Exception as e:
            logger.error(f"Error updating care tips: {e}")
    return redirect(url_for('index'))

@app.route('/diagnose/<plant_id>')
def diagnose(plant_id):
    log = load_log()
    img_path = os.path.join(PLANT_DIR, plant_id)
    if os.path.exists(img_path):
        try:
            diagnosis = get_diagnosis(img_path)
            log[plant_id].setdefault("diagnosis_history", []).append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "content": diagnosis
            })
            log[plant_id]["diagnosis"] = diagnosis
            save_log(log)
        except Exception as e:
            logger.error(f"Error during diagnosis: {e}")
    return redirect(url_for('index'))

@app.route('/water/<plant_id>', methods=["POST"])
def mark_watered(plant_id):
    log = load_log()
    if plant_id in log:
        log[plant_id]["last_watered"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_log(log)
        logger.info(f"💧 Marked '{plant_id}' as watered")
    return redirect(url_for('index'))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    plant_id = request.args.get("plant")
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            img = Image.open(file.stream)
            width, height = img.size

            if height > width:
                flash("⚠️ Please upload a landscape image (wider than tall).")
                return redirect(url_for("upload", plant=plant_id))

            file.stream.seek(0)  # ✅ rewind so we can save properly later
            if plant_id == "new":
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                new_filename = f"plant_{timestamp}.jpg"
                file_path = os.path.join(PLANT_DIR, new_filename)
                file.save(file_path)

                identified_name = identify_plant_name(file_path)
                slug = slugify(identified_name or f"plant_{timestamp}")

                # Rename the file
                new_path = os.path.join(PLANT_DIR, f"{slug}.jpg")
                os.rename(file_path, new_path)

                log = load_log()
                log[f"{slug}.jpg"] = {
                    "name": identified_name or slug,
                    "last_watered": "Never",
                    "care_tips": "<em>Not generated yet.</em>",
                    "diagnosis": "<em>Not diagnosed yet.</em>"
                }
                save_log(log)

                flash(f"✅ Uploaded new plant: {identified_name or slug}")
                return redirect(url_for("index"))
            
            elif plant_id:
                # This is an update for a specific plant
                logger.info(f"📥 Uploading new image for plant ID: {plant_id}")
                file_path = os.path.join(PLANT_DIR, plant_id)
                file.save(file_path)

                # Identify plant name via GPT
                identified_name = identify_plant_name(file_path)
                log = load_log()
                old_name = log[plant_id]["name"]
                if identified_name and identified_name.lower() != old_name.lower():
                    logger.info(f"🔄 Renaming '{old_name}' to '{identified_name}'")
                    new_id = f"{identified_name.lower().replace(' ', '_')}.jpg"
                    new_path = os.path.join(PLANT_DIR, new_id)
                    os.rename(file_path, new_path)
                    log[new_id] = log.pop(plant_id)
                    log[new_id]["name"] = identified_name
                    log[new_id]["image"] = f"/static/plants/{new_id}"
                save_log(log)

                flash(f"✅ Updated image for plant. Name: {identified_name}")
                return redirect(url_for("index"))
            else:
                # If no plant_id: this is the main group upload
                file.save(ALL_PLANTS_IMAGE)
                for f in os.listdir(PLANT_DIR):
                    os.remove(os.path.join(PLANT_DIR, f))
                if os.path.exists(LOG_PATH):
                    os.remove(LOG_PATH)
                flash("✅ New group photo uploaded successfully!")
                return redirect(url_for("index"))
        else:
            flash("⚠️ Please select a file.")
    return render_template("upload.html")

if __name__ == "__main__":
    logger.info("🌿 Plant Care Dashboard starting on http://10.50.1.151 ...")

    log = load_log()
    updated = False
    for filename in os.listdir(PLANT_DIR):
        if filename not in log:
            log[filename] = {
                "name": filename.split(".")[0].replace("_", " ").title(),
                "last_watered": "Never",
                "care_tips": "<em>Not generated yet.</em>",
                "diagnosis": "<em>Not diagnosed yet.</em>"
            }
            logger.info(f"🆕 Synced missing log entry for: {filename}")
            updated = True
    if updated:
        save_log(log)
        logger.info("✅ Plant log synced at startup")

    app.run(host="0.0.0.0", port=5000, debug=True)