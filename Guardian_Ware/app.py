import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


from flask import Flask, render_template, request, jsonify
import base64

from PIL import Image
from twilio.rest import Client
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from skimage.metrics import structural_similarity as ssim
import torch
import json
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import base64

import sqlite3
from flask import Flask, request, jsonify, send_file
from PIL import Image
import cv2
import numpy as np



model_id = "IDEA-Research/grounding-dino-base"
device = "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def get_detected_objs(text_labels , path = 'captured_from_stream.jpg'):
    image = Image.open(path)
    # Check for cats and remote controls


    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]
    detect_objects = []
    for j in zip( result["scores"], result["labels"]):
        if (j[0] > 0.35):
            detect_objects.append(j[1])
    return detect_objects

account_sid = 'AC226ceff2202945620be1ef808ec7afae'
auth_token = 'c3a1711fcc72f61aeed982d4a9dcb322'
twilio_number = '+19473004706' 
to_number = '+919319770015'      # Recipient's phone number

client = Client(account_sid, auth_token)

 # or use your own image-to-text model
trigger_words = ['pen', 'scisor']
first = False
score =0
frame_counter = 0
similarity_threshold = 0.7
frame_skip = 3
debounce_frames = 6

filename = 'captured_from_stream.jpg'

def crop_center(image_path, crop_fraction=0.6):
    """
    Crop the center region of the image.

    Args:
        image_path (str): Path to the image file.
        crop_fraction (float): Fraction of width and height to keep (0 < crop_fraction <= 1).

    Returns:
        PIL.Image: Cropped center portion of the image.
    """
    image = Image.open(image_path)
    w, h = image.size

    crop_w = int(w * crop_fraction)
    crop_h = int(h * crop_fraction)

    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = left + crop_w
    bottom = top + crop_h

    return image.crop((left, top, right, bottom))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

cap_img_path = 'cap_img.jpg'



DATABASE = "frames.db"

# Initialize DB
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS frames (
        session_id TEXT PRIMARY KEY,
        image BLOB
    )''')
    conn.commit()
    conn.close()

init_db()
@app.route('/save')
def save_page():
    return render_template('save_frame.html')

@app.route('/monitor')
def detect_page():
    return render_template('detect.html')


# Save/Update current frame
@app.route('/save_frame', methods=['POST'])
def save_frame():
    session_id = request.form['id']
    image_data = request.form['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('REPLACE INTO frames (session_id, image) VALUES (?, ?)', (session_id, image_bytes))
    conn.commit()
    conn.close()
    return 'Frame saved', 200
last_l = []
# Detect objects in saved frame
@app.route('/detect', methods=['POST'])
def detect():
    global first, different_count, score, cap_img_path, last_l

    session_id = request.form['id']
    trigger_words = request.form.getlist('triggers[]')
    to_number = request.form['phone']

    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('SELECT image FROM frames WHERE session_id=?', (session_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return jsonify({'error': 'No frame found'}), 404

    image_bytes = row[0]
    img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Image decoding failed'}), 400
    cv2.imwrite(filename, frame)
    # Simulate detection
    cropped_img = crop_center(filename, crop_fraction=0.8)
    cropped_img.save(filename)
    l = get_detected_objs(trigger_words, path=filename)
    l.sort()
    if last_l != l and l != []:
        try:
            client.messages.create(
                    body=f"Your dependant may be seeing {l}",
                    from_=twilio_number,
                    to=to_number
                )
        except:
            pass
        print(f"Your dependant may be seeing {l}")
    last_l = l.copy()
    print('Detected',str(l)[1:-1])
    return jsonify({'text': str(l)[1:-1] })

    

    


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use the port Render provides
    app.run(host="0.0.0.0", port=port)

