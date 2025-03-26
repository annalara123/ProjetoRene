import os
import cv2
import numpy as np
import base64
import requests
import threading
import logging
from functools import lru_cache
from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_bounding_box(geometry):

    if geometry["type"] == "Point":
        lat, lon = geometry["coordinates"]
        return lat, lat, lon, lon
    elif geometry["type"] in ["Polygon", "MultiPolygon"]:
        coordinates = geometry.get("coordinates", [])
        if not coordinates:
            return None, None, None, None
        latitudes = []
        longitudes = []
        polygons = coordinates if geometry["type"] == "MultiPolygon" else [coordinates]
        for polygon in polygons:
            for point in polygon[0]:
                lon, lat = point
                latitudes.append(lat)
                longitudes.append(lon)
        return min(latitudes), max(latitudes), min(longitudes), max(longitudes)
    else:
        return None, None, None, None

PLANET_API_KEY = "PLAKc4f887589be347888991c62ffb5b37a0"
PLANET_API_URL = "https://api.planet.com/data/v1/quick-search"

@lru_cache(maxsize=10)
def get_planet_image(lat, lon):
    headers = {"Authorization": f"api-key {PLANET_API_KEY}"}
    payload = {
        "item_types": ["PSScene"],
        "filter": {
            "type": "GeometryFilter",
            "field_name": "geometry",
            "config": {
                "type": "Point",
                "coordinates": [lon, lat]
            }
        }
    }
    try:
        response = requests.post(PLANET_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not data.get("features"):
            return None, None, "Nenhuma imagem encontrada para as coordenadas"
        item = data["features"][0]
        image_url = item["_links"]["thumbnail"]
        geometry = item["geometry"]
        lat_min, lat_max, lon_min, lon_max = get_bounding_box(geometry)
        if lat_min is None:
            return None, None, "Falha ao extrair caixa delimitadora"
        image_response = requests.get(image_url, headers=headers)
        image_response.raise_for_status()
        image_path = os.path.join(UPLOAD_FOLDER, "planet_image.jpg")
        with open(image_path, "wb") as f:
            f.write(image_response.content)
        global_bbox = (lat_min, lat_max, lon_min, lon_max)
        return image_path, global_bbox, None
    except requests.RequestException as e:
        logger.error(f"Erro na requisição: {e}")
        return None, None, f"Erro na requisição: {str(e)}"

def split_image(image, num_rows, num_cols):
    parts = []
    global_height, global_width = image.shape[:2]
    part_height = global_height // num_rows
    part_width = global_width // num_cols
    for i in range(num_rows):
        for j in range(num_cols):
            y_offset = i * part_height
            x_offset = j * part_width
            part_h = global_height - y_offset if i == num_rows - 1 else part_height
            part_w = global_width - x_offset if j == num_cols - 1 else part_width
            sub_image = image[y_offset:y_offset+part_h, x_offset:x_offset+part_w].copy()
            parts.append({
                "image": sub_image,
                "x_offset": x_offset,
                "y_offset": y_offset,
                "width": part_w,
                "height": part_h,
                "global_width": global_width,
                "global_height": global_height
            })
    return parts

def detect_illegal_construction(sub_image, metadata, global_bbox):
    lat_min, lat_max, lon_min, lon_max = global_bbox
    x_offset = metadata["x_offset"]
    y_offset = metadata["y_offset"]
    global_width = metadata["global_width"]
    global_height = metadata["global_height"]

    gray = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            x, y, w, h = cv2.boundingRect(contour)
            global_x = x_offset + x
            global_y = y_offset + y

            lat_top = lat_max - (global_y / (global_height - 1)) * (lat_max - lat_min)
            lat_bottom = lat_max - ((global_y + h - 1) / (global_height - 1)) * (lat_max - lat_min)
            lon_left = lon_min + (global_x / (global_width - 1)) * (lon_max - lon_min)
            lon_right = lon_min + ((global_x + w - 1) / (global_width - 1)) * (lon_max - lon_min)

            google_maps_url = f"https://www.google.com/maps?q={lat_top},{lon_left}"

            detections.append({
                "latitude_top": round(lat_top, 6),
                "longitude_left": round(lon_left, 6),
                "latitude_bottom": round(lat_bottom, 6),
                "longitude_right": round(lon_right, 6),
                "width_pixels": w,
                "height_pixels": h,
                "google_maps_url": google_maps_url
            })
    return detections


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/satellite', methods=['GET'])
def process_satellite():
    try:
        lat = float(request.args.get('lat', 48.97134))
        lon = float(request.args.get('lon', 2.57834))
    except (ValueError, TypeError):
        return jsonify({"error": "Latitude e longitude inválidas"}), 400

    image_path, global_bbox, error = get_planet_image(lat, lon)
    if error:
        return jsonify({"error": error}), 500
    if global_bbox is None:
        return jsonify({"error": "Erro ao processar a caixa delimitadora"}), 500

    full_image = cv2.imread(image_path)
    if full_image is None:
        return jsonify({"error": "Erro ao ler a imagem"}), 500

    parts = split_image(full_image, num_rows=2, num_cols=2)
    detections_results = []
    threads = []

    def process_part(part):
        dets = detect_illegal_construction(part["image"], part, global_bbox)
        detections_results.extend(dets)

    for part in parts:
        t = threading.Thread(target=process_part, args=(part,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    annotated_image = full_image.copy()
    gh, gw = full_image.shape[:2]
    lat_min, lat_max, lon_min, lon_max = global_bbox
    for det in detections_results:
        x1 = int(((det["longitude_left"] - lon_min) / (lon_max - lon_min)) * (gw - 1))
        y1 = int(((lat_max - det["latitude_top"]) / (lat_max - lat_min)) * (gh - 1))
        x2 = int(((det["longitude_right"] - lon_min) / (lon_max - lon_min)) * (gw - 1))
        y2 = int(((lat_max - det["latitude_bottom"]) / (lat_max - lat_min)) * (gh - 1))
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ret, buffer = cv2.imencode('.jpg', annotated_image)
    if not ret:
        return jsonify({"error": "Erro ao codificar a imagem"}), 500
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jsonify({
        "lat": lat,
        "lon": lon,
        "detections": detections_results,
        "processed_image": f"data:image/jpeg;base64,{jpg_as_text}"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)