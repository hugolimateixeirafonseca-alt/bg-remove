"""
padel-alpha-clean  (v9 - chroma remove + clean)
------------------------------------------------
Endpoints:
GET  /                -> health
POST /clean           -> existing alpha-aware clean/upscale pipeline
POST /key-bg-remove   -> remove a solid chroma background (ex: #FF00FF or #00FF00)

/key-bg-remove body:
{
  "url": "https://...",           # required, source image URL (opaque)
  "imgbb_key": "...",             # required
  "color_hex": "#FF00FF",         # optional, explicit chroma colour
  "tolerance": 60,                 # optional, RGB-distance threshold (default 60)
  "feather": 1.2,                  # optional, blur radius on alpha edge (default 1.2)
  "auto_crop_border": false        # optional, if true trims uniform transparent border
}
Response: {"url": "https://i.ibb.co/...", "detected_bg": "#ff00ff"}

/clean body remains compatible with the current project.
"""
import gc
import io
from collections import deque

import numpy as np
import requests
from PIL import Image, ImageFilter
from flask import Flask, jsonify, request

app = Flask(__name__)
Image.MAX_IMAGE_PIXELS = None

MAX_OUTPUT_PX = 4000
MAX_INPUT_PX = 3000


def _fit_within(w, h, max_side):
    longest = max(w, h)
    if longest <= max_side:
        return w, h
    f = max_side / float(longest)
    return max(1, round(w * f)), max(1, round(h * f))


def pick_showcase_bg(garment_set):
    gs = (garment_set or "").strip().lower()
    if gs == "dark":
        return "#1a1a1a"
    return "#ffffff"


def _download_image(url, timeout=120):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def _upload_png_to_imgbb(img, imgbb_key, timeout=120):
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    img = None
    gc.collect()
    buf.seek(0)
    up = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": imgbb_key},
        files={"image": ("design.png", buf, "image/png")},
        timeout=timeout,
    )
    up.raise_for_status()
    out_url = up.json()["data"]["url"]
    buf.close()
    gc.collect()
    return out_url


def _parse_hex_colour(value):
    if not value:
        return None
    v = value.strip().lstrip("#")
    if len(v) != 6:
        raise ValueError("color_hex inválido")
    return tuple(int(v[i:i+2], 16) for i in (0, 2, 4))


def _detect_border_colour(arr):
    h, w, _ = arr.shape
    border = np.concatenate([
        arr[0, :, :3],
        arr[h - 1, :, :3],
        arr[:, 0, :3],
        arr[:, w - 1, :3],
    ], axis=0)
    # round colours slightly to resist antialiasing noise
    rounded = (np.round(border / 16) * 16).astype(np.uint8)
    uniq, counts = np.unique(rounded.reshape(-1, 3), axis=0, return_counts=True)
    dominant = uniq[np.argmax(counts)]
    return tuple(int(x) for x in dominant.tolist())


def _border_connected_mask(near_bg):
    """Flood-fill only pixels connected to the border, so internal similar pixels
    inside the artwork are preserved if not connected to the outer background."""
    h, w = near_bg.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque()

    def push(y, x):
        if 0 <= y < h and 0 <= x < w and near_bg[y, x] and not visited[y, x]:
            visited[y, x] = True
            q.append((y, x))

    for x in range(w):
        push(0, x)
        push(h - 1, x)
    for y in range(h):
        push(y, 0)
        push(y, w - 1)

    while q:
        y, x = q.popleft()
        if y > 0:
            push(y - 1, x)
        if y < h - 1:
            push(y + 1, x)
        if x > 0:
            push(y, x - 1)
        if x < w - 1:
            push(y, x + 1)
    return visited


def _trim_transparent_border(img):
    bbox = img.getbbox()
    if not bbox:
        return img
    return img.crop(bbox)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "padel-alpha-clean", "ver": 9})


@app.route("/key-bg-remove", methods=["POST"])
def key_bg_remove():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    color_hex = data.get("color_hex")
    tolerance = int(float(data.get("tolerance", 60)))
    feather = float(data.get("feather", 1.2))
    auto_crop_border = bool(data.get("auto_crop_border", False))

    if not url or not imgbb_key:
        return jsonify({"error": "faltam 'url' e/ou 'imgbb_key'"}), 400

    try:
        img = _download_image(url)
    except Exception as e:
        return jsonify({"error": f"download falhou: {e}"}), 502

    iw, ih = _fit_within(img.width, img.height, MAX_INPUT_PX)
    if (iw, ih) != img.size:
        img = img.resize((iw, ih), Image.LANCZOS)
        gc.collect()

    arr = np.array(img, dtype=np.uint8)
    rgb = arr[:, :, :3]
    alpha_orig = arr[:, :, 3]

    try:
        target = _parse_hex_colour(color_hex) if color_hex else _detect_border_colour(arr)
    except Exception as e:
        return jsonify({"error": f"cor de chroma inválida: {e}"}), 400

    target_arr = np.array(target, dtype=np.int16)
    dist = np.sqrt(np.sum((rgb.astype(np.int16) - target_arr) ** 2, axis=2))
    near_bg = dist <= tolerance
    bg_mask = _border_connected_mask(near_bg)

    # Start from original alpha; remove border-connected chroma.
    new_alpha = alpha_orig.copy()
    new_alpha[bg_mask] = 0

    out = Image.fromarray(np.dstack([rgb, new_alpha]).astype(np.uint8), mode="RGBA")

    # Feather only slightly, to soften anti-aliased edge transitions.
    if feather > 0:
        a = out.getchannel("A").filter(ImageFilter.GaussianBlur(radius=feather))
        out.putalpha(a)

    if auto_crop_border:
        out = _trim_transparent_border(out)

    try:
        out_url = _upload_png_to_imgbb(out, imgbb_key)
    except Exception as e:
        return jsonify({"error": f"upload ImgBB falhou: {e}"}), 502
    finally:
        out = None
        img = None
        arr = None
        rgb = None
        alpha_orig = None
        new_alpha = None
        gc.collect()

    return jsonify({"url": out_url, "detected_bg": "#%02x%02x%02x" % target})


@app.route("/clean", methods=["POST"])
def clean():
    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url")
    imgbb_key = data.get("imgbb_key")
    scale = float(data.get("scale", 1))
    threshold = int(data.get("threshold", 0))
    erode_px = int(data.get("erode", 0))
    keyline_px = int(data.get("keyline", 0))
    garment_set = data.get("garment_set", "")
    if not url or not imgbb_key:
        return jsonify({"error": "faltam 'url' e/ou 'imgbb_key'"}), 400
    try:
        img = _download_image(url)
    except Exception as e:
        return jsonify({"error": f"download falhou: {e}"}), 502

    showcase_bg = pick_showcase_bg(garment_set)

    iw, ih = _fit_within(img.width, img.height, MAX_INPUT_PX)
    if (iw, ih) != img.size:
        img = img.resize((iw, ih), Image.LANCZOS)
        gc.collect()

    if scale and scale != 1.0:
        target_w, target_h = round(img.width * scale), round(img.height * scale)
        target_w, target_h = _fit_within(target_w, target_h, MAX_OUTPUT_PX)
        if (target_w, target_h) != img.size:
            img = img.resize((max(1, target_w), max(1, target_h)), Image.LANCZOS)
            gc.collect()

    if threshold > 0 or erode_px > 0 or keyline_px > 0:
        import cv2

        arr = np.array(img)
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        del arr

        if threshold > 0:
            opaque = (alpha >= threshold).astype(np.uint8)
        else:
            opaque = (alpha > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        if erode_px > 0:
            opaque = cv2.erode(opaque, kernel, iterations=erode_px)
        if keyline_px > 0:
            outer = cv2.dilate(opaque, kernel, iterations=keyline_px)
            ring = (outer & (1 - opaque)).astype(bool)
            out_rgb = rgb.copy()
            out_rgb[ring] = (255, 255, 255)
            out_alpha = (outer * 255).astype(np.uint8)
        else:
            out_rgb = rgb
            out_alpha = (opaque * 255).astype(np.uint8) if threshold > 0 else alpha
        img = Image.fromarray(np.dstack([out_rgb, out_alpha]), mode="RGBA")
        del rgb, alpha, opaque
        gc.collect()

    try:
        out_url = _upload_png_to_imgbb(img, imgbb_key)
    except Exception as e:
        return jsonify({"error": f"upload ImgBB falhou: {e}"}), 502
    finally:
        img = None
        gc.collect()

    return jsonify({"url": out_url, "showcase_bg": showcase_bg})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
