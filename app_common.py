from __future__ import annotations
import os
import io
import zipfile
from contextlib import suppress
from typing import Dict

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

PREVIEW_DISPLAY_SIZE = (600, 600)
PREVIEW_IMAGE_FORMAT = "JPEG"
COLOR_MODE_OPTIONS = [
    "Standard (Black/White)",
    "Blood Red",
    "Golden Orange",
    "Royal Blue",
    "Forest Green",
    "Unchanged",
]
COLOR_HEX_MAP = {
    "Blood Red": "#780606",
    "Golden Orange": "#FFA500",
    "Royal Blue": "#4169E1",
    "Forest Green": "#228B22",
}

GARMENTS_ALL: Dict[str, Dict[str, object]] = {
    "tshirts": {
        "preview": "WHITE",
        "colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE", "PINK", "WHITE", "YELLOW"],
        "dark_colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE"],
        "asset_dir": "tshirts",
        "guide_dir": "tshirts",
        "display_name": "T-Shirts",
    },
    "hoodies": {
        "preview": "BLACK",
        "colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE", "PINK", "GREY", "YELLOW"],
        "dark_colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE"],
        "asset_dir": "hoodies",
        "guide_dir": "hoodies",
        "display_name": "Hoodies",
    },
    "sweatshirts": {
        "preview": "PINK",
        "colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE", "PINK", "GREY", "YELLOW"],
        "dark_colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE"],
        "asset_dir": "sweatshirts",
        "guide_dir": "sweatshirts",
        "display_name": "Sweatshirts",
    },
    "ringer_tees": {
        "preview": "WHITE-BLACK",
        "colors": ["BLACK-WHITE", "WHITE-BLACK", "WHITE-RED"],
        "dark_colors": ["BLACK-WHITE"],
        "asset_dir": "ringer_tees",
        "guide_dir": "ringer_tees",
        "display_name": "Ringer Tees",
    },
    "model_t1": {
        "preview": "BLACK",
        "colors": ["BLACK", "WHITE"],
        "dark_colors": ["BLACK"],
        "asset_dir": "TSHIRT_1",
        "guide_dir": "TSHIRT_1",
        "display_name": "Model T1",
        "allow_rotation": True,
    },
    "model_t2": {
        "preview": "WHITE",
        "colors": ["WHITE"],
        "dark_colors": [],
        "asset_dir": "TSHIRT_2",
        "guide_dir": "TSHIRT_2",
        "display_name": "Model T2",
        "allow_rotation": True,
    },
    "model_h1": {
        "preview": "BLACK",
        "colors": ["BLACK", "GREEN", "MAROON"],
        "dark_colors": ["BLACK", "GREEN", "MAROON"],
        "asset_dir": "HOODIE_1",
        "guide_dir": "HOODIE_1",
        "display_name": "Model H1",
        "allow_rotation": True,
    },
    "model_h2": {
        "preview": "BLACK",
        "colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "dark_colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "asset_dir": "HOODIE_2",
        "guide_dir": "HOODIE_2",
        "display_name": "Model H2",
        "allow_rotation": True,
    },
    "model_ss1": {
        "preview": "BABY_BLUE",
        "colors": ["BABY_BLUE", "PINK"],
        "dark_colors": ["BABY_BLUE"],
        "asset_dir": "SWEATSHIRT_1",
        "guide_dir": "SWEATSHIRT_1",
        "display_name": "Model SS1",
        "allow_rotation": True,
    },
    "model_ss2": {
        "preview": "BLACK",
        "colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "dark_colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "asset_dir": "SWEATSHIRT_2",
        "guide_dir": "SWEATSHIRT_2",
        "display_name": "Model SS2",
        "allow_rotation": True,
    },
}

FLAT_GARMENT_KEYS = ["tshirts", "hoodies", "sweatshirts", "ringer_tees"]
MODEL_GARMENT_KEYS = ["model_t1", "model_t2", "model_h1", "model_h2", "model_ss1", "model_ss2"]

GARMENTS_FLAT = {key: GARMENTS_ALL[key] for key in FLAT_GARMENT_KEYS}
GARMENTS_MODEL = {key: GARMENTS_ALL[key] for key in MODEL_GARMENT_KEYS}


@st.cache_resource
def _load_image_resource(path: str, last_modified: float) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGBA")


def load_guide_image(guide_dir: str, guide: str) -> Image.Image:
    path = os.path.join("assets", "guides", guide_dir, f"{guide}.png")
    last_modified = os.path.getmtime(path)
    return _load_image_resource(path, last_modified).copy()


def load_shirt_image(asset_dir: str, color: str) -> Image.Image:
    path = os.path.join("assets", asset_dir, f"{color}.jpg")
    last_modified = os.path.getmtime(path)
    return _load_image_resource(path, last_modified).copy()


def sanitize_settings_payload(values) -> dict:
    if not isinstance(values, dict):
        return {}
    sanitized = {}
    for key, value in values.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        sanitized[key] = value
    return sanitized


def remove_prefixed_keys(state_dict, prefix: str) -> list[str]:
    if not isinstance(state_dict, dict):
        return []
    matched = [
        key
        for key in list(state_dict.keys())
        if isinstance(key, str) and key.startswith(prefix)
    ]
    for key in matched:
        state_dict.pop(key, None)
    return matched


def render_preview(cropped, guide_img, shirt_img, settings, color_mode, dark_colors, hex_map):
    alpha = np.array(guide_img.split()[-1])
    mask = alpha < 10
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        box_x0, box_y0 = 0, 0
        box_x1, box_y1 = guide_img.width, guide_img.height
    else:
        box_x0, box_y0 = int(xs.min()), int(ys.min())
        box_x1, box_y1 = int(xs.max()), int(ys.max())
    box_w = max(1, box_x1 - box_x0)
    box_h = max(1, box_y1 - box_y0)

    scale_factor = settings["scale"] / 100
    target_w = max(1, int(box_w * scale_factor))
    target_h = max(1, int(box_h * scale_factor))
    aspect = cropped.width / cropped.height
    if aspect > (target_w / target_h):
        new_w, new_h = target_w, int(target_w / aspect)
    else:
        new_w, new_h = int(target_h * aspect), target_h
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized_alpha = resized.split()[-1]

    if color_mode == "Unchanged":
        fill = resized.copy()
    elif color_mode == "Standard (Black/White)":
        fill_color = "white" if settings["preview"] in dark_colors else "black"
        fill = Image.new("RGBA", resized.size, fill_color)
        fill.putalpha(resized_alpha)
    else:
        fill = Image.new("RGBA", resized.size, hex_map[color_mode])
        fill.putalpha(resized_alpha)

    opacity_value = settings.get("opacity", 95)
    try:
        opacity_value = float(opacity_value)
    except (TypeError, ValueError):
        opacity_value = 95.0
    if opacity_value < 0:
        opacity_value = 0.0
    elif opacity_value > 100:
        opacity_value = 100.0
    if opacity_value != 100.0:
        alpha_scale = opacity_value / 100.0
        fill_alpha = np.array(fill.split()[-1], dtype=np.float32)
        fill_alpha = np.clip(fill_alpha * alpha_scale, 0, 255).astype(np.uint8)
        fill.putalpha(Image.fromarray(fill_alpha, mode="L"))

    angle = settings.get("rotation", 0)
    offset = settings.get("offset", 0)
    if angle:
        center_x = box_x0 + box_w / 2
        center_y = box_y0 + box_h / 2 + offset
        fill = fill.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        px = int(round(center_x - fill.width / 2))
        py = int(round(center_y - fill.height / 2))
    else:
        px = box_x0 + (box_w - fill.width) // 2
        py = box_y0 + (box_h - fill.height) // 2 + offset
    composed = shirt_img.copy()
    composed.paste(fill, (px, py), fill)
    return composed.convert("RGB")


def preview_to_bytes(image: Image.Image) -> bytes:
    preview = image.copy()
    preview.thumbnail(PREVIEW_DISPLAY_SIZE, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    if PREVIEW_IMAGE_FORMAT.upper() == "JPEG":
        preview.save(buffer, format=PREVIEW_IMAGE_FORMAT, quality=85)
    else:
        preview.save(buffer, format=PREVIEW_IMAGE_FORMAT)
    buffer.seek(0)
    return buffer.getvalue()


def open_design_from_bytes(data: bytes) -> Image.Image | None:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def load_cached_design(design_name: str) -> Image.Image | None:
    data = st.session_state.design_cache.get(design_name)
    if data is None:
        design_path = os.path.join("temp_designs", f"{design_name}.png")
        if os.path.exists(design_path):
            with open(design_path, "rb") as f:
                data = f.read()
            st.session_state.design_cache[design_name] = data
    if data is None:
        return None
    return open_design_from_bytes(data)


def run_app(title: str, garments: Dict[str, Dict[str, object]]):
    st.set_page_config(layout="wide")
    st.title(title)

    color_hex_map = COLOR_HEX_MAP

    uploaded_files = st.file_uploader("Upload PNG design files", type=["png"], accept_multiple_files=True)

    active_designs: set[str] = set()

    if "settings" not in st.session_state:
        st.session_state.settings = {}
    if "buffer_ui" not in st.session_state:
        st.session_state.buffer_ui = {}
    if "previews" not in st.session_state:
        st.session_state.previews = {}
    if "copied_settings" not in st.session_state:
        st.session_state.copied_settings = {}
    if "has_rendered_once" not in st.session_state:
        st.session_state.has_rendered_once = {}
    if "design_cache" not in st.session_state:
        st.session_state.design_cache = {}
    if "color_mode" not in st.session_state:
        st.session_state.color_mode = {}

    if uploaded_files:
        if not os.path.exists("temp_designs"):
            os.makedirs("temp_designs")

        tabs = st.tabs([os.path.splitext(os.path.basename(uf.name))[0] for uf in uploaded_files])

        for tab, uploaded_file in zip(tabs, uploaded_files):
            with tab:
                design_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                design_bytes = uploaded_file.getvalue()
                design = open_design_from_bytes(design_bytes)
                if design is None:
                    st.error(f"Unable to read the uploaded design '{uploaded_file.name}'. Please re-upload a valid PNG.")
                    continue
                st.session_state.design_cache[design_name] = design_bytes
                design_path = f"temp_designs/{design_name}.png"
                with open(design_path, "wb") as f:
                    f.write(design_bytes)
                alpha = design.split()[-1]
                bbox = alpha.getbbox()
                if bbox is None:
                    st.warning(f"Design '{uploaded_file.name}' has no visible content and will be skipped.")
                    continue
                cropped = design.crop(bbox)
                active_designs.add(design_name)

                st.markdown(f"### Design: `{design_name}`")
                color_mode_key = f"{design_name}_color_mode_select"
                if design_name not in st.session_state.color_mode:
                    saved_modes = [
                        settings_data.get("color_mode")
                        for combo_key, settings_data in st.session_state.settings.items()
                        if combo_key.startswith(f"{design_name}_")
                    ]
                    initial_color_mode = next(
                        (mode for mode in saved_modes if mode in COLOR_MODE_OPTIONS),
                        COLOR_MODE_OPTIONS[0],
                    )
                    st.session_state.color_mode[design_name] = initial_color_mode
                stored_color_mode = st.session_state.color_mode.get(
                    design_name, COLOR_MODE_OPTIONS[0]
                )
                normalized_color_mode = (
                    stored_color_mode
                    if stored_color_mode in COLOR_MODE_OPTIONS
                    else COLOR_MODE_OPTIONS[0]
                )
                if (
                    color_mode_key not in st.session_state
                    or st.session_state[color_mode_key] not in COLOR_MODE_OPTIONS
                ):
                    st.session_state[color_mode_key] = normalized_color_mode
                if normalized_color_mode != stored_color_mode:
                    st.session_state.color_mode[design_name] = normalized_color_mode
                selected_color_mode = st.selectbox(
                    "🟢Design Color Mode🟢",
                    COLOR_MODE_OPTIONS,
                    key=color_mode_key,
                )
                st.session_state.color_mode[design_name] = selected_color_mode
                design_color_mode = selected_color_mode
                cols = st.columns(len(garments))
                col_idx = 0

                for garment, config in garments.items():
                    display_name = config.get("display_name", garment.replace("_", " ").title())
                    asset_dir = config.get("asset_dir", garment)
                    guide_dir = config.get("guide_dir", asset_dir)
                    combo_key = f"{design_name}_{garment}"
                    if combo_key not in st.session_state.settings:
                        st.session_state.settings[combo_key] = {
                            "scale": 100,
                            "offset": 0,
                            "guide": "STANDARD",
                            "preview": config["preview"],
                            "rotation": 0,
                            "opacity": 95,
                            "color_mode": design_color_mode,
                        }
                    current_settings = st.session_state.settings[combo_key]
                    current_settings["color_mode"] = design_color_mode
                    if "rotation" not in current_settings:
                        current_settings["rotation"] = 0
                    if "opacity" not in current_settings:
                        current_settings["opacity"] = 95
                    if combo_key not in st.session_state.buffer_ui:
                        st.session_state.buffer_ui[combo_key] = sanitize_settings_payload(
                            current_settings
                        )
                    if combo_key not in st.session_state.has_rendered_once:
                        guide_img = load_guide_image(guide_dir, current_settings["guide"])
                        shirt_img = load_shirt_image(asset_dir, current_settings["preview"])
                        preview_image = render_preview(
                            cropped,
                            guide_img,
                            shirt_img,
                            current_settings,
                            current_settings.get("color_mode", design_color_mode),
                            config["dark_colors"],
                            color_hex_map,
                        )
                        st.session_state.previews[combo_key] = preview_to_bytes(preview_image)
                        st.session_state.has_rendered_once[combo_key] = True

                    buf = st.session_state.buffer_ui[combo_key]
                    buf["color_mode"] = design_color_mode
                    if "rotation" not in buf:
                        buf["rotation"] = current_settings.get("rotation", 0)
                    if "opacity" not in buf:
                        buf["opacity"] = current_settings.get("opacity", 95)
                    with st.expander(f"{display_name} Settings for `{design_name}`", expanded=False):
                        guide_folder = os.path.join("assets", "guides", guide_dir)
                        guides = sorted([f.split(".")[0] for f in os.listdir(guide_folder) if f.endswith(".png")])
                        buf["guide"] = st.selectbox("Guide", guides, index=guides.index(buf["guide"]), key=f"{combo_key}_guide")
                        buf["scale"] = st.slider("Scale (%)", 50, 100, buf["scale"], key=f"{combo_key}_scale")
                        buf["offset"] = st.slider("Offset (px)", -100, 100, buf["offset"], key=f"{combo_key}_offset")
                        opacity_default = buf.get("opacity", 95)
                        try:
                            opacity_default = int(round(float(opacity_default)))
                        except (TypeError, ValueError):
                            opacity_default = 95
                        opacity_default = max(0, min(100, opacity_default))
                        buf["opacity"] = st.slider(
                            "Opacity (%)",
                            0,
                            100,
                            opacity_default,
                            key=f"{combo_key}_opacity",
                        )
                        if config.get("allow_rotation"):
                            rotation_value = float(buf.get("rotation", 0))
                            if rotation_value < -10:
                                rotation_value = -10.0
                            elif rotation_value > 10:
                                rotation_value = 10.0
                            buf["rotation"] = st.slider(
                                "Rotation (°)",
                                -10.0,
                                10.0,
                                rotation_value,
                                0.5,
                                key=f"{combo_key}_rotation",
                            )

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📋 Copy {display_name} Settings", key=f"{combo_key}_copy"):
                                copied_payload = sanitize_settings_payload(buf)
                                st.session_state.copied_settings[garment] = {
                                    "values": copied_payload,
                                    "source_design": design_name,
                                }
                                st.success("Copied settings")
                        with col2:
                            if st.button(f"📥 Paste to All {display_name}", key=f"{combo_key}_paste"):
                                if garment not in st.session_state.copied_settings:
                                    st.warning("Copy settings before pasting to other designs.")
                                else:
                                    copied_entry = st.session_state.copied_settings[garment]
                                    source_design = design_name
                                    if isinstance(copied_entry, dict) and "values" in copied_entry:
                                        copied_values = sanitize_settings_payload(
                                            copied_entry.get("values", {})
                                        )
                                        stored_source = copied_entry.get("source_design")
                                        if stored_source:
                                            source_design = stored_source
                                    else:
                                        copied_values = sanitize_settings_payload(copied_entry)
                                    if "rotation" not in copied_values:
                                        copied_values["rotation"] = 0
                                    if "opacity" not in copied_values:
                                        copied_values["opacity"] = buf.get("opacity", 95)
                                    copied_color_mode = copied_values.get("color_mode", design_color_mode)
                                    if copied_color_mode not in COLOR_MODE_OPTIONS:
                                        copied_color_mode = design_color_mode
                                    copied_values["color_mode"] = copied_color_mode
                                    for uf2 in uploaded_files:
                                        other_design_name = os.path.splitext(os.path.basename(uf2.name))[0]
                                        other_key = f"{other_design_name}_{garment}"
                                        st.session_state.buffer_ui[other_key] = sanitize_settings_payload(
                                            copied_values
                                        )
                                        st.session_state.color_mode[other_design_name] = copied_color_mode
                                        st.session_state[f"{other_design_name}_color_mode_select"] = copied_color_mode
                                    st.session_state.copied_settings[garment] = {
                                        "values": sanitize_settings_payload(copied_values),
                                        "source_design": source_design,
                                    }
                                    st.success("Pasted settings")

                        if st.button(f"🔁 Refresh {display_name} Preview", key=f"{combo_key}_refresh"):
                            st.session_state.settings[combo_key] = sanitize_settings_payload(buf)
                            guide_img = load_guide_image(guide_dir, buf["guide"])
                            preview_color = buf.get("preview", config["preview"])
                            shirt_img = load_shirt_image(asset_dir, preview_color)
                            preview_image = render_preview(
                                cropped,
                                guide_img,
                                shirt_img,
                                buf,
                                buf.get("color_mode", design_color_mode),
                                config["dark_colors"],
                                color_hex_map,
                            )
                            st.session_state.previews[combo_key] = preview_to_bytes(preview_image)

                    if combo_key in st.session_state.previews:
                        with cols[col_idx]:
                            preview_data = st.session_state.previews[combo_key]
                            if isinstance(preview_data, Image.Image):
                                preview_data = preview_to_bytes(preview_data)
                                st.session_state.previews[combo_key] = preview_data
                            st.image(preview_data, caption=display_name)
                    col_idx = (col_idx + 1) % len(cols)

        st.markdown("## 🔁 Master Refresh")
        if st.button("🔁 Refresh All Adjusted Previews"):
            for combo_key, buf in st.session_state.buffer_ui.items():
                if combo_key not in st.session_state.settings:
                    continue
                settings_snapshot = st.session_state.settings[combo_key]
                if "rotation" not in buf:
                    buf["rotation"] = settings_snapshot.get("rotation", 0)
                if "rotation" not in settings_snapshot:
                    settings_snapshot["rotation"] = buf["rotation"]
                if "opacity" not in buf:
                    buf["opacity"] = settings_snapshot.get("opacity", 95)
                if "opacity" not in settings_snapshot:
                    settings_snapshot["opacity"] = buf["opacity"]
                design_name, garment = combo_key.rsplit("_", 1)
                design_color_mode = st.session_state.color_mode.get(design_name, COLOR_MODE_OPTIONS[0])
                if design_color_mode not in COLOR_MODE_OPTIONS:
                    design_color_mode = COLOR_MODE_OPTIONS[0]
                buf_color_mode = buf.get("color_mode", design_color_mode)
                if buf_color_mode not in COLOR_MODE_OPTIONS:
                    buf_color_mode = design_color_mode
                buf["color_mode"] = buf_color_mode
                st.session_state.color_mode[design_name] = buf_color_mode
                if buf != settings_snapshot:
                    config = garments.get(garment)
                    if config is None:
                        continue
                    asset_dir = config.get("asset_dir", garment)
                    guide_dir = config.get("guide_dir", asset_dir)
                    design = load_cached_design(design_name)
                    if design is None:
                        st.warning(
                            f"Design '{design_name}' could not be reloaded for refresh. Please re-upload the file if needed."
                        )
                        continue
                    alpha = design.split()[-1]
                    bbox = alpha.getbbox()
                    if bbox is None:
                        st.warning(f"Design '{design_name}' has no visible content and was skipped during refresh.")
                        continue
                    cropped = design.crop(bbox)
                    guide_img = load_guide_image(guide_dir, buf["guide"])
                    preview_color = buf.get("preview", config["preview"])
                    shirt_img = load_shirt_image(asset_dir, preview_color)
                    preview_image = render_preview(
                        cropped,
                        guide_img,
                        shirt_img,
                        buf,
                        buf_color_mode,
                        config["dark_colors"],
                        color_hex_map,
                    )
                    st.session_state.previews[combo_key] = preview_to_bytes(preview_image)
                    st.session_state.settings[combo_key] = sanitize_settings_payload(buf)
            st.success("All adjusted previews refreshed.")

        st.markdown("## 📦 Export All Mockups")
        if st.button("📁 Generate and Download ZIP"):
            output_zip = io.BytesIO()
            with zipfile.ZipFile(output_zip, "w") as zip_buffer:
                for uploaded_file in uploaded_files:
                    design_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]
                    design = load_cached_design(design_name)
                    if design is None:
                        st.warning(
                            f"Skipping '{uploaded_file.name}' because the image data could not be reloaded for export."
                        )
                        continue
                    alpha = design.split()[-1]
                    bbox = alpha.getbbox()
                    if bbox is None:
                        st.warning(f"Skipping '{uploaded_file.name}' because it has no visible content to render.")
                        continue
                    cropped = design.crop(bbox)
                    fallback_color_mode = st.session_state.color_mode.get(design_name, COLOR_MODE_OPTIONS[0])
                    if fallback_color_mode not in COLOR_MODE_OPTIONS:
                        fallback_color_mode = COLOR_MODE_OPTIONS[0]
                        st.session_state.color_mode[design_name] = fallback_color_mode

                    for garment, config in garments.items():
                        combo_key = f"{design_name}_{garment}"
                        base_settings = st.session_state.settings.get(combo_key)
                        if base_settings is None:
                            base_settings = {
                                "scale": 100,
                                "offset": 0,
                                "guide": "STANDARD",
                                "preview": config["preview"],
                                "rotation": 0,
                                "opacity": 95,
                                "color_mode": fallback_color_mode,
                            }
                        else:
                            if "rotation" not in base_settings:
                                base_settings["rotation"] = 0
                            if "opacity" not in base_settings:
                                base_settings["opacity"] = 95
                            if (
                                "color_mode" not in base_settings
                                or base_settings["color_mode"] not in COLOR_MODE_OPTIONS
                            ):
                                base_settings["color_mode"] = fallback_color_mode
                        settings = base_settings.copy()
                        settings["color_mode"] = fallback_color_mode
                        guide_name = settings.get("guide", "STANDARD")
                        asset_dir = config.get("asset_dir", garment)
                        guide_dir = config.get("guide_dir", asset_dir)
                        guide_path = os.path.join("assets", "guides", guide_dir, f"{guide_name}.png")
                        if not os.path.exists(guide_path):
                            continue
                        guide_img = load_guide_image(guide_dir, guide_name)

                        for color in config["colors"]:
                            shirt_path = os.path.join("assets", asset_dir, f"{color}.jpg")
                            if not os.path.exists(shirt_path):
                                continue

                            shirt_img = load_shirt_image(asset_dir, color)
                            color_settings = settings.copy()
                            color_settings["preview"] = color
                            color_mode_to_apply = color_settings.get("color_mode", fallback_color_mode)
                            if color_mode_to_apply not in COLOR_MODE_OPTIONS:
                                color_mode_to_apply = fallback_color_mode
                            composed = render_preview(
                                cropped,
                                guide_img,
                                shirt_img,
                                color_settings,
                                color_mode_to_apply,
                                config["dark_colors"],
                                color_hex_map,
                            )

                            filename = f"{design_name}_{garment}_{color}.jpg"
                            img_bytes = io.BytesIO()
                            composed.save(img_bytes, format="JPEG")
                            zip_buffer.writestr(filename, img_bytes.getvalue())

            output_zip.seek(0)
            st.download_button(
                "⬇️ Download ZIP",
                output_zip.getvalue(),
                file_name="mockups.zip",
                mime="application/zip",
            )

    stale_designs = set(st.session_state.design_cache.keys()) - active_designs
    cleanup_happened = False

    for stale in stale_designs:
        cleanup_happened = True
        prefix = f"{stale}_"
        remove_prefixed_keys(st.session_state.settings, prefix)
        remove_prefixed_keys(st.session_state.buffer_ui, prefix)
        remove_prefixed_keys(st.session_state.previews, prefix)
        remove_prefixed_keys(st.session_state.has_rendered_once, prefix)
        st.session_state.design_cache.pop(stale, None)
        st.session_state.color_mode.pop(stale, None)
        st.session_state.pop(f"{stale}_color_mode_select", None)
        if isinstance(st.session_state.copied_settings, dict):
            copied_to_purge = [
                garment
                for garment, entry in st.session_state.copied_settings.items()
                if isinstance(entry, dict)
                and entry.get("source_design") == stale
            ]
            for garment in copied_to_purge:
                st.session_state.copied_settings.pop(garment, None)
        design_path = os.path.join("temp_designs", f"{stale}.png")
        with suppress(FileNotFoundError):
            os.remove(design_path)

    if not active_designs:
        for state_name in ("settings", "buffer_ui", "previews", "has_rendered_once"):
            state_dict = getattr(st.session_state, state_name, None)
            if isinstance(state_dict, dict) and state_dict:
                state_dict.clear()
                cleanup_happened = True
        if isinstance(st.session_state.copied_settings, dict) and st.session_state.copied_settings:
            st.session_state.copied_settings.clear()
            cleanup_happened = True
        if st.session_state.color_mode:
            st.session_state.color_mode.clear()
            cleanup_happened = True
        cached_design_keys = [
            key for key in st.session_state.keys() if key.endswith("_color_mode_select")
        ]
        for key in cached_design_keys:
            st.session_state.pop(key, None)
            cleanup_happened = True
        if st.session_state.design_cache:
            st.session_state.design_cache.clear()
            cleanup_happened = True

    session_counts = {
        "active_designs": len(active_designs),
        "design_cache": len(st.session_state.design_cache),
        "settings": len(st.session_state.settings),
        "buffer_ui": len(st.session_state.buffer_ui),
        "previews": len(st.session_state.previews),
        "has_rendered_once": len(st.session_state.has_rendered_once),
        "copied_settings": len(st.session_state.copied_settings)
        if isinstance(st.session_state.copied_settings, dict)
        else 0,
    }
    st.session_state.cleanup_summary = session_counts

    if cleanup_happened or not active_designs:
        summary_text = ", ".join(f"{key}={value}" for key, value in session_counts.items())
        prefix = "🧹 Removed stale designs." if cleanup_happened else "📭 No active designs uploaded."
        st.caption(f"{prefix} Session-state sizes — {summary_text}")
