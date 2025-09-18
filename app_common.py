from __future__ import annotations
import os
import io
import zipfile
from typing import Dict

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

PREVIEW_DISPLAY_SIZE = (600, 600)
PREVIEW_IMAGE_FORMAT = "JPEG"
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
    },
    "model_t2": {
        "preview": "WHITE",
        "colors": ["WHITE"],
        "dark_colors": [],
        "asset_dir": "TSHIRT_2",
        "guide_dir": "TSHIRT_2",
        "display_name": "Model T2",
    },
    "model_h1": {
        "preview": "BLACK",
        "colors": ["BLACK", "GREEN", "MAROON"],
        "dark_colors": ["BLACK", "GREEN", "MAROON"],
        "asset_dir": "HOODIE_1",
        "guide_dir": "HOODIE_1",
        "display_name": "Model H1",
    },
    "model_h2": {
        "preview": "BLACK",
        "colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "dark_colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "asset_dir": "HOODIE_2",
        "guide_dir": "HOODIE_2",
        "display_name": "Model H2",
    },
    "model_ss1": {
        "preview": "BABY_BLUE",
        "colors": ["BABY_BLUE", "PINK"],
        "dark_colors": ["BABY_BLUE"],
        "asset_dir": "SWEATSHIRT_1",
        "guide_dir": "SWEATSHIRT_1",
        "display_name": "Model SS1",
    },
    "model_ss2": {
        "preview": "BLACK",
        "colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "dark_colors": ["BLACK", "BROWN", "GREEN", "MAROON", "NAVY_BLUE"],
        "asset_dir": "SWEATSHIRT_2",
        "guide_dir": "SWEATSHIRT_2",
        "display_name": "Model SS2",
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


def render_preview(cropped, guide_img, shirt_img, settings, color_mode, dark_colors, hex_map):
    alpha = np.array(guide_img.split()[-1])
    mask = alpha < 10
    ys, xs = np.where(mask)
    box_x0, box_y0, box_x1, box_y1 = xs.min(), ys.min(), xs.max(), ys.max()
    box_w, box_h = box_x1 - box_x0, box_y1 - box_y0

    target_w = int(box_w * (settings["scale"] / 100))
    target_h = int(box_h * (settings["scale"] / 100))
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

    px = box_x0 + (box_w - new_w) // 2
    py = box_y0 + (box_h - new_h) // 2 + settings["offset"]
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

    color_mode = st.selectbox(
        "🟢Design Color Mode🟢",
        ["Standard (Black/White)", "Blood Red", "Golden Orange", "Royal Blue", "Forest Green", "Unchanged"],
    )
    color_hex_map = COLOR_HEX_MAP

    uploaded_files = st.file_uploader("Upload PNG design files", type=["png"], accept_multiple_files=True)

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

    if uploaded_files:
        if not os.path.exists("temp_designs"):
            os.makedirs("temp_designs")

        tabs = st.tabs([os.path.splitext(os.path.basename(uf.name))[0] for uf in uploaded_files])
        active_designs = set()

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
                        }
                    current_settings = st.session_state.settings[combo_key]
                    if combo_key not in st.session_state.buffer_ui:
                        st.session_state.buffer_ui[combo_key] = current_settings.copy()
                    if combo_key not in st.session_state.has_rendered_once:
                        guide_img = load_guide_image(guide_dir, current_settings["guide"])
                        shirt_img = load_shirt_image(asset_dir, current_settings["preview"])
                        preview_image = render_preview(
                            cropped, guide_img, shirt_img, current_settings, color_mode, config["dark_colors"], color_hex_map
                        )
                        st.session_state.previews[combo_key] = preview_to_bytes(preview_image)
                        st.session_state.has_rendered_once[combo_key] = True

                    buf = st.session_state.buffer_ui[combo_key]
                    with st.expander(f"{display_name} Settings for `{design_name}`", expanded=False):
                        guide_folder = os.path.join("assets", "guides", guide_dir)
                        guides = sorted([f.split(".")[0] for f in os.listdir(guide_folder) if f.endswith(".png")])
                        buf["guide"] = st.selectbox("Guide", guides, index=guides.index(buf["guide"]), key=f"{combo_key}_guide")
                        buf["scale"] = st.slider("Scale (%)", 50, 100, buf["scale"], key=f"{combo_key}_scale")
                        buf["offset"] = st.slider("Offset (px)", -100, 100, buf["offset"], key=f"{combo_key}_offset")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"📋 Copy {display_name} Settings", key=f"{combo_key}_copy"):
                                st.session_state.copied_settings[garment] = buf.copy()
                                st.success("Copied settings")
                        with col2:
                            if st.button(f"📥 Paste to All {display_name}", key=f"{combo_key}_paste"):
                                if garment not in st.session_state.copied_settings:
                                    st.warning("Copy settings before pasting to other designs.")
                                else:
                                    copied_values = st.session_state.copied_settings[garment]
                                    for uf2 in uploaded_files:
                                        other_key = f"{os.path.splitext(os.path.basename(uf2.name))[0]}_{garment}"
                                        st.session_state.buffer_ui[other_key] = copied_values.copy()
                                    st.success("Pasted settings")

                        if st.button(f"🔁 Refresh {display_name} Preview", key=f"{combo_key}_refresh"):
                            st.session_state.settings[combo_key] = buf.copy()
                            guide_img = load_guide_image(guide_dir, buf["guide"])
                            preview_color = buf.get("preview", config["preview"])
                            shirt_img = load_shirt_image(asset_dir, preview_color)
                            preview_image = render_preview(
                                cropped, guide_img, shirt_img, buf, color_mode, config["dark_colors"], color_hex_map
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
                if buf != st.session_state.settings[combo_key]:
                    design_name, garment = combo_key.rsplit("_", 1)
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
                        cropped, guide_img, shirt_img, buf, color_mode, config["dark_colors"], color_hex_map
                    )
                    st.session_state.previews[combo_key] = preview_to_bytes(preview_image)
                    st.session_state.settings[combo_key] = buf.copy()
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

                    for garment, config in garments.items():
                        combo_key = f"{design_name}_{garment}"
                        base_settings = st.session_state.settings.get(
                            combo_key,
                            {"scale": 100, "offset": 0, "guide": "STANDARD", "preview": config["preview"]},
                        )
                        settings = base_settings.copy()
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
                            composed = render_preview(
                                cropped, guide_img, shirt_img, color_settings, color_mode, config["dark_colors"], color_hex_map
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
        for stale in stale_designs:
            st.session_state.design_cache.pop(stale, None)
