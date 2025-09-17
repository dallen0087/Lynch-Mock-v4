import streamlit as st
from PIL import Image
import numpy as np
import zipfile
import io
import os

st.set_page_config(layout="wide")
st.title("🟢THE_SHIRT_MATRIX v4🟢")

garments = {
    "tshirts": {
        "preview": "WHITE",
        "colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE", "PINK", "WHITE", "YELLOW"],
        "dark_colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE"],
    },
    "crop_tops": {
        "preview": "WHITE",
        "colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE", "PINK", "WHITE", "RED"],
        "dark_colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE"],
    },
    "hoodies": {
        "preview": "BLACK",
        "colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE", "PINK", "GREY", "YELLOW"],
        "dark_colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE"],
    },
    "sweatshirts": {
        "preview": "PINK",
        "colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE", "PINK", "GREY", "YELLOW"],
        "dark_colors": ["BABY_BLUE", "BLACK", "GREEN", "MAROON", "NAVY_BLUE"],
    },
    "ringer_tees": {
        "preview": "WHITE-BLACK",
        "colors": ["BLACK-WHITE", "WHITE-BLACK", "WHITE-RED"],
        "dark_colors": ["BLACK-WHITE"],
    },
}

color_mode = st.selectbox(
    "🟢Design Color Mode🟢",
    ["Standard (Black/White)", "Blood Red", "Golden Orange", "Royal Blue", "Forest Green", "Unchanged"],
)
color_hex_map = {
    "Blood Red": "#780606",
    "Golden Orange": "#FFA500",
    "Royal Blue": "#4169E1",
    "Forest Green": "#228B22",
}

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

PREVIEW_DISPLAY_SIZE = (600, 600)
PREVIEW_IMAGE_FORMAT = "JPEG"


@st.cache_resource
def load_guide_image(garment: str, guide: str) -> Image.Image:
    with Image.open(f"assets/guides/{garment}/{guide}.png") as img:
        return img.convert("RGBA")


@st.cache_resource
def load_shirt_image(garment: str, color: str) -> Image.Image:
    with Image.open(f"assets/{garment}/{color}.jpg") as img:
        return img.convert("RGBA")


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


if uploaded_files:
    if not os.path.exists("temp_designs"):
        os.makedirs("temp_designs")

    tabs = st.tabs([f"{uf.name.split('.')[0]}" for uf in uploaded_files])
    refresh_queue = []

    for tab, uploaded_file in zip(tabs, uploaded_files):
        with tab:
            design_name = uploaded_file.name.split(".")[0]
            design_path = f"temp_designs/{design_name}.png"
            with open(design_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            design = Image.open(design_path).convert("RGBA")
            alpha = design.split()[-1]
            bbox = alpha.getbbox()
            cropped = design.crop(bbox)

            st.markdown(f"### Design: `{design_name}`")
            cols = st.columns(len(garments))
            col_idx = 0

            for garment, config in garments.items():
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
                    guide_img = load_guide_image(garment, current_settings["guide"])
                    shirt_img = load_shirt_image(garment, current_settings["preview"])
                    preview_image = render_preview(
                        cropped, guide_img, shirt_img, current_settings, color_mode, config["dark_colors"], color_hex_map
                    )
                    st.session_state.previews[combo_key] = preview_to_bytes(preview_image)
                    st.session_state.has_rendered_once[combo_key] = True

                buf = st.session_state.buffer_ui[combo_key]
                with st.expander(f"{garment.replace('_', ' ').title()} Settings for `{design_name}`", expanded=False):
                    guide_folder = f"assets/guides/{garment}"
                    guides = sorted([f.split(".")[0] for f in os.listdir(guide_folder) if f.endswith(".png")])
                    buf["guide"] = st.selectbox("Guide", guides, index=guides.index(buf["guide"]), key=f"{combo_key}_guide")
                    buf["scale"] = st.slider("Scale (%)", 50, 100, buf["scale"], key=f"{combo_key}_scale")
                    buf["offset"] = st.slider("Offset (px)", -100, 100, buf["offset"], key=f"{combo_key}_offset")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📋 Copy {garment} Settings", key=f"{combo_key}_copy"):
                            st.session_state.copied_settings[garment] = buf.copy()
                            st.success("Copied settings")
                    with col2:
                        if st.button(f"📥 Paste to All {garment.title()}", key=f"{combo_key}_paste"):
                            for uf2 in uploaded_files:
                                other_key = f"{uf2.name.split('.')[0]}_{garment}"
                                st.session_state.buffer_ui[other_key] = st.session_state.copied_settings[garment].copy()
                            st.success("Pasted settings")

                    if st.button(f"🔁 Refresh {garment} Preview", key=f"{combo_key}_refresh"):
                        st.session_state.settings[combo_key] = buf.copy()
                        guide_img = load_guide_image(garment, buf["guide"])
                        preview_color = buf.get("preview", config["preview"])
                        shirt_img = load_shirt_image(garment, preview_color)
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
                        st.image(preview_data, caption=garment.replace("_", " ").title())
                col_idx = (col_idx + 1) % len(cols)

    st.markdown("## 🔁 Master Refresh")
    if st.button("🔁 Refresh All Adjusted Previews"):
        for combo_key, buf in st.session_state.buffer_ui.items():
            if combo_key not in st.session_state.settings:
                continue
            if buf != st.session_state.settings[combo_key]:
                design_name, garment = combo_key.split("_", 1)
                design_path = f"temp_designs/{design_name}.png"
                design = Image.open(design_path).convert("RGBA")
                alpha = design.split()[-1]
                bbox = alpha.getbbox()
                cropped = design.crop(bbox)
                guide_img = load_guide_image(garment, buf["guide"])
                preview_color = buf.get("preview", garments[garment]["preview"])
                shirt_img = load_shirt_image(garment, preview_color)
                preview_image = render_preview(
                    cropped, guide_img, shirt_img, buf, color_mode, garments[garment]["dark_colors"], color_hex_map
                )
                st.session_state.previews[combo_key] = preview_to_bytes(preview_image)
                st.session_state.settings[combo_key] = buf.copy()
        st.success("All adjusted previews refreshed.")

        st.markdown("## 📦 Export All Mockups")
    if st.button("📁 Generate and Download ZIP"):
        output_zip = io.BytesIO()
        with zipfile.ZipFile(output_zip, "w") as zip_buffer:
            for uploaded_file in uploaded_files:
                design_name = uploaded_file.name.split(".")[0]
                design_path = f"temp_designs/{design_name}.png"
                design = Image.open(design_path).convert("RGBA")
                alpha = design.split()[-1]
                bbox = alpha.getbbox()
                cropped = design.crop(bbox)

                for garment, config in garments.items():
                    combo_key = f"{design_name}_{garment}"
                    base_settings = st.session_state.settings.get(
                        combo_key,
                        {"scale": 100, "offset": 0, "guide": "STANDARD", "preview": config["preview"]},
                    )
                    settings = base_settings.copy()
                    guide_name = settings.get("guide", "STANDARD")
                    guide_path = f"assets/guides/{garment}/{guide_name}.png"
                    if not os.path.exists(guide_path):
                        continue
                    guide_img = load_guide_image(garment, guide_name)

                    for color in config["colors"]:
                        shirt_path = f"assets/{garment}/{color}.jpg"
                        if not os.path.exists(shirt_path):
                            continue

                        shirt_img = load_shirt_image(garment, color)
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
