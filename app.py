# ==============================
# IMPORTS
# ==============================
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as RLImage,
    Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
import tempfile
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import BytesIO

# ==============================
# PAGE CONFIG — mobile friendly
# ==============================
st.set_page_config(
    page_title="Fracture Detection — AI Radiology",
    layout="centered",                # better for mobile
    page_icon="🦴",
    initial_sidebar_state="collapsed",  # sidebar hidden initially
)

# ==============================
# CUSTOM CSS — professional medical UI + mobile responsiveness
# ==============================
st.markdown("""
<style>
/* ---- Global ---- */
[data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stHeader"] { background: transparent; }

/* ---- Typography ---- */
h1, h2, h3, h4 { color: #e6edf3; }
p, label, div { color: #8b949e; }

/* ---- Metric cards ---- */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 1.8rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }

/* ---- Buttons ---- */
.stButton button {
    background: #1f6feb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    transition: background 0.2s;
}
.stButton button:hover { background: #388bfd; }

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 20px;
}

/* ---- Progress bar ---- */
[data-testid="stProgress"] > div { background: #1f6feb; border-radius: 4px; }

/* ---- Expander ---- */
[data-testid="stExpander"] { background: #161b22; border: 1px solid #30363d; border-radius: 10px; }

/* ---- Divider ---- */
hr { border-color: #30363d; }

/* ---- Sidebar model info ---- */
.sidebar-pill {
    display: inline-block;
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb55;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    margin-top: 6px;
}

/* ===== MOBILE RESPONSIVE ===== */
@media (max-width: 768px) {
    .main .block-container {
        padding: 0.75rem;
    }
    h1 {
        font-size: 1.6rem !important;
    }
    h2 {
        font-size: 1.3rem !important;
    }
    [data-testid="metric-container"] {
        padding: 8px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    .stButton button {
        padding: 6px 12px;
        font-size: 14px;
    }
    /* Make images responsive */
    .stImage img {
        max-width: 100%;
        height: auto;
    }
    /* Reduce sidebar width on mobile */
    [data-testid="stSidebar"] {
        width: 85vw !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ==============================
# SESSION STATE
# ==============================
if "history" not in st.session_state:
    st.session_state.history = []

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ==============================
# HELPERS
# ==============================
def get_risk(conf):
    if conf > 0.85:
        return "HIGH", "🔴"
    elif conf > 0.60:
        return "MEDIUM", "🟡"
    return "LOW", "🟢"

def resize_image(img, width=480):
    h, w = img.shape[:2]
    ratio = width / w
    return cv2.resize(img, (width, int(h * ratio)))

def extract_features(crop, full_shape, box):
    x1, y1, x2, y2, _ = box
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    mean_i = float(np.mean(gray))
    std_i = float(np.std(gray))
    area_px = (x2 - x1) * (y2 - y1)
    total_px = full_shape[0] * full_shape[1]
    area_pct = area_px / total_px * 100
    return mean_i, std_i, edge_density, area_px, area_pct

def generate_heatmap(img, boxes):
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for (x1, y1, x2, y2, conf) in boxes:
        # gaussian blob centred on detection
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sigma_x = max((x2 - x1) // 2, 1)
        sigma_y = max((y2 - y1) // 2, 1)
        for dy in range(max(0, y1 - sigma_y * 2), min(img.shape[0], y2 + sigma_y * 2)):
            for dx in range(max(0, x1 - sigma_x * 2), min(img.shape[1], x2 + sigma_x * 2)):
                val = conf * np.exp(-0.5 * (
                    ((dx - cx) / sigma_x) ** 2 + ((dy - cy) / sigma_y) ** 2
                ))
                heatmap[dy, dx] += val
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.55, colored, 0.45, 0)

def draw_detections(img, boxes):
    out = img.copy()
    for (x1, y1, x2, y2, conf) in boxes:
        risk, _ = get_risk(conf)
        color = {
            "HIGH": (0, 0, 220),
            "MEDIUM": (0, 160, 240),
            "LOW": (0, 210, 90),
        }[risk]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"Fracture {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out

def make_confidence_chart(history):
    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#161b22")
    ax.set_facecolor("#161b22")
    confs = [h["conf"] for h in history]
    labels = [f"#{i+1}" for i in range(len(history))]
    bar_colors = ["#cf222e" if h["pred"] == "FRACTURE" else "#2da44e" for h in history]
    ax.bar(labels, confs, color=bar_colors, width=0.55)
    ax.axhline(0.85, color="#f0883e", linewidth=1, linestyle="--", alpha=0.8, label="High risk threshold")
    ax.axhline(0.60, color="#d29922", linewidth=1, linestyle="--", alpha=0.8, label="Medium risk threshold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Confidence", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    legend = ax.legend(fontsize=7, facecolor="#1c2128", edgecolor="#30363d", labelcolor="#8b949e")
    fig.tight_layout(pad=0.5)
    return fig

def make_feature_chart(mean_i, std_i, edge_d):
    fig, axes = plt.subplots(1, 3, figsize=(7, 2), facecolor="#161b22")
    labels = ["Mean Intensity", "Std Intensity", "Edge Density"]
    values = [mean_i / 255, std_i / 128, min(edge_d * 10, 1.0)]
    bar_colors = ["#1f6feb", "#388bfd", "#58a6ff"]
    for ax, lbl, val, clr in zip(axes, labels, values, bar_colors):
        ax.set_facecolor("#161b22")
        ax.barh([0], [val], color=clr, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_title(lbl, color="#8b949e", fontsize=8)
        ax.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.text(min(val + 0.05, 0.95), 0, f"{val:.2f}", va="center", color="#e6edf3", fontsize=8)
    fig.tight_layout(pad=0.5)
    return fig

# ==============================
# PDF GENERATION — enhanced
# ==============================
def generate_pdf(prediction, confidence, risk, image_path, features, filename):
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    doc = SimpleDocTemplate(pdf_path, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#0d1117"), spaceAfter=6)
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#57606a"), spaceAfter=4)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11, textColor=colors.HexColor("#24292f"), spaceAfter=8)
    head2_style = ParagraphStyle("Head2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#0d1117"), spaceBefore=14, spaceAfter=6)
    risk_color = {"HIGH": colors.HexColor("#cf222e"), "MEDIUM": colors.HexColor("#d29922"), "LOW": colors.HexColor("#2da44e")}[risk]

    elements = []
    elements.append(Paragraph("Pediatric Wrist Fracture Detection Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  File: {filename}", sub_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#d0d7de")))
    elements.append(Spacer(1, 0.4*cm))
    elements.append(Paragraph("Summary", head2_style))

    data = [
        ["Field", "Value"],
        ["Prediction", prediction],
        ["Confidence Score", f"{confidence:.4f}  ({confidence*100:.1f}%)"],
        ["Risk Level", risk],
        ["Model", "YOLOv8 (best.pt)"],
        ["Analysis Time", datetime.now().strftime("%H:%M:%S")],
    ]
    tbl = Table(data, colWidths=[5*cm, 9*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d1117")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f6f8fa"), colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
        ("TEXTCOLOR", (1, 3), (1, 3), risk_color),
        ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 0.4*cm))

    if features:
        elements.append(Paragraph("Extracted Image Features", head2_style))
        mean_i, std_i, edge_d, area_px, area_pct = features
        feat_data = [
            ["Feature", "Value", "Description"],
            ["Mean Intensity", f"{mean_i:.2f}", "Average pixel brightness in ROI"],
            ["Std Intensity", f"{std_i:.2f}", "Pixel variance (texture measure)"],
            ["Edge Density", f"{edge_d:.4f}", "Ratio of edge pixels (fracture line)"],
            ["Detection Area", f"{area_px:,} px²", f"{area_pct:.1f}% of total image"],
        ]
        ftbl = Table(feat_data, colWidths=[4*cm, 3*cm, 7*cm])
        ftbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d1117")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f6f8fa"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(ftbl)
        elements.append(Spacer(1, 0.4*cm))

    elements.append(Paragraph("Annotated Image", head2_style))
    elements.append(RLImage(image_path, width=10*cm, height=10*cm))
    elements.append(Spacer(1, 0.4*cm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#d0d7de")))
    elements.append(Spacer(1, 0.2*cm))
    elements.append(Paragraph(
        "⚠️  DISCLAIMER: This report is generated by an AI-assisted screening tool and is NOT a substitute "
        "for professional medical diagnosis. All findings must be reviewed by a qualified radiologist or physician.",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#cf222e"))
    ))
    doc.build(elements)
    return pdf_path

# ==============================
# MODEL INFERENCE
# ==============================
def process_image(image):
    results = model(image)[0]
    boxes = []
    confs = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        boxes.append((x1, y1, x2, y2, conf))
        confs.append(conf)
    if confs:
        return boxes, "FRACTURE", max(confs)
    return boxes, "NORMAL", 0.0

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("## 🦴 FractureAI")
    st.markdown('<span class="sidebar-pill">YOLOv8 · best.pt</span>', unsafe_allow_html=True)
    st.divider()
    st.markdown("### Settings")
    conf_threshold = st.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05,
                                help="Detections below this are ignored")
    show_heatmap = st.checkbox("Show attention heatmap", value=True)
    show_features = st.checkbox("Show extracted features", value=True)
    
    st.divider()
    st.markdown("### Display settings")
    # Default width reduced for mobile (280px is comfortable on most phones)
    display_width = st.slider("Image width (px)", 150, 600, 280, 20,
                              help="Smaller values work better on mobile devices")
    
    st.divider()
    st.markdown("### Risk Thresholds")
    st.markdown("""
    | Level | Confidence |
    |-------|-----------|
    | 🔴 HIGH | > 0.85 |
    | 🟡 MEDIUM | 0.60 – 0.85 |
    | 🟢 LOW | < 0.60 |
    """)
    st.divider()

    if st.session_state.history:
        st.markdown("### Session History")
        for i, h in enumerate(st.session_state.history[-5:]):
            emoji = "🔴" if h["pred"] == "FRACTURE" else "🟢"
            st.markdown(f"`#{i+1}` {emoji} **{h['pred']}** — `{h['conf']:.2f}`")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

# ==============================
# MAIN UI
# ==============================
st.title("Pediatric Wrist Fracture Detection")
st.caption("AI-assisted radiology screening · Not a medical diagnosis")
st.divider()

uploaded_file = st.file_uploader(
    "Upload X-ray image",
    type=["jpg", "png", "jpeg"],
    help="Upload a wrist X-ray image. Supports JPG, PNG, JPEG."
)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    col_prev, col_info = st.columns([2, 1])
    with col_prev:
        st.subheader("Uploaded image")
        # Use fixed display width (respects user's slider)
        st.image(resize_image(image, width=display_width), channels="BGR", width=display_width, use_column_width=False)
    with col_info:
        st.subheader("Image info")
        h, w = image.shape[:2]
        st.metric("Width", f"{w} px")
        st.metric("Height", f"{h} px")
        st.metric("Channels", image.shape[2] if len(image.shape) == 3 else 1)
        st.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")

    st.divider()
    analyze_btn = st.button("🔍 Analyze Image", use_container_width=True)

    if analyze_btn:
        with st.spinner("Running YOLOv8 inference..."):
            boxes, prediction, conf = process_image(image)
            # filter by threshold
            boxes = [(x1, y1, x2, y2, c) for (x1, y1, x2, y2, c) in boxes if c >= conf_threshold]
            if boxes:
                conf = max(c for *_, c in boxes)
                prediction = "FRACTURE"
            else:
                conf = 0.0
                prediction = "NORMAL"

        risk, risk_emoji = get_risk(conf)

        # annotated image
        result_img = draw_detections(image.copy(), boxes)

        # save temp for PDF
        temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(temp_img_path, result_img)

        # detection + heatmap side by side (on mobile they will stack because of centered layout)
        cols = st.columns(2 if show_heatmap else 1)
        with cols[0]:
            st.subheader("Detection result")
            st.image(resize_image(result_img, width=display_width), channels="BGR", width=display_width, use_column_width=False)
        if show_heatmap and len(cols) > 1:
            with cols[1]:
                st.subheader("Attention heatmap")
                heatmap_img = generate_heatmap(image.copy(), boxes)
                st.image(resize_image(heatmap_img, width=display_width), channels="BGR", width=display_width, use_column_width=False)

        st.divider()
        st.subheader("Diagnosis summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction", prediction)
        m2.metric("Confidence", f"{conf:.3f}")
        m3.metric("Risk Level", f"{risk_emoji} {risk}")
        m4.metric("Detections", len(boxes))

        if prediction == "FRACTURE":
            st.error(f"🚨 Fracture detected with **{conf*100:.1f}%** confidence. Recommend clinical review.")
        else:
            st.success("✅ No fracture detected. Continue monitoring if symptoms persist.")

        # confidence gauge bar
        st.markdown("**Confidence score**")
        bar_color = "#cf222e" if conf > 0.85 else "#d29922" if conf > 0.6 else "#2da44e"
        st.progress(float(conf))

        # features
        features = None
        if show_features and boxes:
            x1, y1, x2, y2, _ = boxes[0]
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                mean_i, std_i, edge_d, area_px, area_pct = extract_features(crop, image.shape, boxes[0])
                features = (mean_i, std_i, edge_d, area_px, area_pct)

                st.divider()
                st.subheader("Extracted features")
                fig_feat = make_feature_chart(mean_i, std_i, edge_d)
                st.pyplot(fig_feat, use_container_width=True)
                plt.close(fig_feat)

                with st.expander("Raw feature values"):
                    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
                    fc1.metric("Mean Intensity", f"{mean_i:.2f}")
                    fc2.metric("Std Intensity", f"{std_i:.2f}")
                    fc3.metric("Edge Density", f"{edge_d:.4f}")
                    fc4.metric("Area (px²)", f"{area_px:,}")
                    fc5.metric("Area (%)", f"{area_pct:.1f}%")

        # session history tracking
        st.session_state.history.append({
            "filename": uploaded_file.name,
            "pred": prediction,
            "conf": conf,
            "risk": risk,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

        # history chart
        if len(st.session_state.history) > 1:
            st.divider()
            st.subheader("Confidence history (session)")
            fig_hist = make_confidence_chart(st.session_state.history)
            st.pyplot(fig_hist, use_container_width=True)
            plt.close(fig_hist)

        # PDF download
        st.divider()
        with st.spinner("Generating report..."):
            pdf_path = generate_pdf(
                prediction, conf, risk, temp_img_path,
                features, uploaded_file.name
            )
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Full PDF Report",
                data=f,
                file_name=f"fracture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # cleanup
        try:
            os.unlink(temp_img_path)
            os.unlink(pdf_path)
        except Exception:
            pass

        st.divider()
        st.warning("⚠️ AI-assisted result only. Not a substitute for professional medical diagnosis.")

else:
    st.info("👆 Upload a wrist X-ray image above to begin analysis.")