import streamlit as st
import numpy as np
import cv2
from datetime import datetime
from sklearn.cluster import DBSCAN
from pdf2image import convert_from_bytes
from PIL import Image
import io
import tempfile
import os
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🖼️ Spot the Difference")

def convert_pdf_to_png(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    output = io.BytesIO()
    images[0].save(output, format="PNG")
    return output.getvalue()

def calculate_color_difference_percentage(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("As imagens devem ter o mesmo tamanho.")
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_percent = (diff / 255.0) * 100
    avg_diff_per_channel = np.mean(diff_percent, axis=(0, 1))
    avg_overall = np.mean(avg_diff_per_channel)
    return avg_diff_per_channel, avg_overall

def check_and_resize_images(image1, image2):
    if image1.shape != image2.shape:
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        if abs(h1 - h2) <= 100 and abs(w1 - w2) <= 100:
            image2 = cv2.resize(image2, (w1, h1))
        else:
            raise ValueError("Imagens com diferença de tamanho muito grande.")
    return image1, image2

def spot_the_difference(image1_bytes, image2_bytes):
    with tempfile.TemporaryDirectory() as tmpdir:
        img1_np = cv2.imdecode(np.frombuffer(image1_bytes, np.uint8), cv2.IMREAD_COLOR)
        img2_np = cv2.imdecode(np.frombuffer(image2_bytes, np.uint8), cv2.IMREAD_COLOR)
        img1_np, img2_np = check_and_resize_images(img1_np, img2_np)

        gray1 = cv2.cvtColor(img1_np, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_np, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result1, result2 = img1_np.copy(), img2_np.copy()
        rects = []

        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                rects.append([x + w//2, y + h//2, w, h, x, y])
                cv2.rectangle(result1, (x, y), (x + w, y + h), (0, 255, 255), 4)
                cv2.rectangle(result2, (x, y), (x + w, y + h), (255, 0, 255), 4)

        clusters = {}
        if rects:
            centers = np.array([[r[0], r[1]] for r in rects])
            labels = DBSCAN(eps=40, min_samples=1).fit(centers).labels_
            for idx, label in enumerate(labels):
                clusters.setdefault(label, []).append(rects[idx])

        combined_view = np.concatenate((result1, result2), axis=1)
        diff_channel, diff_total = calculate_color_difference_percentage(result1, result2)
        text = f"Diferença média: {diff_total:.2f}% - BGR: {diff_channel[0]:.2f}%, {diff_channel[1]:.2f}%, {diff_channel[2]:.2f}%"

        st.markdown(f"### Resultado da Comparação")
        st.image(combined_view, channels="BGR", caption=text, use_column_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(cv2.cvtColor(combined_view, cv2.COLOR_BGR2RGB))
        ax.set_title(text)
        ax.axis("off")
        st.pyplot(fig)

img1_file = st.file_uploader("Imagem 1 (PNG, JPG ou PDF)", type=["png", "jpg", "jpeg", "pdf"])
img2_file = st.file_uploader("Imagem 2 (PNG, JPG ou PDF)", type=["png", "jpg", "jpeg", "pdf"])

if img1_file and img2_file:
    def preprocess(file):
        if file.name.lower().endswith('.pdf'):
            return convert_pdf_to_png(file.read())
        return file.read()

    img1_bytes = preprocess(img1_file)
    img2_bytes = preprocess(img2_file)

    if st.button("🔍 Comparar"):
        spot_the_difference(img1_bytes, img2_bytes)
