import os
import cv2
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from pdf2image import convert_from_path
from tkinter import filedialog, Tk

# Seleciona arquivo local via janela do sistema
def select_file(prompt="Selecione um arquivo"):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=prompt)
    root.destroy()
    return file_path

# Converte PDF em imagens PNG (uma por página)
def convert_pdf_to_png(pdf_path, output_name_prefix):
    images = convert_from_path(pdf_path)
    output_images = []
    for i, img in enumerate(images):
        output_path = f"{output_name_prefix}_page_{i+1}.png"
        img.save(output_path, "PNG")
        output_images.append(output_path)
    return output_images

# Trata upload de imagem ou PDF
def handle_input(name):
    path = select_file(f"Selecione a imagem ou PDF para '{name}'")
    if path.lower().endswith('.pdf'):
        print(f"{name} é um PDF. Convertendo...")
        png_paths = convert_pdf_to_png(path, name)
        print(f"Usando: {png_paths[0]}")
        return png_paths[0]
    return path

# Calcula diferença percentual de cor
def calculate_color_difference_percentage(img1, img2):
    if img1.shape != img2.shape:
        raise ValueError("As imagens devem ter o mesmo tamanho.")
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_percent = (diff / 255.0) * 100
    avg_diff_per_channel = np.mean(diff_percent, axis=(0, 1))
    avg_overall = np.mean(avg_diff_per_channel)
    return avg_diff_per_channel, avg_overall

# Verifica e redimensiona imagens, se necessário
def check_and_resize_images(image1, image2):
    if image1.shape != image2.shape:
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        if abs(h1 - h2) <= 100 and abs(w1 - w2) <= 100:
            print("Redimensionando a segunda imagem...")
            image2 = cv2.resize(image2, (w1, h1))
        else:
            raise ValueError("Imagens com diferença de tamanho muito grande.")
    return image1, image2

# Função principal
def spot_the_difference(img1_path, img2_path, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    image1, image2 = check_and_resize_images(image1, image2)

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result1, result2 = image1.copy(), image2.copy()
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

    region_dir = os.path.join(output_dir, f"diff_regions_{timestamp}")
    os.makedirs(region_dir, exist_ok=True)

    for i, group in enumerate(clusters.values()):
        xs = [r[4] for r in group]
        ys = [r[5] for r in group]
        ws = [r[2] for r in group]
        hs = [r[3] for r in group]
        x1, y1 = min(xs), min(ys)
        x2 = max([x + w for x, w in zip(xs, ws)])
        y2 = max([y + h for y, h in zip(ys, hs)])
        crop1 = image1[y1:y2, x1:x2]
        crop2 = image2[y1:y2, x1:x2]
        combined = np.concatenate((crop1, crop2), axis=1)
        cv2.imwrite(os.path.join(region_dir, f"region_{i+1}.png"), combined)

    combined_view = np.concatenate((result1, result2), axis=1)
    out_combined = os.path.join(output_dir, f"comparison_{timestamp}.png")
    cv2.imwrite(out_combined, combined_view)

    diff_channel, diff_total = calculate_color_difference_percentage(result1, result2)
    text = f"Diferença média: {diff_total:.2f}% - BGR: {diff_channel[0]:.2f}%, {diff_channel[1]:.2f}%, {diff_channel[2]:.2f}%"

    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(combined_view, cv2.COLOR_BGR2RGB))
    plt.title(text)
    plt.axis('off')
    plt.show()

    print("Diferenças salvas em:", output_dir)

# Execução principal
if __name__ == "__main__":
    print("🖼️ Comparador de Imagens - Spot the Difference")
    img1 = handle_input("Imagem Original")
    img2 = handle_input("Imagem Prova")
    spot_the_difference(img1, img2)
