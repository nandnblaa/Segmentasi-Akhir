import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = r"C:\Users\User\OneDrive\Documents\Phyton\senja.jpeg"

if not os.path.exists(path):
    print("File tidak ditemukan")
    exit()

img_bgr = cv2.imread(path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title("Citra Asli")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale")
plt.axis("off")
plt.show()

def add_gaussian_noise(img, std=15):
    noise = np.random.normal(0, std, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, prob=0.05):
    noisy = img.copy()
    rnd = np.random.rand(*img.shape)
    noisy[rnd < prob/2] = 0
    noisy[rnd > 1 - prob/2] = 255
    return noisy

sp = add_salt_pepper_noise(gray, 0.05)
gauss = add_gaussian_noise(gray, 15)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(sp, cmap='gray')
plt.title("Grayscale + Salt & Pepper (0.05)")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gauss, cmap='gray')
plt.title("Grayscale + Gaussian (std = 15)")
plt.axis("off")

plt.show()

mean_sp = cv2.blur(sp, (3,3))
median_sp = cv2.medianBlur(sp, 3)

def edge_roberts(img):
    kx = np.array([[1,0],[0,-1]], np.float64)
    ky = np.array([[0,1],[-1,0]], np.float64)
    return np.sqrt(
        cv2.filter2D(img, cv2.CV_64F, kx)**2 +
        cv2.filter2D(img, cv2.CV_64F, ky)**2
    )

def edge_prewitt(img):
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float64)
    ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float64)
    return np.sqrt(
        cv2.filter2D(img, cv2.CV_64F, kx)**2 +
        cv2.filter2D(img, cv2.CV_64F, ky)**2
    )

def edge_sobel(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, 3)
    return np.sqrt(gx**2 + gy**2)

def edge_freichen(img):
    s2 = np.sqrt(2)
    kx = np.array([[-1,0,1],[-s2,0,s2],[-1,0,1]], np.float64)
    ky = np.array([[-1,-s2,-1],[0,0,0],[1,s2,1]], np.float64)
    return np.sqrt(
        cv2.filter2D(img, cv2.CV_64F, kx)**2 +
        cv2.filter2D(img, cv2.CV_64F, ky)**2
    )

def normalize(img):
    return ((img / img.max()) * 255).astype(np.uint8)

methods = {
    "Roberts": edge_roberts,
    "Prewitt": edge_prewitt,
    "Sobel": edge_sobel,
    "Frei-Chen": edge_freichen
}

for name, func in methods.items():
    fig, axes = plt.subplots(3, 2, figsize=(12, 16))

    fig.suptitle(
        f"Hasil Segmentasi - Metode {name}",
        fontsize=20,
        y=0.99
    )

    images = [
        normalize(func(gray)),
        normalize(func(sp)),
        normalize(func(gauss)),
        normalize(func(median_sp)),
        normalize(func(mean_sp))
    ]

    labels = [
        "Grayscale",
        "Salt & Pepper",
        "Gaussian",
        "SP + Median Filter",
        "SP + Mean Filter"
    ]

    idx = 0
    for i in range(3):
        for j in range(2):
            ax = axes[i, j]
            if idx < len(images):
                ax.imshow(images[idx], cmap='gray')
                ax.axis("off")

                ax.text(
                    0.5, 0.95,
                    labels[idx],
                    transform=ax.transAxes,
                    fontsize=14,
                    color="white",
                    ha="center",
                    va="top",
                    bbox=dict(facecolor="black", alpha=0.6, pad=4)
                )

                idx += 1
            else:
                ax.axis("off")

    plt.subplots_adjust(
        top=0.93,
        bottom=0.03,
        hspace=0.25,
        wspace=0.05
    )

    plt.show()

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float))**2)

mse_results = []

for name, func in methods.items():
    edge_gray = normalize(func(gray))
    edge_sp = normalize(func(sp))
    edge_gauss = normalize(func(gauss))
    edge_median = normalize(func(median_sp))
    edge_mean = normalize(func(mean_sp))

    mse_results.append([
        name,
        round(mse(edge_gray, edge_sp), 2),
        round(mse(edge_gray, edge_gauss), 2),
        round(mse(edge_gray, edge_median), 2),
        round(mse(edge_gray, edge_mean), 2)
    ])

fig, ax = plt.subplots(figsize=(11,4))
ax.axis('off')

col_labels = [
    "Metode",
    "Gray vs SP",
    "Gray vs Gaussian",
    "Gray vs SP + Median",
    "Gray vs SP + Mean"
]

table = ax.table(
    cellText=mse_results,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

plt.title("Tabel Perbandingan Nilai MSE per Metode Deteksi Tepi", pad=15)
plt.show()

labels = [row[0] for row in mse_results]

sp_vals = [row[1] for row in mse_results]
gauss_vals = [row[2] for row in mse_results]
median_vals = [row[3] for row in mse_results]
mean_vals = [row[4] for row in mse_results]

x = np.arange(len(labels))
w = 0.2

plt.figure(figsize=(10,5))
plt.bar(x - 1.5*w, sp_vals, w, label="Salt & Pepper")
plt.bar(x - 0.5*w, gauss_vals, w, label="Gaussian")
plt.bar(x + 0.5*w, median_vals, w, label="SP + Median")
plt.bar(x + 1.5*w, mean_vals, w, label="SP + Mean")

plt.xticks(x, labels)
plt.ylabel("Nilai MSE")
plt.title("Grafik Perbandingan Nilai MSE per Metode")
plt.legend()
plt.grid(axis='y')
plt.show()

print("Program selesai.")