import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# === Parametreler ===
IMG_SIZE = 64
DATASET_PATH = "/content/faces_extracted/faces"
N_COMPONENTS = 150  # PCA bileşeni sayısı
CLASSES_LIMIT = None  # Tüm sınıfları kullan

# === Veri Yükleme ===
X, y = [], []
labels = sorted(os.listdir(DATASET_PATH))
if CLASSES_LIMIT:
    labels = labels[:CLASSES_LIMIT]

label_map = {name: idx for idx, name in enumerate(labels)}
X, y = [], []

for label in labels:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder): continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img).reshape(-1) / 255.0  # Flatten
            X.append(img_array)
            y.append(label_map[label])
        except Exception as e:
            print(f"Hata: {img_path} → {e}")

X = np.array(X)
y = np.array(y)

# === Eğitim-Test Ayrımı ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === PCA ile Boyut Azaltma ===
pca = PCA(n_components=N_COMPONENTS, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# === SVM Eğitimi ===
svm = SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
svm.fit(X_train_pca, y_train)

# === Tahmin ve Başarı Ölçümü ===
y_pred = svm.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk: {:.2f}%".format(100 * accuracy))

# === Etiket Dönüşümü ===
used_label_indices = np.unique(y_test)
used_labels = [label for i, label in enumerate(labels) if i in used_label_indices]

# === Sınıflandırma Raporu ===
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=used_labels))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred, labels=used_label_indices)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=used_labels, yticklabels=used_labels, cmap="Blues")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("PCA + SVM Confusion Matrix")
plt.show()

# === ROC Eğrisi ===
n_classes = len(np.unique(y))
y_test_bin = label_binarize(y_test, classes=range(n_classes))
y_score = svm.predict_proba(X_test_pca)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    try:
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    except ValueError:
        continue  # O sınıf test setinde yoksa

mean_auc = np.mean(list(roc_auc.values()))
print("Ortalama AUC: {:.4f}".format(mean_auc))

# === ROC Grafiği ===
plt.figure(figsize=(10, 8))
for i in roc_auc:
    plt.plot(fpr[i], tpr[i], lw=1, label=f"{labels[i]} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("PCA + SVM ROC Eğrileri")
plt.legend(loc='lower right')
plt.grid()
plt.show()
