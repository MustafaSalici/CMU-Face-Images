pip install scikit-learn matplotlib seaborn -q
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')
data_dir = "/content/drive/MyDrive/Doktora/Biyometri/Final/cmu+face+images"

import tarfile

tar_path = "/content/drive/MyDrive/Doktora/Biyometri/Final/cmu+face+images/faces.tar.gz"
extract_path = "/content/faces_extracted"

with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

tar_path_2 = "/content/drive/MyDrive/Doktora/Biyometri/Final/cmu+face+images/faces_4.tar.gz"
extract_path_2 = "/content/faces4_extracted"

with tarfile.open(tar_path_2, "r:gz") as tar:
    tar.extractall(path=extract_path_2)


import os

for root, dirs, files in os.walk("/content/faces_extracted/faces"):
    print(f"[faces] {root} - Alt klasörler: {dirs} - Dosya sayısı: {len(files)}")
    break

for root, dirs, files in os.walk("/content/faces4_extracted/faces_4"):
    print(f"[faces_4] {root} - Alt klasörler: {dirs} - Dosya sayısı: {len(files)}")
    break

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
data_dir = "/content/faces_extracted/faces"

X, y = [], []
labels = sorted(os.listdir(data_dir))
label_map = {name: idx for idx, name in enumerate(labels)}

for label in labels:
    folder = os.path.join(data_dir, label)
    if not os.path.isdir(folder): continue
    for img_file in os.listdir(folder):
        img_path = os.path.join(folder, img_file)
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label_map[label])
        except Exception as e:
            print(f"Hata: {img_path} → {e}")

X = np.array(X)
y = to_categorical(np.array(y))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

y_pred = model.predict(X_test)
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Gerçekte test verisinde yer alan sınıfları al
unique_labels = np.unique(y_test_labels)

# Doğrudan bu label'lara karşılık gelen isimleri seç
used_labels = [labels[i] for i in unique_labels]

print(classification_report(y_test_labels, y_pred_labels, target_names=used_labels))

cm = confusion_matrix(y_test_labels, y_pred_labels)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=used_labels, yticklabels=used_labels)
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()
