# MaskGuard AI - Sistem Deteksi Masker Wajah

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Gambaran Umum

**MaskGuard AI** adalah sistem visi komputer cerdas yang secara otomatis mendeteksi masker wajah dan mengklasifikasikannya ke dalam tiga kategori dengan akurasi tinggi. Dibangun dengan Transfer Learning menggunakan MobileNetV2, sistem ini membantu menegakkan protokol keselamatan di ruang publik.

## Fitur Utama

- **Klasifikasi Tiga Kategori**: Mendeteksi tiga kondisi masker:
  - **With Mask** - Menggunakan masker dengan benar
  - **Mask Weared Incorrect** - Penggunaan masker tidak tepat
  - **Without Mask** - Tidak terdeteksi masker

- **Akurasi Tinggi**: Mencapai akurasi lebih dari 95% pada data validasi
- **Siap Real-time**: Dioptimalkan untuk deployment real-time
- **Ramah Perangkat Mobile**: Menggunakan arsitektur MobileNetV2 yang ringan
- **Augmentasi Data**: Peningkatan training dengan transformasi gambar

## Arsitektur Model

```
Input (224x224x3)
↓
MobileNetV2 (Base)
↓
Global Average Pooling
↓
Dense (128, ReLU)
↓
Output (3, Softmax) → [Incorrect, With_Mask, Without_Mask]
```

## Instalasi

### Prasyarat Sistem
```bash
Python 3.8+
TensorFlow 2.0+
OpenCV
NumPy
Matplotlib
Scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/yourusername/maskguard-ai.git
cd maskguard-ai
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Struktur Dataset

Dataset harus diorganisir dengan struktur berikut:

```
Dataset/
├── with_mask/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── without_mask/
│   ├── image1.jpg
│   └── ...
└── mask_weared_incorrect/
    ├── image1.jpg
    └── ...
```

## Penggunaan

### 1. Training Model
```python
from maskguard import MaskDetector

# Inisialisasi detector
detector = MaskDetector()

# Training model
history = detector.train(
    dataset_path="/path/to/dataset",
    epochs=10,
    validation_split=0.2
)
```

### 2. Melakukan Prediksi
```python
# Load model yang sudah ditraining
model = detector.load_model("face_mask_model.h5")

# Prediksi gambar tunggal
result = detector.predict_image("test_image.jpg")
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### 3. Deteksi Real-time
```python
# Memulai deteksi webcam
detector.start_webcam()
```

## Metrik Kinerja

### Laporan Klasifikasi
```
                      precision    recall  f1-score   support

mask_weared_incorrect       0.96      0.94      0.95       320
           with_mask       0.98      0.97      0.98       400
        without_mask       0.95      0.96      0.95       350

            accuracy                           0.96      1070
           macro avg       0.96      0.96      0.96      1070
        weighted avg       0.96      0.96      0.96      1070
```

### Grafik Training
![Training History](images/training_history.png)

### Matriks Konfusi
![Confusion Matrix](images/confusion_matrix.png)

## Demo

### Prediksi Gambar Tunggal
```bash
python demo.py --image test_image.jpg
```

### Deteksi Real-time Webcam
```bash
python webcam_demo.py
```

### Proses Batch
```bash
python batch_process.py --input_folder images/ --output_folder results/
```

## Struktur Proyek

```
maskguard-ai/
├── models/
│   ├── face_mask_model.h5
│   └── mobileNetV2_base/
├── src/
│   ├── __init__.py
│   ├── mask_detector.py
│   ├── train.py
│   └── utils.py
├── datasets/
│   ├── train/
│   └── validation/
├── demos/
│   ├── single_image.py
│   └── webcam_demo.py
├── requirements.txt
├── train_model.ipynb
└── README.md
```

## Kustomisasi

### Menambah Kelas Baru
```python
# Modifikasi output layer untuk kelas baru
model = MaskDetector(num_classes=4)  # Tambah kelas baru
```

### Mengganti Base Model
```python
# Gunakan model pre-trained yang berbeda
detector = MaskDetector(base_model='ResNet50')
```

## Aplikasi

- **Fasilitas Kesehatan**: Memantau kepatuhan masker di rumah sakit
- **Gedung Perkantoran**: Memastikan protokol keselamatan tempat kerja
- **Toko Ritel**: Kontrol akses masuk otomatis
- **Lembaga Pendidikan**: Monitoring keamanan kampus
- **Transportasi Publik**: Penegakan keselamatan angkutan massal

## Berkontribusi

Kami menerima kontribusi! Silakan lihat Panduan Berkontribusi untuk detail.

1. Fork proyek ini
2. Buat feature branch (`git checkout -b feature/FiturAnda`)
3. Commit perubahan Anda (`git commit -m 'Tambahkan FiturAnda'`)
4. Push ke branch (`git push origin feature/FiturAnda`)
5. Buat Pull Request

## Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT - lihat file LICENSE untuk detail.

## Pengakuan

- MobileNetV2 oleh Google Research
- Tim TensorFlow & Keras
- Penyedia dataset dan kontributor
- Komunitas OpenCV untuk tools visi komputer

## Troubleshooting

### Masalah Umum

1. **Memory Error saat Training**
   - Kurangi batch size
   - Gunakan data generator

2. **Webcam Tidak Terdeteksi**
   - Periksa koneksi kamera
   - Pastikan OpenCV terinstall dengan dukungan webcam

3. **Akurasi Rendah**
   - Tambah data training
   - Adjust hyperparameter
   - Coba augmentasi data yang berbeda

## Dukungan

Untuk pertanyaan dan bantuan teknis:

- Dokumentasi: [Link ke dokumentasi lengkap]
- Issues: [GitHub Issues page]
- Email: support@maskguard-ai.com

## Versi

- **v1.0.0** - Rilis stabil pertama
- **v1.1.0** - Optimasi performa real-time
- **v1.2.0** - Tambahan model architecture

## Pembaruan Terakhir

Terakhir diperbarui: November 2025
