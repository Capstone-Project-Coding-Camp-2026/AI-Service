# FinTime Dedicated AI Engine API

API Inference Server berbasis **FastAPI** yang bertugas sebagai *monolithic engine* untuk melayani tiga model Machine Learning utama secara *real-time* untuk ekosistem aplikasi FinTime: **NLP Classification**, **Forecasting**, dan **What-If Lab Simulation**. 

---

## 📖 Tentang Proyek (Overview)

**FinTime** adalah sebuah ekosistem manajemen keuangan pintar (*smart personal finance*) yang tidak hanya mencatat pengeluaran, tetapi juga memberikan wawasan prediktif kepada penggunanya. 

Repositori ini secara khusus memuat **Dedicated AI Engine**, yaitu layanan *backend* mandiri (*inference server*) yang terpisah dari *backend* utama aplikasi. *Engine* ini berfungsi sebagai "otak" analitik yang menangani seluruh komputasi berbasis Machine Learning (TensorFlow/Keras). Terdapat tiga pilar AI utama di dalam *engine* ini:
1. **NLP Classification**: Membaca catatan transaksi berupa teks bebas (*free-text*) dan secara otomatis mengklasifikasikannya ke dalam kategori pengeluaran yang tepat.
2. **Forecasting Engine**: Menganalisis pola transaksi historis (menggunakan data deret waktu dan *rolling features*) untuk memprediksi total pengeluaran di masa depan.
3. **What-If Lab Simulation**: Memberikan rekomendasi cerdas (contoh: *just_buy*, *buy_careful*, *dont_buy*) ketika pengguna ingin melakukan pembelian barang atau mengambil cicilan (*PayLater*), dengan mensimulasikan dampaknya terhadap arus kas dan beban utang (*monthly burden*).

Server ini dirancang untuk skenario *on-premise* dengan memprioritaskan efisiensi latensi. Model dieksekusi secara lokal dan dimuat langsung ke dalam RAM server melalui manajemen *lifespan* FastAPI pada saat proses *booting*.

---

## 🚀 Fitur & Arsitektur Utama

* **In-Memory Model Registry**: Memuat seluruh model ML (`.keras`) dan *metadata* secara terpusat pada saat *startup* server untuk memangkas latensi *inference*.
* **Keras 3 Compatibility Patch**: Menambal (*patching*) secara *hardcode* isu inisialisasi parameter `quantization_config` pada layer `Dense` bawaan Keras 3.
* **Klasifikasi NLP Kustom**: Menggunakan implementasi *regex* untuk tokenisasi teks secara mandiri dan memetakannya langsung ke indeks *vocabulary* yang telah disimpan di metadata tanpa library NLP eksternal.
* **Standarisasi Fitur Otomatis (Forecasting)**: Mengaplikasikan Z-score normalization secara *on-the-fly* menggunakan parameter `mean` dan `scale` yang diekstrak dari *metadata* sebelum melakukan prediksi regresi.
* **Kalkulasi Finansial Dinamis (What-If)**: Menghitung metrik turunan secara *real-time* seperti beban cicilan bulanan, *cashflow*, dan *Expense-to-Income Ratio* (ETR) sebagai ekstraksi fitur tambahan untuk simulasi model.

---

## 💻 Persyaratan Sistem

Untuk menjalankan *engine* ini secara optimal, direkomendasikan menggunakan spesifikasi berikut:
* **Sistem Operasi**: Linux (Ubuntu) atau Windows dengan WSL2 (Windows Subsystem for Linux).
* **Prosesor**: Intel Core i5 12th Gen (atau setara) untuk *handling thread* API yang stabil.
* **Memori**: Minimal 8GB RAM (16GB direkomendasikan karena 3 model dimuat secara bersamaan ke dalam memori).
* **Software**: Python 3.9+, pip, dan Git.

---

## ⚙️ Panduan Instalasi & Cara Menjalankan (Detail)

### 1. Persiapan Repositori & Environment
Sangat disarankan menggunakan *virtual environment* agar dependensi tidak bentrok dengan *project* Python lainnya di komputermu.

```bash
# 1. Kloning repositori
git clone <url-repositori-kamu>
cd <nama-folder-repositori>

# 2. Buat Virtual Environment
python -m venv venv

# 3. Aktivasi Virtual Environment
# Untuk Linux / Windows (WSL2) / macOS:
source venv/bin/activate
# Untuk Windows (Command Prompt / PowerShell native):
# venv\Scripts\activate