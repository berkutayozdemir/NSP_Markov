# Markov Process Text Prediction / Markov Süreci ile Metin Tahmini

This project implements a text prediction model using **Markov Chains**. It is built specifically for the **Stochastic Processes** subject.
The model learns from the text "Nutuk" (by Mustafa Kemal Atatürk) to generate new text sequences based on probability distributions.

**Bu proje Stokastik Süreçler dersi için hazırlanmıştır.** "Nutuk" eseri üzerinden Markov Zinciri modeli eğitilerek, olasılık dağılımlarına dayalı yeni metinler üretilmektedir.

---

## 🇬🇧 English

### Features
*   **Dataset Retrieval**: Automatically downloads the *Nutuk* dataset.
*   **Preprocessing**: Cleaning, lowercasing, and tokenization of Turkish text.
*   **Markov Model**: Builds a state transition matrix (dictionary based).
*   **Advanced Prediction**:
    *   **Laplace Smoothing (Add-alpha)**: Handles unseen word transitions.
    *   **Temperature Sampling**: Controls the "creativity" or randomness of the generation (Low temp = deterministic, High temp = creative).
*   **Visualization**: Transition probabilities visualized using bar charts.

### Dependencies
*   Python 3.x
*   numpy
*   requests
*   pandas
*   seaborn
*   matplotlib

### Usage
1.  Install dependencies: `pip install -r requirements.txt`
2.  Open the Jupyter Notebook: `jupyter notebook NSP_Markov.ipynb`
3.  Run the cells to download data, train the model, and generate text.

---

## 🇹🇷 Türkçe

### Özellikler
*   **Veri Seti**: *Nutuk* veri setini otomatik indirir.
*   **Ön İşleme**: Türkçe metin temizleme, küçük harfe çevirme ve kelime ayırma (tokenization).
*   **Markov Modeli**: Kelime geçiş matrisini oluşturur.
*   **Gelişmiş Tahmin**:
    *   **Laplace Düzeltme (Smoothing)**: Sıfır olasılık sorununu çözer.
    *   **Sıcaklık (Temperature)**: Metin üretimindeki rastgeleliği kontrol eder (Düşük sıcaklık = tutarlı, Yüksek sıcaklık = yaratıcı).
*   **Görselleştirme**: Geçiş olasılıklarını grafiklerle gösterir.

### Kullanım
1.  Kütüphaneleri yükleyin: `pip install -r requirements.txt`
2.  Notebook dosyasını açın: `jupyter notebook NSP_Markov.ipynb`
3.  Hücreleri çalıştırarak modeli eğitebilir ve metin üretebilirsiniz.

## Citation / Atıf
*   **Mehmet Aksoy**, *Nutuk 1. Cilt — Türkçe Doğal Dil İşleme Veri Seti (sürüm v2.0)*, GitHub deposu, 2 Kasım 2025.
    *   URL: https://github.com/mehmetaksoy/Nutuk-Turkce-NLP-Dataset

## License / Lisans
MIT
