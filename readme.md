# Laporan Proyek Machine Learning - Dany Eka Saputra

## Domain Proyek

### Latar Belakang

Proyek ini bertujuan untuk memprediksi tingkat obesitas berdasarkan berbagai fitur dari individu, seperti umur, tinggi badan, berat badan, dan kebiasaan gaya hidup lainnya. Dataset akan digunakan untuk melatih beberapa model machine learning, termasuk Decision Tree, XGBoost, Random Forest, SVM, Logistic Regression, dan LightGBM. Hasil dari model-model tersebut akan dibandingkan dan dievaluasi untuk memilih model yang paling tepat dalam memprediksi tingkat obesitas menggunakan metrik-metrik seperti akurasi, precision, recall, dan f1-score.

### Pentingnya Proyek
Pengembangan sistem prediksi ini penting karena beberapa alasan:

1. Mengembangkan model prediktif akurat untuk prediksi tipe obesitas.
3. Memberikan rekomendasi model prediktif yang paling baik.

### Hasil Riset dan Referensi
Beberapa penelitian telah menunjukkan efektivitas algoritma klasifikasi dalam kemampuan memprediksi tipe obesitas. Algoritma Random Forest dianggap sebagai metode yang efektif dalam memprediksi risiko obesitas[1]. Sementara itu, Artikel lain menjelaskan bagaimana teknik pembelajaran mesin (Machine Learning) dapat digunakan untuk menganalisis faktor-faktor yang mempengaruhi obesitas[2].

## Business Understanding
Obesitas adalah masalah kesehatan masyarakat yang signifikan. Deteksi dini terhadap individu dengan risiko obesitas dapat membantu dalam perencanaan intervensi kesehatan masyarakat dan pengambilan keputusan medis. Oleh karena itu, model prediksi obesitas dapat memberikan dampak positif dalam mengurangi risiko penyakit terkait obesitas dan beban sistem kesehatan.

### Problem Statements
- Bagaimana cara mengklasifikasikan tingkat obesitas individu secara akurat berdasarkan faktor-faktor risiko seperti gaya hidup, riwayat keluarga, dan kebiasaan makan, mengingat  dampak kesehatan masyarakat dari meningkatnya prevalensi obesitas?
- Algoritma machine learning manakah yang memberikan performa terbaik dalam mengklasifikasikan tingkat obesitas berdasarkan data yang tersedia, untuk mendukung diagnosis dini dan strategi kesehatan personal?

### Goals
- Mengembangkan model machine learning yang mampu mengklasifikasikan tingkat obesitas individu (`NObeyesdad`) berdasarkan fitur-fitur yang relevan dengan tingkat akurasi **minimal 85%**.
- Mengidentifikasi model klasifikasi **terbaik** dari beberapa algoritma yang diuji (Decision Tree, XGBoost, Random Forest, SVM, Logistic Regression, LightGBM) berdasarkan metrik evaluasi performa (akurasi, presisi, recall, F1-score) pada data uji.

### Solution statements
- Melakukan analisis data eksploratif (EDA) untuk memahami distribusi data, korelasi antar fitur, dan karakteristik dataset obesitas.
- Menerapkan teknik feature engineering (seperti menghitung BMI) untuk meningkatkan kualitas fitur prediktif.
- Melakukan pra-pemrosesan data termasuk standarisasi fitur numerik dan encoding fitur kategorikal untuk mempersiapkan data bagi model machine learning.
- Mengembangkan, melatih, dan mengevaluasi 6 model klasifikasi (Decision Tree, XGBoost, Random Forest, SVM, Logistic Regression, LightGBM) menggunakan data yang telah diproses.
- Membandingkan performa model menggunakan metrik evaluasi seperti akurasi, presisi, recall, dan F1-score (dengan weighted average untuk menangani kelas tidak seimbang) dan memilih model dengan performa terbaik pada data uji yang memenuhi target akurasi minimal 85%.

## Data Understanding
*Dataset yang digunakan pada proyek ini:*  
https://www.kaggle.com/competitions/playground-series-s4e2/data

### Dataset terdiri dari 20.758 baris dan 18 kolom yang mencakup:
- Kolom ID sebagai identifier
- 16 fitur input yang terbagi menjadi:
- Variabel numerik: Age (Usia), Height (Tinggi), Weight (Berat)
- Variabel kategorikal: family_history_with_overweight (Riwayat Keluarga dengan Obesitas), FAVC (Konsumsi Makanan Berkalori Tinggi), dan lainnya
- 1 kolom target 'NObeyesdad' yang terdiri dari 7 kelas klasifikasi tingkat obesitas yang tidak seimbang.

### Variabel-variabel pada Dataset:
- ID: Pengenal unik
- Gender: Jenis kelamin
- Age: Usia (tahun)
- Height: Tinggi badan (meter)
- Weight: Berat badan (kilogram)
- family_history_with_overweight: Riwayat keluarga dengan obesitas
- FAVC: Konsumsi rutin makanan berkalori tinggi
- FCVC: Frekuensi konsumsi sayuran
- MTRANS: Transportasi yang digunakan
- NCP: Jumlah makanan utama
- CAEC: Konsumsi makanan di antara waktu makan
- SMOKE: Perokok atau tidak
- CH2O: Konsumsi air harian
- SCC: Pemantauan konsumsi kalori
- FAF: Frekuensi aktivitas fisik
- TUE: Waktu penggunaan perangkat teknologi
- CALC: Konsumsi alkohol
- NObeyesdad: Tingkat obesitas yang disimpulkan

### Analisis Kondisi Dataset

1. Missing Value:

   Dataset tidak memiliki nilai yang hilang (missing value) pada seluruh kolom
   Semua data terisi lengkap dari total 20.758 baris

2. Data Duplikat:

   Tidak ditemukan data duplikat

3. Outlier:

   Terdapat outlier pada beberapa variabel numerik:
   Age: Beberapa nilai ekstrem di atas 61 tahun
   Height: Terdapat nilai tidak umum di bawah 1.45 meter dan di atas 1.95 meter
   NCP: Terdapat nilai tidak wajar di atas 4 kali makan per hari
   Outlier tidak dihapus karena masih dalam rentang yang masuk akal dan relevan untuk analisis obesitas

4. Distribusi Data:

   Data kategorikal cukup seimbang, terutama untuk Gender (49.8% laki-laki, 50.2% perempuan)
   Beberapa variabel kategorikal memiliki distribusi tidak merata, seperti SMOKE yang didominasi non-perokok

5. Tipe Data:

   Semua kolom memiliki tipe data yang sesuai
   Variabel numerik bertipe float64/int64
   Variabel kategorikal bertipe object

### Explorasi Data

1. Corellation Matrix
![Correlation Matrix](gambar%20eda/corr.png)

Correlation matrix diatas merepresentasikan hubungan antar fitur, berikut adalah analisisnya:

  - TUE dan Age memiliki korelasi negatif sangat kuat
  - FAF dan Age memiliki korelasi negatif kuat
  - FCVC dan TUE memiliki korelasi negatif kuat
  - Hubungan Weight terhadap TUE dan FAF memeiliki korelasi negatif lemah

2. Histogram
![Histogram](gambar%20eda/distribusi.png)

Histogram-histogram di atas merepresentasikan distribusi data untuk sembilan fitur, berikut analisisnya:
  - Age: Distribusi skewed ke kiri, sebagian besar data berada pada rentang usia 20-30 tahun.
  - Height dan Weight: Terdistribusi normal, dengan puncak sekitar tinggi 1.7 meter dan berat 70-90 kg.
  - FCVC (Frekuensi konsumsi sayur), NCP (Jumlah porsi), dan CH2O (Asupan air): Distribusi sangat terpusat pada nilai diskrit tertentu, menunjukkan perilaku konsumsi konsisten di beberapa level.
  - FAF (Frekuensi aktivitas fisik) dan TUE (Waktu penggunaan teknologi): Skewed, banyak data mendekati nilai 0 untuk aktivitas fisik, sedangkan TUE lebih terdistribusi rata.

3. Distribusi Jenis Kelamin
![Gender Distribution](gambar%20eda/gender%20distribution.png)

  - Data menunjukkan distribusi hampir seimbang antara Female (50.2%) dan Male (49.8%).

4. Distribusi berdasarkan Kategori Obesitas
![Distribusi per Tipe Obesitas](gambar%20eda/distribusi%20per%20tipe%20obesitas.png)

  - Ditemukan bahwa distribusi berdasarkan kategori kelas obesitas tidak merata


## Data Preparation
1. Penanganan Missing Values.

   Dilakukan pemeriksaan nilai yang hilang menggunakan metode .isna().any() pada dataset. Hasil pemeriksaan menunjukkan bahwa tidak terdapat missing value pada seluruh kolom di dataset pelatihan.

2. Penanganan Data Duplikat 

   Dilakukan pemeriksaan data duplikat menggunakan metode .duplicated().sum(). Hasilnya menunjukkan bahwa tidak ditemukan data duplikat dalam dataset pelatihan.

3. Penanganan Outliers:

   Deteksi Outlier: Outlier pada fitur-fitur numerik (Age, Height, Weight, NCP, CH2O, FAF, TUE) dideteksi menggunakan metode Interquartile Range (IQR), dengan visualisasi melalui box plot. Batas bawah (Q1 - 1.5 * IQR) dan batas atas (Q3 + 1.5 * IQR) dihitung untuk setiap fitur numerik.

   Keputusan Penanganan: Meskipun terdeteksi adanya outlier pada beberapa fitur seperti Age, Height, dan NCP, nilai-nilai tersebut tidak dihapus karena dianggap masih berada dalam rentang yang masuk akal dan relevan untuk konteks analisis obesitas.

2. Rekayasa Fitur (Feature Engineering)

   Rekayasa fitur atau feature engineering adalah proses membuat, mengubah, atau memilih fitur yang berkontribusi dalam membangun model machine learning yang berperforma lebih baik[3].

   Terdapat fitur yang dapat kita buat dari fitur yang ada yaitu BMI (Body Mass Index). BMI merupakan indikator yang akurat untuk mengukur proporsi tubuh dan menilai risiko penyakit terkait obesitas, seperti diabetes, hipertensi dan penyakit jantung. Dengan mengkalkulasi BMI, dapat ditentukan tingkat keparahan obesitas dan strategi pencegahan serta pengobatan yang tepat. Selain itu, BMI juga membantu mengidentifikasi individu berisiko tinggi, memantau perubahan berat badan dan menentukan kebutuhan nutrisi. Oleh karena itu, pengkalkulasian BMI menjadi salah satu faktor penting dalam prediksi obesitas. Rumus: BMI = Weight / Height

3. Pembagian Data (Splitting Data)

   Memisahkan dataset menjadi data latih (80%) dan data uji (20%) menggunakan modul train_test_split dari library scikit-learn.

3. Standarisasi dan Encoding

   proses standarisasi dan encoding dilakukan melalui beberapa tahap
   - Pemilihan Kolom : Memilih kolom numerik dan kategorikal dari data latih.
   - Pembuatan Preprocessor : Mengubah variabel kategorikal menjadi representasi numerik menggunakan One-Hot Encoding dan mengubah skala fitur numerik menjadi skala seragam menggunakan StandardScaler.
   - Pembuatan Pipeline : Preprocessor sebagai fungsi melakukan proses standarisasi dan encoding digabungkan sebagai Pipeline dari library scikit-learn.
   - Transformasi Data : Menerapkan preprocessing (standarisasi dan encoding)  pada data latih dan data uji untuk mempersiapkan data bagi model.


## Modeling

1. Decision Tree
   - Decision Tree memiliki keunggulan dalam interpretasi yang mudah dan mendukung data kategorikal serta numerik. Namun, kelemahannya adalah rentan terhadap overfitting jika depth tree terlalu tinggi, sensitif terhadap data yang tidak seimbang, dan kurang stabil[4].

   - Cara kerja Algoritma: 
      1. Membangun pohon keputusan dengan membagi data berdasarkan fitur yang paling informatif
      2. Menggunakan metrik Gini atau Entropy untuk menentukan split terbaik
      3. Melakukan proses rekursif untuk membuat cabang-cabang pohon
      4. Menentukan kelas pada setiap leaf node berdasarkan mayoritas sampel

   - Tahapan:
      1. Inisialisasi model dengan random_state=2024
      2. Melatih model menggunakan data training yang telah diproses
      3. Melakukan prediksi pada data training dan testing
      4. Mengevaluasi performa dengan metrik weighted average

   - Parameter:
      - random_state=2024 untuk reproduktifitas
      - Menggunakan parameter default untuk kriteria split dan kedalaman pohon
      - average="weighted" untuk menangani ketidakseimbangan kelas

2. XGBoost
   XGBoost menawarkan kecepatan, efisiensi, dan kemampuan mengurangi overfitting. Akan tetapi, model ini memiliki kelemahan seperti komputasi intensif, sulit diinterpretasikan, dan memerlukan tuning parameter yang tepat[5].

  - Cara kerja Algoritma:
    1. Membangun model secara bertahap (boosting)
    2. Setiap iterasi fokus pada sampel yang salah diprediksi sebelumnya
    3. Menggunakan gradient descent untuk optimasi
    4. Menggabungkan hasil prediksi dari semua model lemah

  - Tahapan:
    1. Inisialisasi model GradientBoostingClassifier
    2. Melatih model dengan data training yang telah diproses
    3. Melakukan prediksi bertahap
    4. Mengevaluasi hasil dengan metrik weighted average

  - Parameter:
    - Menggunakan parameter default untuk learning rate
    - Menggunakan parameter default untuk jumlah estimator
    - average="weighted" untuk penanganan kelas tidak seimbang

3. Random Forest
   Random Forest memiliki keunggulan dalam stabilitas dan ketahanan terhadap noise. Namun, kelemahannya adalah komputasi intensif, sulit diinterpretasikan, memerlukan banyak data untuk pelatihan, dan dapat terlalu kompleks[6].

  - Cara kerja Algoritma:
    1. Membuat multiple decision tree secara paralel
    2. Setiap tree dilatih dengan subset data random (bagging)
    3. Setiap tree menggunakan subset fitur random
    4. Menggabungkan hasil prediksi melalui voting mayoritas

  - Tahapan:
    1. Inisialisasi RandomForestClassifier dengan n_estimators=100
    2. Melatih model dengan data training
    3. Melakukan ensemble prediction
    4. Evaluasi menggunakan metrik weighted average

  - Parameter:
    - n_estimators=100 (jumlah pohon)
    - Menggunakan parameter default untuk kriteria split
    - average="weighted" untuk kelas tidak seimbang

4. Support Vector Machine (SVM)
   SVM efektif untuk data dengan dimensi tinggi. Kelemahannya adalah komputasi intensif, sulit diinterpretasikan, memerlukan pemilihan kernel yang tepat, dan dapat terlalu kompleks[7].

  - Cara kerja Algoritma:
    1. Mencari hyperplane optimal untuk memisahkan kelas
    2. Menggunakan kernel trick untuk data non-linear
    3. Memaksimalkan margin antara kelas
    4. Menggunakan support vectors untuk klasifikasi

  - Tahapan:
    1. Inisialisasi SVC dengan kernel='rbf'
    2. Melatih model dengan data training terstandarisasi
    3. Melakukan prediksi
    4. Evaluasi dengan metrik weighted average

  - Parameter:
    - kernel='rbf' untuk data non-linear
    - random_state=2024
    - average="weighted" untuk kelas tidak seimbang

5. Logistic Regression
   Logistic Regression memiliki keunggulan dalam interpretasi yang mudah dan sederhana. Namun, kelemahannya adalah memerlukan asumsi linearitas, tidak efektif untuk data kompleks, sensitif terhadap outliers, dan kurang stabil dengan data yang tidak seimbang[8].

  - Cara kerja Algoritma:
    1. Memodelkan probabilitas kelas menggunakan fungsi logistik
    2. Menggunakan multinomial untuk multi-kelas
    3. Optimasi menggunakan gradient descent
    4. Menghasilkan probabilitas untuk setiap kelas

  - Tahapan:
    1. Inisialisasi LogisticRegression dengan multi_class='multinomial'
    2. Melatih model dengan max_iter=1000
    3. Melakukan prediksi probabilistik
    4. Evaluasi menggunakan metrik weighted average

  - Parameter:
    - multi_class='multinomial'
    - max_iter=1000
    - average="weighted"

6. LightGBM
   LightGBM menawarkan waktu pelatihan cepat dan performa optimal. Kelemahannya adalah komputasi intensif, sulit diinterpretasikan, memerlukan tuning parameter yang tepat, dan dapat terlalu kompleks[9].

  - Cara kerja Algoritma:
    1. Menggunakan teknik gradient boosting
    2. Melakukan optimasi leaf-wise
    3. Menggunakan histogram-based learning
    4. Menerapkan parallel learning

  - Tahapan:
    1. Inisialisasi LGBMClassifier
    2. Melatih model dengan data training
    3. Melakukan prediksi bertahap
    4. Evaluasi menggunakan metrik weighted average

  - Parameter:
    - Parameter default untuk learning rate
    - Parameter default untuk jumlah leaves
    - average="weighted" untuk kelas tidak seimbang

## Evaluation

### Metrik Evaluasi

Untuk mengevaluasi performa model, digunakan beberapa metrik yaitu: 

1. Akurasi (Accuracy)

   Akurasi mengukur proporsi prediksi yang benar dari total sampel. Akurasi dihitung dengan rumus:
   Akurasi = (TP + TN) / (TP + TN + FP + FN)
   Di mana:

    - TP (True Positive): Prediksi benar positif
    - TN (True Negative): Prediksi benar negatif
    - FP (False Positive): Prediksi salah positif
    - FN (False Negative): Prediksi salah negatif

2. Precision (Presisi)

   Presisi mengukur proporsi prediksi benar positif dari total prediksi positif. Presisi dihitung dengan rumus:

   Presisi = TP / (TP + FP)

   Presisi penting untuk menilai kemampuan model memprediksi hasil positif yang akurat.

3. Recall (Sensitivitas)

   Recall mengukur proporsi prediksi benar positif dari total sampel positif. Recall dihitung dengan rumus:

   Recall = TP / (TP + FN)

   Recall penting untuk menilai kemampuan model mendeteksi hasil positif.

4. F1-Score

   F1-Score merupakan rata-rata harmonis antara presisi dan recall. F1-Score dihitung dengan rumus:

   F1-Score = 2 * (Presisi * Recall) / (Presisi + Recall)

   F1-Score penting untuk menilai keseimbangan antara presisi dan recall.

   Kategori F1-Score

    - Tinggi (0,9-1,0): Model sangat akurat
    - Sedang (0,7-0,89): Model cukup akurat
    - Rendah (0,5-0,69): Model kurang akurat
    - Sangat rendah (<0,5): Model tidak akurat

   Dalam evaluasi model ini dipertimbangkan:

    - Keseimbangan antara presisi dan recall
    - Kategori F1-Score
    - Nilai akurasi

### Hasil Evaluasi

### Hasil Evaluasi Model (pada Data Uji)

| Model               | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Catatan Overfitting |
|---------------------|----------|----------------------|-------------------|---------------------|---------------------|
| Decision Tree       | 83.67%   | 83.73%               | 83.67%            | 83.71%              | Signifikan          |
| XGBoost             | 89.86%   | 89.81%               | 89.86%            | 89.82%              | Ringan              |
| Random Forest       | 89.79%   | 89.81%               | 89.79%            | 89.77%              | Signifikan          |
| SVM                 | 88.03%   | 87.93%               | 88.03%            | 87.97%              | Rendah              |
| Logistic Regression | 86.51%   | 86.37%               | 86.51%            | 86.42%              | Sangat Rendah       |
| LightGBM            | 89.96%   | 89.98%               | 89.96%            | 89.94%              | Signifikan          |

1. Decision Tree
  - Training Accuracy, Precision, Recall, F1-score: 100%
  - Testing Accuracy, Precision, Recall, F1-score: 84%
  - Perbandingan training dan testing akurasi menunjukkan overfitting yang signifikan. 
  - Precision dan recall yang cukup seimbang pada tiap-tiap training dan testing
  - Dan f1-score pada kategori sedang yang berarti model cukup akurat

2. XGBoost
  - Training Accuracy, Precision, Recall, F1-score: 92%
  - Testing Accuracy, Precision, Recall, F1-score: 90%
  - Perbandingan training dan testing akurasi menunjukkan overfitting namun dengan performa testing yang sangat baik
  - Precision dan recall yang cukup seimbang pada tiap-tiap training dan testing
  - Dan f1-score pada kategori tinggi yang berarti model sangat akurat

3. Random Forest
  - Training Accuracy, Precision, Recall, F1-score: 100%
  - Testing Accuracy, Precision, Recall, F1-score: 90%
  - Perbandingan training dan testing akurasi menunjukkan overfitting namun dengan performa testing yang sangat baik
  - Precision dan recall yang cukup seimbang pada tiap-tiap training dan testing
  - Dan f1-score pada kategori tinggi yang berarti model sangat akurat

4. SVM
  - Training Accuracy, Precision, Recall, F1-score: 90%
  - Testing Accuracy, Precision, Recall, F1-score: 88%
  - Performa stabil antara training dan testing
  - Precision dan recall yang cukup seimbang pada tiap-tiap training dan testing
  - Dan f1-score pada kategori sedang yang berarti model cukup akurat

5. Logistic Regression
  - Training Accuracy, Precision, Recall, F1-score: 86.83%
  - Testing Accuracy, Precision, Recall, F1-score: 87%
  - Performa konsisten antara training dan testing 
  - Precision dan recall yang cukup seimbang pada tiap-tiap training dan testing
  - Dan f1-score pada kategori tinggi yang berarti model model cukup akurat

6. LightGBM
  - Training Accuracy, Precision, Recall, F1-score: 99%
  - Testing Accuracy, Precision, Recall, F1-score: 90%
  - Perbandingan training dan testing akurasi menunjukkan overfitting namun dengan performa testing yang lebih baik
  - Precision dan recall yang cukup seimbang pada tiap-tiap training dan testing
  - Dan f1-score pada kategori tinggi yang berarti model sangat akurat

**Kesimpulan Evaluasi Model**
  - Performa terbaik: XGBoost, Random Forest, dan LightGBM  memberikan performa dan classification report tertinggi, namun LightGBM cukup overfit, sedangkan XGBoost memiliki  performa yang stabil antara tarining dan testing, meskipun ketiganya sama sama memiliki hasil testing akurasi, precision, recall, f1-score pada angka 90%.
  - Overfitting: Decision Tree, Random Forest, dan LightGBM cenderung overfit pada data training.

  catatan: angka akurasi, precision, recall, f1-score telah dilakukan pembulatan ke atas dan ke bawah

### Kesimpulan

Dengan mengembangkan model predictive analysis, saya berhasil menjawab tantangan:

1. Mengatasi masalah dalam membangun model prediktif yang akurat
   Model prediktif mencapai tiap metrik evaluasi dengan angka yang baik menjawab problem statement untuk membangun model prediktif akurat bagi klasifikasi obesitas

2. Mencapai dan melampaui tingkat akurasi minimal 85% 
   Mengembangkan model prediktif dengan tingkat akurasi minimal 85% (menjawab goals)untuk mengklasifikasikan tingkat obesitas seseorang. Model XGBoost, Random Forest, dan LightGBM  mencapai akurasi 90%, melampaui target 85% dari Goals yang telah ditentukan.

3. Telah melakukan perbandingan 6 model machine learning  yaitu 
   Decision Tree, XGBoost, Random Forest, SVM, Logistic Regression, LightGBM menghasilkan solusi optimal dengan XGBoost Dan melakukan evaluasi model menggunakan metrik yang sesuai serta memilih model terbaik berdasarkan hasil evaluasi akurasi

### Implikasi terhadap Business Understanding

* **Apakah sudah menjawab setiap problem statement?**
  1.  Ya, model-model yang dikembangkan (terutama XGBoost, Random Forest, LightGBM) berhasil mengklasifikasikan tingkat obesitas dengan akurasi tinggi (sekitar 90%), menjawab kebutuhan untuk klasifikasi akurat berdasarkan faktor risiko.
  2.  Ya, perbandingan antar 6 algoritma menunjukkan bahwa XGBoost, Random Forest, dan LightGBM memberikan performa terbaik, dengan XGBoost menunjukkan keseimbangan terbaik antara akurasi dan potensi overfitting.

* **Apakah berhasil mencapai setiap goals yang diharapkan?**
  1.  Ya, goal akurasi minimal 85% tercapai dan bahkan terlampaui oleh beberapa model (XGBoost, RF, LightGBM, SVM, Logistic Regression).
  2.  Ya, model terbaik (XGBoost) berhasil diidentifikasi berdasarkan evaluasi metrik performa pada data uji.

* **Apakah setiap solusi statement yang direncanakan berdampak? Jelaskan!**
  1.  **EDA:** Berdampak, membantu memahami data, ketidakseimbangan kelas, dan korelasi, yang menginformasikan pemilihan model dan preprocessing.
  2.  **Feature Engineering (BMI):** Berdampak positif (meskipun tidak diukur secara eksplisit dalam perbandingan model di notebook ini, BMI secara domain knowledge sangat  relevan).
  3.  **Pra-pemrosesan:** Sangat berdampak, memastikan data siap digunakan oleh model dan algoritma seperti SVM dan Logistic Regression dapat bekerja optimal.
  4.  **Pengembangan & Evaluasi 6 Model:** Berdampak, memungkinkan perbandingan objektif dan pemilihan model terbaik.
  5.  **Pemilihan Model Terbaik:** Berdampak, menghasilkan rekomendasi model (XGBoost) yang paling sesuai dengan goals proyek, yaitu akurasi tinggi dan stabilitas (overfitting minimal).


**Rekomendasi:**

  - Berdasarkan hasil pekerjaan ini, saya merekomendasikan pengimplementasian model XGBoost untuk sistem prediksi obesitas, karena mencapai performa paling tinggi dan paling stabil dibanding algoritma lain.



## Referensi

[1] Delpino., et al., "Does machine learning have a high performance to predict obesity among adults and older adults? A systematic review and meta-analysis." Nutrition, Metabolism and Cardiovascular Diseases., 2024. link: https://www.sciencedirect.com/science/article/pii/S0939475324002047
[2] Ferdowsy, et al., "Machine Learning Approaches for Obesity Prediction." Current Research in Behavioral Sciences., 2021. link: https://www.sciencedirect.com/science/article/pii/S2666518221000401
[3] M. Oyamada, "Extracting Feature Engineering Knowledge from Data Science Notebooks," 2019 IEEE International Conference on Big Data (Big Data), Los Angeles, CA, USA, 2019, pp. 6172-6173, doi: 10.1109/BigData47090.2019.9006522.
[4] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," JMLR, 2011. link: https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf
[5] T. Chen, C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016. link: https://arxiv.org/abs/1603.02754
[6] L. Breiman, "Random Forests," Machine Learning, 2001. link: https://link.springer.com/article/10.1023/A:1010933404324
[7] B. Scholkopf et al., "Comparing Support Vector Machines," Neural Computation, 2000. link: https://www.stat.purdue.edu/~yuzhu/stat598m3/Papers/NewSVM.pdf
[8] D. W. Hosmer, S. Lemeshow, "Applied Logistic Regression," Wiley, 1989. link: https://onlinelibrary.wiley.com/doi/book/10.1002/0471722146 
[9] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS, 2017. link: https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf




