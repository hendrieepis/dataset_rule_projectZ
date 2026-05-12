**PROTOKOL PENGAMBILAN DATA**

Dataset Hand Gesture untuk Deteksi Bullying

Menggunakan ZED Stereo Camera

| **Versi Dokumen**    | v1.0 - 2025                                |
|----------------------|--------------------------------------------|
| **Diperuntukkan**    | Mahasiswa S2 (Pengambil Data)              |
| **Disupervisi oleh** | Mahasiswa S3                               |
| **Target Dataset**   | Gerakan tangan jenis pukulan (multi-class) |

> Baca seluruh protokol ini sebelum memulai pengambilan data. Ketidakpatuhan terhadap prosedur akan menyebabkan data ditolak dan harus diulang.

#  1. TUJUAN & GAMBARAN UMUM

Dataset ini digunakan untuk melatih model deep learning yang mampu mengenali gerakan tangan yang berpotensi merupakan tindakan bullying fisik. Data diambil menggunakan ZED Stereo Camera agar tersedia informasi RGB, depth map, dan 3D skeleton secara bersamaan.

### Tujuan Akhir

- Model dapat mengklasifikasikan jenis gerakan pukulan dari sequence video

- Model dapat membedakan gerakan memukul dari gerakan tangan biasa (negative class)

- Dataset cukup beragam (subjek, background, variasi) untuk generalisasi yang baik

### Alur Singkat

1.  Rekam video gerakan selama 1 detik per clip dengan ZED Camera

2.  Setiap clip dibagi menjadi 3-4 sequence (fase gerakan)

3.  Ulangi di 15 background lokasi berbeda

4.  Total 7 subjek mahasiswa yang merekam

5.  Semua data dilabeli sesuai format yang ditentukan

#  2. KELAS GERAKAN (GESTURE CLASS)

Terdapat dua kelompok kelas: Positive Class (gerakan memukul) dan Negative Class (gerakan tangan biasa). Model WAJIB dilatih dengan keduanya agar tidak overfit.

### 2.1 Positive Class - Gerakan Memukul

| **ID Kelas** | **Nama Gesture** | **Deskripsi Gerakan** | **Arah Pukulan** |
|----|----|----|----|
| G01 | JAB | Pukulan lurus ke depan, cepat, satu tangan | Horizontal lurus ke depan |
| G02 | UPPER | Pukulan dari bawah ke atas (uppercut) | Diagonal ke atas ~45° |
| G03 | DOWN | Pukulan serong ke bawah | Diagonal ke bawah ~30° |
| G04 | SMASH | Pukulan kuat ke bawah (overhead) | Vertikal ke bawah |
| G05 | HOOK_R | Pukulan menyamping dari kanan (hook) | Horizontal dari sisi kanan |
| G06 | HOOK_L | Pukulan menyamping dari kiri (hook) | Horizontal dari sisi kiri |
| G07 | SWING | Pukulan memutar lebar (haymaker) | Rotasi horizontal lebar |
| G08 | DOUBLE | Dua pukulan berturutan (jab-cross) | Kiri-kanan bergantian |

### 2.2 Negative Class - Gerakan Bukan Memukul

WAJIB direkam! Ini adalah data pembanding agar model tidak salah klasifikasi gerakan tangan biasa sebagai pukulan.

| **ID Kelas** | **Nama Gesture** | **Deskripsi Gerakan** |
|----|----|----|
| N01 | WAVE | Melambai tangan ke kiri-kanan |
| N02 | POINT | Menunjuk ke suatu arah dengan jari telunjuk |
| N03 | REACH | Meraih/mengambil benda di depan, gerakan lambat |
| N04 | PUSH | Mendorong telapak ke depan (bukan pukulan, lambat) |
| N05 | REST | Tangan diam di sisi tubuh atau bergerak santai |

### 2.3 Definisi Sequence A - B - C

| **Sequence** | **Nama Fase** | **Deskripsi** | **Durasi Frame (30fps)** |
|:--:|----|----|----|
| **A** | Wind-up / Ancang-ancang | Tangan bergerak ke belakang / posisi siap memukul | 7-10 frame |
| **B** | Impact / Eksekusi | Tangan bergerak ke depan dengan cepat menuju target | 5-8 frame |
| **C** | Retract / Kembali | Tangan kembali ke posisi awal / netral | 7-10 frame |

> Aturan PENTING: Jika hanya ada 1 atau 2 sequence yang terlihat, clip TIDAK dianggap valid sebagai gesture memukul. Ketiga sequence A, B, dan C HARUS terekam dalam satu clip.

#  3. SETUP ZED CAMERA

### 3.1 Spesifikasi Rekaman

| **Parameter** | **Nilai yang Digunakan** | **Alasan** |
|----|----|----|
| Resolusi | 720p (HD720) | Keseimbangan kualitas & kecepatan body tracking |
| Frame Rate | 30 FPS | Cukup untuk capture gerakan cepat tanpa blur berlebih |
| Format Simpan | .SVO (ZED native) + export RGB .mp4 | SVO menyimpan semua data termasuk depth |
| Depth Mode | NEURAL atau ULTRA | Akurasi depth maksimal |
| Body Tracking | WAJIB aktif | Untuk export 3D skeleton keypoints |
| Confidence Threshold | Minimum 50 | Pastikan skeleton ter-track dengan baik |
| Jarak Subjek ke Kamera | 1.5 - 3.0 meter | Sweet spot body tracking ZED |

### 3.2 Posisi Kamera

| **Aspek** | **Ketentuan** |
|----|----|
| Ketinggian kamera | Setara dada subjek (sekitar 120-140 cm dari lantai) |
| Sudut kamera | Horizontal / lurus menghadap subjek, tidak miring ke atas/bawah |
| Orientasi kamera | Landscape (horizontal) - JANGAN portrait |
| Fiksasi kamera | Gunakan tripod, kamera TIDAK BOLEH bergerak saat rekam |
| Field of View | Seluruh tubuh subjek dari pinggang ke atas harus terlihat |
| Kondisi pencahayaan | Cukup terang, hindari backlight (cahaya dari belakang subjek) |

### 3.3 Cek Sebelum Rekam

Lakukan checklist ini SETIAP KALI sebelum memulai sesi rekam di lokasi baru:

| **\#** | **Item Cek** | **Status** |
|----|----|----|
| 1 | Body tracking aktif dan skeleton ter-deteksi (confidence \> 50) | \[ \] OK |
| 2 | Seluruh tubuh subjek (pinggang ke atas) masuk frame | \[ \] OK |
| 3 | Pencahayaan cukup, tidak ada bayangan keras di wajah/tangan | \[ \] OK |
| 4 | Background tidak bergerak (orang lewat, kipas angin besar, dsb) | \[ \] OK |
| 5 | Rekam uji 3 detik, cek depth map dan skeleton berjalan normal | \[ \] OK |
| 6 | Nama folder sesuai konvensi penamaan (lihat Seksi 5) | \[ \] OK |
| 7 | Storage tersedia cukup (minimal 10 GB free per sesi) | \[ \] OK |

#  4. PROSEDUR PENGAMBILAN DATA

### 4.1 Persiapan Subjek

6.  Subjek berdiri di titik yang sudah ditandai (jarak 1.5-3 m dari kamera)

7.  Pakaian: hindari warna yang sama persis dengan background (terutama putih polos di dinding putih)

8.  Pastikan tangan dan lengan subjek tidak tertutup jaket tebal / aksesoris besar

9.  Lakukan pemanasan gerakan dulu (jangan langsung masuk sesi rekam)

10. Jelaskan kepada subjek setiap gesture sebelum mulai rekam - tunjukkan contoh

### 4.2 Alur Rekam Per Clip (1 clip = 1 gesture × 1 repetisi)

| **Langkah** | **Tindakan** | **Durasi** |
|----|----|----|
| 1 | Subjek berdiri tegak, posisi netral, tangan di sisi tubuh | 3 detik (buffer sebelum) |
| 2 | Operator bilang: 'SIAP' - subjek ancang-ancang (Sequence A) | ~0.3 detik |
| 3 | Operator bilang: 'YA!' - subjek melakukan gerakan (Sequence B) | ~0.3 detik |
| 4 | Subjek kembali ke posisi netral (Sequence C) | ~0.3 detik |
| 5 | Operator stop rekam | Total rekam = ~1 detik aktual |
| 6 | Simpan file, cek skeleton ter-track di semua frame | ~10 detik |
| 7 | Ulangi dari langkah 1 untuk repetisi berikutnya | \- |

> Satu detik rekam di 30fps = 30 frame. Dari 30 frame ini, pisahkan menjadi Seq A (10 frame), Seq B (8 frame), Seq C (12 frame). Pembagian bisa diadjust saat post-processing, tidak perlu persis saat rekam.

### 4.3 Variasi yang WAJIB Dilakukan Per Gesture

| **Variasi**            | **Pilihan**                    | **Jumlah** |
|------------------------|--------------------------------|------------|
| Tangan dominan         | Kanan / Kiri                   | 2          |
| Kecepatan              | Normal / Cepat                 | 2          |
| Jarak ke kamera        | Dekat (~1.5m) / Sedang (~2.5m) | 2          |
| Repetisi per kombinasi | 5x ulangan                     | 5          |
| Background / Lokasi    | 15 lokasi berbeda              | 15         |
| Subjek                 | 7 mahasiswa berbeda            | 7          |

### 4.4 Target Jumlah Data Keseluruhan

| **Komponen**               | **Jumlah**         | **Keterangan**          |
|----------------------------|--------------------|-------------------------|
| Kelas gesture              | 13 kelas           | 8 positive + 5 negative |
| Subjek                     | 7 orang            | \-                      |
| Lokasi background          | 15 lokasi          | Lihat list Seksi 6      |
| Repetisi per kombinasi     | 5x                 | Agar variasi natural    |
| **TOTAL CLIP (estimasi)**  | **~6.825 clip**    | **= 7 × 13 × 15 × 5**   |
| **TOTAL FRAME (estimasi)** | **~204.750 frame** | **@ 30 frame/clip**     |

> Jika waktu terbatas, prioritaskan dulu: Subjek × Gesture × 5 Repetisi di 7 lokasi. Sisa background bisa diambil belakangan. Koordinasikan dengan supervisor S3 sebelum mengurangi target.

#  5. KONVENSI PENAMAAN FILE & FOLDER

### 5.1 Struktur Folder

> **dataset/**
>
> ├── raw/
>
> │ ├── S01/ S02/ ... S07/ (per subjek)
>
> │ ├── BG01/ BG02/ ... BG15/ (per background)
>
> │ ├── G01_R_001/ (gesture_tangan_repetisi)
>
> │ │ ├── video.svo (raw ZED)
>
> │ │ ├── video.mp4 (export RGB)
>
> │ │ ├── skeleton.json (body tracking ZED)
>
> │ │ └── depth/ (depth frame .png)
>
> │ ├── G01_L_001/ (tangan kiri)
>
> └── annotations/ master_labels.csv

### 5.2 Format Nama Folder Clip

| **Komponen** | **Format** | **Contoh** | **Keterangan** |
|----|----|----|----|
| ID Gesture | G## | G01 | Lihat tabel kelas Seksi 2 |
| ID Negative | N## | N02 | Untuk negative class |
| Tangan | R / L / B | R | Right, Left, Both |
| Repetisi | \### | 003 | 3 digit, mulai dari 001 |
| Contoh folder | G01_R_003 | \- | Gesture jab, tangan kanan, repetisi ke-3 |

### 5.3 Format Nama File Label (master_labels.csv)

> clip_id,subject_id,background_id,gesture_id,hand,repetisi,speed,kamera_jarak_m,ada_occlusion,body_tracking_ok,catatan
>
> S01_BG03_G01_R_001,S01,BG03,G01,R,1,normal,2.0,false,true,

#  6. DAFTAR BACKGROUND LOKASI

Keragaman background penting agar model tidak belajar berdasarkan lingkungan, melainkan berdasarkan gerakan. Gunakan minimal 15 lokasi berbeda.

| **ID** | **Lokasi** | **Tipe Background** | **Catatan** |
|----|----|----|----|
| BG01 | Lab - area meja | Indoor, peralatan lab | Background kompleks |
| BG02 | Lab - area kosong | Indoor, dinding putih | Background sederhana/bersih |
| BG03 | Koridor lantai 1 PENS | Indoor, koridor | Ada tekstur arsitektur |
| BG04 | Koridor lantai 2 PENS | Indoor, koridor atas | Variasi pencahayaan berbeda |
| BG05 | Ruang kelas kosong | Indoor, papan tulis | Background hijau/hitam dominan |
| BG06 | Ruang kelas berisi kursi | Indoor, ramai | Background kompleks |
| BG07 | Lobby/Foyer gedung | Indoor, terbuka lebar | Pencahayaan terang natural |
| BG08 | Tangga gedung | Indoor, tangga | Sudut berbeda, ada railing |
| BG09 | Parkir dalam gedung | Semi-outdoor | Cahaya campuran |
| BG10 | Halaman depan gedung | Outdoor, siang | Cahaya matahari langsung |
| BG11 | Taman kampus | Outdoor, tanaman | Background hijau, natural |
| BG12 | Area outdoor sore hari | Outdoor, sore | Pencahayaan low-light |
| BG13 | Ruang meeting | Indoor, formal | Background netral, rapi |
| BG14 | Lorong sempit | Indoor, sempit | Variasi jarak terbatas |
| BG15 | Area cafetaria/kantin | Indoor, ramai | Background paling kompleks |

> Dokumentasikan foto kondisi background sebelum rekam dan simpan di folder BG##/background_photo.jpg. Ini memudahkan analisis jika ada masalah variasi lingkungan.

#  7. CONSTRAINT & HAL YANG DILARANG

### 7.1 Constraint WAJIB Dipenuhi

| **\#** | **Constraint** | **Nilai** | **Aksi Jika Melanggar** |
|----|----|----|----|
| C1 | Body tracking confidence | \> 50 setiap frame | Hapus clip, rekam ulang |
| C2 | Jarak subjek ke kamera | 1.5 - 3.0 meter | Geser posisi subjek |
| C3 | Framerate aktual | \>= 28 fps | Cek performa PC, tutup aplikasi lain |
| C4 | Tangan terlihat penuh | Tidak ada occlusion saat Seq B | Rekam ulang |
| C5 | Durasi gerakan aktif | ~1 detik (25-35 frame) | Subjek jangan bergerak terlalu lambat/cepat |
| C6 | Subjek harus netral di awal & akhir | Tangan di sisi tubuh | Trim atau rekam ulang |
| C7 | Ketiga sequence A,B,C harus ada | Semua fase terekam | Clip tidak valid, rekam ulang |

### 7.2 Hal yang DILARANG

- Merekam sambil kamera bergerak / handheld tanpa stabilizer

- Menggunakan satu orang yang sama untuk lebih dari 1 ID subjek

- Merekam di kondisi cahaya sangat gelap (malam tanpa lampu yang cukup)

- Memakai pakaian yang sama persis untuk subjek berbeda di lokasi yang sama

- Mengganti nama file/folder setelah pelabelan (akan merusak CSV referensi)

- Menggunakan file SVO yang sudah corrupt / tidak lengkap

- Merekam jika ada orang lain yang bergerak di background dekat kamera

### 7.3 Apa yang Harus Dilakukan Jika Ada Masalah

| **Masalah** | **Solusi** |
|----|----|
| Body tracking hilang di tengah clip | Hapus clip, rekam ulang. Cek jarak dan pencahayaan. |
| Subjek keluar frame saat gerakan | Minta subjek berdiri lebih ke tengah, rekam ulang |
| File SVO corrupt | Cek storage space. Jangan hentikan rekam paksa. Rekam ulang. |
| PC lag / drop frame | Tutup aplikasi lain, cek suhu laptop, kurangi resolusi jika perlu |
| Gerakan tidak natural / terlihat kaku | Berikan lebih banyak latihan sebelum rekam. Jangan paksa. |
| Konflik nama folder | Laporkan ke supervisor S3 SEBELUM mengubah apapun |

#  8. DATA ZED YANG HARUS DIAMBIL

ZED Camera menghasilkan beberapa jenis data sekaligus. Semua jenis data di bawah ini WAJIB disimpan karena digunakan untuk training dengan pendekatan yang berbeda.

| **Tipe Data** | **Format File** | **Isi Data** | **Digunakan Untuk** |
|----|----|----|----|
| RGB Video | video.mp4 | Frame warna biasa (3 channel) | Ekstraksi visual, CNN-based model |
| Raw ZED | video.svo | Format native ZED, berisi semua data | Re-processing jika diperlukan |
| Depth Map | depth/frame_NNN.png | 16-bit grayscale, nilai = jarak (mm) | Isolasi subjek, fitur spasial |
| 3D Skeleton | skeleton.json | Koordinat XYZ 34 keypoint per frame | MAIN FEATURE untuk training |
| Confidence Map | confidence/frame_NNN.png | Seberapa yakin ZED tiap pixel depth-nya | Filter noise depth |

### 8.1 Format skeleton.json

File ini adalah yang paling penting untuk training. Isinya adalah posisi 3D tiap sendi per frame.

> {
>
> "frame_count": 30,
>
> "fps": 30,
>
> "frames": \[
>
> {
>
> "frame_id": 0,
>
> "timestamp_ms": 0,
>
> "body_tracking_confidence": 87.5,
>
> "keypoints": {
>
> "WRIST_RIGHT": {"x": 0.32, "y": 0.15, "z": 1.82},
>
> "ELBOW_RIGHT": {"x": 0.18, "y": 0.08, "z": 1.95},
>
> "SHOULDER_RIGHT": {"x": 0.05, "y": 0.02, "z": 2.10},
>
> *"... (total 34 keypoints)": {}*
>
> }
>
> } ... (30 frame total)

#  9. PEMBAGIAN TUGAS & JADWAL

### 9.2 Jadwal Target 

| **Hari** | **Target** | **Output** |
|----|----|----|
| H1 | Setup ZED, test rekam, finalisasi prosedur | Setup OK, rekam percobaan 10 clip |
| H2-H3 | Rekam BG01-BG05, gestur G01-G04, subjek S01-S03 | ~450 clip |
| H4-H5 | Rekam BG06-BG10, gestur G01-G04, subjek S04-S07 | ~450 clip |
| H6 | Rekam gestur G05-G08 (hook, swing, double) | ~300 clip |
| H7 | Rekam semua negative class N01-N05 | ~525 clip |
| H8 | Rekam sisa background BG11-BG15 | ~sisa clip |
| H9 | QC data: cek CSV, skeleton, file integrity | Dataset validated |
| H10 | Backup + siapkan presentasi minggu depan | Presentasi siap |

### 9.3 Form Laporan Harian

Isi form ini setiap akhir sesi rekam dan kirim ke grup WhatsApp tim:

| **Field**                   | **Isi** |
|-----------------------------|---------|
| Tanggal & Waktu             |         |
| Operator                    |         |
| Subjek hari ini (ID)        |         |
| Lokasi/Background (ID)      |         |
| Gesture yang direkam        |         |
| Jumlah clip berhasil        |         |
| Jumlah clip ditolak/diulang |         |
| Kendala yang ditemui        |         |
| Catatan khusus              |         |

#  10. QUICK REFERENCE CARD

Cetak halaman ini dan tempel di dekat setup kamera sebagai pengingat cepat.

### Kode Gesture Cepat

| **Kode** | **Gesture**                    | **Kode** | **Gesture**      |
|----------|--------------------------------|----------|------------------|
| G01      | Jab (lurus depan)              | N01      | Lambaikan tangan |
| G02      | Uppercut (bawah-atas)          | N02      | Menunjuk jari    |
| G03      | Down punch (atas-bawah serong) | N03      | Meraih benda     |
| G04      | Smash (overhead ke bawah)      | N04      | Mendorong pelan  |
| G05      | Hook kanan                     | N05      | Tangan diam      |
| G06      | Hook kiri                      |          |                  |
| G07      | Swing (haymaker)               |          |                  |
| G08      | Double (jab-cross)             |          |                  |

### Checklist Sebelum Rekam (ringkas)

| **\#** | **Cek Item**                              |
|--------|-------------------------------------------|
| 1      | Body tracking ON dan confidence \> 50?    |
| 2      | Subjek dari pinggang ke atas masuk frame? |
| 3      | Pencahayaan cukup?                        |
| 4      | Kamera di tripod, tidak goyang?           |
| 5      | Nama folder sudah sesuai format?          |
| 6      | SVO storage cukup?                        |

### Urutan Rekam Per Clip

> **POSISI NETRAL → 'SIAP!' → ANCANG-ANCANG (Seq A) → 'YA!' → PUKUL (Seq B) → RETRACT (Seq C) → NETRAL → STOP**
>
> Pertanyaan atau ada data yang meragukan? JANGAN diputuskan sendiri. Hubungi supervisor S3 via WhatsApp/Telegram dengan foto/video clip yang bermasalah.

— Dokumen Protokol Dataset Bullying Gesture Detection —

