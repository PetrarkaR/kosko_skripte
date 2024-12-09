Projekat Detekcija Lopti
Fajlovi Projekta
1. model_camera_run.py
Glavni skript za detekciju lopti

Detekcija objekata u realnom vremenu korišćenjem YOLO modela
Obrađuje unos sa kamere
Detektuje sportske lopte korišćenjem mašinskog učenja
Ključne Karakteristike:

Integracija sa OpenCV
Inferencija preko ONNX Runtime-a
Detekcija objekata u realnom vremenu
Praćenje performansi
Zavisnosti:

OpenCV
NumPy
ONNX Runtime
2. convertor.py
Alat za konverziju video zapisa

Promena rezolucije video fajlova korišćenjem FFmpeg-a
Podržava konfiguraciju prilagođene rezolucije
Očuvava kvalitet video zapisa tokom konverzije
Ključne Karakteristike:

Automatska promena veličine video zapisa
Konfigurabilne izlazne dimenzije
Efikasna obrada korišćenjem FFmpeg-a
Zavisnosti:

FFmpeg
Modul Subprocess
3. analiticka.py
Skript za 3D vizualizaciju

Kreira 3D matematičke vizualizacije
Prikazuje geometriju ravni i otvora
Koristi matplotlib za napredno grafičko prikazivanje
Ključne Karakteristike:

3D prikaz površina
Matematičko modeliranje
Podesivi parametri vizualizacije
Zavisnosti:

NumPy
Matplotlib
Mpl_toolkits
Uputstva za Instalaciju
Preduslovi
Python 3.7+
OpenCV
NumPy
Matplotlib
ONNX Runtime
FFmpeg
Instalacija
# Preporučena instalacija
pip install opencv-python numpy matplotlib onnxruntime
Upotreba
Detekcija Lopti
python model_camera_run.py
Konverzija Video Zapisa
python convertor.py
3D Vizualizacija
python analiticka.py
Napomene
Prilagodite putanje do fajlova po potrebi
Proverite da li su sve zavisnosti instalirane
Indeks uređaja za kameru može varirati
Licenca
[MIT]

Saradnici
[PEtar Ristic]