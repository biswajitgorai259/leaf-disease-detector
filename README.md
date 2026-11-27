# Leaf Check: Plant Disease Detection (Image Processing + ML)

Project 4: Smart Agriculture, Plant Disease Detection Using Leaf Image Processing

Pipeline: **Preprocess -> GrabCut leaf segmentation -> per-spot lesion pinpointing -> feature extraction -> classification**
Tech: Python (Flask, OpenCV, scikit-image, scikit-learn). Front end HTML/CSS/JS. MERN ki zarurat nahi.
Server port: **5005**

## Folder structure

```
leaf-disease-detector/
├── app.py                  # Flask server (port 5005) + API
├── requirements.txt
├── README.md
├── templates/index.html    # page
├── static/
│   ├── css/style.css       # liquid glass UI, desktop + mobile layout
│   ├── js/app.js           # upload, pins, table, CSV, report, history
│   └── img/                # white sketch background (wide + tall)
├── src/
│   ├── preprocess.py       # decode (EXIF fix), resize, noise removal, CLAHE
│   ├── segment.py          # GrabCut leaf mask + lesion pinpointing
│   ├── features.py         # colour + shape + texture (GLCM) features
│   ├── pipeline.py         # sab steps ek saath, spot list aur health score
│   ├── classify.py         # RF / CNN (ONNX) / colour-rule fallback, advice
│   └── sample.py           # "Try sample leaf" ki nakli image
├── train/
│   ├── train_classifier.py # Random Forest training, accuracy, confusion matrix
│   └── make_demo_data.py   # NAKLI data, sirf test ke liye
├── data/                   # PlantVillage yahan rakhein
└── models/                 # trained model yahan save hota hai
```

## Step by step

### Step 1: Python check
Python 3.10 ya naya chahiye. `python --version`

### Step 2: Folder aur virtual environment
```
cd leaf-disease-detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
```

### Step 3: Libraries install
```
pip install -r requirements.txt
```

### Step 4: Server chalayein (port 5005)
```
python app.py
```
- Laptop: `http://localhost:5005`
- Phone: same WiFi pe, terminal mein dikha "Phone" link. Na khule to firewall mein port 5005 allow karein.
- Port badalna ho: `PORT=5005 python app.py` (PowerShell: `$env:PORT=5005; python app.py`)

Abhi **colour-rule prototype** chalta hai. Trained model ke baad "Random Forest (trained)" dikhega.

### Step 5: Apni photo se test
- Ek patta, safed ya plain background, achhi roshni, blur nahi.
- Har spot par numbered pin dikhega. Pin ya table ki row click karke wo spot highlight hota hai.
- Sensitivity slider aur "Boost contrast" try karein. Badalte hi same photo dobara analyse hoti hai.

### Step 6: Asli model (optional, zyada accuracy ke liye)
1. Kaggle se "PlantVillage dataset" download karein.
2. `data/PlantVillage/` mein har bimari ka alag folder rakhein.
3. Training:
```
python train/train_classifier.py --data data/PlantVillage --per-class 300
```
4. Terminal mein **test accuracy** aur har class ki precision/recall dikhegi. `models/leaf_model.joblib` ban jaayega.
5. Server dobara chalayein. Page pe "Random Forest (trained)" dikhna chahiye.

### Step 7 (optional): CNN (ONNX)
Agar aap MobileNet jaisa CNN train karke `models/leaf_cnn.onnx` aur `models/classes.json` rakhte hain, to app use karega. Ye path abhi test nahi kiya gaya hai.

## App ke features
- Upload, drag-drop, paste, phone camera
- Highlighted, Original, Mask views
- Numbered pin har spot par. Pin ya row click se focus.
- Affected %, health score, spot count, chlorosis (peela area) %
- Spot table, CSV export
- Prediction confidence aur colour mix
- Recommended actions
- Download: numbered pin wali image, HTML report
- Recent scans (sirf browser mein)

## Detection kaise hota hai (viva ke liye)
1. **Leaf segmentation:** HSV se patta seed, phir GrabCut se kinara saaf, sabse bada component.
2. **Lesion pinpointing:** patte ke median se gehre dhabbe (local contrast). Brown rang wale dhabbe alag type. Chhota peela dhabba spot, bada peela area chlorosis.
3. **Veins aur green rang** ko bimari se alag rakha gaya hai (patli lines skip, green hue exclude).
4. **Features (28):** HSV stats, histogram, rang ke ratio, spots ki ginti/size/shape, GLCM texture.
5. **Classification:** RF (trained), ya CNN (ONNX), ya colour rules.

## Limitations (khud bata dein)
- Overlap wale do dhabbe kabhi ek pin ban jaate hain.
- Colour rules sirf prototype hain. Asli diagnosis ke liye trained model chahiye.
- PlantVillage lab photos par train hota hai. Khet ki photo par accuracy kam aa sakti hai.
- Accuracy apne training output se hi likhein.

## Dhyan rakhein
- Recommended actions generic hain. Dawai ke liye local agriculture officer / KVK se puchein.
- Images server par save nahi hoti, sirf memory mein process hoti hain.
