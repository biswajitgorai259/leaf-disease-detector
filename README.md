# LeafDoc: Plant Disease Detection (Image Processing + ML)

Project 4: Smart Agriculture, Plant Disease Detection Using Leaf Image Processing

**Pipeline:** Preprocess -> GrabCut leaf segmentation -> per-spot lesion pinpointing -> feature extraction -> classification

**Tech:** Python (Flask, OpenCV, scikit-image, scikit-learn). Front end: HTML, CSS, JavaScript. No MERN stack required.

**Local server port:** 5005

## Folder structure

```
leaf-disease-detector/
├── app.py                  # Flask server (port 5005) + API
├── requirements.txt        # Python libraries
├── Procfile                # Start command for hosting (Render)
├── README.md
├── templates/index.html    # Page
├── static/
│   ├── css/style.css       # Liquid glass UI, desktop and mobile layout
│   ├── js/app.js           # Upload, pins, table, CSV, report, history
│   └── img/                # White sketch background (wide and tall)
├── src/
│   ├── preprocess.py       # Decode (EXIF fix), resize, noise removal, CLAHE
│   ├── segment.py          # GrabCut leaf mask and lesion pinpointing
│   ├── features.py         # Colour, shape and texture (GLCM) features
│   ├── pipeline.py         # Runs all steps together, spot list and health score
│   ├── classify.py         # RF / CNN (ONNX) / colour-rule fallback, advice
│   └── sample.py           # Synthetic image for "Try sample leaf"
├── train/
│   ├── train_classifier.py # Random Forest training, accuracy, confusion matrix
│   ├── make_demo_data.py   # Synthetic data, for testing only
│   └── evaluate_pipeline.py# Accuracy of the image-processing stage on synthetic leaves
├── data/                   # Put PlantVillage here (not uploaded to GitHub)
└── models/                 # Trained model is saved here
```

## Step by step (run on your laptop)

### Step 1: Check Python
Python 3.10 or newer is required. Check with `python --version`.

### Step 2: Create folder and virtual environment
```
cd leaf-disease-detector
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
```

### Step 3: Install libraries
```
pip install -r requirements.txt
```

### Step 4: Start the server (port 5005)
```
python app.py
```
- Laptop: `http://localhost:5005`
- Phone: on the same WiFi, use the "Phone" link shown in the terminal. If it does not open, allow port 5005 in the firewall.
- To change the port: `PORT=5005 python app.py` (PowerShell: `$env:PORT=5005; python app.py`)

Right now the **colour-rule prototype** runs. After training a model, the page will show "Random Forest (trained)".

### Step 5: Test with your own photo
- Use one leaf, on a plain or white background, in good light, without blur.
- A numbered pin appears on each spot. Click a pin or a table row to highlight that spot.
- Try the sensitivity slider and "Boost contrast". The same photo is analysed again as soon as you change them.

### Step 6: Trained model (optional, for better accuracy)
1. Download the "PlantVillage dataset" from Kaggle.
2. Put one folder per disease inside `data/PlantVillage/`.
3. Train:
```
python train/train_classifier.py --data data/PlantVillage --per-class 300
```
4. The terminal prints the **test accuracy** and the precision and recall for each class. It saves `models/leaf_model.joblib`.
5. Restart the server. The page should show "Random Forest (trained)".

### Step 7 (optional): CNN (ONNX)
If you train a CNN such as MobileNet and save it as `models/leaf_cnn.onnx` with `models/classes.json`, the app uses it. This path has not been tested yet.

## Features
- Upload, drag and drop, paste, phone camera
- Highlighted, Original and Mask views
- A numbered pin on each spot. Click a pin or row to focus it.
- Affected %, health score, spot count, chlorosis (yellow area) %
- Spot table and CSV export
- Prediction confidence and colour mix
- Recommended actions
- Downloads: pinned image with numbers, HTML report
- Recent scans (stored only in the browser)

## How detection works (for viva)
1. **Leaf segmentation:** A seed mask is made from HSV colour. GrabCut then cleans the edge and the largest component is kept.
2. **Lesion pinpointing:** Dark spots are found by comparing each pixel with the local median of the leaf. Brown spots are a separate type. A small yellow spot is a spot, and a large yellow area is chlorosis.
3. **Veins and green colour** are kept separate from disease (thin lines are skipped and green hue is excluded).
4. **Features (28):** HSV statistics, hue histogram, colour ratios, spot count, size and shape, and GLCM texture.
5. **Classification:** Random Forest (trained), or CNN (ONNX), or colour rules.

## Limitations (be honest about these in the viva)
- Two spots that overlap can become one pin.
- Colour rules are only a prototype. A trained model is needed for a real diagnosis.
- PlantVillage images are taken in a lab. Accuracy on field photos can be lower.
- Report accuracy only from your own training output.

## Please note
- Recommended actions are generic. For medicines, ask your local agriculture officer or Krishi Vigyan Kendra (KVK).
- Images are never saved on the server. They are processed only in memory.

## Deploy (Render)
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn -w 1 -b 0.0.0.0:$PORT app:app --timeout 120`
- Free tier sleeps after about 15 minutes of inactivity, so the first request can be slow.
-