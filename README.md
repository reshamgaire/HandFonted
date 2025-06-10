# ✍️ HandFonted

**Turn your handwriting into a functional .ttf font file. Try it now on the live web application!**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-handfonted.xyz-brightgreen?style=for-the-badge&logo=rocket)](https://handfonted.xyz)

This repository contains the source code for the command-line tool that powers the HandFonted web app. It provides a complete pipeline to take an image of handwritten characters, segment them, classify them using a PyTorch model, and build a working TrueType font.

---

### Table of Contents
* [How It Works](#how-it-works)
* [Features](#features)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)

---

### How It Works

The HandFonted pipeline consists of three main stages:
1.  **Segmentation (`character_segmentation.py`):** Uses OpenCV to perform image processing (adaptive thresholding, morphological operations) to find and extract individual character images from the source. It includes a smart heuristic to merge dots with 'i' and 'j' characters.
2.  **Classification (`character_classification.py`):** A custom `ResInceptionNet` model (built with PyTorch) classifies each character image. It uses the Hungarian algorithm (`linear_sum_assignment`) to ensure each character from the input sheet is uniquely assigned to a letter.
3.  **Font Creation (`font_creation.py`):** The classified images are vectorized using `scikit-image`. These vector outlines are then used to replace the corresponding glyphs in a base template font (`.ttf`), generating a new, custom font file.

---

### Features
- **End-to-End Pipeline:** From a single image to a usable `.ttf` font.
- **Custom Deep Learning Model:** A hybrid ResNet-Inception model for accurate character classification.
- **Intelligent Dot Merging:** Correctly handles dotted characters like 'i' and 'j'.
- **Vectorization:** Converts pixel images into smooth, scalable font glyphs.
- **Customizable:** Control font name, style, and stroke thickness.

---

### Getting Started

#### Prerequisites
- Python 3.8+
- An image of your handwriting.

#### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/reshamgaire/handfonted.git
    cd handfonted
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
---

### Usage

Run the main script from the command line.

**Basic Usage:**
```bash
python main.py --input-image "path/to/your/handwriting.jpg" --output-path "output/my_font.ttf"
```

**Customized Usage:**
```bash
python main.py \
    --input-image "examples/good_example.jpg" \
    --output-path "output/ReshamHand.ttf" \
    --font-name "Resham Hand" \
    --font-style "Regular" \
    --thickness 110 \
    --model-path "resources/best_ResInceptionNet_model0.8811.pth" \
    --base-font "resources/arial.ttf"
```

---

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---