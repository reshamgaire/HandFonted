# ✍️ HandFonted

**Turn your handwriting into a functional .ttf font file. Try it now on the live web application!**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-handfonted.xyz-brightgreen?style=for-the-badge&logo=rocket)](https://handfonted.xyz)


HandFonted uses a deep learning pipeline to segment, classify, and vectorize handwritten characters from a single image, producing a high-quality font file with uniform stroke thickness and balanced metrics.

---

## 🚀 How it Works

The transformation from pixels to a font file happens in three distinct stages:

### 1. Intelligent Segmentation (`character_segmentation.py`)
* **Detection:** Utilizes **PaddleOCR (PP-OCRv5)** to locate text lines within the input image.
* **Isolation:** Employs Connected Components Analysis (CCA) to separate individual characters.
* **Dot Merging:** A custom geometric logic identifies and merges disconnected components, such as the dots on lowercase **'i'** and **'j'**, based on their vertical alignment and horizontal proximity.

### 2. Optimized Classification (`character_classification.py`)
* **Architecture:** Uses **ResInceptionNet**, a custom PyTorch model combining ResNet's skip connections with Inception’s multi-scale feature extraction.
* **Global Assignment:** Unlike standard OCR that predicts characters independently, HandFonted uses the **Hungarian Algorithm** (`linear_sum_assignment`) to map segmented images to the 52 classes (A-Z, a-z). This ensures an optimal, one-to-one mapping for the entire alphabet.

### 3. Font Engineering (`font_creation.py`)
* **Stroke Normalization:** To ensure the font looks cohesive, raw character images first undergo **morphological smoothing** to remove edge anomalies and noise. They are then processed using **Skeletonization** and **Distance Transforms** to achieve a user-defined uniform stroke thickness.
* **Vectorization:** Character outlines are extracted with sub-pixel precision, simplified using the **Ramer-Douglas-Peucker algorithm** (to remove wobbly edges), and converted into smooth vector glyphs. 
* **Auto-Metrics:** The system dynamically calculates **Left/Right Side Bearings (LSB/RSB)** and Advance Widths based on the character's "ink" distribution to ensure natural letter spacing.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/reshamgaire/HandFonted.git
   cd HandFonted
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires PyTorch, PaddleOCR, FontTools, OpenCV, and Scikit-Image.*

3. **Download Models:**
   Make sure the pre-trained `best_ResInceptionNet_model.pth` and PaddleOCR inference models are in the `resources/` directory.

---

## 💻 Usage

Run the full pipeline via the CLI:

```bash
python main.py \
    --input-image "examples/my_handwriting.jpg" \
    --output-path "output/my_custom_font.ttf" \
    --font-name "Jane Doe Hand" \
    --thickness 100
```

### Arguments:
| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input-image` | Path to the handwriting image. | (Required) |
| `--output-path` | Path to save the `.ttf` file. | (Required) |
| `--thickness` | Desired stroke weight (higher is bolder). | 100 |
| `--font-name` | The name of your font family. | "My Handwriting" |
| `--base-font` | The template `.ttf` for metrics. | `resources/arial.ttf` |

---

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
