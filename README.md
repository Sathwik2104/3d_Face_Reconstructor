# 🎭 3D Face Reconstructor

An end-to-end deep learning pipeline for **monocular 3D face reconstruction** from a single 2D in-the-wild portrait image. Leveraging a deep ResNet regressor and parametric **3D Morphable Models (3DMM - Basel Face Model 2019)**, this repository provides data preparation, model training, CLI inference, 3D mesh export (`.obj`), and an interactive web application built with **Streamlit**.

---

## 🌟 Key Features

- **Monocular 3D Reconstruction:** Reconstruct dense 3D facial geometry, pose, texture, and spherical harmonics lighting from a single unconstrained 2D photograph.
- **Parametric 3DMM Representation:** Built on Basel Face Model (BFM 2019) parametric basis vectors:
  - Identity shape coefficients ($\alpha \in \mathbb{R}^{80}$)
  - Expression coefficients ($\beta \in \mathbb{R}^{64}$)
  - Texture / albedo coefficients ($\delta \in \mathbb{R}^{80}$)
  - Rigid pose parameters ($R \in \mathbb{R}^3, t \in \mathbb{R}^3$)
  - Spherical Harmonics lighting parameters ($l \in \mathbb{R}^{27}$)
- **Interactive Web App (Streamlit):** Upload any portrait photo, view the generated 3D facial mesh interactively in the browser, and download the 3D model.
- **Universal 3D Export:** Exports standard `.obj` Wavefront meshes with vertices, vertex normals, texture coordinates, and triangle face definitions compatible with Blender, Unity, Unreal Engine, and 3D printing software.
- **Complete ML Pipeline:** Includes dataset preprocessing, landmark alignment, data augmentation (Albumentations), and training scripts with multi-task loss functions.

---

## 🏗️ Architecture & Pipeline

```text
  Input 2D Image
        │
        ▼
 Face Detection & Landmark Alignment (68 Points)
        │
        ▼
 ResNet Deep Feature Encoder / Regressor (network.py)
        │
        ├──> Shape Coefficients (α)
        ├──> Expression Coefficients (β)
        ├──> Texture Coefficients (δ)
        ├──> Rigid Head Pose (R, t)
        └──> Illumination / Spherical Harmonics (l)
        │
        ▼
 3DMM Mesh Synthesis (BFM 2019 basis vectors + face_model.obj)
        │
        ▼
 3D Wavefront Mesh (.obj) / Interactive Streamlit 3D Viewer
```

---

## 📁 Repository Structure

```text
3d_Face_Reconstructor/
├── .gitignore
├── face_model.obj                  # Canonical base 3D face template mesh
├── requirements.txt                # Python project dependencies
├── apps/
│   └── streamlit_app.py            # Interactive Streamlit 3D web UI
├── configs/
│   └── train.yaml                  # Training & inference hyperparameter configs
├── models/
│   └── network.py                  # ResNet-based 3DMM coefficient regressor
└── scripts/
    ├── convert_bfm2019.py          # BFM 2019 .mat to PyTorch tensor converter
    ├── extract_coeffs_from_mat.py  # Parameter extraction utilities
    ├── dataset.py                  # PyTorch Dataset & batch loader
    ├── prepare_data.py             # Image cropping, alignment, and landmark processing
    ├── test_data.py                # Data verification and pipeline validation
    ├── mesh_utils.py               # Normals, Euler rotations, and OBJ exporter utilities
    ├── infer.py                    # Standalone CLI inference script
    └── train.py                    # Multi-loss model training loop
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (recommended for training and fast inference)

### 2. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Sathwik2104/3d_Face_Reconstructor.git
cd 3d_Face_Reconstructor

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Interactive Web Application (Streamlit)
Launch the Streamlit web interface to reconstruct faces via browser:
```bash
streamlit run apps/streamlit_app.py
```
- Open `http://localhost:8501` in your browser.
- Upload any portrait photo.
- Inspect the reconstructed 3D mesh directly in 3D space.
- Download the resulting `.obj` file.

### 2. Command-Line Inference
Run inference directly on a target image from the terminal:
```bash
python scripts/infer.py --image path/to/portrait.jpg --config configs/train.yaml
```
The output 3D `.obj` file will be saved to your output directory.

### 3. Model Training
To train or fine-tune the model on your own face dataset:
```bash
# 1. Prepare and align the dataset
python scripts/prepare_data.py

# 2. Start model training
python scripts/train.py --config configs/train.yaml
```

Training hyperparameters (batch size, learning rates, loss weights for landmark error, photometric consistency, and coefficient regularization) can be adjusted in [`configs/train.yaml`](configs/train.yaml).

---

## 🛠️ Tech Stack

- **Deep Learning:** PyTorch, Torchvision
- **Computer Vision:** OpenCV, Albumentations, Pillow, SciPy
- **3D Geometry & Rendering:** PyVista, Trimesh, NumPy
- **Web Interface:** Streamlit
- **Configuration:** PyYAML

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
