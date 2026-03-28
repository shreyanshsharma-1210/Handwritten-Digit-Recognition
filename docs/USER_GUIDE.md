# User Guide: Handwritten Digit Recognition

This guide provides instructions on how to set up, train, and run the digit recognition project.

## 1. Setup

### Prerequisites
- Python 3.8+
- Active internet connection (for initial MNIST download)

### Installation
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shreyanshsharma-1210/Handwritten-Digit-Recognition.git
    cd Handwritten-Digit-Recognition
    ```
2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Linux/Mac:
    source .venv/bin/activate
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 2. Training Models
You can train each phase of the project using the following scripts. Each script automatically saves logs, metrics, and the best model to the project root.

### Phase 2: Simple Neural Network
```bash
python run_phase2.py
```

### Phase 3: Convolutional Neural Networks (CNNs)
This trains both the basic and advanced architectures.
```bash
python run_phase3.py
```

---

## 3. Comparative Analysis
Once you have trained the models, run the benchmark script to generate a comparison report.
```bash
python src/compare_models.py
```
- **Results**: See `results/reports/model_comparison.csv`.
- **Figures**: See accuracy and latency plots in `results/figures/`.

---

## 4. Launching the Web Application
The web app allows for interactive digit drawing and real-time inference with explainability toggles.

```bash
streamlit run web_app/app.py
```
### Features:
1.  **Canvas**: Draw any digit (0-9).
2.  **Upload**: Upload a photo of a handwritten digit.
3.  **Samples**: Click on one of the 10 provided MNIST samples to see model predictions.
4.  **Grad-CAM**: Toggle "Show Grad-CAM" to visualize the model's spatial attention.

---

## 5. Running Tests
To verify the integrity of the data pipeline and model architectures, run:
```bash
pytest tests/
```
