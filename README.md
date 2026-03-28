# Handwritten Digit Recognition with Explainable AI (XAI)

A comprehensive deep learning project for recognizing handwritten digits from the MNIST dataset using multiple architectures (Simple NN, Basic CNN, Advanced CNN) and featuring advanced Explainable AI (XAI) visualizations.

![Project Banner](results/figures/cnn_basic_training_curves.png)

## 🚀 Features
- **Multiple Architectures**: Baseline Dense NN, LeNet-5 Style CNN, and a deep Advanced CNN with BatchNorm.
- **Data Augmentation**: Robust pipeline including Elastic Deformations, Rotations, Shifts, and Zooms.
- **Explainable AI (XAI)**: Integrated Grad-CAM and Saliency Maps to visualize model attention.
- **Interactive Web App**: Streamlit-based application for live drawing, image uploads, and real-time inference.
- **Benchmark Suite**: Comprehensive model comparison framework (Accuracy, Latency, Size).
- **Unit Tested**: Full test suite for preprocessing, augmentation, and model logic.

## 📊 Performance Summary
| Model | Accuracy | Params | Size (MB) | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **Simple NN** | ~98.8% | 535k | 6.2 | 0.10 |
| **Basic CNN** | ~99.2% | 225k | 2.6 | 0.16 |
| **Advanced CNN**| ~98.9% | 130k | 1.6 | 0.34 |

## 🛠️ Installation & Usage

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Running the Web App
```bash
streamlit run web_app/app.py
```

### Reproducing Results
- **Phase 2 (NN)**: `python run_phase2.py`
- **Phase 3 (CNN)**: `python run_phase3.py`
- **Phase 4 (Benchmarks)**: `python src/compare_models.py`

## 📂 Documentation
- **[Architecture Guide](docs/ARCHITECTURE.md)**: Details on data flow and model design.
- **[API Reference](docs/API_DOCUMENTATION.md)**: Function-level documentation.
- **[User Guide](docs/USER_GUIDE.md)**: Detailed setup and usage instructions.

## 🧪 Testing
```bash
pytest tests/
```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
