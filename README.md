# SentimentScope

SentimentScope is a Python-based tool for detecting the sentiment (positive, negative) of comments or paragraphs, ideal for analyzing e-commerce feedback. It uses the `cardiffnlp/twitter-roberta-base-sentiment` model for accurate sentiment analysis, with an older `FacebookAI/roberta-base` model included for reference and comparing. The project supports testing on datasets like e-commerce reviews, with large datasets hosted externally to comply with GitHub limits.

## System Requirements
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS 12+
- **Memory**: 16GB RAM recommended (8GB minimum)
- **GPU**: Optional (e.g., GTX 1650 Ti for faster inference)
- **Disk Space**: ~2GB for models and dependencies, plus dataset storage

### Python
- **Python 3.12** (Core functionality: sentiment analysis)
  ```bash
  # Windows: Use Anaconda (recommended)
  conda create -n thesis python=3.11
  conda activate thesis
### Project Dependencies
- **Python Libraries**:
- **transformers (Hugging Face for RoBERTa models)**
- **torch (PyTorch for model inference)**
- **pandas (Dataset handling)**
- **python-dotenv (Environment variables)**
- **Full list in requirements.txt**

### DATASET:
- **DATA**: https://drive.google.com/file/d/11F0meGrucbIiS9aXeJTdsc8H5zqSC9Le/view?usp=sharing
