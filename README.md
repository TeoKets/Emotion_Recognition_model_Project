# 🎭 Emotion Recognition Model (TensorFlow + CNN + GUI + API)

This repository contains a full **Emotion Recognition System** built with:

- **TensorFlow / Keras**  
- **Convolutional Neural Networks (CNNs)**  
- **Custom preprocessing pipeline**  
- **REST API for model inference**  
- **Desktop GUI (image upload → prediction)**  
- **TensorBoard logs for deep training analysis**

The model predicts **human emotions from images**, trained on grayscale face datasets.

---

# 📊 Model Training Insights (TensorBoard)

Training was monitored using **TensorBoard**, including scalar metrics, histograms, and time-series weight evolution.

---

## 📈 **Epoch Accuracy**
Training vs Validation Accuracy  
![epoch_accuracy](docs/images/epoch_accuracy.png)

---

## 📉 **Epoch Loss**
Training vs Validation Loss  
![epoch_loss](docs/images/epoch_loss.png)

---

## 🔍 **Evaluation Loss vs Iterations**
Batch-level loss curve  
![evaluation_loss](docs/images/evaluation_loss_vs_iterations.png)

---

## 🧠 **BatchNorm Parameters (beta, gamma, bias)**

### Beta Histogram  
![beta_histogram](docs/images/beta_histogram.png)

### Bias Histogram  
![bias_histogram](docs/images/bias_histogram.png)

### Gamma Histogram  
![gamma_histogram](docs/images/gamma_histogram.png)

---

## ⚙ **Kernel & BatchNorm Running Stats**

### Kernel Weights Histogram  
![kernel_hist](docs/images/kernel_histogram.png)

### Moving Mean  
![moving_mean](docs/images/moving_mean_hist.png)

### Moving Variance  
![moving_variance](docs/images/moving_variance_hist.png)

---

# 🧬 Model Architecture

The network is a **Convolutional Neural Network (CNN)** optimized for emotion detection from grayscale facial images.

Typical architecture:

