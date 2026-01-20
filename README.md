# CIFAR-10 Üzerinde 5 Model Karşılaştırması (Aynı Şartlar)

Bu projede **CIFAR-10** veri seti üzerinde, farklı derin öğrenme mimarilerinin
görüntü sınıflandırma performansları **adil karşılaştırma (fair comparison)**
yaklaşımıyla analiz edilmiştir.

## Amaç
CIFAR-10 üzerinde farklı mimarilerin sınıflandırma performanslarını karşılaştırmak.

Değerlendirilen metrikler:
- **Accuracy**
- **Precision (macro)**
- **Recall (macro)**
- **F1-score (macro)**
- **ROC-AUC (multiclass OVR)**

## Kullanılan Modeller
- **SimpleCNN**
- **CustomCNN**
- **ResNet-18**
- **ViT-Tiny**
- **ConvNeXt-Tiny**
---

## Kurulum

### Gereksinimler
- Python **3.11+**
- (Önerilir) CUDA destekli GPU

### Kurulum
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
````

---

## Proje Yapısı

```text
deep-learning-project/
├─ data/                         # CIFAR-10 otomatik indirilir
├─ results/
│  ├─ runs/                      # Her model için run klasörleri
│  └─ presentation_assets/       # Sunum PNG + JSON + TXT
└─ src/
   ├─ config.py
   ├─ utils.py
   ├─ model.py
   ├─ train.py
   ├─ eval.py
   ├─ make_assets.py
   ├─ infer_grid.py
   ├─ metrics_report.py
   ├─ metrics_report_png.py
   ├─ metrics_png_from_txt.py
   └─ write_metrics.py
```

---

## Veri Kümesi ve Ön İşleme

**Veri seti:** CIFAR-10 (10 sınıf)

### Fair Comparison (Aynı Şartlar)

* Tüm modeller **aynı giriş boyutunu** görür: **224×224**

  * CIFAR-10 (32×32) → Resize / RandomResizedCrop(224)
* Aynı normalize:

  * Mean: `(0.4914, 0.4822, 0.4465)`
  * Std:  `(0.2470, 0.2435, 0.2616)`
* Augmentation (ON):

  * `RandomResizedCrop(224)`
  * `RandomHorizontalFlip()`

---

## Eğitim ve Değerlendirme

### Tek Model Eğitimi

```bash
python -m src.train --model resnet18 --epochs 3
python -m src.train --model vit_tiny --epochs 3
python -m src.train --model convnext_tiny --epochs 3
```

### Test + Confusion Matrix

```bash
python -m src.eval --ckpt results/runs/<RUN_NAME>/checkpoints/best.pt
```

### Test Örnekleri (Inference Grid)

```bash
python -m src.infer_grid --ckpt results/runs/<RUN_NAME>/checkpoints/best.pt
```

### Tüm Modelleri Tek Akışta Çalıştır (Önerilen)

```bash
python -m src.make_assets --epochs 3
```

> Epoch arttıkça (özellikle ViT / ConvNeXt) eğitim süresi ciddi şekilde uzar.

---

## Sonuçlar ve Görseller

Sunum için üretilen tüm görseller:

```text
results/presentation_assets/
```

### Eğitim Eğrileri + Confusion Matrix (Yan Yana)

#### SimpleCNN (Aug ON)

<p align="center">
  <img src="results/presentation_assets/simplecnn_aug_on_loss.png" width="32%" />
  <img src="results/presentation_assets/simplecnn_aug_on_accuracy.png" width="32%" />
  <img src="results/presentation_assets/simplecnn_aug_on_confusion_matrix.png" width="32%" />
</p>

#### CustomCNN (Aug ON)

<p align="center">
  <img src="results/presentation_assets/customcnn_aug_on_loss.png" width="32%" />
  <img src="results/presentation_assets/customcnn_aug_on_accuracy.png" width="32%" />
  <img src="results/presentation_assets/customcnn_aug_on_confusion_matrix.png" width="32%" />
</p>

#### ResNet-18 (Aug ON)

<p align="center">
  <img src="results/presentation_assets/resnet18_aug_on_loss.png" width="32%" />
  <img src="results/presentation_assets/resnet18_aug_on_accuracy.png" width="32%" />
  <img src="results/presentation_assets/resnet18_aug_on_confusion_matrix.png" width="32%" />
</p>

#### ResNet-18 (Aug OFF)

<p align="center">
  <img src="results/presentation_assets/resnet18_aug_off_loss.png" width="32%" />
  <img src="results/presentation_assets/resnet18_aug_off_accuracy.png" width="32%" />
  <img src="results/presentation_assets/resnet18_aug_off_confusion_matrix.png" width="32%" />
</p>

#### ViT-Tiny (Aug ON)

<p align="center">
  <img src="results/presentation_assets/vit_tiny_aug_on_loss.png" width="32%" />
  <img src="results/presentation_assets/vit_tiny_aug_on_accuracy.png" width="32%" />
  <img src="results/presentation_assets/vit_tiny_aug_on_confusion_matrix.png" width="32%" />
</p>

#### ConvNeXt-Tiny (Aug ON)

<p align="center">
  <img src="results/presentation_assets/convnext_tiny_aug_on_loss.png" width="32%" />
  <img src="results/presentation_assets/convnext_tiny_aug_on_accuracy.png" width="32%" />
  <img src="results/presentation_assets/convnext_tiny_aug_on_confusion_matrix.png" width="32%" />
</p>

### Tüm Modellerin Karşılaştırması

<p align="center">
  <img src="results/presentation_assets/compare_all_models.png" width="75%" />
</p>

---

## Metrikler

Raporlanan metrikler:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)
* ROC-AUC (multiclass OVR)

### Mevcut Run’lardan (Yeniden Eğitim Yapmadan)

#### Checkpoint → metrics.txt

```bash
python -m src.metrics_report --runs results/runs/...
```

#### Checkpoint → TXT + PNG tablo + grafikler

```bash
python -m src.metrics_report_png --runs results/runs/...
```

Çıktılar:

```text
results/presentation_assets/metrics/
├─ metrics.txt
├─ metrics_table.png
├─ bar_accuracy.png
├─ bar_precision_macro.png
├─ bar_recall_macro.png
├─ bar_f1_macro.png
└─ bar_roc_auc_ovr_macro.png
```

---

## Çıktılar

Her run klasörü:

```text
results/runs/<RUN_NAME>/
├─ checkpoints/best.pt
├─ curves/loss.png
├─ curves/accuracy.png
├─ figures/confusion_matrix.png
├─ figures/inference_grid.png
└─ metrics/
   ├─ train_summary.json
   └─ eval_summary.json
```

---

## Tekrarlanabilirlik
* Sabit seed (`seed=42`)
* Aynı ön işleme ve giriş boyutu (224×224)
* Checkpoint’ler durduğu sürece **yeniden eğitim gerekmez**

---

## Sınırlamalar
* CIFAR-10’un 32×32 olması → 224×224 büyütme maliyetlidir
* Epoch arttıkça eğitim süresi çok uzar
* ROC-AUC multiclass OVR bazı ortamlarda hesaplanamayabilir (N/A)

---

## Ders Bilgileri
* **Ders:** Derin Öğrenme ve Uygulamaları
* **Konu:** CIFAR-10 Üzerinde 5 Model Karşılaştırması (Aynı Şartlar)
