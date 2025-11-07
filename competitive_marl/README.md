# Competitive MARL - 3D Pursuit-Evasion

**Temiz, modüler, Kaggle GPU eğitimine optimize edilmiş competitive multi-agent reinforcement learning sistemi.**

Clean, modular competitive multi-agent reinforcement learning system optimized for Kaggle GPU training.

---

## 🎯 Özellikler / Features

### TR
- ✅ **3D HUD Görünümü**: demo_3d stilinde egocentric perspektif
- ✅ **Competitive MARL**: Her iki taraf da RL ile eğitiliyor (pursuer vs evader)
- ✅ **Kaggle Optimize**: GPU eğitimi için rendering olmadan hızlı eğitim
- ✅ **Local Test**: Rendering ile test ve görselleştirme
- ✅ **Modüler Yapı**: Temiz, yeniden kullanılabilir bileşenler
- ✅ **Alternating Training**: Sırayla eğitim (pursuer → evader → pursuer...)
- ✅ **Focus-Based Rewards**: 30% odaklanma alanı ile ödüllendirme

### EN
- ✅ **3D HUD View**: Egocentric perspective in demo_3d style
- ✅ **Competitive MARL**: Both sides trained with RL (pursuer vs evader)
- ✅ **Kaggle Optimized**: Fast training without rendering for GPU
- ✅ **Local Testing**: Testing and visualization with rendering
- ✅ **Modular Structure**: Clean, reusable components
- ✅ **Alternating Training**: Sequential training (pursuer → evader → pursuer...)
- ✅ **Focus-Based Rewards**: Rewards based on 30% focus area

---

## 📁 Klasör Yapısı / Folder Structure

```
competitive_marl/
├── environment/
│   └── pursuit_evasion_3d.py    # 3D pursuit-evasion environment
├── agents/
│   ├── pursuer_agent.py          # Pursuer (agent) RL wrapper
│   └── evader_agent.py           # Evader (target) RL wrapper
├── training/
│   └── train_kaggle.py           # Kaggle GPU training (NO RENDER)
├── testing/
│   └── test_with_render.py       # Local testing (WITH RENDER)
├── utils/
│   └── hud_renderer.py           # 3D HUD visualization
├── models/                        # Saved models go here
│   ├── pursuer_latest.zip
│   └── evader_latest.zip
├── config.py                      # Configuration
└── README.md
```

---

## 🚀 Hızlı Başlangıç / Quick Start

### 1️⃣ Kaggle'da Eğitim / Training on Kaggle

```bash
# Kaggle notebook'ta GPU açık olarak:
python training/train_kaggle.py --rounds 50 --steps-per-round 10000
```

**Kaggle Ayarları:**
- ✅ GPU: Açık (T4 or P100)
- ✅ Internet: Açık (pip install için)
- ✅ Rendering: KAPALI (hızlı eğitim için)

**Çıktı:**
```
models/pursuer_latest.zip
models/evader_latest.zip
```

### 2️⃣ Local'de Test / Testing Locally

Kaggle'dan modelleri indirdikten sonra:

```bash
# Modellerle test (rendering ile)
python testing/test_with_render.py \
    --pursuer models/pursuer_latest.zip \
    --evader models/evader_latest.zip \
    --episodes 5
```

**Gereksinimler:**
- ✅ pygame (rendering için)
- ✅ Eğitilmiş modeller (Kaggle'dan indirilmiş)

---

## ⚙️ Konfigürasyon / Configuration

`config.py` dosyasında tüm parametreler:

```python
# Çevre Parametreleri / Environment Parameters
ENV_CONFIG = {
    "view_size": 30.0,              # FOV boyutu
    "success_threshold": 9.0,       # 30% odaklanma alanı
    "target_size": 4.0,             # Target boyutu (daha büyük)
    "max_steps": 1000,              # Maksimum adım
}

# Eğitim Parametreleri / Training Parameters
TRAINING_CONFIG = {
    "pursuer_steps_per_round": 10000,   # Pursuer adımları
    "evader_steps_per_round": 10000,    # Evader adımları
    "num_rounds": 50,                    # Toplam tur sayısı
}
```

---

## 🧠 Eğitim Stratejisi / Training Strategy

### Alternating Training (Sıralı Eğitim)

```
Round 1:
  1. Train Pursuer (10K steps) vs current Evader
  2. Save Pursuer model
  3. Train Evader (10K steps) vs updated Pursuer
  4. Save Evader model

Round 2:
  1. Train Pursuer vs updated Evader
  2. Save Pursuer model
  3. Train Evader vs updated Pursuer
  4. Save Evader model

...

Round 50:
  Final models saved
```

**Avantajları:**
- ✅ Her ajan karşısındaki en güncel versiyona karşı öğrenir
- ✅ Stability: Eşzamanlı eğitimden daha stabil
- ✅ Kaggle-friendly: GPU'da hızlı çalışır

---

## 📊 Ödül Sistemi / Reward System

### Pursuer (Takipçi)
```python
# Odakta (< 9.0 units):
+0.1 per step              # Sürekli ödül
+10.0 bonus                # 5 saniye odakta kalma bonusu

# Odak dışında:
-0.01 * distance           # Mesafe cezası
-2.0 penalty               # Bonus'a yakınken kaçma cezası
-100.0 penalty             # Hedefi tamamen kaybetme
```

### Evader (Kaçan)
```python
# Odakta (< 9.0 units):
-0.1 per step              # Yakalanma cezası
-10.0 penalty              # 5 saniye yakalanma cezası

# Odak dışında:
+0.05 * distance           # Uzaklaşma ödülü
+2.0 bonus                 # Bonus'a yakınken kaçma bonusu
+100.0 bonus               # Tamamen kaçma bonusu
```

**Competitive:** Pursuer ve Evader zıt ödüller alır!

---

## 🎮 HUD Görünümü / HUD View

### 3D Egocentric HUD
```
┌─────────────────────────────────┐
│ PURSUER HUD                      │
│ Vel: 45.2                        │
│ Dist: 12.3                       │
│ T-Vel: 38.7                      │
│                                  │
│ FOCUSED ✓                        │
│ Focus: 35/50 (70%)               │
│          ┌───┐                   │
│          │ ● │  ← Target         │
│    ╋     └───┘                   │
│   Agent                          │
│                                  │
│ Step: 543                        │
│ Reward: 125.3                    │
└─────────────────────────────────┘
```

**Özellikler:**
- ✅ Agent her zaman merkezde (egocentric)
- ✅ Target derinliğe göre büyüklükte
- ✅ Velocity vektörleri (yeşil = agent, kırmızı = target)
- ✅ Focus durumu (yeşil = odakta, kırmızı = dışarıda)
- ✅ Real-time istatistikler

---

## 📦 Gereksinimler / Requirements

### Kaggle için / For Kaggle
```
gymnasium
stable-baselines3[extra]
torch
numpy
```

### Local test için / For Local Testing
```
gymnasium
stable-baselines3[extra]
torch
numpy
pygame  # HUD rendering için
```

### Kurulum / Installation
```bash
pip install gymnasium stable-baselines3[extra] torch numpy pygame
```

---

## 💾 Model Kaydetme / Model Saving

### Kaggle'da
```python
# Otomatik kaydediliyor / Automatically saved:
models/pursuer_latest.zip
models/evader_latest.zip

# Her round sonrası / After each round
```

### Modelleri İndirme / Downloading Models
```python
# Kaggle notebook'ta:
# 1. Sağ panel → Output
# 2. models/ klasörünü indir
# 3. Local'de test et
```

---

## 🔬 Test Sonuçları / Test Results

```bash
python testing/test_with_render.py --episodes 10
```

### Örnek Çıktı / Example Output
```
Episode 1/10
  Pursuer Reward: 125.3
  Evader Reward: -98.7
  Focus Time: 432 steps (72.0%)

TESTING SUMMARY
Episodes Completed: 10

Pursuer Performance:
  Mean Reward: 118.5 ± 15.2
  Min/Max: 95.3 / 145.7

Evader Performance:
  Mean Reward: -105.2 ± 18.3
  Min/Max: -132.4 / -85.1

Focus Statistics:
  Mean Focus Time: 68.3% ± 8.5%
  Min/Max: 55.2% / 78.9%
```

---

## 🐛 Troubleshooting

### Kaggle'da "No module named 'pygame'"
```
✅ Normal! Kaggle eğitimi rendering kullanmaz.
✅ Sadece local test için pygame gerekli.
```

### "CUDA out of memory"
```
# Batch size'ı küçült:
PURSUER_CONFIG = {
    "batch_size": 128  # 256 yerine
}
```

### Model bulunamadı
```bash
# Modellerin doğru klasörde olduğundan emin ol:
ls models/
# pursuer_latest.zip
# evader_latest.zip
```

---

## 📝 Kaggle Notebook Örneği / Kaggle Notebook Example

```python
# 1. Setup
!git clone https://github.com/your-repo/PIDRL.git
%cd PIDRL/competitive_marl
!pip install gymnasium stable-baselines3[extra] torch

# 2. Train
!python training/train_kaggle.py --rounds 50 --steps-per-round 10000

# 3. Download models from Output panel
```

---

## 🎯 İleri Seviye / Advanced

### Custom Config
```python
# Kendi config'ini oluştur:
from config import get_config

config = get_config()
config["env"]["max_steps"] = 2000  # Daha uzun episodlar
config["training"]["num_rounds"] = 100  # Daha fazla eğitim
```

### Simultaneous Training
```python
# config.py'de:
TRAINING_CONFIG = {
    "mode": "simultaneous",  # alternating yerine
    "total_timesteps": 1000000,
}
```

---

## 📊 TensorBoard

```bash
# Eğitim sırasında:
tensorboard --logdir logs/tensorboard/

# Browser'da aç:
http://localhost:6006
```

---

## 🤝 Contributing

Yeni özellikler veya iyileştirmeler için pull request açabilirsiniz!

Feel free to open pull requests for new features or improvements!

---

## 📄 License

MIT License - Projeyi serbestçe kullanabilirsiniz.

---

## ⭐ Citation

Bu projeyi kullanırsanız lütfen referans verin:

```bibtex
@software{competitive_marl_3d,
  title={Competitive MARL: 3D Pursuit-Evasion},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/PIDRL}
}
```

---

## 📧 İletişim / Contact

Sorularınız için issue açabilirsiniz.

For questions, please open an issue.

---

**Başarılar! / Good Luck! 🚀**
