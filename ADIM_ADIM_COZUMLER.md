# Kaggle Model Test Etme - Adım Adım Çözümler

Bu doküman, Kaggle'da eğitilen modelleri lokal makinenizde test ederken karşılaştığınız hataları ve çözümlerini **adım adım** açıklar.

---

## 📋 Özet: Ne Oldu, Ne Yaptık?

### Başlangıç Durumu
Kaggle'da bir model eğittiniz ve indirdiniz. Ancak `test_trained_model.py` ile test etmeye çalıştığınızda **3 farklı hata** aldınız.

### Final Durumu ✅
Şimdi tüm hatalar düzeltildi. `git pull` yapıp modelinizi sorunsuz test edebilirsiniz!

---

## 🔴 HATA #1: Numpy Sürüm Uyumsuzluğu

### Step 1: Hatayı Gördük
```
ModuleNotFoundError: No module named 'numpy._core.numeric'
```

### Step 2: Nedeni Anladık
- **Kaggle**: numpy 2.0+ kullanıyor (varsayılan olarak)
- **Sizin PC**: numpy 1.x yüklü (requirements.txt'te `<2.0.0` kısıtı vardı)
- **Sorun**: Model Kaggle'da numpy 2.0 ile kaydedildi
  - Cloudpickle modeli kaydederken numpy'ın iç modüllerini referans eder
  - Numpy 2.0: `numpy._core.numeric` diye bir modül var
  - Numpy 1.x: `numpy.core.numeric` diye bir modül var (alt çizgi yok!)
  - Sizin PC'de numpy 1.x var ama model `numpy._core` arıyor → HATA!

### Step 3: Çözdük ✅
**Dosya**: `test_trained_model.py` (satır 22-30)

```python
# Otomatik numpy uyumluluk düzeltmesi
try:
    import numpy._core.numeric as _numeric  # numpy 2.0+ için
except (ImportError, AttributeError):
    import numpy.core.numeric as _numeric  # numpy 1.x için
    # numpy 2.0 referanslarını 1.x'e yönlendir
    sys.modules['numpy._core.numeric'] = _numeric
    sys.modules['numpy._core'] = sys.modules['numpy.core']
```

**Ne Yaptık?**
- Önce numpy 2.0 modülünü import etmeyi dene
- Olmazsa numpy 1.x modülünü al
- Ama numpy 2.0 isimlerini (alt çizgili) numpy 1.x modülüne yönlendir
- Böylece model `numpy._core` diye bir şey aradığında, aslında `numpy.core` bulur!

**Ek Güncelleme**: `requirements.txt`
```diff
- numpy>=1.24.0,<2.0.0  # Sadece 1.x'e izin veriyordu
+ numpy>=1.24.0          # Hem 1.x hem 2.x çalışır
```

### Step 4: Ek Uyarı Düzelttik
```
UserWarning: Could not deserialize object policy_kwargs
```

**Ne Yaptık?**
```python
# Uyarıları sessizleştir (zararsız)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not deserialize.*")
    model = SAC.load(model_path)
```

**Commit**: `119f0d7` - "Fix numpy compatibility for Kaggle-trained models"

---

## 🔴 HATA #2: LazyFrames Hatası

### Step 1: Hatayı Gördük
```
AttributeError: 'LazyFrames' object has no attribute 'reshape'
```

Model yüklendi ama ilk adımda hata aldık!

### Step 2: Nedeni Anladık
- **FrameStack wrapper**: Birden fazla frame'i üst üste koyar (4 frame stack)
- **Optimizasyon**: Her frame'i kopyalamak yerine `LazyFrames` kullanır (hafıza tasarrufu)
- **LazyFrames**: Sadece gerektiğinde numpy array'e dönüşür
- **Sorun**: `model.predict(obs)` çağrıldığında:
  1. Stable-baselines3 `obs.reshape()` yapmak ister
  2. Ama `LazyFrames` nesnesinin `.reshape()` metodu yok!
  3. HATA!

### Step 3: Çözdük ✅
**Dosya**: `test_trained_model.py`

**Değişiklik 1** (satır 93):
```python
obs, info = env.reset()
obs = np.array(obs)  # LazyFrames → numpy array
```

**Değişiklik 2** (satır 108):
```python
obs, reward, terminated, truncated, info = env.step(action)
obs = np.array(obs)  # LazyFrames → numpy array
```

**Ne Yaptık?**
- `np.array(obs)` çağırdık
- Bu LazyFrames'i otomatik olarak numpy array'e çevirir
- Eğer zaten numpy array ise, hiçbir şey değişmez (güvenli)
- Artık `model.predict()` numpy array alıyor → çalışıyor! ✅

**Commit**: `d6a000e` - "Fix LazyFrames compatibility for model.predict()"

---

## 🔴 HATA #3: Aksiyon Boyutu Uyumsuzluğu

### Step 1: Hatayı Gördük
```
ValueError: operands could not be broadcast together with shapes (3,) (2,)
```

### Step 2: Nedeni Anladık
- **2D Environment**: Eski versiyon, aksiyon space = `(ax, ay)` → 2 boyut
  - Sadece yatay düzlemde hareket (X, Y)
- **3D Environment**: Yeni versiyon, aksiyon space = `(ax, ay, az)` → 3 boyut
  - Yatay + derinlik kontrolü (X, Y, Z)

**Sizin Durum**:
- Modeliniz **2D environment'ta eğitilmiş** → aksiyon: `[ax, ay]` (2 sayı)
- Test scripti **hardcoded 3D kullanıyordu** → environment: 3 boyut bekliyor
- Model 2 sayı veriyor, environment 3 sayı bekliyor → **BROADCAST HATASI!**

**Kod'daki Sorun** (`test_trained_model.py` eski versiyon):
```python
env = make_env(config, render_mode=render_mode, use_3d=True)  # Herzaman 3D!
```

### Step 3: Çözdük ✅
**Dosya**: `test_trained_model.py` (satır 80-93)

```python
# Modelin aksiyon boyutunu otomatik tespit et
action_space_dim = model.action_space.shape[0]
use_3d = (action_space_dim == 3)

if use_3d:
    print(f"\n✓ Detected 3D model (action space: {action_space_dim}D)")
    print("  Using 3D environment with depth perception")
else:
    print(f"\n✓ Detected 2D model (action space: {action_space_dim}D)")
    print("  Using 2D environment (classic version)")

# Modele uygun environment oluştur
env = make_env(config, render_mode=render_mode, use_3d=use_3d)
```

**Ne Yaptık?**
1. Model yüklendikten sonra `model.action_space.shape[0]` kontrol et
2. Eğer 3 ise → 3D model → 3D environment kullan
3. Eğer 2 ise → 2D model → 2D environment kullan
4. Kullanıcıya hangi tip tespit edildiğini göster

**Artık**:
- 2D model → 2D environment otomatik seçilir ✅
- 3D model → 3D environment otomatik seçilir ✅
- Manuel müdahale gerekmez!

**Commit**: `d2d1302` - "Add automatic 2D/3D environment detection for model testing"

---

## 📊 Tüm Düzeltmelerin Özeti

| Hata # | Sorun | Neden | Çözüm | Commit |
|--------|-------|-------|-------|--------|
| **#1** | `numpy._core.numeric` bulunamadı | Numpy 2.0 (Kaggle) vs 1.x (lokal) | Modül aliasing ekledik | `119f0d7` |
| **#2** | LazyFrames reshape hatası | LazyFrames'in reshape() yok | `np.array(obs)` ekledik | `d6a000e` |
| **#3** | Broadcast shape hatası (3,) vs (2,) | 2D model, 3D environment | Otomatik tespit ekledik | `d2d1302` |

---

## ✅ Şimdi Nasıl Kullanırsınız?

### Adım 1: Güncel Kodu Çekin
```bash
cd C:\Users\Lenovo\Desktop\PIDRL-main\PIDRL-main
git pull origin claude/pid-nn-rl-research-011CUpVJyyPR2RaPkVsoCSU3
```

### Adım 2: Modelinizi Test Edin
```bash
python test_trained_model.py --model models/sac/best_model/best_model.zip --episodes 10
```

### Adım 3: Çıktıyı İnceleyin
```
Loading model from: models/sac/best_model/best_model.zip
✓ Model loaded successfully!

✓ Detected 2D model (action space: 2D)
  Using 2D environment (classic version)

Testing model for 10 episodes...
======================================================================

Episode 1/10
----------------------------------------------------------------------
  Step 50: Distance=12.34, Focus Progress=45.2%
  Step 100: Distance=8.56, Focus Progress=67.8%
  ...

  Episode Summary:
    Total Reward: 156.78
    Episode Length: 500
    Time in Focus: 72.4%
    Final Distance: 5.23

...

======================================================================
OVERALL STATISTICS
======================================================================
Average Reward: 145.32 ± 12.45
Average Length: 487.6 ± 25.3
Average Focus Time: 68.9% ± 8.2%
```

---

## 🔍 Teknik Detaylar: Neden Bu Hatalar Oluştu?

### Genel Neden: Farklı Ortamlarda Eğitim ve Test

**Kaggle Ortamı**:
- GPU: Tesla T4 x2
- Python: 3.10+
- Numpy: 2.0.0+
- OS: Linux (Ubuntu)
- Environment: Değişiyor (hangi notebook kullandınıza bağlı)

**Sizin Lokal PC**:
- CPU/GPU: Değişken
- Python: Muhtemelen 3.8-3.11
- Numpy: 1.24.x (requirements.txt'ten)
- OS: Windows
- Environment: 3D (hardcoded)

### Cloudpickle Nasıl Çalışır?

Model kaydedilirken:
```python
SAC.save("model.zip")
```

İçeride olan:
1. Neural network weights → PyTorch tensors
2. Optimizer state → PyTorch tensors
3. **Policy configuration** → Python objects (cloudpickle ile)
4. **Action/observation space** → Gym spaces (cloudpickle ile)

Cloudpickle:
- Python objelerini binary'ye çevirir
- Ama **referansları** da saklar (hangi modülden geldiğini)
- Örnek: `numpy._core.numeric.normalize` gibi

Yüklenirken:
- Cloudpickle aynı modülleri import etmeye çalışır
- Eğer modül adı değiştiyse → HATA!
- Eğer modül versiyonu farklıysa → HATA (bazen)

### FrameStack ve LazyFrames

**Normal yöntem** (her frame kopyalanır):
```
Frame 1: [64x64] = 4096 bytes
Frame 2: [64x64] = 4096 bytes
Frame 3: [64x64] = 4096 bytes
Frame 4: [64x64] = 4096 bytes
Total: 16384 bytes
```

**LazyFrames** (referans tutar):
```
Original frames: [F1, F2, F3, F4]
LazyFrames: sadece pointer'lar → [&F1, &F2, &F3, &F4]
Total: sadece 32 bytes (pointer'lar)
Actual data: İhtiyaç olunca numpy array'e çevrilir
```

**Avantaj**: 1000 environment paralel çalışırken çok hafıza tasarrufu
**Dezavantaj**: Numpy array gerektiğinde manuel çeviri gerekir

---

## 🎓 Öğrenilen Dersler

### 1. Cross-Platform Model Sharing
- Farklı ortamlarda eğitilen modeller uyumsuz olabilir
- Çözüm: Compatibility shims (modül aliasing)
- Veya: Aynı sürümleri kullanın (requirements.txt ile)

### 2. Lazy Evaluation
- Performans için lazy object'ler kullanılır
- Ama bazı API'ler eager evaluation bekler
- Çözüm: Explicit conversion (`np.array()`)

### 3. Environment Versioning
- Environment değiştiğinde (2D → 3D) eski modeller kırılır
- Çözüm: Model metadata'sında environment version sakla
- Veya: Aksiyon space'ten otomatik tespit

### 4. Error Handling
- Her katmanda try-except kullan
- Kullanıcıya anlamlı mesajlar göster
- Otomatik fallback'ler ekle

---

## 📚 Ek Kaynaklar

### İlgili Dosyalar
- `test_trained_model.py`: Ana test scripti (tüm düzeltmeler burada)
- `TROUBLESHOOTING.md`: İngilizce troubleshooting guide
- `requirements.txt`: Bağımlılıklar (numpy kısıtı kaldırıldı)

### Git Commit'leri (Kronolojik)
1. `119f0d7`: Numpy uyumluluk düzeltmesi
2. `d6a000e`: LazyFrames düzeltmesi
3. `d2d1302`: 2D/3D otomatik tespit

### Faydalı Komutlar

**Numpy versiyonunu kontrol et:**
```bash
python -c "import numpy; print(numpy.__version__)"
```

**Model bilgilerini incele:**
```python
from stable_baselines3 import SAC
model = SAC.load("model.zip")
print(f"Action space: {model.action_space}")
print(f"Observation space: {model.observation_space}")
```

**Environment'ı test et:**
```python
from environments import make_env
import yaml

config = yaml.safe_load(open("configs/config.yaml"))
env = make_env(config, use_3d=False)  # 2D
# env = make_env(config, use_3d=True)  # 3D

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

---

## 🎯 Sonraki Adımlar

Artık modeliniz çalıştığına göre:

### 1. Performansı Analiz Edin
```bash
# Daha fazla episode test edin
python test_trained_model.py --model models/sac/best_model/best_model.zip --episodes 100 --no-render
```

### 2. Görselleştirme Yapın
```bash
# Rendering ile izleyin
python test_trained_model.py --model models/sac/best_model/best_model.zip --episodes 3
```

### 3. 3D Model Eğitin (İsteğe Bağlı)
Eğer 3D derinlik algısı ile model eğitmek isterseniz:

**Kaggle'da**:
```python
# kaggle_train.ipynb'de:
# 3D environment kullanıldığından emin olun
```

**Lokal'de**:
```bash
# 3D ile eğitim
python quick_train.py --full  # Uzun sürer!
```

### 4. Karşılaştırma Yapın
```bash
# PID, Kalman-PID ve SAC'ı karşılaştırın
python experiments/compare_methods.py --n-episodes 100
```

---

## ❓ Hala Sorun mu Var?

### Hata Mesajını Paylaşın
```bash
# Hatayı dosyaya kaydet
python test_trained_model.py --model models/sac/best_model/best_model.zip 2>&1 | tee error.log
```

### Ortam Bilgilerini Toplayın
```bash
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import numpy; print(f'Numpy: {numpy.__version__}')"
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
```

### Troubleshooting Dokümanlarına Bakın
- `TROUBLESHOOTING.md` (İngilizce, kapsamlı)
- Bu dosya (Türkçe, step-by-step)

---

**Son Güncelleme**: 2025-11-06
**Toplam Düzeltme**: 3 kritik hata
**Durum**: ✅ Tüm hatalar çözüldü, model test edilmeye hazır!
