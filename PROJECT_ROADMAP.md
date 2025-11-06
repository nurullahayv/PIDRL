# PIDRL Project Roadmap - 5 Aşamalı Geliştirme Planı

## 🎯 Proje Vizyonu

Hierarchical Multi-Agent Reinforcement Learning sistemi ile dogfight simülasyonu.

---

## 📋 AŞAMA 1: Temel Competitive 3D Pursuit-Evasion

### Hedef
- Sadece 3D environment (2D sistemler kaldırılacak)
- Agent: RL ile takip (pursuer)
- Target: PID ile kaçış (evader - karenin dışına çıkmaya çalışır)
- Her ikisi de acceleration vectörü üretir
- Relative motion: target_acc - agent_acc

### Yapılacaklar
- [ ] 2D environment ve ilgili dosyaları kaldır
  - `environments/pursuit_evasion_env.py` → SİL
  - `demo.py` → SİL (zaten silindi)
  - 2D PID controllers → SİL
- [ ] 3D environment'ı güncelle
  - Agent ve target için acceleration vektörleri
  - Relative motion hesaplama
- [ ] Target için PID evader controller
  - Amaç: Karenin kenarına git ve dışarı çık
  - Input: Kendi pozisyonu
  - Output: Acceleration vectör (ax, ay, az)
- [ ] Agent için RL training
  - SAC algoritması
  - Reward: Target'ı FOV içinde tut
- [ ] Test ve visualizasyon

### Dosya Yapısı
```
environments/
  └── pursuit_evasion_env_3d.py  (ana environment)
controllers/
  ├── pid_controller_3d.py        (agent için - opsiyonel)
  ├── kalman_pid_controller_3d.py (agent için - opsiyonel)
  └── target_evader_pid.py        (NEW - target için)
agents/
  └── sac_agent.py                (agent için RL)
```

### Başarı Kriteri
- ✅ Agent RL ile target'ı takip edebiliyor
- ✅ Target PID ile kaçabiliyor (dışarı çıkmaya çalışıyor)
- ✅ Competitive reward sistemi çalışıyor
- ✅ Training stabil ve etkili

**Tahmini Süre**: 1-2 gün

---

## 📋 AŞAMA 2: Target için RL Agent (Competitive MARL)

### Hedef
- Target da RL ile öğrenir (SAC)
- İki RL agent birbirine karşı (adversarial training)
- Self-play veya population-based training

### Yapılacaklar
- [ ] Target için RL agent
  - SAC agent (agent ile aynı)
  - Reward: Agent'tan kaç, FOV dışında kal
- [ ] Training pipeline
  - Self-play: İki agent birlikte eğitiliyor
  - Curriculum learning: Kolay → zor
- [ ] Multi-agent training
  - Parallel environments
  - Experience sharing (opsiyonel)
- [ ] Evaluation
  - Agent vs Agent
  - Performance metrics (escape rate, capture rate)

### Dosya Yapısı
```
agents/
  ├── sac_agent.py
  └── multi_agent_trainer.py  (NEW - competitive training)
experiments/
  └── train_competitive.py     (NEW - self-play training)
```

### Başarı Kriteri
- ✅ Her iki agent de RL ile öğreniyor
- ✅ Adversarial training stabil
- ✅ Agent'lar gittikçe gelişiyor (arms race)
- ✅ Win rate ~50% civarında dengelenmiş

**Tahmini Süre**: 2-3 gün

---

## 📋 AŞAMA 3: 3D Arena + Search & Pursuit Modes

### Hedef
- Geniş 3D arena (örn: 1000x1000x1000 birim)
- Uçak modelleri veya küreler
- İki mod:
  1. **Search mode**: Birbirini görmüyorlar, arama yapıyorlar
  2. **Pursuit mode**: FOV'a girince takip başlıyor
- Realistic FOV (cone-based, limited range)

### Yapılacaklar
- [ ] 3D Arena environment
  - Büyük hareket alanı
  - 3D pozisyon ve yönelim (position + orientation)
  - Uçak fizik modeli (yaw, pitch, roll)
- [ ] FOV sistemi
  - Cone-based görüş alanı (azimuth, elevation)
  - Range limitation
  - Visibility check
- [ ] Search behavior
  - Random search pattern
  - Intelligent search (RL-based veya rule-based)
  - Sensor modeling
- [ ] Mode switching
  - Search → Pursuit (target detected)
  - Pursuit → Search (target lost)
- [ ] Visualization
  - 3D rendering (pygame 3D veya OpenGL)
  - Uçak modelleri
  - FOV cone gösterimi

### Dosya Yapısı
```
environments/
  ├── arena_3d.py           (NEW - büyük 3D arena)
  ├── pursuit_mode.py       (mevcut pursuit-evasion refactor)
  └── search_mode.py        (NEW - search behavior)
utils/
  ├── fov_cone.py           (NEW - cone-based FOV)
  ├── aircraft_model.py     (NEW - uçak fizik)
  └── visibility.py         (NEW - visibility check)
rendering/
  ├── renderer_3d.py        (NEW - 3D visualization)
  └── assets/               (NEW - uçak modelleri)
```

### Başarı Kriteri
- ✅ Agent'lar büyük arenada hareket edebiliyor
- ✅ Search mode'da birbirini bulabiliyor
- ✅ Pursuit mode'a geçiş smooth
- ✅ 3D görselleştirme çalışıyor

**Tahmini Süre**: 3-4 gün

---

## 📋 AŞAMA 4: Multi-Agent Dogfight (N vs N)

### Hedef
- Çok sayıda uçak (örn: 4 vs 4 veya free-for-all)
- Her agent diğerlerini kitlemeye çalışır
- Target selection (hangi düşmanı takip edeceğine karar ver)
- Hierarchical decision making
- Her agent için ayrı takip ekranı

### Yapılacaklar
- [ ] Multi-agent environment
  - N agent desteği
  - Global state + local observations
  - Collision detection
- [ ] Target selection
  - High-level policy: Hangi hedefi seç?
  - Factors: Mesafe, görüş açısı, tehdit seviyesi
- [ ] Team coordination (opsiyonel)
  - Communication
  - Formation flying
- [ ] Hierarchical structure
  - High-level: Taktik karar (hangi hedef?)
  - Low-level: Takip kontrolü (pursuit-evasion)
- [ ] Multi-screen visualization
  - Her agent için split-screen
  - 4-6 agent için grid layout
  - Real-time switch between views

### Dosya Yapısı
```
environments/
  └── multi_agent_dogfight.py  (NEW - N vs N)
agents/
  ├── hierarchical_agent.py    (NEW - high + low level)
  └── target_selector.py       (NEW - target selection)
rendering/
  └── multi_view_renderer.py   (NEW - split screen)
experiments/
  └── train_multi_agent.py     (NEW - multi-agent training)
```

### Başarı Kriteri
- ✅ N agent aynı anda çalışıyor
- ✅ Target selection akıllıca yapılıyor
- ✅ Agent'lar engage/disengage kararı verebiliyor
- ✅ Multi-screen görselleştirme çalışıyor

**Tahmini Süre**: 4-5 gün

---

## 📋 AŞAMA 5: Hierarchical RL + No-Fly Zones

### Hedef
- HRL sistemi:
  - High-level: Strateji (attack, evade, patrol, reposition)
  - Mid-level: Taktik (target selection, maneuver type)
  - Low-level: Motor control (pursuit-evasion skills)
- No-fly zones: Hava savunma sistemleri (SAM sites)
- Büyük harita

### Yapılacaklar
- [ ] HRL architecture
  - Options framework veya Feudal RL
  - High-level policy (abstract actions)
  - Low-level policies (primitive skills)
- [ ] Strategic behaviors
  - **Attack**: Düşmana yaklaş ve engage et
  - **Evade**: Tehlikeden kaç
  - **Patrol**: Alanı koru
  - **Reposition**: Avantajlı pozisyon al
- [ ] No-fly zones
  - Yarım küre şeklinde yasak bölgeler
  - SAM sistemi modeling (detection range, firing)
  - Penalty for entering
- [ ] Map design
  - Stratejik noktalar (waypoints)
  - Terrain (opsiyonel - dağlar, vadiler)
  - Multiple no-fly zones
- [ ] Training
  - Curriculum learning (basit → kompleks)
  - Multi-task learning
  - Transfer learning (low-level skills reuse)

### Dosya Yapısı
```
agents/
  ├── hrl_agent.py              (NEW - hierarchical agent)
  ├── high_level_policy.py      (NEW - strategy)
  ├── mid_level_policy.py       (NEW - tactics)
  └── low_level_policies/       (pursuit, evade, etc.)
environments/
  ├── strategic_dogfight.py     (NEW - full system)
  └── no_fly_zones.py           (NEW - SAM systems)
utils/
  └── map_generator.py          (NEW - map creation)
experiments/
  └── train_hrl.py              (NEW - HRL training)
```

### Başarı Kriteri
- ✅ Agent'lar stratejik kararlar verebiliyor
- ✅ No-fly zone'lardan kaçınıyor
- ✅ Hierarchical policies etkili çalışıyor
- ✅ Complex scenarios'larda başarılı

**Tahmini Süre**: 5-7 gün

---

## 📊 Genel Zaman Çizelgesi

| Aşama | Açıklama | Tahmini Süre | Kümülatif |
|-------|----------|--------------|-----------|
| **Aşama 1** | Competitive 3D (RL vs PID) | 1-2 gün | 2 gün |
| **Aşama 2** | Target RL (RL vs RL) | 2-3 gün | 5 gün |
| **Aşama 3** | 3D Arena + Search/Pursuit | 3-4 gün | 9 gün |
| **Aşama 4** | Multi-Agent Dogfight | 4-5 gün | 14 gün |
| **Aşama 5** | HRL + No-Fly Zones | 5-7 gün | 21 gün |

**Toplam**: ~3 hafta (yoğun çalışma ile)

---

## 🔧 Teknik Detaylar

### Ortak Bileşenler
- **RL Algorithm**: SAC (Soft Actor-Critic)
  - Continuous action space
  - Off-policy learning
  - Sample efficient
- **Framework**: Stable-Baselines3
- **Rendering**: Pygame (2D HUD) + OpenGL/PyVista (3D arena)
- **Physics**: Custom (simplified aircraft dynamics)

### Her Aşama İçin
- Training scripts
- Evaluation scripts
- Visualization tools
- Unit tests
- Documentation

---

## 🎓 Öğrenme Değeri

Bu proje şunları öğretir:
1. **Aşama 1-2**: Competitive MARL, self-play
2. **Aşama 3**: State machines, mode switching
3. **Aşama 4**: Multi-agent coordination, target selection
4. **Aşama 5**: Hierarchical RL, strategic decision making

---

## 🚀 Yaklaşım

### Geliştirme Sırası
1. ✅ Her aşamayı sırayla tamamla
2. ✅ Her aşama sonunda test ve validation
3. ✅ Bir sonraki aşamaya geçmeden önce stabil hale getir
4. ✅ Git branch'leri kullan (phase-1, phase-2, etc.)

### Branch Stratejisi
```
main
├── phase-1-competitive-3d
├── phase-2-competitive-marl
├── phase-3-arena-search
├── phase-4-multi-agent
└── phase-5-hrl
```

### Iterative Development
- Her aşamada MVP (Minimum Viable Product) yaklaşımı
- Önce çalışır hale getir, sonra optimize et
- Continuous testing

---

## 📝 Sonraki Adım

**ŞİMDİ**: Aşama 1'i başlatalım!

Onayınızı bekliyorum. Aşama 1'e başlayalım mı?

### Aşama 1 Checklist:
- [ ] 2D dosyaları temizle
- [ ] Target evader PID controller yaz
- [ ] 3D environment'ı güncelle (acceleration vectörleri)
- [ ] Training pipeline kur
- [ ] Test ve demo

**Tahmini Süre**: 1-2 gün
**Başlamak için onay bekliyor**: ✋
