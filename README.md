# Multi-Source VLM Fusion for Indoor Geolocation Reliability

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/abstract/document/11359628/)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)](docker-compose.yml)

Official implementation of **"Multi-Source Visual Language Model Fusion for Indoor Geolocation Reliability"** (IEEE Access 2026).

---

## 📄 Overview

Indoor geolocation from images is challenging due to lack of distinctive landmarks. Black-box deep learning models provide confident predictions without justification—risky for forensic or intelligence applications.

This system validates predictions through **multi-source AI fusion**:

- **Geographic Prediction**: Fine-tuned DeiT-384 vision transformer (Geoai) with deep hashing for country classification (14 countries)
- **Material Recognition**: MATERobot analyzes construction materials to validate regional patterns
- **Architectural Segmentation**: LangSAM (Grounding DINO + SAM) identifies key elements (windows, doors, floors, ceilings)
- **VLM Synthesis**: LLaVA v1.6 generates natural language explanations with reliability assessments
- **Interactive Investigation**: Chat interface for interrogating predictions ("Why not France?")

**Performance**: 82.2% accuracy at 5% coverage across 6 countries  
**Paper**: [IEEE Access](https://ieeexplore.ieee.org/abstract/document/11359628/)

---

## 🏗️ System Architecture

Docker microservices orchestrated via `docker-compose`:

| Service | Container | Description |
|---------|-----------|-------------|
| **Geolocation Engine** | `geo-llava` | Geoai for feature extraction, Faiss similarity search, attention visualization, LLaVA synthesis. Serves web UI + REST API. |
| **Material Recognition** | `geo-materials` | Detects construction materials to validate regional architectural patterns. |
| **Architectural Segmentation** | `geo-segmentation` | Identifies/localizes architectural elements using Grounding DINO and SAM. |
| **Vector Database** | `geo-mongo` | Stores deep hashing feature vectors (128-bit geographic, 512-bit segment codes). |
| **Data Migration** | `geo-db-migration` | Loads pre-computed embeddings into MongoDB on first startup. |

---

## 🌍 Supported Countries

**Paper evaluation** (6 countries): Argentina, Chile, France, Germany, Japan, Norway

**Full implementation** (14 countries):
- **Europe**: Germany, Poland, Norway, France, Hungary
- **Asia**: Pakistan, Kazakhstan, Japan, South Korea  
- **Latin America**: Bolivia, Chile, Argentina, Colombia, Peru

---

## 🛠️ Installation & Setup

**Requirements**: NVIDIA GPU (14GB+ VRAM), Docker

### 1. Clone Repository
```bash
git clone https://github.com/Nkonstan/indoor-geolocalisation.git
cd indoor-geolocalisation
```

### 2. Configuration
```bash
touch .env
```

Add to `.env`:
```env
SECRET_KEY=your_secure_random_key_here
MONGODB_URI=mongodb://mongodb:27017/
```

### 3. Download Models

Required structure:
```
models/
├── llava-v1.6-mistral-7b/       
├── clip-vit-large-patch14-336/  
├── grounding-dino-base/         
├── sam-checkpoints/             
└── hash-models/
    ├── indoor-geoai             
    └── dhn_model_512bits_36     
```

**Download links**:
- llava-v1.6-mistral-7b: https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md
- CLIP: https://huggingface.co/openai/clip-vit-large-patch14-336
- Grounding DINO: https://huggingface.co/IDEA-Research/grounding-dino-base
- SAM: https://github.com/facebookresearch/segment-anything#model-checkpoints
- Indoor GeoAI: https://huggingface.co/nikokons/indoor-geoai
- DHN 512: https://huggingface.co/nikokons/dhn_model_512bits_36

### 4. Build and Run
```bash
docker compose up --build -d
```

**Services**:
- Web Interface: http://localhost:5006
- MATERobot API: http://localhost:5001
- LangSAM API: http://localhost:5002

---

## 💻 Usage

Open http://localhost:5006, upload an indoor image (JPG/PNG). The system:

1. **Predicts country** via indoor-geoai + Faiss search
2. **Extracts evidence** via MATERobot (materials) + LangSAM (architecture)
3. **Generates explanation** via LLaVA v1.6 with reliability score

Results include attention map, material distribution, segmentation, and chat interface.

**REST API**: See `/process` and `/send_message` endpoints for programmatic access.

---

## 📊 Evaluation Dataset

2,397 verified residential images across 6 countries (Argentina, Chile, France, Germany, Japan, Norway).

---

## 📜 Citation
```bibtex
@ARTICLE{Konstantinou2026MultiSource,
  author={Konstantinou, Nikolaos and Semertzidis, Theodoros and Daras, Petros},
  journal={IEEE Access}, 
  title={Multi-Source Visual Language Model Fusion for Indoor Geolocation Reliability}, 
  year={2026},
  volume={14},
  pages={13202-13217},
  doi={10.1109/ACCESS.2026.3656619}
}
```

---

## 👥 Authors

**Nikolaos Konstantinou** (nkonstantinou@iti.gr), Theodoros Semertzidis, Petros Daras  
*Visual Computing Laboratory, CERTH, Thessaloniki, Greece*

---

## 🤝 Acknowledgments

Supported by the **EU Horizon Framework** through **VANGUARD Project** (Grant 101121282).