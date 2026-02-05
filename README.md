# Multi-Source VLM Fusion for Indoor Geolocation Reliability

This repository contains the official implementation of the paper **"Multi-Source Visual Language Model Fusion for Indoor Geolocation Reliability"**.

## 📄 Overview

Indoor geolocation from images is a complex challenge due to the lack of distinctive landmarks in residential spaces. Black-box deep learning models often provide high-confidence predictions without justification, making them risky for forensic or intelligence applications.

This repository implements an **interactive analysis tool** that validates these black-box predictions. It orchestrates a suite of AI microservices to provide a transparent, evidence-based assessment of an indoor image's location:

1. **Orchestrated Microservices:** A Docker-based system that coordinates specialized agents for **Material Recognition (MATERobot)**, **Architectural Segmentation (LangSAM)**, and **Deep Hashing (GeoAI)**.
2. **VLM-Powered Interpretation:** Uses **LLaVA v1.6** as a cognitive engine to synthesize these technical signals into natural language explanations, offering users interpretable reasoning (e.g., identifying specific window styles or flooring materials) alongside a reliability assessment.
3. **Interactive Investigation:** Provides a web interface where users can upload images, view real-time analysis, and interrogate the system via a chat interface to understand *why* a location was chosen or rejected.

The underlying framework, as detailed in our IEEE Access paper, demonstrates that this multi-source approach achieves **82.2% accuracy at 5% coverage**, significantly outperforming single-source baselines.

---

## 🏗️ System Architecture

The system is built as a set of Dockerized microservices orchestrated via `docker-compose`.

### Core Services

| Service | Container Name | Description |
| --- | --- | --- |
| **LLaVA App** | `geo-llava` | The main orchestrator and VLM inference engine. Handles web routes (`main.py`) and fuses data from other services. |
| **MATERobot** | `geo-materials` | Detects construction materials (wood, paint, tile, etc.) to validate regional architectural patterns. |
| **LangSAM** | `geo-segmentation` | Performs semantic segmentation to identify and count architectural elements (windows, doors, floors, ceilings). |
| **MongoDB** | `geo-mongo` | Stores geolocation feature vectors (Deep Hashing) and metadata for retrieval. |
| **Migration** | `geo-db-migration` | Initializes the MongoDB with feature vectors and label data on startup. |

---

## 🚀 Key Features

* **Broad Classification Support:** The implementation supports classification across **14 countries** spanning Europe, Asia, and Latin America (see *Supported Countries* below).
* **Interpretable Geolocation:** Instead of just a country label, the system provides a breakdown of *why* a location was predicted based on architectural evidence.
* **Material Analysis:** Quantifies material usage.
* **Architectural Segmentation:** Identifies key elements and compares them against a large database of curated elements.
* **Interactive Chat:** Users can ask follow-up questions about the image (e.g., "Why is this not France?") via the LLaVA-integrated interface.

---

## 🌍 Supported Countries

While the paper evaluates the methodology on a strict subset of 6 countries, this codebase comes pre-configured to classify images from **14 distinct countries**:

| Continent | Supported Countries |
| --- | --- |
| **Europe** | Germany, Poland, Norway, France, Hungary |
| **Asia** | Pakistan, Kazakhstan, Japan, South Korea |
| **Latin America** | Bolivia, Chile, Argentina, Colombia, Peru |

---

## 🛠️ Installation & Setup

### Prerequisites

* **NVIDIA GPU** (Required for inference)
* **Docker Desktop** & **Docker Compose**
* **NVIDIA Container Toolkit** (for GPU passthrough to Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/indoor-geolocation-reliability.git
cd indoor-geolocation-reliability

```

### 2. Configuration (Crucial)

Create a `.env` file in the root directory to store your secrets. You can copy the structure below:

```bash
# Create .env file
touch .env

```

**Content of `.env`:**

```env
SECRET_KEY=your_secure_random_key_here
MONGODB_URI=mongodb://mongodb:27017/

```

### 3. Download Model Weights

You must download the pre-trained weights and place them in the `models/` directory. The system expects the following structure:

```text
models/
├── llava-v1.6-mistral-7b/       # LLaVA VLM weights
├── clip-vit-large-patch14-336/  # CLIP encoder
├── grounding-dino-base/         # For LangSAM
├── sam-checkpoints/             # SAM weights
└── hash-models/                 # Custom GeoAI deep hashing models
    ├── indoor-geoai             # Main geolocation model (14 countries)
    └── dhn_model_512bits_36     # Secondary deep hashing model

```
#### GeoAI Hash Models

The `hash-models/` directory contains the custom deep hashing networks used for indoor image geolocation. Pre-trained weights for these models are hosted on Hugging Face:

- **Indoor GeoAI (Primary Model)**  
  Deep hashing geolocation model trained on indoor imagery across **14 countries**.  
  Download: https://huggingface.co/nikokons/indoor-geoai  

- **Secondary Model**  
  Download: https://huggingface.co/nikokons/dhn_model_512bits_36  

  Place the files in:
  ```text
  models/hash-models/indoor-geoai/
```


### 4. Build and Run

Start the entire stack using Docker Compose:

```bash
docker compose up --build -d

```

* The **main API** (Web UI) will be available at: `http://localhost:5006`
* **MATERobot API**: `http://localhost:5001`
* **LangSAM API**: `http://localhost:5002`

---

## 💻 Usage

### Web Interface

1. Navigate to `http://localhost:5006`.
2. Upload an indoor image (supported formats: JPG, PNG).
3. The system will process the image through the pipeline:
* **Step 1:** GeoAI predicts the country.
* **Step 2:** MATERobot & LangSAM extract auxiliary features.
* **Step 3:** LLaVA generates a reliability assessment and description.


4. View the results, including the attention map, material distribution, and architectural segmentation on the unified dashboard.

### API Endpoints (`main.py`)

* **`POST /process`**: Main pipeline entry point. Upload an image to get the full analysis (Prediction + Reliability + Context).

* **`POST /send_message`**: Conversational endpoint for the chat interface.

---

## 📊 Evaluation Dataset

The framework was evaluated in the associated paper using a curated dataset of **2,397 verified residential images** across a representative subset of six countries:

* 🇦🇷 **Argentina**
* 🇨🇱 **Chile**
* 🇫🇷 **France**
* 🇩🇪 **Germany**
* 🇯🇵 **Japan**
* 🇳🇴 **Norway**

---

## 📜 Citation

If you use this code or dataset in your research, please cite our IEEE Access paper:
Paper Link: [https://ieeexplore.ieee.org/abstract/document/11359628/](https://ieeexplore.ieee.org/abstract/document/11359628/)

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

## 👥 Authors

* **Nikolaos Konstantinou** (nkonstantinou@iti.gr)
* **Theodoros Semertzidis**
* **Petros Daras**

*Visual Computing Laboratory, Centre for Research and Technology Hellas (CERTH), Thessaloniki, Greece.*

---

## 🤝 Acknowledgments

This work was supported by the **European Union Horizon-Research and Innovation Framework Programme** through the **VANGUARD Project** under Grant 101121282.