# FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y python3 python3-pip libgl1-mesa-glx libglib2.0-0 g++ git wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MINICONDA_SHA256=8eb5999c2f7ac6189690d95ae5ec911032fa6697ae4b34eb3235802086566d78

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /root/miniconda.sh \
    && echo "${MINICONDA_SHA256} /root/miniconda.sh" | sha256sum -c - \
    && if [ -d /opt/conda ]; then /bin/bash /root/miniconda.sh -b -u -p /opt/conda; else /bin/bash /root/miniconda.sh -b -p /opt/conda; fi \
    && rm /root/miniconda.sh \
    && /opt/conda/bin/conda clean -t

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Ensure the LLaVA directory does not already exist
RUN rm -rf /app/LLaVA

# Clone the LLaVA repository
RUN git clone https://github.com/haotian-liu/LLaVA.git /app/LLaVA \
    && cd /app/LLaVA \
    && conda create -n llava python=3.10 -y \
    && echo "source activate llava" > ~/.bashrc \
    && pip install --upgrade pip \
    && pip install -e .

# COPY llava-v1.6-mistral-7b /app/llava-v1.6-mistral-7b
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
EXPOSE 5000

CMD ["python3", "-u", "./app.py"]


