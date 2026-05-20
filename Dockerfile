# 1. Usa a imagem oficial do Python 3.14
FROM python:3.14-slim

# 2. Configurações para o Python 
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Instala as ferramentas de sistema necessarias
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    unzip \
    jq \
    make \
    git \
    chromium \
    chromium-driver \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# 3.1  Força o Chromium a aceitar rodar como root no Docker
RUN mv /usr/bin/chromium /usr/bin/chromium-real && \
    echo '#!/bin/bash\nexec /usr/bin/chromium-real --no-sandbox --disable-dev-shm-usage "$@"' > /usr/bin/chromium && \
    chmod +x /usr/bin/chromium
WORKDIR /app

# 4. INSTALAÇÃO DO SCRIPTLATTES  isolado e fixado em uma versão exata
RUN git clone https://github.com/jpmenachalco/scriptLattes.git tools/scriptLattes && \
    cd tools/scriptLattes && \
    git checkout fb713a1 && \
    make install

# 5. INSTALAÇÃO DAS BIBLIOTECAS 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ====================================================================

# 6.  COPIA O CÓDIGO
# Quando você alterar o start.py ou os arquivos .config, o Docker vai usar o cache 
# de todas as etapas pesadas lá em cima e vai recriar apenas esta etapa!
COPY . /app/

# 7. Trava de setup
RUN touch .setup_concluido.lock

# 8. Comando inicial
CMD ["python", "start.py"]