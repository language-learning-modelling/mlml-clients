apt-get update && apt-get install -y \
  git \
  wget \
  curl \
  unzip \
  vim \
  tmux \
  python3-pip \
  python3-venv &&
  apt-get clean &&
  rm -rf /var/lib/apt/lists/*
