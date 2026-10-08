FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    python3 \
    python3-pip \
    libfontconfig1 \
    libgl1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Install Godot 4 Headless CLI
RUN wget https://github.com/godotengine/godot/releases/download/4.2.1-stable/Godot_v4.2.1-stable_linux.x86_64.zip \
    && unzip Godot_v4.2.1-stable_linux.x86_64.zip \
    && mv Godot_v4.2.1-stable_linux.x86_64 /usr/local/bin/godot4 \
    && chmod +x /usr/local/bin/godot4 \
    && rm Godot_v4.2.1-stable_linux.x86_64.zip

WORKDIR /workspace
