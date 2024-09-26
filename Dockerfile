FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    diffusers==0.30.3 transformers==4.45.0 accelerate==0.34.2 bitsandbytes==0.44.0 sentencepiece==0.2.0 protobuf==5.28.2 && \
    GIT_LFS_SKIP_SMUDGE=1 git clone -b dev https://github.com/camenduru/FLUX-Controlnet-Inpainting /content/outpaint && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/scheduler/scheduler_config.json -d /content/model/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/text_encoder/config.json -d /content/model/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/text_encoder/model.safetensors -d /content/model/text_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/text_encoder_2/config.json -d /content/model/text_encoder_2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/text_encoder_2/model-00001-of-00002.safetensors -d /content/model/text_encoder_2 -o model-00001-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/text_encoder_2/model-00002-of-00002.safetensors -d /content/model/text_encoder_2 -o model-00002-of-00002.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/text_encoder_2/model.safetensors.index.json -d /content/model/text_encoder_2 -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/tokenizer/merges.txt -d /content/model/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/tokenizer/special_tokens_map.json -d /content/model/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/tokenizer/tokenizer_config.json -d /content/model/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/tokenizer/vocab.json -d /content/model/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/tokenizer_2/special_tokens_map.json -d /content/model/tokenizer_2 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/tokenizer_2/spiece.model -d /content/model/tokenizer_2 -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/tokenizer_2/tokenizer.json -d /content/model/tokenizer_2 -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/tokenizer_2/tokenizer_config.json -d /content/model/tokenizer_2 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/transformer/config.json -d /content/model/transformer -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/transformer/diffusion_pytorch_model-00001-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00001-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/transformer/diffusion_pytorch_model-00002-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00002-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/transformer/diffusion_pytorch_model-00003-of-00003.safetensors -d /content/model/transformer -o diffusion_pytorch_model-00003-of-00003.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/transformer/diffusion_pytorch_model.safetensors.index.json -d /content/model/transformer -o diffusion_pytorch_model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/vae/config.json -d /content/model/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/model/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/FLUX.1-dev-diffusers/raw/main/model_index.json -d /content/model -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/FLUX.1-dev-Controlnet-Inpainting-Alpha/raw/main/config.json -d /content/controlnet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/FLUX.1-dev-Controlnet-Inpainting-Alpha/resolve/main/diffusion_pytorch_model.safetensors -d /content/controlnet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/config.json -d /content/llama -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/generation_config.json -d /content/llama -o generation_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00001-of-00004.safetensors -d /content/llama -o model-00001-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00002-of-00004.safetensors -d /content/llama -o model-00002-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00003-of-00004.safetensors -d /content/llama -o model-00003-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/llama/model-00004-of-00004.safetensors -d /content/llama -o model-00004-of-00004.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/model.safetensors.index.json -d /content/llama -o model.safetensors.index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/special_tokens_map.json -d /content/llama -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/tokenizer.json -d /content/llama -o tokenizer.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/llama/tokenizer_config.json -d /content/llama -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/config.json -d /content/siglip -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/siglip/model.safetensors -d /content/siglip -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/preprocessor_config.json -d /content/siglip -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/special_tokens_map.json -d /content/siglip -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/siglip/spiece.model -d /content/siglip -o spiece.model && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/siglip/tokenizer_config.json -d /content/siglip -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/raw/main/adapter/config.yaml -d /content/adapter -o config.yaml && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/joy-caption/resolve/main/adapter/image_adapter.pt -d /content/adapter -o image_adapter.pt

COPY ./worker_runpod.py /content/outpaint/worker_runpod.py
WORKDIR /content/outpaint
CMD python worker_runpod.py