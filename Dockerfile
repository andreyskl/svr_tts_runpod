FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# рабочая папка
WORKDIR /app

# базовые пакеты
RUN apt-get update && apt-get install -y \
    python3 python3-pip git ffmpeg wget && \
    rm -rf /var/lib/apt/lists/*

# зависимости
RUN pip3 install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    fastapi uvicorn[standard] gradio soundfile numpy

# клонируем SVR-TTS
RUN git clone https://github.com/Selectorrr/svr_tts.git
WORKDIR /app/svr_tts

# фикс лимита 250 -> 2500
RUN sed -i 's/MAX_TEXT_LEN = 250/MAX_TEXT_LEN = 2500/' */*tokenizer.py || true

# ставим пакет
RUN pip3 install -e .

WORKDIR /app

# создаём api+gradio в одном файле
RUN echo '\
import gradio as gr\n\
from fastapi import FastAPI, UploadFile\n\
from fastapi.responses import FileResponse\n\
import uvicorn, asyncio, os, tempfile, soundfile as sf\n\
from svr_tts import SVRTTS\n\
\n\
app = FastAPI()\n\
model = SVRTTS()\n\
\n\
@app.post("/synthesize")\n\
async def synthesize(text: str):\n\
    wav = model.tts(text)\n\
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")\n\
    sf.write(tmp.name, wav, 22050)\n\
    return FileResponse(tmp.name, media_type="audio/wav")\n\
\n\
def gradio_ui():\n\
    def infer(text):\n\
        wav = model.tts(text)\n\
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")\n\
        sf.write(tmp.name, wav, 22050)\n\
        return tmp.name\n\
    iface = gr.Interface(fn=infer, inputs="text", outputs="audio")\n\
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)\n\
\n\
if __name__ == "__main__":\n\
    loop = asyncio.get_event_loop()\n\
    loop.create_task(uvicorn.run(app, host="0.0.0.0", port=8000))\n\
    gradio_ui()\n\
' > /app/server.py

EXPOSE 8000 7860

CMD ["python3", "server.py"]
