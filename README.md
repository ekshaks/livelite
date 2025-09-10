# WebRTC based audio video pipelines

## Installation

```bash
git clone https://github.com/ekshaks/livelite.git
cd livelite
pip install -r requirements.txt
```

## Run basic pipeline

```bash
cd livelite
python -m apps.basic_pipeline
```

Open browser at http://localhost:9000. Click on "Start Streaming" and start speaking. 
You should see the transcribed text and the response from the agent.


## Assets

- screenshot of UI



## Feature Support

- [ ] STT APIs 
- [ ] TTS APIs
- [ ] Google GenAI operators
- [ ] Pipecat - stt
- [ ] Livekit