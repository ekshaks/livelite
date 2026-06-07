// trackManager.js
export const TrackManager = {
  pc: null,
  dataChannel: null,

  // transceivers for stable slots
  videoTransceiver: null,
  audioTransceiver: null,

  // currently active tracks
  videoTrack: null,
  audioTrack: null,
  remoteAudioEl: null,

  onDataMessage: null, // callback

// Inside initConnection

  async initConnection(onDataMessage=null) {
    this.pc = new RTCPeerConnection();
    this.onDataMessage = onDataMessage;

    // reserve lanes for audio + video in SDP
    this.audioTransceiver = this.pc.addTransceiver("audio");
    this.videoTransceiver = this.pc.addTransceiver("video");

    // setup data channel
    this.dataChannel = this.pc.createDataChannel("server_text");
    this.dataChannel.onmessage = (e) => {
      console.log("Server:", e.data);
      if (this.onDataMessage) this.onDataMessage(e.data);
    };

    // handle ICE candidates (optional if using trickle ICE later)
    this.pc.onicecandidate = (event) => {
      if (event.candidate) {
        //console.log("ICE candidate:", event.candidate);
      }
    };

    this.pc.ontrack = (event) => {
      if (event.track.kind !== "audio") return;
      console.log("Remote assistant audio track received");
      const stream = event.streams[0] || new MediaStream([event.track]);
      const audioEl = this.ensureRemoteAudioElement();
      audioEl.srcObject = stream;
      audioEl.play().catch((err) => {
        console.warn("Remote audio autoplay blocked:", err);
      });
    };

    // create offer
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    const res = await fetch("/offer", {
      method: "POST",
      body: JSON.stringify(offer),
      headers: { "Content-Type": "application/json" }
    });
    const answer = await res.json();
    await this.pc.setRemoteDescription(answer);
  },

  /** -------------------- VIDEO -------------------- **/

  async startVideo(localVideoEl, useBackCamera=true) {
    if (this.videoTrack) {
      // already active
      this.videoTrack.enabled = true;
      return;
    }
    const constraints = {
      video: {
        //facingMode: useBackCamera ? { exact: "environment" } : "user"
        facingMode: useBackCamera ? "environment" : "user"
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    this.videoTrack = stream.getVideoTracks()[0];
    await this.videoTransceiver.sender.replaceTrack(this.videoTrack);

    // show preview
    localVideoEl.srcObject = new MediaStream([this.videoTrack]);
  },

  async stopVideo(localVideoEl) {
    if (this.videoTrack) {
      await this.videoTransceiver.sender.replaceTrack(null);
      this.videoTrack.stop();
      this.videoTrack = null;
    }
    if (localVideoEl) {
      localVideoEl.srcObject = null;
    }
  },

  /** -------------------- AUDIO -------------------- **/

  async initAudio() {
    if (this.audioTrack) {
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });
    this.audioTrack = stream.getAudioTracks()[0];
    await this.audioTransceiver.sender.replaceTrack(this.audioTrack);
  },

  muteAudio() {
    if (this.audioTrack) {
      this.audioTrack.enabled = false;
    }
  },

  unmuteAudio() {
    if (this.audioTrack) {
      this.audioTrack.enabled = true;
    }
  },

  ensureRemoteAudioElement() {
    if (this.remoteAudioEl) {
      return this.remoteAudioEl;
    }

    const audioEl = document.createElement("audio");
    audioEl.id = "remoteAssistantAudio";
    audioEl.autoplay = true;
    audioEl.playsInline = true;
    audioEl.style.display = "none";
    document.body.appendChild(audioEl);
    this.remoteAudioEl = audioEl;
    return audioEl;
  }
};
