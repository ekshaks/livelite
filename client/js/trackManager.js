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

  async initConnection() {
    this.pc = new RTCPeerConnection();

    // reserve lanes for audio + video in SDP
    this.audioTransceiver = this.pc.addTransceiver("audio");
    this.videoTransceiver = this.pc.addTransceiver("video");

    // setup data channel
    this.dataChannel = this.pc.createDataChannel("server_text");
    this.dataChannel.onmessage = (e) => {
      console.log("Server:", e.data);
    };

    // handle ICE candidates (optional if using trickle ICE later)
    this.pc.onicecandidate = (event) => {
      if (event.candidate) {
        //console.log("ICE candidate:", event.candidate);
      }
    };

    // create offer
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    const res = await fetch("http://localhost:9000/offer", {
      method: "POST",
      body: JSON.stringify(offer),
      headers: { "Content-Type": "application/json" }
    });
    const answer = await res.json();
    await this.pc.setRemoteDescription(answer);
  },

  /** -------------------- VIDEO -------------------- **/

  async startVideo(localVideoEl) {
    if (this.videoTrack) {
      // already active
      this.videoTrack.enabled = true;
      return;
    }
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
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
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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
  }
};
