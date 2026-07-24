import { createInteractions } from "./interaction.js";
import { createMediaControls } from "./mediaControls.js";
import { TrackManager } from "./trackManager.js";
import { createTranscriptView } from "./transcriptView.js";

const UI_API_VERSION = 1;

const elements = {
  appName: document.getElementById("appName"),
  audioButton: document.getElementById("muteAudioBtn"),
  celebration: document.getElementById("celebration"),
  connectionStatus: document.getElementById("connectionStatus"),
  localVideo: document.getElementById("localVideo"),
  minimizeVideoButton: document.getElementById("minimizeVideoBtn"),
  statusText: document.getElementById("statusText"),
  transcript: document.getElementById("transcriptionText"),
  videoButton: document.getElementById("stopVideoBtn"),
  videoPanel: document.getElementById("videoPanel"),
  videoState: document.getElementById("videoState"),
  workspace: document.querySelector(".workspace"),
};

function setStatus(message, state = "connected") {
  elements.statusText.textContent = message;
  elements.connectionStatus.dataset.state = state;
}

async function loadClientConfig() {
  try {
    const response = await fetch("/client-config");
    if (!response.ok) return {};
    const config = await response.json();
    if (typeof config.app_name === "string" && config.app_name.trim()) {
      elements.appName.textContent = config.app_name.trim();
      document.title = config.app_name.trim();
    }
    return config;
  } catch (error) {
    console.warn("Client configuration unavailable", error);
    return {};
  }
}

const transcriptView = createTranscriptView(elements.transcript);
const interactions = createInteractions(elements.celebration);

const messageHandlers = new Map();

function registerMessageHandler(type, handler) {
  if (typeof type !== "string" || typeof handler !== "function") {
    throw new TypeError("registerMessageHandler requires a message type and handler");
  }
  const handlers = messageHandlers.get(type) || [];
  handlers.push(handler);
  messageHandlers.set(type, handlers);
  return () => {
    const remaining = (messageHandlers.get(type) || []).filter(
      (candidate) => candidate !== handler,
    );
    messageHandlers.set(type, remaining);
  };
}

registerMessageHandler("transcript", (message) => transcriptView.append(message));
registerMessageHandler(
  "game_feedback",
  (message) => interactions.handleGameFeedback(message),
);

function loadStylesheet(url) {
  return new Promise((resolve) => {
    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = url;
    stylesheet.onload = resolve;
    stylesheet.onerror = () => {
      console.error(`App stylesheet failed to load: ${url}`);
      resolve();
    };
    document.head.appendChild(stylesheet);
  });
}

async function loadAppUI(config) {
  if (config.ui_stylesheet) {
    await loadStylesheet(config.ui_stylesheet);
  }
  if (!config.ui_module) return;
  if (config.ui_api_version !== UI_API_VERSION) {
    console.error(
      `Unsupported app UI API version: ${config.ui_api_version}`,
    );
    return;
  }

  try {
    const appModule = await import(config.ui_module);
    if (typeof appModule.register !== "function") {
      throw new TypeError("App UI module must export register(context)");
    }
    await appModule.register(Object.freeze({
      appConfig: Object.freeze({ ...config }),
      interactionRoot: elements.celebration,
      showCelebration: interactions.showCelebration,
      registerMessageHandler,
      sendClientMessage: (message) => TrackManager.sendData(message),
      uiApiVersion: UI_API_VERSION,
    }));
  } catch (error) {
    console.error("App UI module failed to register", error);
  }
}

function onDataMessage(rawData) {
  try {
    const message = JSON.parse(rawData);
    const type = message.type || "transcript";
    for (const handler of messageHandlers.get(type) || []) {
      try {
        handler(message);
      } catch (error) {
        console.error(`Client message handler failed for ${type}`, error);
      }
    }
  } catch (error) {
    console.error("Invalid data-channel message", error);
  }
}

createMediaControls({
  trackManager: TrackManager,
  elements,
  onStatus: setStatus,
});

(async () => {
  const config = await loadClientConfig();
  await loadAppUI(config);
  setStatus("Connecting…", "connecting");
  try {
    await TrackManager.initConnection(onDataMessage);
    setStatus("Connected", "connected");
  } catch (error) {
    console.error("Connection failed", error);
    setStatus("Connection failed", "error");
  }
})();
