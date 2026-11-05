var PRESETS = {"home": {"tools": [{"name": "set_lights", "description": "Turn lights on or off or dim them.", "parameters": {"type": "object", "properties": {"room": {"type": "string", "description": "Which room."}, "state": {"type": "string", "enum": ["on", "off"]}, "brightness": {"type": "integer", "description": "Percent 1-100."}}, "required": ["room", "state"]}}, {"name": "set_thermostat", "description": "Set the target temperature.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "temperature": {"type": "number"}, "mode": {"type": "string", "enum": ["heat", "cool", "auto"]}}, "required": ["room", "temperature"]}}, {"name": "lock_door", "description": "Lock a door.", "parameters": {"type": "object", "properties": {"door": {"type": "string", "description": "Which door."}}, "required": ["door"]}}], "q": "dim the bedroom lights to 20 percent and lock the front door"}, "robot": {"tools": [{"name": "move", "description": "Drive the robot in a direction.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]}, "distance_m": {"type": "number", "description": "Distance in meters."}}, "required": ["direction", "distance_m"]}}, {"name": "rotate", "description": "Rotate the robot in place.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}, "degrees": {"type": "number"}}, "required": ["direction", "degrees"]}}, {"name": "gripper", "description": "Open or close the gripper.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}}], "q": "move forward 2 meters, turn left 90 degrees, then close the gripper"}, "device": {"tools": [{"name": "open_website", "description": "Open a website in a new tab.", "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "The site to open."}}, "required": ["url"]}}, {"name": "set_theme", "description": "Change the page accent color.", "parameters": {"type": "object", "properties": {"accent_color": {"type": "string", "description": "A color name or hex."}}, "required": ["accent_color"]}}, {"name": "speak", "description": "Say something out loud through the device speakers.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}], "q": "make the accent green, open github.com, and say hello"}, "gallery": {"tools": [{"name": "create_album", "description": "Create a new photo album.", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}, {"name": "move_photos", "description": "Move photos into an album.", "parameters": {"type": "object", "properties": {"album": {"type": "string"}, "filter": {"type": "string", "description": "Which photos, e.g. 'last weekend', 'screenshots'."}}, "required": ["album", "filter"]}}, {"name": "delete_photos", "description": "Delete photos.", "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}, "required": ["filter"]}}], "q": "create an album called Summer 2026 and move the photos from last weekend into it"}, "email": {"tools": [{"name": "send_email", "description": "Send an email.", "parameters": {"type": "object", "properties": {"to": {"type": "string", "description": "Email address."}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"]}}], "q": "send the receipt to finance@cactus.dev with subject expenses: Blue Bottle Coffee, $14.50, August 4th"}, "currency": {"tools": [{"name": "convert_currency", "description": "Convert an amount between currencies.", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string", "description": "ISO code, e.g. USD."}, "to_currency": {"type": "string", "description": "ISO code, e.g. EUR."}}, "required": ["amount", "from_currency", "to_currency"]}}], "q": "how much is 250 dollars in euros"}, "document": {"tools": [{"name": "record_booking", "description": "Record the details of a hotel booking.", "parameters": {"type": "object", "properties": {"hotel": {"type": "string"}, "confirmation_number": {"type": "string"}, "guest_name": {"type": "string"}, "room_type": {"type": "string"}, "check_in": {"type": "string"}, "check_out": {"type": "string"}, "total": {"type": "number"}, "currency": {"type": "string"}}, "required": ["confirmation_number", "check_in", "check_out", "total"]}}], "q": "extract the booking from this email: Dear Mr Okafor, thank you for choosing the Harbor Light Hotel. This email confirms reservation HL-88213 for a deluxe sea-view room, checking in on 14 September 2026 and checking out on 18 September 2026. The total for your stay is 642.50 euros, payable at checkout. Breakfast is included from 7am, and our shuttle meets the 9:40 ferry on request. We look forward to welcoming you. Warm regards, Reception."}, "sentiment": {"tools": [{"name": "classify_sentiment", "description": "Classify the sentiment of a message.", "parameters": {"type": "object", "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral", "mixed"]}}, "required": ["sentiment"]}}], "q": "classify the sentiment of this message: this is the worst purchase I have ever made, it broke in a day and support ignored me"}, "crowded": {"tools": [{"name": "set_lights", "description": "Control lights.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "state": {"type": "string", "enum": ["on", "off"]}}, "required": ["room", "state"]}}, {"name": "set_thermostat", "description": "Set temperature.", "parameters": {"type": "object", "properties": {"temperature": {"type": "number"}}, "required": ["temperature"]}}, {"name": "lock_door", "description": "Lock a door.", "parameters": {"type": "object", "properties": {"door": {"type": "string"}}, "required": ["door"]}}, {"name": "play_music", "description": "Play music.", "parameters": {"type": "object", "properties": {"genre": {"type": "string"}}, "required": []}}, {"name": "set_timer", "description": "Set a timer.", "parameters": {"type": "object", "properties": {"time_human": {"type": "string"}}, "required": ["time_human"]}}, {"name": "send_message", "description": "Send a text.", "parameters": {"type": "object", "properties": {"recipient": {"type": "string"}, "message": {"type": "string"}}, "required": ["recipient", "message"]}}, {"name": "create_note", "description": "Create a note.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}, {"name": "start_vacuum", "description": "Start the robot vacuum.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}}, "required": []}}, {"name": "water_plants", "description": "Run the irrigation system.", "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": []}}, {"name": "open_blinds", "description": "Open or close the blinds.", "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "position": {"type": "string", "enum": ["open", "closed", "half"]}}, "required": ["room", "position"]}}, {"name": "preheat_oven", "description": "Preheat the oven.", "parameters": {"type": "object", "properties": {"temperature_c": {"type": "number"}}, "required": ["temperature_c"]}}, {"name": "charge_car", "description": "Start charging the car.", "parameters": {"type": "object", "properties": {"limit_percent": {"type": "integer"}}, "required": []}}], "q": "close the bedroom blinds halfway"}, "repeat": {"tools": [{"name": "add_to_list", "description": "Add one item to the shopping list.", "parameters": {"type": "object", "properties": {"item": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["item"]}}], "q": "add 2 milk and 6 eggs to the shopping list"}, "array": {"tools": [{"name": "add_items", "description": "Add items to the shopping list.", "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}, "required": ["items"]}}], "q": "add milk, eggs, and bread to the shopping list"}, "flight": {"tools": [{"name": "search_flights", "description": "Search for flights.", "parameters": {"type": "object", "properties": {"from_city": {"type": "string"}, "to_city": {"type": "string"}, "date": {"type": "string"}, "direct_only": {"type": "boolean", "description": "Only non-stop flights."}, "passengers": {"type": "integer"}}, "required": ["from_city", "to_city"]}}], "q": "find non-stop flights from lagos to nairobi on december 3rd for two passengers"}, "refuse": {"tools": [{"name": "move", "description": "Drive the robot in a direction.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["forward", "backward", "left", "right"]}, "distance_m": {"type": "number", "description": "Distance in meters."}}, "required": ["direction", "distance_m"]}}, {"name": "rotate", "description": "Rotate the robot in place.", "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}, "degrees": {"type": "number"}}, "required": ["direction", "degrees"]}}, {"name": "gripper", "description": "Open or close the gripper.", "parameters": {"type": "object", "properties": {"action": {"type": "string", "enum": ["open", "close"]}}, "required": ["action"]}}], "q": "write me a short poem about the moon"}};
var LABELS = {"home": "Smart home", "robot": "Robot", "gallery": "Gallery", "device": "Device control", "email": "Extract email", "currency": "Currency", "document": "Document extraction", "sentiment": "Sentiment", "crowded": "12 tools routing", "repeat": "Repeated calls", "array": "Array argument", "flight": "Flight form", "refuse": "Off-topic refusal"};

var _toastTimer = null;

function showError(msg) {
  if (_toastTimer) clearTimeout(_toastTimer);
  document.getElementById("toastMsg").textContent = msg;
  document.getElementById("toast").classList.add("visible");
  _toastTimer = setTimeout(dismissToast, 8000);
}

function dismissToast() {
  document.getElementById("toast").classList.remove("visible");
}

function loadToolsFile(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function () {
    try {
      JSON.parse(reader.result);
      document.getElementById("tools").value = reader.result;
      newChat();
    } catch (e) {
      showError("Invalid JSON file");
    }
  };
  reader.readAsText(file);
  input.value = "";
}

function fetchModelName() {
  fetch("/model").then(function (r) { return r.json(); })
    .then(function (d) { document.getElementById("modelName").textContent = d.name || ""; })
    .catch(function () {});
}

function loadModelFile(input) {
  var file = input.files[0];
  if (!file) return;
  var name = document.getElementById("modelName");
  name.textContent = "loading " + file.name + "...";
  fetch("/load-model", { method: "POST", headers: { "X-Filename": file.name }, body: file })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (d.error) { showError(d.error); fetchModelName(); }
      else { name.textContent = d.name; newChat(); }
    })
    .catch(function (e) { showError("Upload failed: " + e.message); fetchModelName(); });
  input.value = "";
}

var conversation = document.getElementById("conversation");
var emptyState = document.getElementById("emptyState");

function addTurn(query, data) {
  if (emptyState) { emptyState.remove(); emptyState = null; }
  var turn = document.createElement("div");
  turn.className = "turn";

  var q = document.createElement("div");
  q.className = "turn-query";
  q.textContent = query;
  turn.appendChild(q);

  var pre = document.createElement("pre");
  pre.className = "turn-result";
  var calls = data.function_calls;
  if (data.type === "refuse" || (Array.isArray(calls) && calls.length === 0)) {
    pre.classList.add("refused");
    pre.textContent = "no tool call (off-topic / refused)";
  } else {
    pre.textContent = JSON.stringify(calls || data, null, 2);
  }
  turn.appendChild(pre);

  if (data.reasoning) {
    var reason = document.createElement("div");
    reason.className = "turn-reasoning";
    var label = document.createElement("div");
    label.className = "turn-reasoning-label";
    label.textContent = "Reasoning trace";
    var body = document.createElement("div");
    body.className = "turn-reasoning-body";
    body.textContent = data.reasoning;
    reason.appendChild(label);
    reason.appendChild(body);
    turn.appendChild(reason);
  }

  var bits = [];
  if (data.confidence !== undefined && data.confidence !== null)
    bits.push("confidence " + Number(data.confidence).toFixed(4));
  if (data.decode_tps) bits.push(Math.round(data.decode_tps) + " tok/s");
  if (bits.length) {
    var meta = document.createElement("div");
    meta.className = "turn-meta";
    meta.textContent = bits.join("  ·  ");
    turn.appendChild(meta);
  }

  conversation.appendChild(turn);
  conversation.scrollTop = conversation.scrollHeight;
}

async function send() {
  var input = document.getElementById("query");
  var btn = document.getElementById("sendBtn");
  var query = input.value.trim();
  if (!query) return;
  var tools = document.getElementById("tools").value.trim() || "[]";
  try { JSON.parse(tools); } catch (e) { showError("Invalid tools JSON"); return; }

  input.disabled = true;
  btn.disabled = true;
  try {
    var r = await fetch("/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, tools: tools }),
    });
    var data = await r.json();
    if (data.error) { showError(data.error); }
    else { addTurn(query, data); input.value = ""; }
  } catch (e) {
    showError("Request failed: " + e.message);
  } finally {
    input.disabled = false;
    btn.disabled = false;
    input.focus();
  }
}

function newChat() {
  fetch("/reset", { method: "POST" }).catch(function () {});
  conversation.innerHTML = "";
  emptyState = document.createElement("div");
  emptyState.className = "empty";
  emptyState.id = "emptyState";
  emptyState.textContent = "Pick a preset or type a query, then Run.";
  conversation.appendChild(emptyState);
}

function applyPreset(key) {
  var p = PRESETS[key];
  document.getElementById("tools").value = JSON.stringify(p.tools, null, 2);
  document.getElementById("query").value = p.q;
  newChat();
  document.getElementById("query").focus();
}

var presetBox = document.getElementById("presets");
Object.keys(LABELS).forEach(function (key) {
  if (!PRESETS[key]) return;
  var b = document.createElement("button");
  b.className = "preset-btn";
  b.textContent = LABELS[key];
  b.onclick = function () { applyPreset(key); };
  presetBox.appendChild(b);
});

function togglePanel() {
  var sidebar = document.getElementById("sidebar");
  sidebar.classList.toggle("open");
  document.querySelector(".tools-toggle span").textContent =
    sidebar.classList.contains("open") ? "Query" : "Tools";
}

var _pollTimer = null;
var _ftRunning = false;

function openFinetuneModal() {
  var tools = document.getElementById("tools").value.trim() || "[]";
  try { JSON.parse(tools); } catch (e) { showError("Invalid tools JSON"); return; }
  _resetModal();
  document.getElementById("modalOverlay").classList.add("visible");
  document.getElementById("ftApiKey").focus();
}

function closeModal(e) {
  if (e && e.target && e.target !== document.getElementById("modalOverlay")) return;
  if (_ftRunning) return;
  document.getElementById("modalOverlay").classList.remove("visible");
}

function _resetModal() {
  document.getElementById("ftSteps").classList.remove("visible");
  document.getElementById("ftProgress").textContent = "";
  var start = document.getElementById("ftStartBtn");
  start.disabled = false;
  start.textContent = "Start Finetune";
  start.style.display = "";
  document.getElementById("modalCloseBtn").style.display = "";
  var dl = document.getElementById("ftDownload");
  if (dl) dl.remove();
  document.querySelectorAll(".modal-step").forEach(function (s) {
    s.classList.remove("active", "done");
  });
}

async function startFinetune() {
  var apiKey = document.getElementById("ftApiKey").value.trim();
  if (!apiKey) { showError("OpenRouter API key is required"); return; }
  var tools = document.getElementById("tools").value.trim() || "[]";
  try { JSON.parse(tools); } catch (e) { showError("Invalid tools JSON"); return; }
  var samples = parseInt(document.getElementById("ftSamples").value, 10) || 200;

  var btn = document.getElementById("ftStartBtn");
  btn.disabled = true;
  btn.textContent = "Starting...";
  document.getElementById("modalCloseBtn").style.display = "none";
  document.getElementById("ftSteps").classList.add("visible");
  _ftRunning = true;

  try {
    var r = await fetch("/finetune", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tools: tools, api_key: apiKey, samples: samples }),
    });
    var data = await r.json();
    if (data.error) { showError(data.error); _ftRunning = false; _resetModal(); return; }
    _pollTimer = setInterval(pollFinetune, 2000);
  } catch (e) {
    showError("Request failed: " + e.message);
    _ftRunning = false;
    _resetModal();
  }
}

function _updateSteps(current) {
  var steps = document.querySelectorAll(".modal-step");
  var past = true;
  steps.forEach(function (s) {
    var name = s.getAttribute("data-step");
    s.classList.remove("active", "done");
    if (name === current) { s.classList.add("active"); past = false; }
    else if (past) { s.classList.add("done"); }
  });
}

var _stepLabels = {
  "generating data": "Generating data...",
  "training": "Training...",
  "building": "Building .cact...",
};

async function pollFinetune() {
  try {
    var r = await fetch("/finetune/status");
    var data = await r.json();
    _updateSteps(data.step);
    var btn = document.getElementById("ftStartBtn");
    btn.textContent = _stepLabels[data.step] || data.step;
    if (data.log && data.log.length)
      document.getElementById("ftProgress").textContent = data.log[data.log.length - 1];

    if (!data.running) {
      clearInterval(_pollTimer);
      _pollTimer = null;
      _ftRunning = false;
      document.getElementById("modalCloseBtn").style.display = "";
      if (data.step === "done") {
        document.querySelectorAll(".modal-step").forEach(function (s) {
          s.classList.remove("active"); s.classList.add("done");
        });
        btn.style.display = "none";
        fetchModelName();
        if (data.checkpoint) {
          var dl = document.createElement("a");
          dl.id = "ftDownload";
          dl.className = "modal-download";
          dl.href = "/download/" + data.checkpoint;
          dl.download = data.checkpoint;
          dl.textContent = "Download " + data.checkpoint;
          document.getElementById("ftFooter").appendChild(dl);
        }
      } else {
        showError("Finetune failed — " + (data.error || "unknown error"));
        btn.textContent = "Retry";
        btn.disabled = false;
      }
    }
  } catch (e) {
    clearInterval(_pollTimer);
    _pollTimer = null;
    _ftRunning = false;
    document.getElementById("modalCloseBtn").style.display = "";
    showError("Lost connection to server");
    document.getElementById("ftStartBtn").textContent = "Retry";
    document.getElementById("ftStartBtn").disabled = false;
  }
}

document.getElementById("query").addEventListener("keydown", function (e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
document.addEventListener("keydown", function (e) { if (e.key === "Escape") closeModal(); });

(function () {
  var handle = document.getElementById("resizeHandle");
  var sidebar = document.getElementById("sidebar");
  var dragging = false;
  handle.addEventListener("mousedown", function (e) {
    if (window.innerWidth <= 768) return;
    e.preventDefault();
    dragging = true;
    handle.classList.add("active");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    sidebar.style.width = Math.min(Math.max(e.clientX, 200), window.innerWidth * 0.6) + "px";
  });
  window.addEventListener("mouseup", function () {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("active");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
  });
  window.addEventListener("resize", function () {
    if (window.innerWidth <= 768) sidebar.style.width = "";
  });
})();

fetchModelName();
applyPreset("home");
