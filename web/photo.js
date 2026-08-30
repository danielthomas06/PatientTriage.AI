"use strict";

// Photo capture, shared by kiosk.js and app.js.
//
// "Take a photo" used to be an <input type=file capture="environment">,
// which only actually opens a camera on a mobile browser -- on a laptop
// (Chrome, Edge, anywhere without that mobile affordance) the `capture`
// attribute is simply ignored and it opens the ordinary file picker, same as
// "Upload a photo" did. Reported directly: on a laptop demo, "Take a photo"
// was just opening a folder. This replaces it with a real in-page camera --
// getUserMedia, a live preview, a Capture button -- so it actually takes a
// photo on the machine these demos are actually run on.
//
// "Upload a photo" stays a plain <input type=file>; only "Take a photo"
// needed to change.

function _buildCameraDialog() {
  const dlg = document.createElement("dialog");
  dlg.className = "camera-dlg";
  dlg.innerHTML = `
    <div class="camera-body">
      <video autoplay playsinline muted></video>
      <canvas hidden></canvas>
      <p class="camera-err" hidden></p>
    </div>
    <div class="camera-actions">
      <button type="button" class="camera-cancel">Cancel</button>
      <button type="button" class="camera-snap primary">Capture</button>
    </div>`;
  document.body.appendChild(dlg);
  return dlg;
}

let _dlg = null;
let _stream = null;

function _stopStream() {
  if (_stream) {
    _stream.getTracks().forEach((t) => t.stop());
    _stream = null;
  }
}

async function _openCamera(onCaptured) {
  if (!_dlg) _dlg = _buildCameraDialog();
  const video = _dlg.querySelector("video");
  const canvas = _dlg.querySelector("canvas");
  const err = _dlg.querySelector(".camera-err");
  err.hidden = true;
  video.hidden = false;
  _dlg.querySelector(".camera-snap").hidden = false;

  const close = () => { _stopStream(); _dlg.close(); };
  _dlg.querySelector(".camera-cancel").onclick = close;
  _dlg.querySelector(".camera-snap").onclick = () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => { close(); if (blob) onCaptured(blob); }, "image/jpeg", 0.85);
  };

  _dlg.showModal();
  try {
    // "environment" (the back/outward camera) if the device has more than
    // one; a laptop with a single webcam just gets that one regardless.
    _stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
    video.srcObject = _stream;
  } catch (e) {
    video.hidden = true;
    _dlg.querySelector(".camera-snap").hidden = true;
    err.textContent = "Couldn't access the camera (" + e.message + ") -- use Upload a photo instead.";
    err.hidden = false;
  }
}

// attachPhotoCapture(takeBtn, uploadBtn, uploadInput, onFile) wires both
// entry points to one callback -- onFile(blobOrFile) -- so the caller has a
// single place to handle "a photo arrived", regardless of which produced it.
// uploadInput is a hidden <input type=file>; uploadBtn is the visible button
// that has to open it -- a hidden file input never opens on its own, a click
// has to be forwarded to it explicitly.
function attachPhotoCapture(takeBtn, uploadBtn, uploadInput, onFile) {
  if (takeBtn) {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      takeBtn.addEventListener("click", () => _openCamera(onFile));
    } else {
      takeBtn.hidden = true;   // no camera API at all here -- upload is the only path
    }
  }
  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener("click", () => uploadInput.click());
    uploadInput.addEventListener("change", () => {
      const file = uploadInput.files[0];
      uploadInput.value = "";
      if (file) onFile(file);
    });
  }
}

// Resize/compress before upload -- works for both a picked file and a
// captured canvas blob, so a multi-MB phone photo (or a full-resolution
// webcam frame) never gets shipped whole to either model tier.
function resizeImage(file, maxDim = 1280, quality = 0.82) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("couldn't process that image")),
        "image/jpeg", quality);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("couldn't read that image")); };
    img.src = url;
  });
}

async function uploadPhoto(ref, blob) {
  const r = await fetch(`/api/photo?ref=${encodeURIComponent(ref)}`, {
    method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob,
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "couldn't add that photo");
  return data;
}
