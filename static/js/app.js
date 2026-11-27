/* Leaf Check front end (port 5005). Saara image processing server par hota hai. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };

  var CAPTIONS = {
    highlighted: "Red outlines mark lesion regions. Tap a numbered pin to see that spot.",
    original: "Photo after resize and noise removal. Numbered pins mark each spot.",
    mask: "White = lesion pixels found by the segmentation step."
  };
  var ALT = {
    highlighted: "Leaf with lesions outlined in red",
    original: "Original leaf photo",
    mask: "Mask of lesion pixels"
  };

  var state = { file: null, data: null, view: "highlighted", token: 0, fresh: true, focus: 0 };

  /* ---------- helpers ---------- */
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function setStatus(t) { $("status").textContent = t || ""; }
  function showError(t) { var e = $("error"); e.textContent = t; e.hidden = false; }
  function hideError() { $("error").hidden = true; }
  function setBusy(b) {
    $("pick").disabled = b;
    $("sample").disabled = b;
    document.body.classList.toggle("busy", b);
  }
  function save(href, name) {
    var a = document.createElement("a");
    a.href = href; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
  }

  /* Bari phone photo chhoti karke bhejo: upload tez, EXIF rotation canvas theek kar deta hai */
  function shrink(file, max) {
    return new Promise(function (resolve) {
      if (!/^image\/(jpeg|png|webp)/.test(file.type) || file.size < 400 * 1024) { resolve(file); return; }
      var url = URL.createObjectURL(file), img = new Image();
      img.onload = function () {
        URL.revokeObjectURL(url);
        var s = Math.min(1, max / Math.max(img.naturalWidth, img.naturalHeight));
        if (s >= 1) { resolve(file); return; }
        var c = document.createElement("canvas");
        c.width = Math.round(img.naturalWidth * s);
        c.height = Math.round(img.naturalHeight * s);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        c.toBlob(function (b) { resolve(b || file); }, "image/jpeg", 0.9);
      };
      img.onerror = function () { URL.revokeObjectURL(url); resolve(file); };
      img.src = url;
    });
  }

  /* ---------- analyse ---------- */
  async function analyze() {
    var my = ++state.token;
    hideError();
    setBusy(true);
    setStatus(state.file ? "Analysing " + state.file.name + "…" : "Analysing sample leaf…");
    try {
      var sens = $("sens").value, clahe = $("clahe").checked ? "1" : "0", res;
      if (state.file) {
        var fd = new FormData();
        fd.append("image", await shrink(state.file, 1280), state.file.name || "leaf.jpg");
        fd.append("sensitivity", sens);
        fd.append("clahe", clahe);
        res = await fetch("/api/analyze", { method: "POST", body: fd });
      } else {
        res = await fetch("/api/sample?sensitivity=" + sens + "&clahe=" + clahe);
      }
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(data.error || "Something went wrong. Please try another photo.");
      if (my !== state.token) return;
      state.data = data;
      render(data);
      if (state.fresh) { addHistory(data); state.fresh = false; }
      setStatus(state.file ? state.file.name : "Sample leaf");
    } catch (e) {
      if (my !== state.token) return;
      showError(e && e.message && e.message !== "Failed to fetch"
        ? e.message
        : "Could not reach the server. Is app.py running on port 5005?");
      setStatus("");
    } finally {
      if (my === state.token) setBusy(false);
    }
  }

  /* ---------- render ---------- */
  function bars(el, rows, emptyText) {
    if (!rows.length) { el.innerHTML = '<p class="empty-note">' + esc(emptyText) + "</p>"; return; }
    el.innerHTML = rows.map(function (r) {
      var v = Math.max(0, Math.min(100, r.value));
      return '<div class="bar"><span class="n">' + esc(r.name) + '</span><span class="t"><i style="width:' + v +
        '%"></i></span><span class="v">' + v.toFixed(1) + "%</span></div>";
    }).join("");
  }

  function renderSpots(d) {
    $("pins").innerHTML = d.pins.map(function (p) {
      return '<button type="button" class="pin t-' + p.type + '" style="left:' + (p.x * 100).toFixed(2) +
        "%;top:" + (p.y * 100).toFixed(2) + '%" data-id="' + p.id + '" aria-label="Spot ' + p.id + ", " +
        esc(p.label) + ", " + p.area + ' percent of leaf"><span>' + p.id + "</span></button>";
    }).join("");
    $("spotRows").innerHTML = d.pins.length
      ? d.pins.map(function (p) {
          return '<tr data-id="' + p.id + '" tabindex="0"><td><span class="num t-' + p.type + '">' + p.id +
            "</span></td><td>" + esc(p.label) + '</td><td class="num2">' + p.area.toFixed(2) + "%</td></tr>";
        }).join("")
      : '<tr><td colspan="3" class="empty">No lesions found. Leaf looks clean.</td></tr>';
    $("spotNote").textContent = d.spot_count > d.pins.length
      ? "Showing the " + d.pins.length + " largest of " + d.spot_count + " spots."
      : d.spot_count + " spot(s) found.";
  }

  function focusSpot(id) {
    state.focus = id;
    document.querySelectorAll(".pin").forEach(function (p) {
      p.classList.toggle("on", +p.getAttribute("data-id") === id);
    });
    document.querySelectorAll("#spotRows tr[data-id]").forEach(function (r) {
      r.classList.toggle("on", +r.getAttribute("data-id") === id);
    });
    var row = id ? document.querySelector('#spotRows tr[data-id="' + id + '"]') : null;
    if (row) row.scrollIntoView({ block: "nearest" });
  }

  function showView(v) {
    state.view = v;
    document.querySelectorAll(".tabs button").forEach(function (b) {
      var on = b.getAttribute("data-view") === v;
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.tabIndex = on ? 0 : -1;
    });
    $("stage").setAttribute("data-view", v);
    $("caption").textContent = CAPTIONS[v];
    if (state.data) $("img").src = state.data.images[v];
    $("img").alt = ALT[v];
  }

  function render(d) {
    $("engine").textContent = d.engine;
    $("engineLine").textContent = d.engine;
    $("label").textContent = d.label;
    $("desc").textContent = d.desc;
    var sev = $("sev");
    sev.textContent = "Severity: " + d.severity;
    sev.setAttribute("data-sev", d.severity);
    $("m-pct").textContent = d.pct.toFixed(1) + "%";
    $("m-health").textContent = d.health + "%";
    $("m-spots").textContent = String(d.spot_count);
    $("m-chl").textContent = d.chlorosis.toFixed(1) + "%";
    $("marker").style.left = Math.min(100, (d.pct / 30) * 100) + "%";
    bars($("preds"), d.predictions.map(function (p) { return { name: p.name, value: p.conf }; }), "No prediction available.");
    bars($("mix"), d.mix.map(function (m) { return { name: m.name, value: m.pct }; }), "No disease colours detected.");
    $("actions").innerHTML = d.actions.map(function (a) { return "<li>" + esc(a) + "</li>"; }).join("");
    renderSpots(d);
    showView(state.view);
    focusSpot(0);
  }

  /* pin ya table row par click: wahi spot focus */
  document.addEventListener("click", function (e) {
    var el = e.target.closest ? e.target.closest("[data-id]") : null;
    if (!el) return;
    if (el.classList.contains("pin") || el.tagName === "TR") {
      var id = +el.getAttribute("data-id");
      focusSpot(state.focus === id ? 0 : id);
    }
  });
  document.addEventListener("keydown", function (e) {
    if ((e.key === "Enter" || e.key === " ") && e.target.tagName === "TR") {
      e.preventDefault();
      focusSpot(+e.target.getAttribute("data-id"));
    }
  });

  /* ---------- input ---------- */
  function choose(file) {
    if (!file) return;
    if (!/^image\//.test(file.type)) { showError("That is not an image. Please choose a JPG or PNG."); return; }
    state.file = file;
    state.fresh = true;
    analyze();
  }
  $("pick").addEventListener("click", function () { $("file").click(); });
  $("file").addEventListener("change", function (e) { choose(e.target.files[0]); e.target.value = ""; });
  $("sample").addEventListener("click", function () { state.file = null; state.fresh = true; analyze(); });

  var drop = $("drop");
  ["dragenter", "dragover"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add("over"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove("over"); });
  });
  drop.addEventListener("drop", function (e) { choose(e.dataTransfer.files[0]); });
  document.addEventListener("paste", function (e) {
    var items = (e.clipboardData && e.clipboardData.items) || [];
    for (var i = 0; i < items.length; i++) {
      if (items[i].type.indexOf("image/") === 0) { choose(items[i].getAsFile()); break; }
    }
  });

  /* settings badalne par usi photo ko dobara analyse karo */
  var timer = 0;
  function again() {
    clearTimeout(timer);
    timer = setTimeout(function () { state.fresh = false; analyze(); }, 250);
  }
  $("sens").addEventListener("input", function () { $("sensVal").textContent = parseFloat(this.value).toFixed(2); again(); });
  $("clahe").addEventListener("change", again);

  /* tabs (arrow keys bhi chalte hain) */
  var tabs = Array.prototype.slice.call(document.querySelectorAll(".tabs button"));
  tabs.forEach(function (b, i) {
    b.addEventListener("click", function () { showView(b.getAttribute("data-view")); });
    b.addEventListener("keydown", function (e) {
      var j = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : -1;
      if (j < 0) return;
      j = (j + tabs.length) % tabs.length;
      tabs[j].focus();
      showView(tabs[j].getAttribute("data-view"));
      e.preventDefault();
    });
  });

  /* ---------- downloads ---------- */
  $("dlImg").addEventListener("click", function () {
    if (state.data) save(state.data.images.annotated, "leaf-pinned.jpg");
  });
  $("csv").addEventListener("click", function () {
    var d = state.data; if (!d) return;
    var rows = [["pin", "type", "area_percent", "x_norm", "y_norm"]];
    d.pins.forEach(function (p) { rows.push([p.id, p.label, p.area, p.x, p.y]); });
    var txt = rows.map(function (r) { return r.join(","); }).join("\n");
    save("data:text/csv;charset=utf-8," + encodeURIComponent(txt), "leaf-spots.csv");
  });
  $("dlRep").addEventListener("click", function () {
    var d = state.data; if (!d) return;
    var list = function (a) { return a.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join(""); };
    var spotRows = d.pins.map(function (p) {
      return "<tr><td>" + p.id + "</td><td>" + esc(p.label) + "</td><td>" + p.area.toFixed(2) + "%</td></tr>";
    }).join("");
    var html = '<!doctype html><meta charset="utf-8"><title>Leaf Check report</title>' +
      '<style>body{font-family:system-ui,Arial,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;color:#10231a}' +
      "img{max-width:100%;border-radius:12px;border:1px solid #ccd}td,th{padding:6px 14px 6px 0;text-align:left}" +
      "h1{margin-bottom:4px}</style>" +
      "<h1>Leaf Check report</h1><p>" + esc(new Date().toLocaleString()) + "</p>" +
      '<img src="' + d.images.annotated + '" alt="Leaf with numbered spot pins">' +
      "<h2>" + esc(d.label) + "</h2><p>" + esc(d.desc) + "</p><table>" +
      "<tr><td>Severity</td><td><b>" + esc(d.severity) + "</b></td></tr>" +
      "<tr><td>Affected area</td><td><b>" + d.pct.toFixed(1) + "%</b></td></tr>" +
      "<tr><td>Health score</td><td><b>" + d.health + "%</b></td></tr>" +
      "<tr><td>Spots</td><td><b>" + d.spot_count + "</b></td></tr>" +
      "<tr><td>Chlorosis area</td><td>" + d.chlorosis.toFixed(1) + "%</td></tr>" +
      "<tr><td>Method</td><td>" + esc(d.engine) + "</td></tr></table>" +
      "<h3>Spots</h3><table><tr><th>Pin</th><th>Type</th><th>Area</th></tr>" + spotRows + "</table>" +
      "<h3>Predictions</h3><ul>" + list(d.predictions.map(function (p) { return p.name + " - " + p.conf.toFixed(1) + "%"; })) + "</ul>" +
      "<h3>Recommended actions</h3><ul>" + list(d.actions) + "</ul>" +
      "<p><small>This is an aid, not a replacement for a crop specialist's advice.</small></p>";
    save("data:text/html;charset=utf-8," + encodeURIComponent(html), "leaf-report.html");
  });

  /* ---------- history (sirf is browser mein, localStorage) ---------- */
  var KEY = "leafcheck.history";
  function loadHist() { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch (e) { return []; } }
  function saveHist(h) { try { localStorage.setItem(KEY, JSON.stringify(h)); } catch (e) { /* ignore */ } }
  var hist = loadHist();

  function drawHist() {
    $("history").hidden = hist.length === 0;
    $("histList").innerHTML = hist.map(function (h) {
      return '<li><img src="' + h.thumb + '" alt=""><div class="t"><b>' + esc(h.label) + "</b><span>" +
        esc(h.pct) + " affected · " + esc(h.spots) + " spots · " + esc(h.sev) + " · " + esc(h.time) +
        "</span></div></li>";
    }).join("");
  }
  function addHistory(d) {
    var img = new Image();
    img.onload = function () {
      var c = document.createElement("canvas");
      c.width = 104; c.height = 104;
      var g = c.getContext("2d");
      g.fillStyle = "#fff"; g.fillRect(0, 0, 104, 104);
      var s = Math.min(104 / img.width, 104 / img.height);
      g.drawImage(img, (104 - img.width * s) / 2, (104 - img.height * s) / 2, img.width * s, img.height * s);
      var t = new Date();
      hist.unshift({
        thumb: c.toDataURL("image/jpeg", 0.7),
        label: d.label,
        pct: d.pct.toFixed(1) + "%",
        spots: String(d.spot_count),
        sev: d.severity,
        time: ("0" + t.getHours()).slice(-2) + ":" + ("0" + t.getMinutes()).slice(-2)
      });
      hist = hist.slice(0, 6);
      saveHist(hist);
      drawHist();
    };
    img.src = d.images.original;
  }
  $("clearHist").addEventListener("click", function () { hist = []; saveHist(hist); drawHist(); });

  /* ---------- start ---------- */
  drawHist();
  state.fresh = false;
  analyze(); /* shuru mein sample leaf dikhao, taaki page khali na lage */
})();
