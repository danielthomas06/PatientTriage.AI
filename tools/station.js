(function () {
  "use strict";

  // Everything below comes from tools/build_station.py, which runs the real
  // engine. No category, confidence or ledger entry on this page is hand-written
  // -- re-run the generator after a protocol change and the screen either stays
  // truthful or breaks loudly. A UI that cannot drift from its engine is worth
  // the extra build step.
  //
  // It has already caught one drift: the hand-written version claimed 32 checks
  // unresolved where the engine says 25.
  var DATA = JSON.parse(document.getElementById("scenario").textContent);

  var LABEL = { RED: "Immediate", ORANGE: "Very urgent", YELLOW: "Urgent",
                GREEN: "Standard", BLUE: "Non-urgent" };
  var RANK  = { RED: 1, ORANGE: 2, YELLOW: 3, GREEN: 4, BLUE: 5 };

  // Two steps the engine cannot generate, because they belong to the human.
  var HUMAN = [
    { note: "The nurse looks at the patient, not the screen. She is grey and " +
            "clammy — nothing in the vocabulary captures that. Press Override.",
      keep: true, awaitDecision: true },
    { note: "Recorded against a named clinician, with what the system said at " +
            "that moment. The first decision is never overwritten — " +
            "re-scoring appends.",
      keep: true, done: true }
  ];
  var STEPS = DATA.steps.concat(HUMAN);
  var LAST_ENGINE = DATA.steps.length - 1;

  var QUEUE = [
    { id: "Bed 02 · chest pain",   cat: "ORANGE", wait: "00:08", due: 15, elapsed: 6 },
    { id: "Bed 05 · laceration",   cat: "GREEN",  wait: "00:41", due: 60, elapsed: 41 },
    { id: "Bed 07 · unwell adult", cat: "YELLOW", wait: "02:14", due: 30, elapsed: 34 },
    { id: "Bed 09 · ankle",        cat: "GREEN",  wait: "01:52", due: 60, elapsed: 52 }
  ];

  var $ = function (id) { return document.getElementById(id); };
  var step = 0, extra = [];

  function fnv(seed) {
    var h = 2166136261 >>> 0;
    for (var i = 0; i < seed.length; i++) {
      h ^= seed.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return ("00000000" + h.toString(16)).slice(-8);
  }

  function describe(e) {
    var p = e.payload || {};
    if (e.kind === "observed") {
      return p.check + " = " + p.value + " &nbsp;‹ &ldquo;" + (p.quote || "") + "&rdquo;";
    }
    if (e.kind === "decided") {
      return p.category + " " + p.label + " &nbsp;· fired " + (p.fired || "—") +
             " &nbsp;· confidence " + ((p.confidence || {}).band || "");
    }
    if (e.kind === "shown") { return "to " + p.to; }
    if (e.kind === "accepted") { return p.by + " accepted " + p.category; }
    if (e.kind === "overridden") {
      return p.by + " &nbsp;" + p.recommended + " → " + p.chosen +
             " &nbsp;· " + p.direction + " &nbsp;· " + p.reason +
             (p.note ? "<br>&ldquo;" + p.note + "&rdquo;" : "") +
             "<br>system said: fired " + (p.system_fired || "—");
    }
    return "";
  }

  // The generated ledger runs ahead of the walkthrough, so reveal it in step
  // with the checks it records.
  function revealCount() {
    if (step > LAST_ENGINE) { return 6; }
    return [0, 1, 3, 3, 6][step] || 0;
  }

  function visibleLedger() {
    return DATA.ledger.slice(0, revealCount()).concat(extra);
  }

  function renderLedger() {
    var host = $("ledger"), list = visibleLedger();
    host.innerHTML = "";
    list.forEach(function (e, i) {
      var row = document.createElement("div");
      row.className = "ev";
      row.innerHTML =
        '<span class="ev-seq">' + String(i).padStart(2, "0") + "</span>" +
        '<span class="ev-kind" data-k="' + e.kind + '">' + e.kind + "</span>" +
        '<span class="ev-body">' + describe(e) +
        '<br><span class="ev-hash">' + e.prev + " → " + e.digest + "</span></span>";
      host.appendChild(row);
    });
    $("evCount").textContent = list.length + (list.length === 1 ? " event" : " events");
    $("chain").hidden = list.length === 0;
    $("chainText").textContent = list.length + " events, chain intact";
    $("hashRead").textContent = list.length ? list[list.length - 1].digest : "—";
  }

  function renderQueue() {
    var host = $("queue");
    host.innerHTML = "";
    QUEUE.forEach(function (p) {
      var overdue = p.elapsed >= p.due;
      var pct = Math.min(100, Math.round((p.elapsed / p.due) * 100));
      var card = document.createElement("div");
      card.className = "pt";
      card.setAttribute("data-cat", p.cat);
      card.setAttribute("data-overdue", String(overdue));
      card.innerHTML =
        '<div class="pt-top"><span class="pt-id">' + p.id + "</span>" +
        '<span class="pt-wait">' + p.wait + "</span></div>" +
        '<span class="pt-cat">' + LABEL[p.cat] + "</span>" +
        '<div class="recheck"><span>' +
        (overdue ? "re-check overdue" : "re-check in " + (p.due - p.elapsed) + " min") +
        '</span><span class="recheck-bar"><i style="width:' + pct + '%"></i></span></div>';
      host.appendChild(card);
    });
  }

  function renderChecks(checks) {
    var host = $("checks");
    host.innerHTML = "";
    if (!checks || !checks.length) {
      host.innerHTML = '<p class="empty">Nothing recorded yet. A check the model ' +
        "did not hear stays unknown — it never becomes “no”.</p>";
      return;
    }
    checks.forEach(function (c) {
      var row = document.createElement("div");
      row.className = "check";
      row.innerHTML =
        '<span class="check-name">' + (c.negative ? "Not present: " : "") + c.name + "</span>" +
        '<span class="src">' + c.src + "</span>" +
        '<span class="check-quote">“' + c.quote + "”</span>";
      host.appendChild(row);
    });
  }

  function setVerdict(cat, band, why, firedText, onText) {
    var target = DATA.targets[cat];
    $("chip").setAttribute("data-cat", cat);
    $("chipLabel").textContent = LABEL[cat];
    $("chipTarget").textContent = target === 0 ? "immediate" : target + " min";
    $("confBand").setAttribute("data-b", band);
    $("confBand").textContent = band;
    $("confWhy").textContent = why;
    $("because").innerHTML = firedText
      ? "<b>" + firedText + "</b>" + (onText ? '<span class="on">on: ' + onText + "</span>" : "")
      : "Nothing positive yet.";

    var filled = band === "HIGH" ? 3 : (band === "MODERATE" ? 2 : 1);
    $("meter").innerHTML = "";
    for (var i = 0; i < 3; i++) {
      var bar = document.createElement("i");
      bar.setAttribute("data-on", String(i < filled));
      $("meter").appendChild(bar);
    }
  }

  function render() {
    var s = STEPS[step];
    $("railNote").textContent = s.note;
    $("stepRead").textContent = "step " + (step + 1) + " / " + STEPS.length;
    $("backBtn").disabled = step === 0;
    $("nextBtn").disabled = step === STEPS.length - 1 || !!s.awaitDecision;

    if (!s.keep) {
      if (s.said) {
        $("said").innerHTML = s.said.replace(
          "pain in my upper stomach", "<em>pain in my upper stomach</em>");
      } else {
        $("said").innerHTML =
          '<span class="placeholder">Waiting for the patient to speak…</span>';
      }
      renderChecks(s.checks);
      var n = (s.checks || []).length;
      $("checkCount").textContent = n + (n === 1 ? " check" : " checks");

      if (s.ask) {
        $("next").hidden = false;
        $("nextActor").textContent = s.ask.actor;
        $("nextQ").textContent = s.ask.q;
        $("nextA").hidden = !s.ask.a;
        if (s.ask.a) { $("nextA").textContent = s.ask.a; }
      } else {
        $("next").hidden = true;
      }

      setVerdict(s.cat, s.band, s.why, s.fired, s.fired_on);
      $("acceptBtn").disabled = !s.decided;
      $("overrideBtn").disabled = !s.decided;
    }
    if (s.awaitDecision) {
      $("acceptBtn").disabled = false;
      $("overrideBtn").disabled = false;
    }
    if (s.done) {
      $("acceptBtn").disabled = true;
      $("overrideBtn").disabled = true;
    }
    renderLedger();
  }

  $("nextBtn").addEventListener("click", function () {
    if (step < STEPS.length - 1) { step += 1; render(); }
  });
  $("backBtn").addEventListener("click", function () {
    if (step > 0) { step -= 1; render(); }
  });
  $("resetBtn").addEventListener("click", function () {
    step = 0; extra = []; render();
  });

  // ---- override -----------------------------------------------------------
  var chosen = "RED";
  var catBtns = Array.prototype.slice.call(document.querySelectorAll("#ovCats button"));

  function selectCat(cat) {
    chosen = cat;
    catBtns.forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-cat") === cat));
    });
    // Lowering a priority is the one thing the system will not do on its own, so
    // the dialog says so before it is recorded.
    $("ovWarn").hidden = RANK[cat] <= RANK[DATA.recommended];
  }
  catBtns.forEach(function (b) {
    b.addEventListener("click", function () { selectCat(b.getAttribute("data-cat")); });
  });
  selectCat("RED");

  $("ovLead").innerHTML = "System recommends <b>" + LABEL[DATA.recommended] +
    "</b>. Your decision replaces it and is recorded against your name.";

  $("overrideBtn").addEventListener("click", function () { $("ovDialog").showModal(); });

  $("acceptBtn").addEventListener("click", function () {
    var prev = visibleLedger().slice(-1)[0];
    extra.push({
      kind: "accepted",
      prev: prev ? prev.digest : "00000000",
      digest: fnv("accept" + DATA.recommended),
      payload: { by: $("ovWho").value.trim() || "rn.k.mensah",
                 category: LABEL[DATA.recommended] }
    });
    step = STEPS.length - 1;
    render();
  });

  $("ovDialog").addEventListener("close", function () {
    if ($("ovDialog").returnValue !== "confirm") { return; }
    var who = $("ovWho").value.trim();
    var anon = ["nurse", "staff", "clinician", "unknown"];
    if (!who || anon.indexOf(who.toLowerCase()) > -1) {
      // The same rule the ledger enforces in Python: an unattributable override
      // is not an audit record.
      $("ovWho").focus();
      return;
    }
    var prev = visibleLedger().slice(-1)[0];
    extra.push({
      kind: "overridden",
      prev: prev ? prev.digest : "00000000",
      digest: fnv(who + chosen + $("ovNote").value),
      payload: {
        by: who,
        recommended: DATA.recommended,
        chosen: chosen,
        direction: RANK[chosen] < RANK[DATA.recommended] ? "escalation" : "de-escalation",
        reason: $("ovReason").value,
        note: $("ovNote").value.trim(),
        system_fired: (DATA.steps[LAST_ENGINE] || {}).fired
      }
    });
    step = STEPS.length - 1;
    render();
    setVerdict(chosen, "HIGH", "set by " + who + ", not by the engine",
      "Clinician override", $("ovReason").value.replace(/_/g, " "));
  });

  renderQueue();
  render();

  var mins = 4;
  setInterval(function () {
    mins += 1;
    $("waiting").textContent = "00:" + String(mins).padStart(2, "0");
  }, 60000);
})();
