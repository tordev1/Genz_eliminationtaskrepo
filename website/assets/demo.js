/* Live demo page logic for Team Genz.
 *
 * Talks to the Flask app in website/demo_server.py (POST /api/detect_events,
 * multipart form field "video") which imports the repo's own solution.py
 * and returns real solution.detect_events() output as
 * { events: [[start_sec, end_sec, label], ...] }.
 *
 * This is a real network call with a real "processing" state -- there is no
 * fake/canned animation here. If the fetch fails outright (most likely
 * because this page was opened as a file:// document instead of through
 * `python website/demo_server.py`), we say so plainly instead of pretending
 * it worked.
 *
 * Event-class colors are read from the CSS custom properties defined once in
 * assets/style.css (--ev-<label>), so the demo timeline, the results-page
 * mockup and the architecture legend all stay in sync with one palette.
 */
(function () {
  "use strict";

  var MAX_BYTES = 200 * 1024 * 1024; // 200 MB, matches the limit stated on the page
  var MAX_SECONDS = 30; // kept short for this demo's hosting; server re-checks with an actual video probe

  var fileInput = document.getElementById("demo-file");
  var runBtn = document.getElementById("demo-run");
  var statusLine = document.getElementById("demo-status");
  var spinner = document.getElementById("demo-spinner");
  var statusText = document.getElementById("demo-status-text");
  var timelineWrap = document.getElementById("demo-timeline-wrap");
  var timelineAxis = document.getElementById("demo-timeline-axis");
  var timelineTicks = document.getElementById("demo-timeline-ticks");
  var legendEl = document.getElementById("demo-legend");
  var tbody = document.getElementById("demo-table-body");
  var emptyState = document.getElementById("demo-empty-state");

  if (!runBtn) return; // not on this page

  var FALLBACK_COLOR = "#898781";

  function classColor(label) {
    var v = getComputedStyle(document.documentElement)
      .getPropertyValue("--ev-" + label)
      .trim();
    return v || FALLBACK_COLOR;
  }

  function setStatus(mode, text, spinning) {
    statusLine.classList.remove("error", "ok");
    if (mode) statusLine.classList.add(mode);
    statusText.textContent = text;
    spinner.classList.toggle("active", !!spinning);
  }

  function fmt(t) {
    return Number(t).toFixed(2) + "s";
  }

  function resetResults() {
    timelineWrap.classList.remove("visible");
    timelineAxis.innerHTML = "";
    timelineTicks.innerHTML = "";
    legendEl.innerHTML = "";
    tbody.innerHTML = "";
    emptyState.style.display = "none";
  }

  function renderEvents(events) {
    resetResults();

    if (!events || events.length === 0) {
      emptyState.style.display = "block";
      emptyState.textContent =
        "No events detected in this clip (this can be a correct result -- " +
        "e.g. a quiet stretch of road -- not necessarily a failure).";
      return;
    }

    var totalEnd = 0;
    events.forEach(function (e) {
      if (e[1] > totalEnd) totalEnd = e[1];
    });
    if (totalEnd <= 0) totalEnd = 1;

    timelineWrap.classList.add("visible");

    var seenLabels = {};
    events.forEach(function (e) {
      var start = e[0], end = e[1], label = e[2];
      var seg = document.createElement("div");
      seg.className = "timeline-seg";
      var leftPct = (start / totalEnd) * 100;
      var widthPct = Math.max(((end - start) / totalEnd) * 100, 0.4);
      seg.style.left = leftPct + "%";
      seg.style.width = widthPct + "%";
      seg.style.background = classColor(label);
      seg.title = label + ": " + fmt(start) + " – " + fmt(end);
      timelineAxis.appendChild(seg);
      seenLabels[label] = true;
    });

    // ticks: 0, mid, end
    [0, totalEnd / 2, totalEnd].forEach(function (t) {
      var span = document.createElement("span");
      span.textContent = fmt(t);
      timelineTicks.appendChild(span);
    });

    // legend: swatch + text label for every class that actually appears
    // (color never carries meaning alone -- see assets/style.css .legend).
    Object.keys(seenLabels).sort().forEach(function (label) {
      var item = document.createElement("span");
      item.className = "item";
      var sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = classColor(label);
      var txt = document.createElement("span");
      txt.textContent = label;
      item.appendChild(sw);
      item.appendChild(txt);
      legendEl.appendChild(item);
    });

    // table (also serves as the plain-text fallback for the timeline)
    events
      .slice()
      .sort(function (a, b) { return a[0] - b[0]; })
      .forEach(function (e) {
        var tr = document.createElement("tr");

        var tdSwatch = document.createElement("td");
        var sw = document.createElement("span");
        sw.className = "swatch";
        sw.style.display = "inline-block";
        sw.style.background = classColor(e[2]);
        tdSwatch.appendChild(sw);

        var tdLabel = document.createElement("td");
        tdLabel.textContent = e[2];

        var tdStart = document.createElement("td");
        tdStart.textContent = fmt(e[0]);

        var tdEnd = document.createElement("td");
        tdEnd.textContent = fmt(e[1]);

        var tdDur = document.createElement("td");
        tdDur.textContent = fmt(e[1] - e[0]);

        tr.appendChild(tdSwatch);
        tr.appendChild(tdLabel);
        tr.appendChild(tdStart);
        tr.appendChild(tdEnd);
        tr.appendChild(tdDur);
        tbody.appendChild(tr);
      });
  }

  function runDemo() {
    var file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus("error", "Choose an .mp4 file first.", false);
      return;
    }
    if (!/\.mp4$/i.test(file.name)) {
      setStatus("error", "Only .mp4 files are supported by this demo.", false);
      return;
    }
    if (file.size > MAX_BYTES) {
      setStatus(
        "error",
        "That file is " + (file.size / (1024 * 1024)).toFixed(0) +
          "MB; this demo is limited to 200MB.",
        false
      );
      return;
    }

    resetResults();
    runBtn.disabled = true;
    setStatus(
      null,
      "Uploading and running detect_events()… this runs on CPU and can take " +
        "up to a minute or two for a full clip.",
      true
    );

    var fd = new FormData();
    fd.append("video", file, file.name);

    fetch("/api/detect_events", { method: "POST", body: fd })
      .then(function (res) {
        return res.json().then(function (data) {
          return { ok: res.ok, data: data };
        }).catch(function () {
          return { ok: res.ok, data: null };
        });
      })
      .then(function (result) {
        runBtn.disabled = false;
        if (!result.ok || !result.data) {
          var msg =
            (result.data && result.data.error) ||
            "The server returned an error (HTTP status without a JSON body).";
          setStatus("error", msg, false);
          return;
        }
        var events = result.data.events || [];
        setStatus(
          "ok",
          "Done — " + events.length + " event(s) found (Part A only; " +
            "risk-curve visualization: coming soon).",
          false
        );
        renderEvents(events);
      })
      .catch(function () {
        runBtn.disabled = false;
        setStatus(
          "error",
          "Could not reach the demo server. This page needs to be served by " +
            "website/demo_server.py (python website/demo_server.py, then open " +
            "http://localhost:5000) -- opening index.html directly as a file " +
            "will not run the model. See website/README.md.",
          false
        );
      });
  }

  runBtn.addEventListener("click", function (ev) {
    ev.preventDefault();
    runDemo();
  });

  var maxSecondsNote = document.getElementById("demo-max-seconds");
  if (maxSecondsNote) maxSecondsNote.textContent = String(MAX_SECONDS);
})();
