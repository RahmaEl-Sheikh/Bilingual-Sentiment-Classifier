/* script.js — bilingual sentiment analyzer frontend */

const EXAMPLES = [
  { lang: "en", label: "EN · positive", text: "The room was spotless and the staff went out of their way to help us. Would stay again in a heartbeat." },
  { lang: "en", label: "EN · negative", text: "We waited 40 minutes for a table and the food arrived cold. Not coming back." },
  { lang: "ar", label: "AR · إيجابي", text: "الفندق رائع جدا والموظفين متعاونين والغرفة نظيفة، تجربة ممتازة بكل المقاييس" },
  { lang: "ar", label: "AR · سلبي", text: "للأسف الخدمة سيئة جدا والانتظار طويل ولن أكرر التجربة مرة أخرى" },
];

const el = (id) => document.getElementById(id);

function buildExamples() {
  const wrap = el("examples");
  EXAMPLES.forEach((ex) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "example-chip";
    btn.textContent = ex.label;
    if (ex.lang === "ar") btn.lang = "ar";
    btn.addEventListener("click", () => {
      el("text-input").value = ex.text;
      el("text-input").focus();
      runAnalysis();
    });
    wrap.appendChild(btn);
  });
}

async function loadMetrics() {
  try {
    const res = await fetch("/api/metrics");
    if (!res.ok) return;
    const data = await res.json();
    const stats = document.querySelectorAll("#hero-stats .stat-num");
    stats[0].textContent = Math.round(data.en_f1 * 100) + "%";
    stats[1].textContent = Math.round(data.ar_f1 * 100) + "%";
  } catch (e) {
    /* hero still reads fine with the static training-set stat if this fails */
  }
}

function setLoading(isLoading) {
  const btn = el("analyze-btn");
  btn.classList.toggle("is-loading", isLoading);
  btn.disabled = isLoading;
}

function showError(message) {
  el("error-area").hidden = false;
  el("error-text").textContent = message;
  el("result-area").hidden = true;
}

function renderResult(data) {
  el("error-area").hidden = true;
  el("result-area").hidden = false;

  const isPositive = data.sentiment === "positive";
  const stamp = el("verdict-stamp");
  stamp.classList.remove("is-positive", "is-negative");
  stamp.classList.add(isPositive ? "is-positive" : "is-negative");

  el("verdict-en").textContent = isPositive ? "Positive" : "Negative";
  el("verdict-ar").textContent = isPositive ? "إيجابي" : "سلبي";

  const pct = Math.round((data.confidence || 0) * 100);
  el("confidence-value").textContent = pct + "%";
  const fill = el("confidence-fill");
  fill.style.width = pct + "%";
  fill.classList.remove("is-positive", "is-negative");
  fill.classList.add(isPositive ? "is-positive" : "is-negative");

  el("meta-language").textContent = data.language || "—";
  el("meta-model").textContent = data.model_name || (data.language === "Arabic" ? "logistic regression (char n-grams)" : "linear SVM (word n-grams)");

  const cleaned = (data.cleaned_text || "").trim();
  const tokenList = el("token-list");
  tokenList.innerHTML = "";
  if (cleaned) {
    cleaned.split(/\s+/).forEach((tok) => {
      const span = document.createElement("span");
      span.className = "token";
      span.textContent = tok;
      tokenList.appendChild(span);
    });
    const originalWords = data.original_word_count || null;
    el("pipeline-note").textContent = originalWords
      ? `${originalWords} words in your text → ${cleaned.split(/\s+/).length} tokens after removing links, stopwords, and punctuation, then normalizing what's left:`
      : `Tokens the model actually scored, after cleaning and normalization:`;
  } else {
    el("pipeline-note").textContent = "Nothing usable survived cleaning (e.g. only a link or stopwords) — the model returned a default reading.";
  }
}

async function runAnalysis() {
  const text = el("text-input").value.trim();
  el("input-error").hidden = true;

  if (!text) {
    el("input-error").textContent = "Type or paste something to analyze first.";
    el("input-error").hidden = false;
    return;
  }

  setLoading(true);
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong analyzing that text.");
      return;
    }
    data.original_word_count = text.split(/\s+/).filter(Boolean).length;
    renderResult(data);
  } catch (e) {
    showError("Couldn't reach the analysis server. Is app.py still running?");
  } finally {
    setLoading(false);
  }
}

el("analyze-btn").addEventListener("click", runAnalysis);
el("text-input").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") runAnalysis();
});

buildExamples();
loadMetrics();
