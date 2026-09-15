"use strict";
const DATA = JSON.parse(document.getElementById("review-data").textContent);
const blind = DATA.task === "blind", $ = id => document.getElementById(id);
const clone = value => JSON.parse(JSON.stringify(value));
const storageKey = "mbh-review-v1-" + DATA.task + "-" + DATA.dataset_id;
const source = new Map(DATA.rows.map(r => [r.key, r]));
const groups = blind ? DATA.rows.map(r => [r]) : [...new Set(DATA.rows.map(r => r.appellation))]
  .map(name => DATA.rows.filter(r => r.appellation === name));
let review = {schema_version: 1, task: DATA.task, dataset_id: DATA.dataset_id, sheet_id: DATA.sheet_id,
  annotator: "", submitted: false, attestation: false, rows: DATA.rows.map(r => ({key: r.key,
    source: clone(r), status: "undecided", notes: "", ...(blind ? {triples: []} : {corrected_referent: ""})}))};
let position = 0, dirty = false, timer, blockedStorage = false, confirmAction = null;
let byKey = new Map(review.rows.map(r => [r.key, r]));

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  if (text !== undefined) node.textContent = text;
  return node;
}
function append(parent, ...nodes) {parent.append(...nodes); return parent;}
function field(label, input) {return append(el("label", {class: "field-label"}), el("span", {}, label), input);}
function msg(text, error = false) {$('message').textContent = text; $('message').classList.toggle("error", error);}
function complete(r) {return blind ? r.status === "none" || r.status === "triples" && r.triples.length > 0 &&
  r.triples.every(t => [t.subject, t.predicate, t.object].every(x => x.trim())) : r.status !== "undecided";}
function canonical(v) {return JSON.stringify(v, function(k, x) {return x && typeof x === "object" && !Array.isArray(x) ?
  Object.fromEntries(Object.keys(x).sort().map(key => [key, x[key]])) : x;});}
function keys(value, expected) {return value && !Array.isArray(value) && typeof value === "object" &&
  Object.keys(value).sort().join("|") === [...expected].sort().join("|");}
function check(incoming, requireSubmitted = false) {
  if (!keys(incoming, ["schema_version", "task", "dataset_id", "sheet_id", "annotator", "submitted", "attestation", "rows"]))
    throw Error("Unrecognised review format. Current draft preserved.");
  for (const k of ["schema_version", "task", "dataset_id", "sheet_id"])
    if (incoming[k] !== DATA[k]) throw Error("This file belongs to a different task, dataset, or version.");
  if (typeof incoming.annotator !== "string" || typeof incoming.submitted !== "boolean" || typeof incoming.attestation !== "boolean")
    throw Error("Invalid review metadata.");
  if (!Array.isArray(incoming.rows) || incoming.rows.length !== DATA.rows.length) throw Error("The file is missing source records.");
  const seen = new Set();
  for (const r of incoming.rows) {
    if (!keys(r, ["key", "source", "status", "notes", blind ? "triples" : "corrected_referent"]) ||
        !source.has(r.key) || seen.has(r.key) || canonical(r.source) !== canonical(source.get(r.key)) || typeof r.notes !== "string")
      throw Error("Duplicate, missing, or changed source record. Import rejected.");
    seen.add(r.key);
    if (blind) {
      if (!["undecided", "triples", "none"].includes(r.status) || !Array.isArray(r.triples)) throw Error("Invalid sentence decision.");
      for (const t of r.triples) {
        if (!keys(t, ["subject", "predicate", "object", "unclear"]) || typeof t.unclear !== "boolean" ||
            ![t.subject, t.predicate, t.object].every(x => typeof x === "string") ||
            t.predicate !== "" && !Object.hasOwn(DATA.predicates, t.predicate)) throw Error("Invalid triple or predicate.");
      }
      if (r.status === "none" && r.triples.length) throw Error("A no-relation decision contains triples.");
    } else if (!["undecided", "yes", "no", "not_a_person", "unclear"].includes(r.status) || typeof r.corrected_referent !== "string")
      throw Error("Invalid candidate decision.");
  }
  if ((incoming.submitted || requireSubmitted) && (!incoming.submitted || !incoming.attestation ||
      !incoming.annotator.trim() || !incoming.rows.every(complete))) throw Error("Complete all decisions and supply your name and attestation first.");
  return incoming;
}
function save() {
  if (blockedStorage) return;
  try {localStorage.setItem(storageKey, JSON.stringify({review, position})); $('save-label').textContent = "Draft recovered on this browser";}
  catch (_) {$('save-label').textContent = "Recovery unavailable; export your draft";}
}
function edited() {review.submitted = false; review.attestation = false; dirty = true; clearTimeout(timer); timer = setTimeout(save, 180); updateProgress();}
function updateProgress() {
  const n = review.rows.filter(complete).length;
  $('count').replaceChildren(document.createTextNode(n + " "), el("span", {}, "/ " + review.rows.length + " reviewed"));
  $('progress').max = review.rows.length; $('progress').value = n;
  $('progress-note').textContent = (review.rows.length - n) + " remaining";
  $('final-state').textContent = review.submitted ? "Review submitted for export" : "Draft / awaiting review";
  const done = groups[position].every(s => complete(byKey.get(s.key)));
  $('record-state').textContent = done ? "Reviewed" : "Unfinished";
  $('record-state').classList.toggle("warn", !done);
}
function choices(r, options, changed) {
  const box = el("div", {class: "decisions", role: "group", "aria-label": blind ? "Sentence decision" : "Decision for " + r.source.candidate_entity});
  for (const [value, label] of options) {
    const input = el("input", {type: "radio", name: "decision-" + r.key, value, checked: r.status === value,
      onchange: () => changed(value)});
    box.append(append(el("label"), input, document.createTextNode(label)));
  }
  return box;
}
function note(r) {return append(el("div", {class: "notes"}), field("Evidence notes", el("textarea", {
  value: r.notes, placeholder: "Uncertainty, conflicting usage, or a passage-specific qualification", oninput: e => {r.notes = e.target.value; edited();}})));}
function triple(r, t, i) {
  const box = el("div", {class: "triple"});
  const remove = el("button", {class: "icon", title: "Remove triple " + (i + 1), "aria-label": "Remove triple " + (i + 1),
    onclick: () => {r.triples.splice(i, 1); edited(); render();}}, "\u00d7");
  box.append(append(el("div", {class: "triple-top"}), el("strong", {class: "small"}, "RELATION " + (i + 1)), remove));
  const names = el("div", {class: "pair"});
  for (const [key, label] of [["subject", "Subject"], ["object", "Object"]]) names.append(field(label, el("input", {
    value: t[key], placeholder: "Name or description in the passage", "aria-label": label + " " + (i + 1),
    oninput: e => {t[key] = e.target.value; edited();}})));
  const help = el("p", {class: "predicate-help"}, DATA.predicates[t.predicate] || "Choose the relation stated in this passage.");
  const select = el("select", {"aria-label": "Predicate " + (i + 1), onchange: e => {
    t.predicate = e.target.value; help.textContent = DATA.predicates[t.predicate] || "Choose a relation."; edited();}});
  select.append(el("option", {value: ""}, "Select relation"));
  for (const p of Object.keys(DATA.predicates)) select.append(el("option", {value: p}, p));
  select.value = t.predicate;
  box.append(names, append(el("div", {class: "predicate"}), field("Predicate", select), help),
    append(el("label", {class: "check"}), el("input", {type: "checkbox", checked: t.unclear,
      onchange: e => {t.unclear = e.target.checked; edited();}}), document.createTextNode("Referent or relation unclear")));
  return box;
}
function citation(s) {return s.parva + (s.adhyaya === null || s.adhyaya === "" ? " / chapter uncited" : " / chapter " + s.adhyaya);}
function blindWork(s) {
  const r = byKey.get(s.key), work = el("div", {class: "work"});
  const text = append(el("article", {class: "passage"}), el("div", {class: "citation"}, citation(s)), el("p", {class: "prose"}, s.sentence));
  const decisions = el("section", {class: "decision"});
  decisions.append(el("h2", {}, "What does this passage state?"), choices(r,
    [["undecided", "Undecided"], ["triples", "Relations stated"], ["none", "No relation"]], value => {
      if (value === "none" && r.triples.length) {
        render(); confirm("Clear these relations?", "Marking this passage as no relation will remove its existing triples.", () => {
          r.triples = []; r.status = "none"; edited(); render();}); return;
      }
      r.status = value;
      if (value === "triples" && !r.triples.length) r.triples.push({subject: "", predicate: "", object: "", unclear: false});
      edited(); render();
    }));
  r.triples.forEach((t, i) => decisions.append(triple(r, t, i)));
  if (r.status !== "none") decisions.append(el("button", {onclick: () => {
    r.status = "triples"; r.triples.push({subject: "", predicate: "", object: "", unclear: false}); edited(); render();
  }}, "+ Add relation"));
  decisions.append(note(r)); return append(work, text, decisions);
}
function evidence(s) {
  const box = el("div", {class: "evidence"});
  return append(box, el("div", {class: "citation"}, citation(s)), el("p", {class: "prose"}, s.sentence));
}
function candidate(s) {
  const r = byKey.get(s.key), box = el("section", {class: "candidate"});
  const heading = append(el("div"), el("h2", {}, s.candidate_entity), el("div", {class: "small muted"}, "Candidate referent"));
  box.append(append(el("div", {class: "candidate-header"}), heading,
    el("span", {class: "badge" + (s.sightings.length ? "" : " warn")}, s.sightings.length ? s.total_sightings + " sightings" : "uncited")));
  const passages = el("div");
  if (!s.sightings.length) passages.append(el("p", {class: "muted"}, "No supporting apposition passage is available for this pair."));
  else {s.sightings.slice(0, 3).forEach(sighting => passages.append(evidence(sighting)));
    if (s.sightings.length > 3) {const more = append(el("details"), el("summary", {}, "More passages (" + (s.sightings.length - 3) + ")"));
      s.sightings.slice(3).forEach(sighting => more.append(evidence(sighting))); passages.append(more);}
    if (s.total_sightings > s.sightings.length) passages.append(el("p", {class: "small muted"}, "Showing " + s.sightings.length + " of " + s.total_sightings + " available sightings."));}
  const labels = append(el("details"), el("summary", {}, "Source labels (unverified)"), el("p", {},
    s.appellation + ": " + (s.appellation_label || "unlabelled") + "; " + s.candidate_entity + ": " + (s.entity_label || "unlabelled")));
  passages.append(labels);
  const controls = el("div");
  controls.append(el("h3", {}, "Does this appellation refer to this candidate?"), choices(r,
    [["undecided", "Undecided"], ["yes", "Yes"], ["no", "No"], ["not_a_person", "Not a person"], ["unclear", "Unclear"]],
    value => {r.status = value; edited();}), field("Corrected referent (optional)", el("input", {
      value: r.corrected_referent, oninput: e => {r.corrected_referent = e.target.value; edited();}})), note(r));
  return append(box, append(el("div", {class: "candidate-body"}), passages, controls));
}
function render() {
  const group = groups[position];
  $('heading').textContent = blind ? "Read the passage. Record the relation." : group[0].appellation;
  $('subheading').textContent = blind ? "Debroy translation / BORI critical edition" : group.length + " candidate" + (group.length === 1 ? "" : "s") + " / contextual usages may support several referents";
  $('position').textContent = (position + 1) + " / " + groups.length;
  $('prev').disabled = position === 0; $('next').disabled = position === groups.length - 1;
  $('workspace').replaceChildren(...(blind ? [blindWork(group[0])] : group.map(candidate)));
  if (!blind) $('entity-picker').value = String(position);
  updateProgress();
}
function navigate(n) {position = Math.max(0, Math.min(n, groups.length - 1)); render(); save();}
function confirm(title, text, action, attest = false) {
  $('confirm-title').textContent = title; $('confirm-text').textContent = text;
  $('attest-wrap').hidden = !attest; $('attest').checked = false;
  $('confirm-action').textContent = attest ? "Export completed review" : "Confirm";
  $('confirm-action').disabled = attest; confirmAction = action; $('confirm').showModal();
}
function download(submitted) {
  const out = clone(review); out.submitted = submitted; out.attestation = submitted;
  check(out, submitted);
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob), a = el("a", {href: url, download: blind ? "phase5_blind.json" : "phase2_entities.json"});
  document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  review = out; byKey = new Map(review.rows.map(r => [r.key, r])); dirty = false; save(); render();
  msg((submitted ? "Completed review" : "Draft") + " download requested. Keep the JSON file as your saved record.");
}
$('prev').onclick = () => navigate(position - 1); $('next').onclick = () => navigate(position + 1);
$('next-unfinished').onclick = () => {for (let offset = 1; offset <= groups.length; offset++) {
  const n = (position + offset) % groups.length;
  if (groups[n].some(s => !complete(byKey.get(s.key)))) {navigate(n); return;}}
  msg("All decisions are recorded. Submit the review when ready.");};
$('annotator').oninput = e => {review.annotator = e.target.value; edited();};
$('export').onclick = () => {try {download(false);} catch (e) {msg(e.message, true);}};
$('submit').onclick = () => {
  if (!review.annotator.trim()) {msg("Enter an annotator name before submitting.", true); $('annotator').focus(); return;}
  if (!review.rows.every(complete)) {msg("Some decisions are unfinished. Export a draft or complete the remaining records.", true); return;}
  confirm("Submit completed review", "This export marks every record as reviewed. It will be checked before scoring.", () => download(true), true);
};
$('attest').onchange = () => {$('confirm-action').disabled = !$('attest').checked;};
$('cancel').onclick = () => $('confirm').close();
$('confirm-action').onclick = () => {$('confirm').close(); try {confirmAction();} catch (e) {msg(e.message, true);}};
$('import').onclick = () => $('file').click();
$('file').onchange = async e => {
  const file = e.target.files[0]; e.target.value = ""; if (!file) return;
  try {
    if (file.size > 10 * 1024 * 1024) throw Error("Review file exceeds 10 MB.");
    const incoming = check(JSON.parse(await file.text()));
    confirm("Import saved review", incoming.rows.filter(complete).length + " decisions by " + (incoming.annotator || "an unnamed reviewer") +
      ". This replaces the current draft. Export it first if you need to retain it.", () => {
        review = incoming; byKey = new Map(review.rows.map(r => [r.key, r]));
        position = Math.max(0, groups.findIndex(g => g.some(s => !complete(byKey.get(s.key)))));
        $('annotator').value = review.annotator; blockedStorage = false; dirty = false; save(); render(); msg("Review imported; source records verified.");
      });
  } catch (error) {msg(error.message + " Current draft preserved.", true);}
};
$('task-name').textContent = blind ? "Blind relation review" : "Entity resolution review";
$('phase').textContent = blind ? "PHASE 5 / BLIND CALIBRATION" : "PHASE 2 / CANDIDATE CENSUS";
$('standard').textContent = blind ? "Use only this passage. Keep descriptions unresolved when the referent is uncertain." : "Read each candidate against its passages. Shared appellations can have multiple valid bearers.";
$('attest-text').textContent = blind ? "I annotated these passages without consulting model outputs or the private calibration key." : "I reviewed each candidate against the available evidence; these are my own decisions.";
const rules = blind ? ["Annotate only what the passage states. Future, conditional, questioned events and similes do not establish an event.",
  "Quoted claims count as claims by their speaker. Note contested facts; leave unsupported relations out.",
  "Keep names and descriptions as written. Mark unclear on any unresolved triple. Do not guess a referent.",
  "Ignore readable chapter-marker debris. Record no relation explicitly when the inventory does not apply.",
  "Keep the private calibration key closed until this review is complete."] :
  ["A yes means this appellation can refer to this candidate in the cited context. It does not mean the names are interchangeable everywhere.",
   "Use unclear for insufficient or conflicting evidence. Source labels are unverified annotations.",
   "Not a person records a category mismatch. A corrected referent stays a suggestion until separately reviewed."];
const ruleList = el("ol", {class: "rules"}); rules.forEach(r => ruleList.append(el("li", {}, r))); $('rule-content').append(ruleList);
if (blind) for (const [p, definition] of Object.entries(DATA.predicates)) $('rule-content').append(append(el("p"), el("strong", {}, p + ": "), document.createTextNode(definition)));
else {
  $('rules').querySelector('summary').textContent = "Entity review rules"; $('entity-search-wrap').hidden = false;
  function options() {const term = $('entity-search').value.toLowerCase(); $('entity-picker').replaceChildren();
    groups.forEach((g, i) => {if (g[0].appellation.toLowerCase().includes(term)) $('entity-picker').append(el("option", {value: i}, g[0].appellation));});}
  options(); $('entity-search').oninput = options;
  $('entity-picker').onchange = e => {if (e.target.value !== "") navigate(Number(e.target.value));};
}
try {
  const saved = localStorage.getItem(storageKey);
  if (saved) {const parsed = JSON.parse(saved); review = check(parsed.review);
    byKey = new Map(review.rows.map(r => [r.key, r])); position = Number.isInteger(parsed.position) && parsed.position >= 0 && parsed.position < groups.length ? parsed.position : 0;
    msg("Recovered the saved draft. Export a file to keep it outside this browser.");}
} catch (_) {blockedStorage = true; msg("Browser recovery is unavailable or invalid. Import your exported JSON to resume; any existing recovery data is preserved.", true); $('save-label').textContent = "Recovery unavailable; export a file";}
$('annotator').value = review.annotator; render();
window.addEventListener("beforeunload", e => {clearTimeout(timer); save(); if (dirty) {e.preventDefault(); e.returnValue = "";}});
window.addEventListener("storage", e => {if (e.key === storageKey) {blockedStorage = true; $('save-label').textContent = "Another tab changed this draft"; msg("Another tab changed the recovery copy. Export this tab's draft before importing a newer copy.", true);}});
