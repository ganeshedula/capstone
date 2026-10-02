import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const ROOT = '/Users/ganesh/B-tech/capstone_Nerual';
const TMP_DIR = path.join(ROOT, '.presentation_build');
const FINAL_PPTX = path.join(ROOT, 'output', 'capstone_project_review_v9.pptx');
const SKILL_DIR = '/Users/ganesh/.codex/plugins/cache/openai-primary-runtime/presentations/26.930.11008/skills/presentations';
const { resolvePresentationFont } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const C = {
  navy: '#10253F', ink: '#13283F', blue: '#2864C5', teal: '#148A8A',
  aqua: '#DDF3F1', bluePale: '#E8F0FC', paper: '#F7F9FC', white: '#FFFFFF',
  gray: '#536477', mid: '#A8B6C4', line: '#D6DEE8', orange: '#C7782B',
  red: '#9B4B49', light: '#EDF2F7', paleGold: '#F5EBDD',
};
const FONT = resolvePresentationFont();
const ppt = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function shape(slide, geometry, x, y, w, h, fill = 'none', lineFill = 'none', lineWidth = 0, name) {
  return slide.shapes.add({
    geometry, name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: 'solid', fill: lineFill, width: lineWidth },
  });
}
function rect(slide, x, y, w, h, fill, opts = {}) {
  return shape(slide, opts.geometry || 'rect', x, y, w, h, fill, opts.line || 'none', opts.lineWidth || 0, opts.name);
}
function text(slide, s, x, y, w, h, size = 24, color = C.ink, bold = false, opts = {}) {
  const sh = shape(slide, 'textbox', x, y, w, h, 'none', 'none', 0, opts.name);
  sh.text = s;
  sh.text.style = {
    typeface: FONT, fontSize: size, bold, color,
    alignment: opts.align || 'left', verticalAlignment: opts.valign || 'top',
    autoFit: opts.autoFit || 'shrinkText', wrap: 'square',
    lineSpacing: opts.lineSpacing || 1.05,
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  return sh;
}
function circle(slide, x, y, d, fill, outline = 'none', lw = 0) {
  return shape(slide, 'ellipse', x, y, d, d, fill, outline, lw);
}
function line(slide, x1, y1, x2, y2, color = C.line, width = 2) {
  return slide.shapes.add({
    geometry: 'line',
    position: {
      left: Math.min(x1, x2), top: Math.min(y1, y2),
      width: Math.abs(x2 - x1), height: Math.abs(y2 - y1),
      verticalFlip: y2 < y1,
    },
    fill: 'none', line: { style: 'solid', fill: color, width },
  });
}
function arrow(slide, x1, y1, x2, y2, color = C.blue, width = 2) {
  line(slide, x1, y1, x2 - 7, y2, color, width);
  const tri = shape(slide, 'triangle', x2 - 12, y2 - 6, 12, 12, color, 'none', 0);
  tri.position = { left: x2 - 12, top: y2 - 6, width: 12, height: 12, rotation: 90 };
}
function arrowDown(slide, x, y1, y2, color = C.blue, width = 2) {
  line(slide, x, y1, x, y2 - 8, color, width);
  const head = shape(slide, 'triangle', x - 6, y2 - 13, 12, 12, color, 'none', 0);
  head.position = { left: x - 6, top: y2 - 13, width: 12, height: 12, rotation: 180 };
}
function baseSlide(title, num, subtitle = '') {
  const slide = ppt.slides.add();
  slide.background.fill = C.paper;
  rect(slide, 0, 0, 12, 720, C.blue);
  text(slide, title, 58, 38, 1160, 54, 34, C.navy, true);
  if (subtitle) text(slide, subtitle, 60, 92, 1150, 36, 17, C.gray, false);
  line(slide, 60, 676, 1220, 676, C.line, 1);
  text(slide, 'VIT-AP UNIVERSITY  /  SCOPE', 60, 686, 480, 20, 12, C.gray, true);
  text(slide, String(num).padStart(2, '0'), 1164, 684, 56, 22, 13, C.gray, true, { align: 'right' });
  return slide;
}
function label(slide, s, x, y, w, color = C.blue) {
  text(slide, s.toUpperCase(), x, y, w, 20, 13, color, true);
}
function node(slide, x, y, w, h, title, sub, accent = C.blue, fill = C.white) {
  rect(slide, x, y, w, h, fill, { line: C.line, lineWidth: 1, geometry: 'roundRect' });
  rect(slide, x, y, 5, h, accent);
  text(slide, title, x + 18, y + 15, w - 34, 30, 19, C.navy, true);
  if (sub) text(slide, sub, x + 18, y + 48, w - 34, h - 57, 14, C.gray, false);
}
function notes(slide, content) { slide.speakerNotes.textFrame.setText(content); }

// 1. Title
{
  const s = ppt.slides.add();
  s.background.fill = C.paper;
  rect(s, 0, 0, 780, 720, C.navy);
  rect(s, 780, 0, 500, 720, '#EEF3F8');
  rect(s, 64, 72, 66, 5, '#56C4BE');
  text(s, 'Layer-Contrastive Decoding\nwith Epistemic Uncertainty\nand Self-Consistency for\nSmall Language Models', 64, 78, 660, 158, 34, C.white, true, { lineSpacing: 0.95 });
  text(s, 'TinyLlama reliability pipeline', 66, 246, 630, 34, 20, '#B9D8F0', false);
  text(s, 'B.Tech Capstone Project Review', 66, 301, 520, 31, 18, C.white, true);
  line(s, 66, 346, 690, 346, '#54708C', 1);
  label(s, 'Presented by', 66, 367, 220, '#72D0C7');
  text(s, 'E. Ganesh Kumar Reddy  ·  23BCE8969\nB. Sai Susmitha  ·  23BCE9884\nP. Dhanush Datta  ·  23BCE9258\nE. Sri Vardhan Reddy  ·  23BCE8972', 66, 397, 638, 126, 16, C.white, false, { lineSpacing: 1.08 });
  label(s, 'Guided by', 66, 545, 170, '#72D0C7');
  text(s, 'Palacharla Ravi Kumar  |  Assistant Professor Sr. Grade-2\nSCOPE, VIT-AP University', 66, 573, 640, 56, 15, C.white, false);
  text(s, 'VIT-AP UNIVERSITY', 836, 58, 360, 26, 15, C.blue, true, { align: 'right' });
  text(s, 'Layer contrast\nand uncertainty', 836, 112, 350, 68, 28, C.navy, true, { align: 'right' });
  // Editable network motif, representing intermediate model layers.
  const cols = [[860, 248], [970, 214], [1082, 242], [1170, 208]];
  const ys = [[260, 354, 448], [230, 307, 384, 461], [260, 354, 448], [282, 378, 474]];
  for (let ci = 0; ci < cols.length - 1; ci++) {
    for (const y1 of ys[ci]) for (const y2 of ys[ci + 1]) {
      line(s, cols[ci][0] + 13, y1 + 13, cols[ci + 1][0] + 13, y2 + 13, '#C5D4E1', 1);
    }
  }
  for (let ci = 0; ci < cols.length; ci++) {
    ys[ci].forEach((yy, j) => circle(s, cols[ci][0], yy, 26, (ci === 3 ? C.teal : ci === 0 ? C.blue : '#7295B8'), C.white, 2));
  }
  text(s, 'PROMPT', 840, 532, 120, 22, 12, C.gray, true, { align: 'center' });
  text(s, 'HIDDEN LAYERS', 964, 532, 174, 22, 12, C.gray, true, { align: 'center' });
  text(s, 'ANSWER', 1140, 532, 100, 22, 12, C.gray, true, { align: 'center' });
  notes(s, 'Project title supplied by the user: “Layer-Contrastive Decoding with Epistemic Uncertainty and Self-Consistency for Small Language Models.” The technical subtitle identifies the implemented TinyLlama reliability pipeline.');
}

// 2. Project overview
{
  const s = baseSlide('Project Overview', 2, 'A small language model can answer questions. This project studies how decoding and uncertainty signals can help evaluate those answers.');
  label(s, 'Project in one line', 68, 156, 280);
  text(s, 'Compare layer-contrastive decoding and Epinet-based uncertainty for TinyLlama answer generation.', 68, 184, 1110, 66, 26, C.navy, true);
  const y = 316;
  node(s, 78, y, 300, 130, 'Question prompt', 'A user asks a short factual question.', C.blue, C.white);
  node(s, 490, y, 300, 130, 'TinyLlama + DoLa', 'Contrast final-layer scores with a selected earlier layer.', C.teal, C.white);
  node(s, 902, y, 300, 130, 'Answer + signals', 'Generate a response and inspect uncertainty or candidate agreement.', C.orange, C.white);
  arrow(s, 378, y + 65, 482, y + 65, C.blue, 3);
  arrow(s, 790, y + 65, 894, y + 65, C.blue, 3);
  label(s, 'Implemented technologies', 68, 510, 280);
  text(s, 'Python   /   PyTorch   /   Hugging Face Transformers & Datasets   /   NumPy', 68, 542, 1120, 30, 18, C.gray, true);
  text(s, 'Methods: TinyLlama   ·   DoLa   ·   Epistemic Neural Network (Epinet)   ·   Self-consistency', 68, 590, 1130, 28, 16, C.gray, false);
  notes(s, 'Project details are based on README.md, config.yaml, main.py, run_experiment.py, Neural Network/dola.py, and Neural Network/enn_torch.py. No benchmark result is asserted.');
}

// 3. Problem and motivation
{
  const s = baseSlide('Problem Statement & Motivation', 3, 'Fluent language generation does not by itself show whether a response is correct or reliable.');
  label(s, 'Problem', 72, 154, 150);
  text(s, 'A single model response offers limited evidence about answer reliability.', 72, 184, 485, 70, 27, C.navy, true);
  text(s, 'The project tests complementary signals at two levels: token scores across model layers and agreement across sampled answers.', 72, 270, 485, 90, 19, C.gray, false);
  // visual: answer with uncertain validity
  rect(s, 680, 150, 512, 210, C.white, { line: C.line, lineWidth: 1, geometry: 'roundRect' });
  label(s, 'Generation can sound certain', 712, 178, 420, C.orange);
  text(s, '“The answer is …”', 712, 216, 418, 40, 27, C.navy, true);
  line(s, 712, 273, 1158, 273, C.line, 1);
  text(s, 'Surface fluency  ≠  measured correctness', 712, 296, 440, 30, 18, C.red, true);
  label(s, 'Challenges addressed by the implementation', 72, 414, 560);
  const points = [
    ['One-shot decoding', 'No answer aggregation across candidates'],
    ['Layer disagreement', 'Intermediate and final predictions can differ'],
    ['Uncertainty meaning', 'Entropy is not automatically correctness probability'],
  ];
  points.forEach((p, i) => {
    const yy = 452 + i * 58;
    circle(s, 76, yy + 2, 28, i === 2 ? C.paleGold : C.bluePale);
    text(s, String(i + 1), 76, yy + 7, 28, 18, 13, C.blue, true, { align: 'center' });
    text(s, p[0], 120, yy, 188, 22, 17, C.navy, true);
    text(s, p[1], 316, yy, 822, 26, 16, C.gray, false);
  });
  notes(s, 'Motivation is framed as an evaluation problem, not as a claim that the project already improves reliability. TruthfulQA describes how language models can reproduce common falsehoods: Lin, Hilton, and Evans, ACL 2022, https://aclanthology.org/2022.acl-long.229/.');
}

// 4. Objectives
{
  const s = baseSlide('Project Objectives', 4, 'The implementation connects decoding, uncertainty estimation, and answer-level comparison.');
  const items = [
    ['01', 'Establish a Base baseline', 'Use the local TinyLlama chat model for question answering.'],
    ['02', 'Add layer contrast', 'Select an earlier layer using Jensen–Shannon divergence and apply DoLa.'],
    ['03', 'Train an Epinet', 'Learn a next-token correction from C4 features while keeping the base model fixed.'],
    ['04', 'Aggregate candidate answers', 'Compare plurality voting with ENN-entropy-weighted voting.'],
    ['05', 'Measure quality and confidence', 'Evaluate exact match and calibration metrics when run outputs are available.'],
  ];
  items.forEach((it, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 72 + col * 590, y = 156 + row * 148;
    circle(s, x, y + 4, 42, i === 4 ? C.teal : C.blue);
    text(s, it[0], x, y + 15, 42, 18, 13, C.white, true, { align: 'center' });
    text(s, it[1], x + 62, y, 470, 32, 20, C.navy, true);
    text(s, it[2], x + 62, y + 40, 478, 56, 16, C.gray, false);
  });
  rect(s, 662, 452, 512, 94, C.aqua, { geometry: 'roundRect' });
  text(s, 'Evaluation is part of the objective. Saved runs are small and do not establish a general performance gain.', 686, 470, 464, 62, 17, C.ink, true);
  notes(s, 'Objectives reflect the implemented commands and methods in main.py and run_experiment.py. The last objective is intentionally evaluative and makes no claim of achieved improvement.');
}

// 5. Related work
{
  const s = baseSlide('Existing Methods & Research Gap', 5, 'The project combines established ideas for a local small-model evaluation pipeline.');
  const cols = [
    { x: 70, title: 'Base language model', sub: 'Next-token prediction', body: 'Produces a single continuation under greedy decoding or sampling.', ref: 'Zhang et al., 2024' },
    { x: 368, title: 'DoLa', sub: 'Layer-level contrast', body: 'Contrasts mature and premature layer distributions during decoding.', ref: 'Chuang et al., ICLR 2024' },
    { x: 666, title: 'Epinet / ENN', sub: 'Epistemic signal', body: 'Indexes predictions by a sampled epistemic variable to expose model uncertainty.', ref: 'Osband et al., 2021' },
    { x: 964, title: 'Self-consistency', sub: 'Answer aggregation', body: 'Samples several candidates and selects a normalized plurality answer.', ref: 'Wang et al., ICLR 2023' },
  ];
  for (const [i, c] of cols.entries()) {
    rect(s, c.x, 162, 244, 228, C.white, { line: C.line, lineWidth: 1, geometry: 'roundRect' });
    rect(s, c.x, 162, 244, 5, [C.blue, C.teal, C.orange, C.blue][i]);
    text(s, c.title, c.x + 18, 185, 208, 31, 19, C.navy, true);
    text(s, c.sub, c.x + 18, 222, 208, 24, 14, [C.blue, C.teal, C.orange, C.blue][i], true);
    text(s, c.body, c.x + 18, 260, 208, 80, 15, C.gray, false);
    text(s, c.ref, c.x + 18, 356, 212, 19, 12, C.gray, true);
  }
  label(s, 'Research gap in this capstone', 72, 432, 400);
  text(s, 'The repository brings these components together. Small C4 and TruthfulQA runs are recorded, but they do not establish a general reliability gain.', 72, 462, 1116, 74, 22, C.navy, true);
  line(s, 72, 566, 1194, 566, C.line, 1);
  text(s, 'References: TinyLlama (Zhang et al., 2024)  ·  DoLa (Chuang et al., 2024)  ·  ENN (Osband et al., 2021)  ·  Self-consistency (Wang et al., 2023)', 72, 585, 1125, 42, 14, C.gray, false);
  notes(s, 'Sources: Zhang et al., “TinyLlama: An Open-Source Small Language Model,” https://arxiv.org/abs/2401.02385. Chuang et al., “DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models,” ICLR 2024, https://openreview.net/forum?id=Th6NyL07na. Osband et al., “Epistemic Neural Networks,” https://arxiv.org/abs/2107.08924. Wang et al., “Self-Consistency Improves Chain of Thought Reasoning in Language Models,” ICLR 2023, https://arxiv.org/abs/2203.11171. These references motivate individual methods; the deck does not claim prior work validated this exact combination.');
}

// 6. Methodology
{
  const s = baseSlide('Proposed System & Methodology', 6, 'Two paths meet at evaluation: C4 trains the Epinet; questions exercise answer generation and voting.');
  label(s, 'Feature training path', 70, 146, 270, C.teal);
  node(s, 70, 176, 246, 104, 'English C4', 'Stream or local shard', C.teal);
  node(s, 366, 176, 246, 104, 'Feature extraction', 'TinyLlama hidden states', C.teal);
  node(s, 662, 176, 246, 104, 'Document split', '90/10 train/validation', C.teal);
  node(s, 958, 176, 246, 104, 'Train Epinet', 'Next-token cross-entropy', C.teal);
  arrow(s, 316, 228, 358, 228, C.teal, 2); arrow(s, 612, 228, 654, 228, C.teal, 2); arrow(s, 908, 228, 950, 228, C.teal, 2);
  line(s, 70, 318, 1204, 318, C.line, 1);
  label(s, 'Question-answering path', 70, 348, 330, C.blue);
  node(s, 70, 380, 246, 110, 'Question prompt', 'Local QA set or TruthfulQA', C.blue);
  node(s, 366, 380, 246, 110, 'Base / DoLa', 'Generate candidate answer(s)', C.blue);
  node(s, 662, 380, 246, 110, 'ENN scoring', 'Predictive entropy per answer', C.orange);
  node(s, 958, 380, 246, 110, 'Voting + evaluation', 'Plurality / weighted vote', C.blue);
  arrow(s, 316, 434, 358, 434, C.blue, 2); arrow(s, 612, 434, 654, 434, C.blue, 2); arrow(s, 908, 434, 950, 434, C.blue, 2);
  arrowDown(s, 1080, 280, 373, C.teal, 2);
  text(s, 'The ENN confidence weight is an entropy heuristic. Calibration against correctness must be measured.', 72, 540, 1116, 54, 17, C.gray, false);
  notes(s, 'C4 feature preparation and document split: Neural Network/data_prep.py. Model and DoLa decoding: Neural Network/dola.py. Epinet training and sampled entropy: Neural Network/enn_torch.py. Candidate aggregation: run_experiment.py and Neural Network/self_consistency.py. The diagram represents implemented code paths, not evidence of successful trained runs. C4 background reference: Raffel et al., JMLR 2020, https://jmlr.org/papers/v21/20-074.html.');
}

// 7. Model architecture
{
  const s = baseSlide('Model & System Architecture', 7, 'DoLa changes token scores. The Epinet samples token distributions. Voting combines answer strings.');
  // DoLa path: both layer sources converge on JSD selection, then contrastive logits.
  label(s, 'A  Layer-contrastive decoding', 66, 146, 430, C.blue);
  rect(s, 76, 204, 210, 58, C.blue, { geometry: 'roundRect' });
  text(s, 'Mature block 21', 92, 212, 182, 22, 17, C.white, true);
  text(s, 'Final-layer logits', 92, 237, 182, 18, 13, C.white, false);
  rect(s, 76, 292, 210, 58, C.teal, { geometry: 'roundRect' });
  text(s, 'Candidate blocks', 92, 300, 182, 22, 17, C.white, true);
  text(s, '0, 4, 8, 12, 16, 20', 92, 325, 182, 18, 13, C.white, false);
  node(s, 340, 242, 152, 72, 'JSD selector', 'Choose premature layer', C.teal);
  rect(s, 520, 242, 164, 72, C.white, { line: C.line, lineWidth: 1, geometry: 'roundRect' });
  rect(s, 520, 242, 5, 72, C.blue);
  text(s, 'DoLa contrast', 538, 258, 136, 21, 16, C.navy, true);
  text(s, 'mature − α · premature', 538, 285, 136, 18, 11, C.gray, false);
  arrow(s, 286, 233, 327, 265, C.blue, 2);
  arrow(s, 286, 321, 327, 291, C.teal, 2);
  arrow(s, 492, 278, 512, 278, C.blue, 2);
  node(s, 164, 382, 456, 64, 'Next-token distribution', 'softmax(contrastive logits)', C.blue, C.bluePale);
  arrowDown(s, 602, 314, 374, C.blue, 2);
  // Epinet path
  line(s, 712, 146, 712, 468, C.line, 1);
  label(s, 'B  Epistemic Neural Network', 748, 146, 440, C.teal);
  node(s, 750, 190, 410, 67, 'Input feature x', 'Concatenate mature + premature (2 × 2,048)', C.blue);
  arrowDown(s, 955, 257, 278, C.blue, 2);
  node(s, 750, 285, 196, 94, 'Frozen prior', 'Random indexed MLP', C.gray, C.white);
  node(s, 964, 285, 196, 94, 'Learnable branch', 'Indexed MLP, width 512', C.teal, C.white);
  rect(s, 750, 406, 410, 62, C.white, { line: C.line, lineWidth: 1, geometry: 'roundRect' });
  rect(s, 750, 406, 5, 62, C.orange);
  text(s, 'Combined correction → vocabulary logits', 768, 416, 374, 22, 16, C.navy, true);
  text(s, 'Shared z; K = 3 epistemic samples by default', 768, 442, 374, 16, 11, C.gray, false);
  arrowDown(s, 852, 379, 398, C.gray, 2); arrowDown(s, 1060, 379, 398, C.teal, 2);
  text(s, 'Epinet outputs token entropy / an epistemic proxy; neither directly measures answer correctness. Audit: 8.66M trainable parameters.', 750, 476, 430, 40, 13, C.gray, false);
  label(s, 'Answer-level decision path', 70, 506, 360, C.blue);
  const stages = [
    { x: 70, title: 'DoLa distribution', sub: 'token probabilities', col: C.blue },
    { x: 306, title: 'Sample N answers', sub: 'candidate strings', col: C.teal },
    { x: 542, title: 'ENN scores', sub: 'entropy heuristic', col: C.orange },
    { x: 778, title: 'Aggregate votes', sub: 'plurality / weighted', col: C.blue },
    { x: 1014, title: 'Final answer', sub: 'selected response', col: C.teal },
  ];
  for (const st of stages) {
    rect(s, st.x, 538, 194, 58, C.white, { line: C.line, lineWidth: 1, geometry: 'roundRect' });
    rect(s, st.x, 538, 5, 58, st.col);
    text(s, st.title, st.x + 14, 547, 170, 20, 15, C.navy, true);
    text(s, st.sub, st.x + 14, 571, 170, 16, 12, C.gray, false);
  }
  for (let i = 0; i < stages.length - 1; i++) arrow(s, stages[i].x + 194, 567, stages[i + 1].x - 7, 567, C.mid, 2);
  text(s, 'Plurality uses answer counts; weighted voting uses entropy-derived scores.', 72, 612, 1110, 24, 14, C.gray, false);
  notes(s, 'Architecture details are read from Neural Network/dola.py and Neural Network/enn_torch.py. The configured TinyLlama structure uses a mature index of 21, candidate premature indices [0,4,8,12,16,20], hidden width 2,048, Epinet width 512, epistemic index size 6, and three samples by default. The answer path shows sampling candidate strings, scoring with the entropy heuristic, and aggregating votes; entropy is not itself answer correctness. The model_audit_experiments.json records 8,664,064 trainable Epinet parameters.');
}

// 8. Implementation and setup
{
  const s = baseSlide('Implementation & Experimental Setup', 8, 'Configured defaults and available artifacts are separated from completed experimental evidence.');
  const leftX = 72, rightX = 682;
  label(s, 'Software stack', leftX, 150, 280);
  text(s, 'Python  ·  PyTorch  ·  Transformers\nDatasets  ·  NumPy', leftX, 180, 500, 62, 19, C.navy, true);
  label(s, 'Data artifacts', leftX, 286, 260, C.teal);
  text(s, 'C4 feature cache\n1,841 token examples  /  20 document IDs\nTrain 1,637  /  validation 204\n2,048 dimensions  /  max length 128', leftX, 318, 520, 112, 17, C.ink, false);
  text(s, 'Local QA file: 10 labeled questions', leftX, 448, 520, 28, 16, C.gray, true);
  label(s, 'ENN training configuration', rightX, 150, 420, C.orange);
  text(s, 'AdamW  /  learning rate 1 × 10⁻⁴\nBatch size 1  /  2 epochs  /  3 z-samples\nCross-entropy  /  weight decay 1 × 10⁻⁴\nFinal train loss 2.3298  /  validation 2.5698', rightX, 180, 526, 112, 16, C.ink, false);
  label(s, 'Evaluation and availability', rightX, 310, 420, C.blue);
  text(s, 'TruthfulQA MC1/MC2 and C4 token metrics are saved.\n\nENN checkpoint: checkpoints/enn_best.pt\nRecorded backend: MPS  /  device model and RAM: [TO BE PROVIDED]', rightX, 342, 526, 118, 16, C.ink, false);
  rect(s, 72, 522, 1120, 88, C.paleGold, { geometry: 'roundRect' });
  text(s, 'Recorded software: Python 3.10.20  /  PyTorch 2.11.0  /  Transformers 5.17.0', 96, 544, 1066, 28, 18, C.navy, true);
  text(s, 'Training history is saved. Exact C4 source revision, machine model and RAM are not recorded.', 96, 577, 1060, 20, 14, C.gray, false);
  notes(s, 'Values verified from config.yaml, Neural Network/features_cache.npz, eval_questions.json, README.md, results/training_history.json, results/model_audit_experiments.json, and checkpoints/enn_best.pt. Cache metadata: version 4, 20 requested texts, max_length 128, val_fraction 0.1; arrays contain 1,841 rows, mature/premature shapes (1841,2048), labels (1841,), and 20 document IDs. Saved audit records 1,637 train and 204 validation examples, deterministic document-level split seed 42, two epochs, batch 1, lr 1e-4, K=3, final train loss 2.3298, final validation loss 2.5698, MPS backend, Python 3.10.20, PyTorch 2.11.0, Transformers 5.17.0, Datasets 4.8.5, NumPy 1.26.4, and 8,664,064 trainable Epinet parameters. Physical machine model and RAM and exact C4 revision are not stored.');
}

// 9. Results
{
  const s = baseSlide('Results & Performance Analysis', 9, 'Saved evaluations show small and mixed effects. Each comparison below keeps its evaluation slice and settings visible.');
  text(s, 'Held-out C4 next-token evaluation', 74, 145, 620, 28, 21, C.navy, true);
  text(s, '100 texts  /  9,431 tokens  /  α = 0.05', 74, 177, 1084, 24, 15, C.gray, false);
  const t = s.tables.add({
    rows: 3, columns: 6, left: 72, top: 212, width: 1124, height: 142,
    values: [
      ['Method', 'ENN weight', 'Top-1', 'Top-5', 'Perplexity', 'Tokens'],
      ['DoLa', '0.0', '50.22%', '72.84%', '10.7034', '9,431'],
      ['DoLa + ENN', '0.2', '50.31%', '72.89%', '10.7189', '9,431'],
    ],
  });
  t.styleOptions = { headerRow: true, bandedRows: true, firstColumn: true };
  t.borders.assign({ style: 'solid', fill: C.line, width: 1 });
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 6; c++) {
      const cell = t.getCell(r, c);
      cell.text.style = {
        typeface: FONT, fontSize: r === 0 ? 16 : 16,
        bold: r === 0 || c === 0,
        color: r === 0 ? C.white : c === 0 ? C.navy : C.gray,
        alignment: c === 0 ? 'left' : 'center', verticalAlignment: 'middle',
        autoFit: 'shrinkText', wrap: 'square',
      };
      if (r === 0) cell.fill = C.navy;
      else if (r % 2 === 0) cell.fill = C.bluePale;
      else cell.fill = C.white;
    }
  }
  text(s, 'On this slice, ENN weight 0.2 raised top-1 by 0.09 percentage points and top-5 by 0.05 points, while perplexity rose by 0.0155. Differences are descriptive only.', 76, 368, 1110, 48, 15, C.gray, false);
  line(s, 74, 426, 1194, 426, C.line, 1);
  text(s, 'TruthfulQA MC1 / MC2  ·  10 questions per slice', 74, 442, 700, 27, 20, C.navy, true);
  text(s, 'Q0–9  |  α=1, weight=1  |  Base / DoLa / DoLa+ENN  |  MC1: 0.70 / 0.40 / 0.40  |  MC2: 0.4931 / 0.4546 / 0.4511', 76, 480, 1110, 38, 14, C.ink, false);
  text(s, 'Q10–19  |  α=0.05, weight=0  |  Base / DoLa / DoLa+ENN  |  MC1: 0.30 / 0.30 / 0.30  |  MC2: 0.4076 / 0.4066 / 0.4066', 76, 523, 1110, 38, 14, C.ink, false);
  rect(s, 72, 580, 1124, 52, C.paleGold, { geometry: 'roundRect' });
  text(s, 'Small, differently configured slices. No reliable overall improvement can be concluded.', 92, 592, 1082, 25, 17, C.navy, true);
  notes(s, 'Sources: output/c4_eval_20260930_enn02/c4_eval_results.json records n_texts=100, n_tokens=9431, alpha=0.05, ENN weight=0.2, before (DoLa) top-1 0.5022, top-5 0.7284, perplexity 10.7034; after (DoLa+ENN) top-1 0.5031, top-5 0.7289, perplexity 10.7189. These are descriptive next-token metrics, not answer accuracy. results/model_audit_experiments.json records two separate TruthfulQA 10-question slices: indices 0–9 at alpha=1, weight=1 with MC1/MC2 Base .7/.4931, DoLa .4/.4546, DoLa+ENN .4/.4511; indices 10–19 at alpha=.05, weight=0 with Base .3/.4076, DoLa .3/.4066, DoLa+ENN .3/.4066. The zero weight makes DoLa+ENN score-identical to DoLa in the second slice. The two slices differ in hyperparameters and are not pooled or directly compared.');
}

// 10. Conclusion and future work
{
  const s = baseSlide('Conclusion & Future Work', 10, 'The current contribution is an implementable comparison pipeline; empirical conclusions require the next evaluation run.');
  label(s, 'Conclusion', 72, 154, 220, C.blue);
  text(s, 'The system combines TinyLlama, dynamic DoLa layer contrast, an Epinet trained on C4 next-token features, and answer-level voting.', 72, 188, 510, 106, 23, C.navy, true);
  text(s, 'Recorded C4 and TruthfulQA evaluations show mixed results. The small slices do not establish a general factuality or reliability gain.', 72, 318, 510, 82, 17, C.gray, false);
  line(s, 630, 154, 630, 514, C.line, 1);
  label(s, 'Future work', 678, 154, 240, C.teal);
  const f = [
    'Train and archive the active PyTorch ENN checkpoint',
    'Run held-out TruthfulQA and larger QA evaluations',
    'Report repeated seeds, calibration, and risk–coverage',
    'Profile inference time and memory on documented hardware',
  ];
  f.forEach((a, i) => {
    const yy = 194 + i * 70;
    circle(s, 680, yy + 1, 26, C.aqua);
    text(s, String(i + 1), 680, yy + 6, 26, 17, 12, C.teal, true, { align: 'center' });
    text(s, a, 724, yy, 458, 48, 17, C.ink, false);
  });
  line(s, 72, 558, 1192, 558, C.line, 1);
  text(s, 'Thank You', 72, 580, 510, 42, 30, C.navy, true);
  text(s, 'Questions & Discussion', 72, 626, 510, 28, 18, C.blue, true);
  text(s, 'SCOPE  ·  VIT-AP UNIVERSITY', 784, 605, 400, 24, 15, C.gray, true, { align: 'right' });
  notes(s, 'Conclusion and future work reflect the absence of current benchmark outputs and checkpoint. No accuracy or reliability improvement is claimed.');
}

const candidatePath = path.join(TMP_DIR, 'candidate-v9.pptx');
await (await PresentationFile.exportPptx(ppt)).save(candidatePath);
// Also render per-slide previews for visual review.
const previewDir = path.join(TMP_DIR, 'previews');
await fs.mkdir(previewDir, { recursive: true });
for (let i = 0; i < ppt.slides.items.length; i++) {
  const slide = ppt.slides.items[i];
  const blob = await ppt.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(previewDir, `slide-${String(i + 1).padStart(2, '0')}.png`), new Uint8Array(await blob.arrayBuffer()));
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(path.join(previewDir, `slide-${String(i + 1).padStart(2, '0')}.layout.json`), await layout.text());
}

const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: [9],
  workspaceDir: ROOT,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: '/Users/ganesh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
  integrityValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL_DIR, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit', '--require-native-table-slide', '9'],
  fontPolicy: { basis: 'design', families: [FONT] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(TMP_DIR, 'capstone_project_review_v9.validation.json'),
});
console.log(JSON.stringify({ finalPath: FINAL_PPTX, font: FONT, slides: ppt.slides.items.length, finalizer: result }, null, 2));
