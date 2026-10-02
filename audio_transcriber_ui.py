"""
Audio Transcriber Studio
========================
Upload audios -> auto-transcribe (English, Whisper, local & free) -> review/edit
transcripts while listening -> rename + pair audio/.txt with the same name -> ZIP.

Setup (once):
    pip install faster-whisper gradio pandas
    + install ffmpeg (only needed for the "convert to WAV" option)

Run:
    python audio_transcriber_ui.py
Open http://127.0.0.1:7860
"""

import csv
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd
import av
from faster_whisper import WhisperModel

# Ensure compatibility across all PyAV versions
_orig_av_open = av.open
def _safe_av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)
av.open = _safe_av_open

MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "distil-large-v3", "large-v3"]
COLS = ["#", "original file", "transcript (editable)", "seconds"]
NAMING = [
    "Sequential  ->  clip_0001",
    "Sequential + words  ->  clip_0001_hello_how_are_you",
    "Keep original name",
]
_models = {}


DEFAULT_MODEL_DIR = "D:/whisper_models"


def get_model(name, model_dir, device="cpu"):
    folder_str = (model_dir or "").strip() or DEFAULT_MODEL_DIR
    folder = Path(folder_str).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    key = (name, str(folder), device)
    if key not in _models:
        try:
            compute_type = "float16" if device == "cuda" else ("int8" if device == "cpu" else "auto")
            _models[key] = WhisperModel(name, device=device, compute_type=compute_type,
                                        download_root=str(folder))
        except Exception:
            _models[key] = WhisperModel(name, device="cpu", compute_type="int8",
                                        download_root=str(folder))
    return _models[key]


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(p).name)]


def slugify(text, n=6):
    return "_".join(re.findall(r"[a-z0-9]+", text.lower())[:n]) or "untitled"


def clean(text, lower, strip_punct):
    if strip_punct:
        text = re.sub(r"[^\w\s']", "", text)
    if lower:
        text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- step 1
def transcribe(files, model_name, model_dir, device_choice, beam, vad, sort_files, progress=gr.Progress(track_tqdm=True)):
    if not files:
        raise gr.Error("Upload at least one audio file first.")
    
    total_files = len(files)
    progress(0.0, desc="Preparing and sorting files...")
    files = sorted(files, key=natural_key) if sort_files else list(files)
    
    progress(0.05, desc=f"Loading Whisper model '{model_name}' on {device_choice}...")
    try:
        model = get_model(model_name, model_dir, device=device_choice)
    except Exception as e:
        raise gr.Error(f"Can't load model folder '{model_dir}': {e}")
    
    rows = []
    for i, p in enumerate(files, 1):
        filename = Path(p).name
        progress((i - 1) / total_files, desc=f"Transcribing {i}/{total_files}: {filename}")
        try:
            segs, info = model.transcribe(p, language="en", beam_size=int(beam), vad_filter=vad)
            text = " ".join(s.text.strip() for s in segs).strip()
        except RuntimeError as err:
            if "cublas" in str(err).lower() or "cuda" in str(err).lower():
                # Automatic fallback to CPU if CUDA library is missing
                progress((i - 1) / total_files, desc=f"CUDA error; switching to CPU for {filename}...")
                model = get_model(model_name, model_dir, device="cpu")
                segs, info = model.transcribe(p, language="en", beam_size=int(beam), vad_filter=vad)
                text = " ".join(s.text.strip() for s in segs).strip()
            else:
                raise err

        rows.append([i, filename, text, round(info.duration, 1)])
        progress(i / total_files, desc=f"Done {i}/{total_files}: {filename}")
    
    progress(1.0, desc="Finalizing transcripts...")
    df = pd.DataFrame(rows, columns=COLS)
    total = sum(r[3] for r in rows)
    msg = f"✅ **{len(rows)} files transcribed** ({total/60:.1f} min of audio). Review below, then go to **Rename & save**."
    return df, files, msg, gr.Tabs(selected="review")


def play_selected(srcs, evt: gr.SelectData):
    i = evt.index[0]
    return srcs[i] if srcs and i < len(srcs) else None


# ---------------------------------------------------------------- step 2
def save(df, srcs, prefix, start, pad, naming, audio_mode, out_dir,
         formats, lower, strip_punct, skip_empty, overwrite, progress=gr.Progress(track_tqdm=True)):
    if df is None or not len(df) or not srcs:
        raise gr.Error("Nothing to save. Transcribe some files first.")
    if len(df) != len(srcs):
        raise gr.Error("Row count changed. Please don't add/delete rows; re-run transcription.")
    if audio_mode.startswith("Convert") and not shutil.which("ffmpeg"):
        raise gr.Error("ffmpeg not found. Install it or choose 'Copy as-is'.")

    progress(0.0, desc="Setting up output directory...")
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    written, meta, preview = [], [], []
    n, skipped = int(start or 1), 0
    formats = formats or []
    total_rows = len(df)

    for idx, ((_, row), src) in enumerate(zip(df.iterrows(), srcs), 1):
        raw_text = row[COLS[2]]
        text_str = "" if pd.isna(raw_text) else str(raw_text)
        text = clean(text_str, lower, strip_punct)
        if skip_empty and not text:
            skipped += 1
            continue
        orig = str(row[COLS[1]])
        if naming == NAMING[2]:
            base = re.sub(r"[^\w\-. ]", "_", Path(orig).stem)
        else:
            parts = [prefix.strip(), f"{n:0{int(pad)}d}"]
            if naming == NAMING[1]:
                parts.append(slugify(text))
            base = "_".join(p for p in parts if p)
        ext = ".wav" if audio_mode.startswith("Convert") else Path(src).suffix.lower()

        k, final = 1, base
        while not overwrite and ((out / f"{final}{ext}").exists() or (out / f"{final}.txt").exists()):
            k += 1
            final = f"{base}_dup{k}"
        audio_dst, txt_dst = out / f"{final}{ext}", out / f"{final}.txt"

        progress(idx / total_rows, desc=f"Saving {idx}/{total_rows}: {audio_dst.name}")
        if audio_mode.startswith("Convert"):
            subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", str(audio_dst)],
                           check=True, capture_output=True)
        else:
            shutil.copy2(src, audio_dst)
        txt_dst.write_text(text + "\n", encoding="utf-8")
        written += [audio_dst, txt_dst]
        meta.append({"file": audio_dst.name, "transcript": text, "original": orig,
                     "seconds": row[COLS[3]]})
        preview.append([orig, audio_dst.name, txt_dst.name])
        n += 1

    progress(0.9, desc="Writing metadata files...")
    if any("csv" in f for f in formats):
        mp = out / "metadata.csv"
        new = not mp.exists()
        with open(mp, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter="|")
            if new:
                w.writerow(["file", "transcript"])
            w.writerows([[m["file"], m["transcript"]] for m in meta])
        written.append(mp)
    if any("jsonl" in f for f in formats):
        jp = out / "metadata.jsonl"
        with open(jp, "a", encoding="utf-8") as fh:
            for m in meta:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        written.append(jp)

    progress(0.95, desc="Creating ZIP archive...")
    zp = out.parent / f"{out.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in dict.fromkeys(written):
            z.write(p, arcname=p.name)

    progress(1.0, desc="Completed!")
    msg = f"**Saved {len(meta)} pairs** to `{out}`"
    if skipped:
        msg += f"  ·  skipped {skipped} empty"
    return msg, pd.DataFrame(preview, columns=["original", "new audio", "new transcript"]), str(zp)


# ---------------------------------------------------------------- UI
CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#hero h1 {margin-bottom: 0;} #hero p {opacity: .7; margin-top: 4px;}
"""
theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", radius_size="lg")

with gr.Blocks(theme=theme, css=CSS, title="Audio Transcriber Studio") as app:
    srcs = gr.State([])
    with gr.Column(elem_id="hero"):
        gr.Markdown("# 🎙️ Audio Transcriber Studio\nUpload → transcribe → review → rename & pair → download.")

    with gr.Tabs() as tabs:
        with gr.Tab("1 · Upload & transcribe", id="upload"):
            files = gr.File(label="Drop audio files here", file_count="multiple",
                            file_types=["audio"], type="filepath")
            with gr.Accordion("Transcription settings", open=False):
                with gr.Row():
                    model_dd = gr.Dropdown(MODELS, value="small.en", label="Model",
                                           info="Bigger = more accurate but slower. small.en is a good start.")
                    device_dd = gr.Dropdown(["cpu", "cuda", "auto"], value="cpu", label="Device",
                                            info="CPU is recommended and stable. CUDA requires cuBLAS 12 DLLs.")
                    beam = gr.Slider(1, 10, value=5, step=1, label="Beam size",
                                     info="Higher = slightly better accuracy, slower")
                model_dir = gr.Textbox(DEFAULT_MODEL_DIR, label="Model download folder",
                                       info="Models are downloaded here (use forward slashes, e.g. D:/whisper_models)")
                with gr.Row():
                    vad = gr.Checkbox(True, label="Skip silence (VAD)")
                    sort_files = gr.Checkbox(True, label="Natural sort by filename (clip2 before clip10)")
            go = gr.Button("Transcribe", variant="primary", size="lg")
            status1 = gr.Markdown()

        with gr.Tab("2 · Review transcripts", id="review"):
            gr.Markdown("Click a row to **play** it. Edit the transcript column to fix mistakes.")
            player = gr.Audio(label="Preview", interactive=False)
            table = gr.Dataframe(headers=COLS, datatype=["number", "str", "str", "number"],
                                 interactive=True, wrap=True, column_widths=["6%", "24%", "60%", "10%"])

        with gr.Tab("3 · Rename & save", id="save"):
            with gr.Row():
                naming = gr.Radio(NAMING, value=NAMING[0], label="Naming style")
            with gr.Row():
                prefix = gr.Textbox("clip", label="Prefix")
                start = gr.Number(1, precision=0, label="Start number")
                pad = gr.Slider(1, 8, value=4, step=1, label="Number padding (0001)")
                out_dir = gr.Textbox("dataset", label="Output folder")
            with gr.Row():
                audio_mode = gr.Radio(["Copy as-is", "Convert to 16 kHz mono WAV (needs ffmpeg)"],
                                      value="Copy as-is", label="Audio output")
                formats = gr.CheckboxGroup(["metadata.csv (pipe-separated)", "metadata.jsonl"],
                                           value=["metadata.csv (pipe-separated)"],
                                           label="Extra metadata files")
            with gr.Row():
                lower = gr.Checkbox(False, label="Lowercase transcripts")
                strip_p = gr.Checkbox(False, label="Remove punctuation")
                skip_empty = gr.Checkbox(True, label="Skip empty transcripts")
                overwrite = gr.Checkbox(False, label="Overwrite existing (else adds _dup2…)")
            save_btn = gr.Button("Rename, pair & save", variant="primary", size="lg")
            status2 = gr.Markdown()
            result = gr.Dataframe(label="What was saved", interactive=False, wrap=True)
            zip_out = gr.File(label="Download ZIP")

    go.click(transcribe, [files, model_dd, model_dir, device_dd, beam, vad, sort_files],
             [table, srcs, status1, tabs])
    table.select(play_selected, [srcs], [player])
    save_btn.click(save,
                   [table, srcs, prefix, start, pad, naming, audio_mode, out_dir,
                    formats, lower, strip_p, skip_empty, overwrite],
                   [status2, result, zip_out])

if __name__ == "__main__":
    app.launch()
