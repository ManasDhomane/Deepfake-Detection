import { useEffect, useRef, useState } from "react";

const ACCEPT = "image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const input = useRef(null);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  function pick(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError("");
  }

  async function analyze() {
    setBusy(true);
    setResult(null);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const res = await fetch("/api/predict", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Analysis failed.");
      setResult(data);
    } catch (e) {
      setError(e.message === "Failed to fetch" ? "Cannot reach the server. Start the Flask backend on port 5000." : e.message);
    } finally {
      setBusy(false);
    }
  }

  const isVideo = file?.type.startsWith("video/");
  const fake = result?.label === "fake";

  return (
    <main>
      <header>
        <h1>Is this face real?</h1>
        <p>Upload a photo or short video. A neural network checks the face for signs of manipulation and gives a verdict with a confidence score.</p>
      </header>

      <section className="bench">
        <div
          className={`drop ${drag ? "over" : ""} ${preview ? "filled" : ""}`}
          onClick={() => input.current.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
          role="button"
          tabIndex={0}
          aria-label="Choose a file to analyze"
        >
          <input ref={input} type="file" accept={ACCEPT} hidden onChange={(e) => pick(e.target.files[0])} />
          {!preview && <p><strong>Drop a file here</strong> or click to browse<br /><small>JPG, PNG, WEBP, MP4, MOV, AVI, MKV, WEBM · up to 100 MB</small></p>}
          {preview && (isVideo ? <video src={preview} controls muted onClick={(e) => e.stopPropagation()} /> : <img src={preview} alt="Selected upload" />)}
        </div>

        <div className="panel" aria-live="polite">
          {!file && <p className="hint">Your result will appear here.</p>}
          {file && !result && !error && (
            <>
              <p className="filename">{file.name}</p>
              <button onClick={analyze} disabled={busy}>{busy ? "Analyzing…" : "Analyze file"}</button>
              {busy && isVideo && <p className="hint">Videos take longer because 16 frames are checked.</p>}
            </>
          )}
          {error && (
            <>
              <p className="error">{error}</p>
              <button onClick={analyze} disabled={busy}>Try again</button>
            </>
          )}
          {result && (
            <div className={`verdict ${fake ? "fake" : "real"}`}>
              <h2>{fake ? "Likely manipulated" : "Likely authentic"}</h2>
              <p className="conf"><b>{result.confidence}%</b> confidence</p>
              <div className="meter" role="img" aria-label={`Fake probability ${Math.round(result.fake_probability * 100)} percent`}>
                <span style={{ left: `${result.fake_probability * 100}%` }} />
              </div>
              <div className="scale"><span>real</span><span>fake</span></div>

              {result.kind === "video" && (
                <>
                  <h3>Frame by frame ({result.frames_analyzed} frames with a face)</h3>
                  <div className="frames">
                    {result.frame_scores.map((s, i) => (
                      <span key={i} title={`Frame ${i + 1}: ${(s * 100).toFixed(0)}% fake`}
                        className={s >= 0.5 ? "f" : "r"} style={{ height: `${Math.max(s, 0.04) * 100}%` }} />
                    ))}
                  </div>
                </>
              )}
              <button className="ghost" onClick={() => { setFile(null); setPreview(""); setResult(null); }}>Check another file</button>
            </div>
          )}
        </div>
      </section>

      <footer>This tool reads facial artifacts only. It does not analyze audio or fully AI-generated images, and heavy compression lowers accuracy. Treat the result as one signal, not proof.</footer>
    </main>
  );
}
