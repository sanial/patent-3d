import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist';
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { usePatentUpload, type ParsePatentResponse } from './usePatentUpload';
import { useClaimsAnalysis } from './useClaimsAnalysis';
import './PatentUploadPanel.css';

GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

type Tab = 'pages' | 'figures' | 'claims';

interface PatentUploadPanelProps {
  apiUrl?: string;
  onResult?: (resp: ParsePatentResponse) => void;
}

interface PagePreview {
  pageNumber: number;
  dataUrl: string | null; // null while rendering
}

export function PatentUploadPanel({ apiUrl, onResult }: PatentUploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [pages, setPages] = useState<PagePreview[]>([]);
  const [pdfFileName, setPdfFileName] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('pages');
  const [includeFiguresInAnalysis, setIncludeFiguresInAnalysis] = useState(true);
  const renderTokenRef = useRef(0);

  const { upload, status, progress, error, result, reset } = usePatentUpload({
    apiUrl,
    onResult,
  });
  const {
    analyze,
    reset: resetAnalysis,
    status: analysisStatus,
    error: analysisError,
    analysis,
  } = useClaimsAnalysis({ apiUrl });

  const renderPdfPages = useCallback(async (file: File) => {
    const token = ++renderTokenRef.current;
    setPages([]);
    setPdfFileName(file.name);

    const buf = await file.arrayBuffer();
    if (token !== renderTokenRef.current) return;

    const pdf = await getDocument({ data: buf }).promise;
    if (token !== renderTokenRef.current) return;

    // Seed placeholders so the user sees the page count immediately.
    setPages(
      Array.from({ length: pdf.numPages }, (_, i) => ({
        pageNumber: i + 1,
        dataUrl: null,
      }))
    );

    // Render pages sequentially — gives a "loading" cascade effect.
    for (let i = 1; i <= pdf.numPages; i++) {
      if (token !== renderTokenRef.current) return;
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 0.4 });
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) continue;
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      // pdf.js v4 requires `canvas` in the render params
      await page.render({ canvas, canvasContext: ctx, viewport } as Parameters<typeof page.render>[0]).promise;
      const dataUrl = canvas.toDataURL('image/png');
      if (token !== renderTokenRef.current) return;
      setPages((prev) => prev.map((p) => (p.pageNumber === i ? { ...p, dataUrl } : p)));
    }
  }, []);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const pdf = Array.from(files).find((f) => f.name.toLowerCase().endsWith('.pdf'));
      if (!pdf) return;
      void renderPdfPages(pdf);
      void upload(pdf);
    },
    [upload, renderPdfPages]
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleReset = useCallback(() => {
    renderTokenRef.current++;
    setPages([]);
    setPdfFileName(null);
    setTab('pages');
    reset();
    resetAnalysis();
  }, [reset, resetAnalysis]);

  // Make sure we don't leak worker tasks if the component unmounts.
  useEffect(() => () => { renderTokenRef.current++; }, []);

  const extractedFigures = result?.result.extractedFigures ?? [];
  const structure = result?.result.structure ?? null;
  const parsedClaims = structure?.claims ?? [];
  const refDefinitions = structure?.refDefinitions ?? {};

  const refCount = result ? Object.keys(result.patentData).length : 0;
  const renderedCount = pages.filter((p) => p.dataUrl).length;

  // Claims to display: prefer Gemma analysis if available, else parsed claims.
  const displayClaims = useMemo(() => {
    if (analysis?.claims?.length) {
      return analysis.claims.map((a) => ({
        number: a.number,
        type: a.type,
        summary: a.summary,
        refs: a.ref_numbers,
        dependsOn: a.dependsOn,
      }));
    }
    return parsedClaims.map((c) => ({
      number: c.number,
      type: c.type,
      summary: c.body.length > 240 ? c.body.slice(0, 240) + '…' : c.body,
      refs: c.refs,
      dependsOn: c.dependsOn,
    }));
  }, [analysis, parsedClaims]);

  const handleAnalyze = useCallback(() => {
    if (!parsedClaims.length) return;
    void analyze(parsedClaims, refDefinitions, extractedFigures, {
      includeFigures: includeFiguresInAnalysis,
    });
  }, [analyze, parsedClaims, refDefinitions, extractedFigures, includeFiguresInAnalysis]);

  return (
    <div className="patent-upload-panel">
      <div className="pup-header">
        <span className="pup-title">Patent OCR</span>
        {(status === 'done' || pdfFileName) && (
          <button className="pup-reset" onClick={handleReset} type="button">
            New upload
          </button>
        )}
      </div>

      {!pdfFileName && (
        <div
          className={`pup-dropzone ${dragOver ? 'pup-dropzone--over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,application/pdf"
            hidden
            onChange={(e) => handleFiles(e.target.files)}
          />
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          <strong>Drop patent PDF here</strong>
          <p>or click to browse</p>
        </div>
      )}

      {pdfFileName && (
        <div className="pup-filename" title={pdfFileName}>{pdfFileName}</div>
      )}

      {(status === 'uploading' || status === 'parsing') && (
        <div className="pup-progress">
          <div className="pup-progress-bar" style={{ width: `${Math.round(progress * 100)}%` }} />
          <span className="pup-progress-label">
            {status === 'uploading' ? 'Uploading…' : 'Running OCR (PyMuPDF + EasyOCR)…'}
          </span>
        </div>
      )}

      {status === 'error' && error && <div className="pup-error">{error}</div>}

      {(pages.length > 0 || result) && (
        <div className="pup-tabs">
          <button
            type="button"
            className={`pup-tab ${tab === 'pages' ? 'pup-tab--active' : ''}`}
            onClick={() => setTab('pages')}
          >
            Pages<span className="pup-tab-count">{pages.length || ''}</span>
          </button>
          <button
            type="button"
            className={`pup-tab ${tab === 'figures' ? 'pup-tab--active' : ''}`}
            onClick={() => setTab('figures')}
          >
            Figures
            <span className="pup-tab-count">{extractedFigures.length || ''}</span>
          </button>
          <button
            type="button"
            className={`pup-tab ${tab === 'claims' ? 'pup-tab--active' : ''}`}
            onClick={() => setTab('claims')}
          >
            Claims<span className="pup-tab-count">{parsedClaims.length || ''}</span>
          </button>
        </div>
      )}

      {tab === 'pages' && pages.length > 0 && (
        <div className="pup-pages-section">
          <div className="pup-pages-header">
            <span>Page previews</span>
            <span className="pup-pages-count">
              {renderedCount} / {pages.length}
            </span>
          </div>
          <div className="pup-pages-grid">
            {pages.map((p) => (
              <div key={p.pageNumber} className="pup-page">
                {p.dataUrl ? (
                  <img src={p.dataUrl} alt={`Page ${p.pageNumber}`} />
                ) : (
                  <div className="pup-page-skeleton" />
                )}
                <span className="pup-page-num">{p.pageNumber}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {tab === 'figures' && (
        <>
          {extractedFigures.length === 0 && status === 'done' && (
            <div className="pup-empty">No figures extracted.</div>
          )}
          {extractedFigures.length === 0 && status !== 'done' && (
            <div className="pup-empty">Figures will appear after OCR completes.</div>
          )}
          {extractedFigures.length > 0 && (
            <div className="pup-figures-grid">
              {extractedFigures.map((f) => (
                <div key={f.figureId} className="pup-figure">
                  {f.pngDataUrl ? (
                    <img src={f.pngDataUrl} alt={f.captionText || f.figureId} />
                  ) : null}
                  <div className="pup-figure-caption">
                    {f.captionText || f.figureId}
                    <span style={{ marginLeft: 6, color: '#8a8a90', fontSize: 10 }}>
                      p.{f.page + 1}
                    </span>
                  </div>
                  {f.refNumbersOriginallyInside.length > 0 && (
                    <div className="pup-figure-refs">
                      {f.refNumbersOriginallyInside.map((r) => (
                        <span key={r} className="pup-figure-ref-chip">{r}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {tab === 'claims' && (
        <>
          <div className="pup-analyze-bar">
            <button
              type="button"
              className="pup-analyze-btn"
              onClick={handleAnalyze}
              disabled={
                !parsedClaims.length ||
                analysisStatus === 'running' ||
                status !== 'done'
              }
            >
              {analysisStatus === 'running'
                ? 'Analyzing…'
                : analysis
                ? 'Re-analyze'
                : 'Analyze with Gemma 4'}
            </button>
            <label className="pup-analyze-toggle">
              <input
                type="checkbox"
                checked={includeFiguresInAnalysis}
                onChange={(e) => setIncludeFiguresInAnalysis(e.target.checked)}
              />
              include figures
            </label>
          </div>

          {analysisError && <div className="pup-error">{analysisError}</div>}

          {analysis?.novelty_keywords?.length ? (
            <div className="pup-keywords">
              {analysis.novelty_keywords.map((k) => (
                <span key={k} className="pup-keyword">{k}</span>
              ))}
            </div>
          ) : null}

          {displayClaims.length === 0 ? (
            <div className="pup-empty">
              {status === 'done' ? 'No claims detected.' : 'Claims will appear after OCR completes.'}
            </div>
          ) : (
            <ul className="pup-claim-list">
              {displayClaims.map((c) => (
                <li
                  key={c.number}
                  className={`pup-claim ${c.dependsOn ? 'pup-claim--dependent' : ''}`}
                >
                  <div className="pup-claim-head">
                    <span className="pup-claim-num">{c.number}.</span>
                    <span className="pup-claim-type">
                      {c.type}
                      {c.dependsOn ? ` · depends on ${c.dependsOn}` : ''}
                    </span>
                  </div>
                  <p className="pup-claim-summary">{c.summary}</p>
                  {c.refs.length > 0 && (
                    <div className="pup-claim-refs">
                      {c.refs.map((r) => (
                        <span key={r} className="pup-figure-ref-chip">{r}</span>
                      ))}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}

      {status === 'done' && result && (
        <div className="pup-result">
          <div className="pup-result-row">
            <span className="pup-result-key">Title</span>
            <span className="pup-result-val">{result.result.title || '—'}</span>
          </div>
          <div className="pup-result-row">
            <span className="pup-result-key">Claims</span>
            <span className="pup-result-val">{parsedClaims.length}</span>
          </div>
          <div className="pup-result-row">
            <span className="pup-result-key">References</span>
            <span className="pup-result-val">{refCount}</span>
          </div>
          <div className="pup-result-row">
            <span className="pup-result-key">Figures</span>
            <span className="pup-result-val">{extractedFigures.length}</span>
          </div>
        </div>
      )}
    </div>
  );
}
