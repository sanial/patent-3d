import { useCallback, useState } from 'react';

export interface RefEntry {
  ref: string;
  label: string;
  description: string;
  snippets: string[];
  pages: number[];
}

export interface FigureHit {
  page: number;
  text: string;
  bbox: [number, number, number, number];
  confidence: number;
}

export interface ExtractedFigure {
  figureId: string;
  page: number;
  bbox: [number, number, number, number];
  captionText: string;
  refNumbersOriginallyInside: string[];
  pngDataUrl: string;
  svg: string;
}

export interface ParsedClaim {
  number: number;
  type: 'independent' | 'dependent' | string;
  body: string;
  refs: string[];
  dependsOn: number | null;
}

export interface StructuredPatent {
  sections: Record<string, string>;
  claims: ParsedClaim[];
  refDefinitions: Record<string, string>;
}

export interface PatentOCRResult {
  title: string;
  abstract: string;
  claims: string[];
  refEntries: Record<string, RefEntry>;
  figures: FigureHit[];
  extractedFigures: ExtractedFigure[];
  structure: StructuredPatent | null;
}

export interface PatentDataEntry {
  label: string;
  description: string;
  claims: number[];
  pages: number[];
}

export interface ParsePatentResponse {
  filename: string;
  result: PatentOCRResult;
  patentData: Record<string, PatentDataEntry>;
}

export type UploadStatus = 'idle' | 'uploading' | 'parsing' | 'done' | 'error';

interface UsePatentUploadOptions {
  apiUrl?: string;
  onResult?: (resp: ParsePatentResponse) => void;
}

export function usePatentUpload({ apiUrl, onResult }: UsePatentUploadOptions = {}) {
  const baseUrl =
    apiUrl ??
    (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_OCR_API_URL ??
    'http://localhost:8000';

  const [status, setStatus] = useState<UploadStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ParsePatentResponse | null>(null);

  const reset = useCallback(() => {
    setStatus('idle');
    setProgress(0);
    setError(null);
    setResult(null);
  }, []);

  const upload = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setError('Only PDF files are supported');
        setStatus('error');
        return;
      }

      setError(null);
      setResult(null);
      setStatus('uploading');
      setProgress(0);

      const buildForm = () => {
        const f = new FormData();
        f.append('file', file);
        return f;
      };

      // Fire the fast figure extractor in parallel — it usually finishes
      // well before /api/parse-patent and lets the Figures tab populate
      // within seconds instead of waiting for the full pipeline.
      const figuresPromise = (async () => {
        try {
          console.log('[patent] fast figure extractor: requesting');
          const r = await fetch(
            `${baseUrl}/api/extract-figures?use_gemma=true`,
            { method: 'POST', body: buildForm() }
          );
          if (!r.ok) {
            const detail = await r.text();
            console.warn('[patent] fast figure extractor failed', r.status, detail);
            return null;
          }
          const j = (await r.json()) as { extractedFigures?: ExtractedFigure[] };
          console.log('[patent] fast figure extractor returned', j.extractedFigures?.length, 'figures');
          return j.extractedFigures ?? null;
        } catch (e) {
          console.warn('[patent] fast figure extractor threw', e);
          return null;
        }
      })();

      // As soon as the fast extractor returns, surface its figures
      // even if /api/parse-patent is still running.
      void figuresPromise.then((figs) => {
        if (!figs || figs.length === 0) return;
        setResult((prev) => {
          if (prev) {
            return {
              ...prev,
              result: { ...prev.result, extractedFigures: figs },
            };
          }
          // No full result yet — stash a placeholder so the Figures tab works.
          return {
            filename: file.name,
            result: {
              title: '',
              abstract: '',
              claims: [],
              refEntries: {},
              figures: [],
              extractedFigures: figs,
              structure: null,
            },
            patentData: {},
          };
        });
      });

      // Fire the fast claim extractor (Gemma 4 vision on the last few
      // pages) in parallel too. Claims tab populates within seconds,
      // bypassing the slow OCR path entirely.
      const claimsPromise = (async () => {
        try {
          console.log('[patent] fast claim extractor: requesting');
          const r = await fetch(
            `${baseUrl}/api/extract-claims?last_pages=4`,
            { method: 'POST', body: buildForm() }
          );
          if (!r.ok) {
            const detail = await r.text();
            console.warn('[patent] fast claim extractor failed', r.status, detail);
            return null;
          }
          const j = (await r.json()) as { claims?: ParsedClaim[] };
          console.log('[patent] fast claim extractor returned', j.claims?.length, 'claims');
          return j.claims ?? null;
        } catch (e) {
          console.warn('[patent] fast claim extractor threw', e);
          return null;
        }
      })();

      void claimsPromise.then((claims) => {
        if (!claims || claims.length === 0) return;
        setResult((prev) => {
          const claimBodies = claims.map((c) => c.body);
          if (prev) {
            const structure = prev.result.structure
              ? { ...prev.result.structure, claims }
              : { sections: {}, claims, refDefinitions: {} };
            return {
              ...prev,
              result: {
                ...prev.result,
                claims: claimBodies,
                structure,
              },
            };
          }
          return {
            filename: file.name,
            result: {
              title: '',
              abstract: '',
              claims: claimBodies,
              refEntries: {},
              figures: [],
              extractedFigures: [],
              structure: { sections: {}, claims, refDefinitions: {} },
            },
            patentData: {},
          };
        });
      });

      try {
        const resp = await fetch(`${baseUrl}/api/parse-patent`, {
          method: 'POST',
          body: buildForm(),
        });
        setStatus('parsing');
        setProgress(0.5);

        if (!resp.ok) {
          const detail = await resp.text();
          throw new Error(`Server ${resp.status}: ${detail}`);
        }

        const json = (await resp.json()) as ParsePatentResponse;
        // If the fast extractor already returned figures, prefer those —
        // they're more reliable than the inline ones from parse-patent.
        const fastFigs = await figuresPromise;
        const fastClaims = await claimsPromise;
        let merged: ParsePatentResponse =
          fastFigs && fastFigs.length > 0
            ? {
                ...json,
                result: { ...json.result, extractedFigures: fastFigs },
              }
            : json;
        // Prefer fast Gemma-vision claims when parse-patent came up empty.
        const parsedClaimCount = merged.result.structure?.claims?.length ?? 0;
        if (fastClaims && fastClaims.length > 0 && parsedClaimCount === 0) {
          const claimBodies = fastClaims.map((c) => c.body);
          const structure = merged.result.structure
            ? { ...merged.result.structure, claims: fastClaims }
            : { sections: {}, claims: fastClaims, refDefinitions: {} };
          merged = {
            ...merged,
            result: {
              ...merged.result,
              claims: claimBodies,
              structure,
            },
          };
        }
        setResult(merged);
        setProgress(1);
        setStatus('done');
        onResult?.(merged);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setStatus('error');
      }
    },
    [baseUrl, onResult]
  );

  return { upload, reset, status, progress, error, result };
}
