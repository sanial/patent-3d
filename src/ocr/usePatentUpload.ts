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

      const form = new FormData();
      form.append('file', file);

      try {
        const resp = await fetch(`${baseUrl}/api/parse-patent`, {
          method: 'POST',
          body: form,
        });
        setStatus('parsing');
        setProgress(0.5);

        if (!resp.ok) {
          const detail = await resp.text();
          throw new Error(`Server ${resp.status}: ${detail}`);
        }

        const json = (await resp.json()) as ParsePatentResponse;
        setResult(json);
        setProgress(1);
        setStatus('done');
        onResult?.(json);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setStatus('error');
      }
    },
    [baseUrl, onResult]
  );

  return { upload, reset, status, progress, error, result };
}
