import { useCallback, useState } from 'react';
import type { ExtractedFigure, ParsedClaim } from './usePatentUpload';

export interface ClaimAnalysis {
  number: number;
  type: string;
  summary: string;
  key_elements: string[];
  ref_numbers: string[];
  dependsOn: number | null;
}

export interface ComponentInfo {
  role: string;
  appears_in_claims: number[];
}

export interface ClaimsAnalysisResult {
  claims: ClaimAnalysis[];
  component_summary: Record<string, ComponentInfo>;
  novelty_keywords: string[];
  figure_ref_reconciliation: Record<string, string>;
}

export type AnalysisStatus = 'idle' | 'running' | 'done' | 'error';

interface UseClaimsAnalysisOptions {
  apiUrl?: string;
}

export function useClaimsAnalysis({ apiUrl }: UseClaimsAnalysisOptions = {}) {
  const baseUrl =
    apiUrl ??
    (import.meta as unknown as { env?: Record<string, string> }).env?.VITE_OCR_API_URL ??
    'http://localhost:8000';

  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<ClaimsAnalysisResult | null>(null);

  const reset = useCallback(() => {
    setStatus('idle');
    setError(null);
    setAnalysis(null);
  }, []);

  const analyze = useCallback(
    async (
      claims: ParsedClaim[],
      refDefinitions: Record<string, string>,
      figures: ExtractedFigure[] | undefined,
      opts: { includeFigures?: boolean } = {}
    ) => {
      if (!claims.length) return;
      setStatus('running');
      setError(null);

      const figure_pngs_b64 =
        opts.includeFigures && figures
          ? figures
              .map((f) => f.pngDataUrl.replace(/^data:image\/png;base64,/, ''))
              .filter(Boolean)
          : null;

      try {
        const resp = await fetch(`${baseUrl}/api/analyze-claims`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            claims: claims.map((c) => ({ number: c.number, body: c.body })),
            ref_definitions: refDefinitions,
            figure_pngs_b64,
          }),
        });
        if (!resp.ok) {
          const detail = await resp.text();
          throw new Error(`Server ${resp.status}: ${detail}`);
        }
        const json = (await resp.json()) as { analysis: ClaimsAnalysisResult };
        setAnalysis(json.analysis);
        setStatus('done');
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setStatus('error');
      }
    },
    [baseUrl]
  );

  return { analyze, reset, status, error, analysis };
}
