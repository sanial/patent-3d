import { PATENT_DATA } from '../data/patentData';
import styles from './ClaimsPanel.module.css';

interface ClaimsPanelProps {
  selectedRef: string | null;
  visibilityMap: Record<string, boolean>;
  onSelect: (ref: string) => void;
  onToggleVisibility: (ref: string) => void;
  onIsolate: () => void;
  onShowAll: () => void;
  onFocus: () => void;
  onFileDrop: (files: FileList) => void;
}

export function ClaimsPanel({
  selectedRef,
  visibilityMap,
  onSelect,
  onToggleVisibility,
  onIsolate,
  onShowAll,
  onFocus,
  onFileDrop,
}: ClaimsPanelProps) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.sidebarHeader}>
        <div className={styles.sidebarTitle}>Components</div>
      </div>

      <div className={styles.compList}>
        {Object.entries(PATENT_DATA).map(([ref, data]) => (
          <div
            key={ref}
            className={`${styles.compItem} ${selectedRef === ref ? styles.active : ''}`}
            onClick={() => onSelect(ref)}
          >
            <div className={styles.compBadge} style={{ background: `${data.color}22`, color: data.color }}>
              {ref}
            </div>
            <div className={styles.compInfo}>
              <div className={styles.compRef}>{ref} — {data.name}</div>
              <div className={styles.compName}>Claims: {data.claims.join(', ')}</div>
            </div>
            <button
              className={`${styles.compVis} ${visibilityMap[ref] === false ? styles.hiddenIcon : ''}`}
              title="Toggle visibility"
              aria-label="Toggle visibility"
              onClick={(e) => { e.stopPropagation(); onToggleVisibility(ref); }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </button>
          </div>
        ))}
      </div>

      {/* Upload zone */}
      <label className={styles.uploadZone}>
        <input
          type="file"
          accept=".glb,.obj,.gltf"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && onFileDrop(e.target.files)}
        />
        <div className={styles.upIcon}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        </div>
        <strong>Drop GLB / OBJ here</strong>
        <p>or click to browse</p>
      </label>

      {/* Detail panel */}
      <div className={styles.detail}>
        {!selectedRef || !PATENT_DATA[selectedRef] ? (
          <div className={styles.detailEmpty}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            </svg>
            <p>Click a component to see patent details</p>
          </div>
        ) : (() => {
          const d = PATENT_DATA[selectedRef];
          const primaryClaim = d.claims[0];
          return (
            <>
              <div className={styles.detailRef}>
                <div className={styles.compBadge} style={{ background: `${d.color}22`, color: d.color, fontSize: 14 }}>
                  {selectedRef}
                </div>
                <span className={styles.detailRefNum}>{d.name}</span>
              </div>
              <p className={styles.detailDesc}>{d.desc}</p>
              <div className={styles.detailSection}>
                <div className={styles.detailLabel}>Patent Claims</div>
                <div className={styles.claimChips}>
                  {d.claims.map((c) => (
                    <span key={c} className={`${styles.chip} ${c === primaryClaim ? styles.primary : ''}`}>
                      Claim {c}
                    </span>
                  ))}
                </div>
              </div>
              <div className={styles.detailSection}>
                <div className={styles.detailLabel}>Properties</div>
                {Object.entries(d.props).map(([k, v]) => (
                  <div key={k} className={styles.propRow}>
                    <span className={styles.propKey}>{k}</span>
                    <span className={styles.propVal}>{v}</span>
                  </div>
                ))}
              </div>
            </>
          );
        })()}
      </div>

      <div className={styles.controls}>
        <button className={styles.btn} onClick={onIsolate}>Isolate</button>
        <button className={styles.btn} onClick={onShowAll}>Show All</button>
        <button className={`${styles.btn} ${styles.btnRight}`} onClick={onFocus}>Focus</button>
      </div>
    </aside>
  );
}
