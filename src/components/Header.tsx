import styles from './Header.module.css';

interface HeaderProps {
  onWireframe: () => void;
  onAnnotations: () => void;
  onReset: () => void;
  onExport: () => void;
  wireframeActive: boolean;
  annotationsActive: boolean;
}

export function Header({ onWireframe, onAnnotations, onReset, onExport, wireframeActive, annotationsActive }: HeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-label="Patent 3D Viewer">
          <rect x="1" y="5" width="12" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
          <rect x="9" y="1" width="12" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.5" fill="var(--bg)" strokeDasharray="2 1.5" />
          <line x1="5" y1="11" x2="15" y2="5" stroke="currentColor" strokeWidth="1" strokeDasharray="2 1.5" opacity="0.5" />
        </svg>
        <span className={styles.logoText}>Patent Viewer <span className={styles.logoSub}>3D</span></span>
      </div>
      <div className={styles.right}>
        <button
          className={`${styles.btnIcon} ${wireframeActive ? styles.active : ''}`}
          onClick={onWireframe}
          title="Toggle wireframe"
          aria-label="Toggle wireframe"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
          </svg>
        </button>
        <button
          className={`${styles.btnIcon} ${annotationsActive ? styles.active : ''}`}
          onClick={onAnnotations}
          title="Toggle annotations"
          aria-label="Toggle annotations"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
        <button className={styles.btnIcon} onClick={onReset} title="Reset camera" aria-label="Reset camera">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
          </svg>
        </button>
        <button className={styles.btnPrimary} onClick={onExport}>Export GLB</button>
      </div>
    </header>
  );
}
