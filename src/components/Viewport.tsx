import { forwardRef } from 'react';
import styles from './Viewport.module.css';

interface ViewportProps {
  selectedRef: string | null;
}

export const Viewport = forwardRef<HTMLDivElement, ViewportProps>(({ selectedRef }, ref) => {
  return (
    <div className={styles.viewport} ref={ref}>
      <div className={styles.overlay}>
        <span className={styles.hint}>Drag to orbit · Scroll to zoom · Right-drag to pan</span>
        {selectedRef && <span className={styles.hint}>Ref {selectedRef} selected</span>}
      </div>
    </div>
  );
});
Viewport.displayName = 'Viewport';
