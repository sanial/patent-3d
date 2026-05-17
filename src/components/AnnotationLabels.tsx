import { useEffect, useRef, useState, useCallback } from 'react';
import { createPortal } from 'react-dom';
import * as THREE from 'three';
import { PATENT_DATA } from '../data/patentData';
import styles from './AnnotationLabels.module.css';

interface LabelState {
  ref: string;
  x: number;
  y: number;
  inFront: boolean;
}

// World-space anchor positions for each annotated component
const ANCHORS: Record<string, [number, number, number]> = {
  '101': [-2.0, 0.5, 0.3],
  '102': [2.0, 0.5, 0.3],
  '130': [0, 3.3, 0],
  '108': [3.4, 0.3, 0],
  '105': [3.6, -2.5, 0],
};

interface Props {
  visible: boolean;
  viewportRef: React.RefObject<HTMLDivElement | null>;
  onSelect: (ref: string) => void;
  onRegisterUpdater: (fn: (camera: THREE.Camera) => void) => void;
}

export function AnnotationLabels({ visible, viewportRef, onSelect, onRegisterUpdater }: Props) {
  const [labels, setLabels] = useState<LabelState[]>([]);
  const [portalTarget, setPortalTarget] = useState<HTMLDivElement | null>(null);

  // Stable Vector3 instances, one per anchor
  const anchorVecsRef = useRef(
    Object.entries(ANCHORS).map(([ref, pos]) => ({
      ref,
      vec: new THREE.Vector3(...pos),
    }))
  );

  const update = useCallback(
    (camera: THREE.Camera) => {
      const mount = viewportRef.current;
      if (!mount) return;
      const w = mount.clientWidth;
      const h = mount.clientHeight;

      setLabels(
        anchorVecsRef.current.map(({ ref, vec }) => {
          const projected = vec.clone().project(camera);
          return {
            ref,
            x: ((projected.x + 1) / 2) * w,
            y: ((-projected.y + 1) / 2) * h,
            inFront: projected.z < 1,
          };
        })
      );
    },
    [viewportRef]
  );

  useEffect(() => {
    onRegisterUpdater(update);
  }, [onRegisterUpdater, update]);

  // Resolve portal target once the viewport mounts
  useEffect(() => {
    setPortalTarget(viewportRef.current);
  }, [viewportRef]);

  if (!visible || !portalTarget) return null;

  return createPortal(
    <div className={styles.overlay}>
      {labels
        .filter((l) => l.inFront)
        .map(({ ref, x, y }) => {
          const data = PATENT_DATA[ref];
          if (!data) return null;
          return (
            <button
              key={ref}
              className={styles.label}
              style={{ transform: `translate(${x}px, ${y}px)` }}
              onClick={() => onSelect(ref)}
            >
              <span className={styles.badge} style={{ borderColor: data.color, color: data.color }}>
                {ref}
              </span>
              <span className={styles.name}>{data.name}</span>
            </button>
          );
        })}
    </div>,
    portalTarget
  );
}
