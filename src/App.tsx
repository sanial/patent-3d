import { useState, useRef } from 'react';
import { Header } from './components/Header';
import { Viewport } from './components/Viewport';
import { ClaimsPanel } from './components/ClaimsPanel';
import { AnnotationLabels } from './components/AnnotationLabels';
import { PatentUploadPanel } from './ocr/PatentUploadPanel';
import type { ParsePatentResponse } from './ocr/usePatentUpload';
import { useThreeScene } from './hooks/useThreeScene';
import styles from './App.module.css';

export default function App() {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  const [wireframeActive, setWireframeActive] = useState(false);
  const [annotationsActive, setAnnotationsActive] = useState(true);
  const [visibilityMap, setVisibilityMap] = useState<Record<string, boolean>>({});

  const scene = useThreeScene(viewportRef, (ref) => {
    setSelectedRef((prev) => {
      if (prev) scene.setHighlight(prev, false);
      if (ref) scene.setHighlight(ref, true);
      return ref;
    });
  });

  const handleSelect2 = (ref: string) => {
    if (selectedRef) scene.setHighlight(selectedRef, false);
    scene.setHighlight(ref, true);
    setSelectedRef(ref);
  };

  const handleWireframe = () => {
    const on = scene.toggleWireframe();
    setWireframeActive(!!on);
  };

  const handleToggleVis = (ref: string) => {
    const isVisible = scene.toggleVisibility(ref);
    setVisibilityMap((prev) => ({ ...prev, [ref]: isVisible }));
  };

  const handleShowAll = () => {
    scene.showAll();
    setVisibilityMap({});
  };

  return (
    <div className={styles.app}>
      <Header
        onWireframe={handleWireframe}
        onAnnotations={() => setAnnotationsActive((a) => !a)}
        onReset={scene.resetCamera}
        onExport={scene.exportGLB}
        wireframeActive={wireframeActive}
        annotationsActive={annotationsActive}
      />
      <Viewport ref={viewportRef} selectedRef={selectedRef} />
      <aside className={styles.leftSidebar}>
        <PatentUploadPanel
          onResult={(resp: ParsePatentResponse) => {
            console.log('Patent parsed:', resp);
          }}
        />
      </aside>
      <ClaimsPanel
        selectedRef={selectedRef}
        visibilityMap={visibilityMap}
        onSelect={handleSelect2}
        onToggleVisibility={handleToggleVis}
        onIsolate={() => selectedRef && scene.isolateComponent(selectedRef)}
        onShowAll={handleShowAll}
        onFocus={() => selectedRef && scene.focusComponent(selectedRef)}
        onFileDrop={scene.loadFiles}
      />
      <AnnotationLabels
        visible={annotationsActive}
        viewportRef={viewportRef}
        onSelect={handleSelect2}
        onRegisterUpdater={scene.setAnnotationUpdater}
      />
    </div>
  );
}
