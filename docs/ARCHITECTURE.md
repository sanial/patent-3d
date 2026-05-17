# Patent-3D Viewer — Architecture

How the components are visualized in Three.js and how the layers interact.

## The layer stack

The viewport is actually **three overlapping layers** sharing the same DOM container ([src/components/Viewport.tsx](../src/components/Viewport.tsx)):

1. **WebGL canvas** — the Three.js `renderer.domElement`, appended into `viewportRef` by `useThreeScene`.
2. **CSS overlay** — small hint text rendered by React.
3. **Annotation labels** — HTML buttons portaled into the same viewport `div`, positioned in screen-space.

React owns the DOM tree; Three.js owns one `<canvas>` child inside it. They never touch each other's nodes — they only communicate through the imperative API returned by the hook.

## The Three.js scene ([useThreeScene.ts](../src/hooks/useThreeScene.ts))

Inside a single `useEffect`, the hook builds:

- `WebGLRenderer` (shadows, ACES tone mapping, sRGB) appended into `mountRef`
- `Scene` with fog + dark background
- `PerspectiveCamera` at `(6, 4, 8)`
- `OrbitControls` for drag/zoom/pan
- 3 directional lights (key/fill/rim) + ambient + a `GridHelper` as the "floor"

Then it calls a set of `build*` functions, each of which constructs a `THREE.Group` of primitive meshes (`BoxGeometry`, `CylinderGeometry`, `SphereGeometry`) and attaches metadata:

```ts
g.userData = { refNum: '101', patentData: PATENT_DATA['101'] };
componentMeshesRef.current['101'] = g;
```

That `componentMeshesRef` map (ref number → Group) is the **bridge between the patent data and the 3D objects**. Every imperative operation — highlight, isolate, focus, toggle visibility — is just a lookup in this map plus a mutation on the group.

The build pipeline is:

- `buildSensorModule` × 2 → groups `'101'`, `'102'`
- `buildController` → group `'130'`
- `buildReferenceLine` → group `'108'`
- `buildTray` → group `'105'`
- `buildPlants`, `buildBeams` → decorative, no refNum

## How interaction flows

**Click → selection.** The mount listens for `click`. `handleViewportClick` uses a `Raycaster` against every mesh in the scene, then walks `parent` pointers until it finds one with `userData.refNum`. That refNum is passed to the `onSelect` callback supplied by [App.tsx](../src/App.tsx), which stores it in `selectedRef` state and calls `scene.setHighlight(prev, false)` / `setHighlight(ref, true)` — the highlight just sets `emissive` on every non-transparent material in the group.

**ClaimsPanel / AnnotationLabels → 3D.** They call `onSelect(ref)` (regular React prop), which routes to `handleSelect2` in `App.tsx`, which mutates the same highlight + state. Buttons like Isolate / Focus / Show All call the corresponding hook methods (`isolateComponent`, `focusComponent`, `showAll`) which act directly on the groups in `componentMeshesRef`.

## How the annotation labels stay aligned

This is the most interesting cross-layer coupling. [AnnotationLabels](../src/components/AnnotationLabels.tsx) has a hardcoded `ANCHORS` map of world-space `Vector3`s for each ref. On mount it registers an `update(camera)` callback via `onRegisterUpdater`:

```ts
// In useThreeScene's render loop:
const animate = () => {
  controls.update();
  renderer.render(scene, camera);
  annUpdateRef.current?.(camera);   // ← per-frame callback
};
```

Every frame, `update` projects each anchor through the camera, converts NDC to pixel coords, and `setLabels(...)`. React re-renders the buttons with `transform: translate(x, y)`. The `inFront: projected.z < 1` check hides labels when the anchor is behind the camera.

So the label layer is essentially a **React-rendered HUD driven by a Three.js render-loop tick** — no canvas drawing, no `CSS2DRenderer`, just `setState` per frame.

## Data flow summary

```
PATENT_DATA ──┐
              ├──► useThreeScene builds groups, stores in componentMeshesRef
              │         │
              │         ├──► raycaster click ──► onSelect ──► App state
              │         │
              │         └──► render loop ──► annUpdateRef ──► AnnotationLabels setState
              │
              ├──► ClaimsPanel (reads via selectedRef)
              └──► AnnotationLabels (reads via ref → PATENT_DATA[ref])

App state (selectedRef, wireframeActive, visibilityMap)
   │
   └──► imperative calls into scene.{setHighlight, toggleWireframe,
                                     isolateComponent, focusComponent, ...}
```

## Key architectural choice

**React state is the source of truth for UI**, but **`componentMeshesRef` is the source of truth for 3D objects**. The hook exposes a small imperative API so React never has to re-render the scene — it just pokes it.
