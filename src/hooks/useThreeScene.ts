import { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { PATENT_DATA } from '../data/patentData';

type SelectCallback = (ref: string | null) => void;

export function useThreeScene(
  mountRef: React.RefObject<HTMLDivElement | null>,
  onSelect: SelectCallback
) {
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const componentMeshesRef = useRef<Record<string, THREE.Group>>({});
  const wireframeOnRef = useRef(false);
  const rafRef = useRef<number>(0);
  const annUpdateRef = useRef<((camera: THREE.Camera) => void) | null>(null);

  // ── Helpers ──────────────────────────────────────────────────────
  const makeMat = (hex: string, metalness = 0.3, roughness = 0.5) =>
    new THREE.MeshStandardMaterial({ color: new THREE.Color(hex), metalness, roughness });

  function addMesh(
    geo: THREE.BufferGeometry,
    mat: THREE.Material,
    parent: THREE.Object3D,
    pos: [number, number, number] = [0, 0, 0],
    scale: [number, number, number] = [1, 1, 1],
    rot: [number, number, number] = [0, 0, 0]
  ) {
    const m = new THREE.Mesh(geo, mat);
    m.position.set(...pos);
    m.scale.set(...scale);
    m.rotation.set(...rot.map((d) => (d * Math.PI) / 180) as [number, number, number]);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    return m;
  }

  // ── Scene builders ────────────────────────────────────────────────
  function buildSensorModule(scene: THREE.Scene, refNum: string, tx: number, rotZ: number, beamDir: number) {
    const matBlack    = makeMat('#141416', 0.6, 0.4);
    const matDarkGray = makeMat('#2a2a2e', 0.5, 0.5);
    const matWire     = makeMat('#3a3a40', 0.8, 0.3);
    const matGlass    = new THREE.MeshStandardMaterial({ color: 0x6ab0cc, metalness: 0.1, roughness: 0.05, transparent: true, opacity: 0.7 });

    const g = new THREE.Group();
    g.position.x = tx;
    g.rotation.z = (rotZ * Math.PI) / 180;
    g.userData = { refNum, patentData: PATENT_DATA[refNum] };

    addMesh(new THREE.BoxGeometry(0.8, 0.55, 0.55), matBlack, g);
    addMesh(new THREE.CylinderGeometry(0.13, 0.13, 0.32, 16), matDarkGray, g, [beamDir * 0.54, -0.18, 0], [1, 1, 1], [0, 0, 90]);
    addMesh(new THREE.SphereGeometry(0.1, 16, 12), matGlass, g, [beamDir * 0.72, -0.18, 0]);
    ([-0.14, 0, 0.14] as number[]).forEach((px) => addMesh(new THREE.CylinderGeometry(0.022, 0.022, 0.2, 8), matWire, g, [px, -0.36, 0]));
    addMesh(new THREE.BoxGeometry(0.07, 0.32, 0.38), matDarkGray, g, [-beamDir * 0.44, 0.08, 0]);
    addMesh(new THREE.BoxGeometry(0.76, 0.5, 0.02), matBlack, g, [0, 0, 0.28]);

    scene.add(g);
    componentMeshesRef.current[refNum] = g;
  }

  function buildController(scene: THREE.Scene) {
    const matDarkGray = makeMat('#2a2a2e', 0.5, 0.5);
    const matBlack    = makeMat('#141416', 0.6, 0.4);
    const matWire     = makeMat('#3a3a40', 0.8, 0.3);

    const g = new THREE.Group();
    g.position.set(0, 2.8, 0);
    g.userData = { refNum: '130', patentData: PATENT_DATA['130'] };

    addMesh(new THREE.BoxGeometry(1.4, 0.9, 0.8), matDarkGray, g);
    addMesh(new THREE.BoxGeometry(1.35, 0.85, 0.02), matBlack, g, [0, 0, 0.41]);
    addMesh(new THREE.CylinderGeometry(0.055, 0.055, 0.22, 8), matWire, g, [0, -0.56, 0]);
    ([[0.35, 0.1], [-0.35, 0.1], [0.35, -0.1], [-0.35, -0.1]] as [number, number][]).forEach(([lx, ly]) => {
      addMesh(new THREE.SphereGeometry(0.028, 8, 6), makeMat('#4f98a3', 0, 0.2), g, [lx, ly, 0.41]);
    });

    scene.add(g);
    componentMeshesRef.current['130'] = g;
  }

  function buildReferenceLine(scene: THREE.Scene) {
    const matWire = makeMat('#3a3a40', 0.8, 0.3);
    const g = new THREE.Group();
    g.userData = { refNum: '108', patentData: PATENT_DATA['108'] };
    for (let x = -3; x <= 3; x += 0.35) {
      addMesh(new THREE.CylinderGeometry(0.01, 0.01, 0.2, 6), matWire, g, [x, 0, 0], [1, 1, 1], [0, 0, 90]);
    }
    scene.add(g);
    componentMeshesRef.current['108'] = g;
  }

  function buildTray(scene: THREE.Scene) {
    const matWhite = makeMat('#e8e8e4', 0.0, 0.6);
    const g = new THREE.Group();
    g.position.set(0, -2.65, 0);
    g.userData = { refNum: '105', patentData: PATENT_DATA['105'] };
    addMesh(new THREE.BoxGeometry(7.2, 0.16, 1.0), matWhite, g);
    addMesh(new THREE.BoxGeometry(0.07, 0.12, 1.0), matWhite, g, [-3.57, 0.12, 0]);
    addMesh(new THREE.BoxGeometry(0.07, 0.12, 1.0), matWhite, g, [3.57, 0.12, 0]);
    scene.add(g);
    componentMeshesRef.current['105'] = g;
  }

  function buildPlants(scene: THREE.Scene) {
    const matGreen = makeMat('#2d8c3e', 0.0, 0.8);
    ([-2.7, -1.8, -0.9, 0, 0.9, 1.8, 2.7] as number[]).forEach((px) => {
      const g = new THREE.Group();
      g.position.set(px, -2.15, 0);
      addMesh(new THREE.CylinderGeometry(0.022, 0.028, 0.32, 6), matGreen, g, [0, 0.16, 0]);
      ([[- 0.11, 0.28], [0.11, 0.2], [0, 0.38]] as [number, number][]).forEach(([lx, ly], i) => {
        const leaf = new THREE.Mesh(new THREE.SphereGeometry(0.09, 8, 6), matGreen);
        leaf.position.set(lx, ly, 0);
        leaf.scale.set(1.4, 0.65, 0.45 + i * 0.05);
        g.add(leaf);
      });
      scene.add(g);
    });
  }

  function buildBeams(scene: THREE.Scene) {
    const matWire = makeMat('#3a3a40', 0.8, 0.3);
    const beamMat = makeMat('#4f98a3', 0, 1);

    const b109g = new THREE.Group();
    for (let t = 0; t < 1; t += 0.12) {
      const seg = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.28, 6), beamMat);
      seg.position.set(-2 + t * 2, -t * 2.2, 0);
      seg.rotation.z = (-55 * Math.PI) / 180;
      b109g.add(seg);
    }
    scene.add(b109g);

    const b110g = new THREE.Group();
    for (let t = 0; t < 1; t += 0.12) {
      const seg = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.008, 0.28, 6), beamMat);
      seg.position.set(2 - t * 2, -t * 2.2, 0);
      seg.rotation.z = (55 * Math.PI) / 180;
      b110g.add(seg);
    }
    scene.add(b110g);

    addMesh(new THREE.CylinderGeometry(0.018, 0.018, 2.0, 8), matWire, scene, [0, 1.5, 0]);
    for (let x = -3.2; x <= 3.2; x += 0.4) {
      addMesh(new THREE.BoxGeometry(0.22, 0.008, 0.9), makeMat('#2d8c3e', 0, 1), scene, [x, -2.1, 0]);
    }
  }

  // ── Public methods ────────────────────────────────────────────────
  const setHighlight = useCallback((refNum: string, on: boolean) => {
    const g = componentMeshesRef.current[refNum];
    if (!g) return;
    g.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material && !(child.material as THREE.MeshStandardMaterial).transparent) {
        const mat = child.material as THREE.MeshStandardMaterial;
        mat.emissive = on ? new THREE.Color(0x4f98a3) : new THREE.Color(0x000000);
        mat.emissiveIntensity = on ? 0.25 : 0;
      }
    });
  }, []);

  const toggleWireframe = useCallback(() => {
    wireframeOnRef.current = !wireframeOnRef.current;
    const scene = sceneRef.current;
    if (!scene) return;
    scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh && obj.material) {
        (obj.material as THREE.MeshStandardMaterial).wireframe = wireframeOnRef.current;
      }
    });
    return wireframeOnRef.current;
  }, []);

  const resetCamera = useCallback(() => {
    cameraRef.current?.position.set(6, 4, 8);
    controlsRef.current?.target.set(0, 0, 0);
    controlsRef.current?.update();
  }, []);

  const isolateComponent = useCallback((refNum: string) => {
    Object.keys(componentMeshesRef.current).forEach((ref) => {
      const g = componentMeshesRef.current[ref];
      if (g) g.visible = ref === refNum;
    });
  }, []);

  const showAll = useCallback(() => {
    Object.values(componentMeshesRef.current).forEach((g) => { if (g) g.visible = true; });
  }, []);

  const toggleVisibility = useCallback((refNum: string): boolean => {
    const g = componentMeshesRef.current[refNum];
    if (!g) return true;
    g.visible = !g.visible;
    return g.visible;
  }, []);

  const focusComponent = useCallback((refNum: string) => {
    const g = componentMeshesRef.current[refNum];
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!g || !camera || !controls) return;
    const box = new THREE.Box3().setFromObject(g);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(new THREE.Vector3(1, 0.6, 1).normalize(), size * 2.2);
    controls.update();
  }, []);

  const exportGLB = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const exporter = new GLTFExporter();
    exporter.parse(
      scene,
      (glb) => {
        const blob = new Blob([glb as ArrayBuffer], { type: 'application/octet-stream' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'patent_fig1_scene.glb';
        a.click();
      },
      (err) => console.error(err),
      { binary: true }
    );
  }, []);

  const loadFiles = useCallback((files: FileList) => {
    const scene = sceneRef.current;
    if (!scene) return;
    const gltfLoader = new GLTFLoader();
    const objLoader = new OBJLoader();
    Array.from(files).forEach((file) => {
      const url = URL.createObjectURL(file);
      const name = file.name.replace(/\.[^.]+$/, '');
      if (file.name.endsWith('.glb') || file.name.endsWith('.gltf')) {
        gltfLoader.load(url, (gltf) => {
          const model = gltf.scene;
          const box = new THREE.Box3().setFromObject(model);
          const size = box.getSize(new THREE.Vector3()).length();
          model.scale.multiplyScalar(2 / size);
          const center = box.getCenter(new THREE.Vector3());
          model.position.sub(center).setY(1);
          const refMatch = name.match(/(\d{3})/);
          const ref = refMatch ? refMatch[1] : 'EXT';
          model.userData = { refNum: ref, patentData: PATENT_DATA[ref] || { name, color: '#bb9a4f', claims: [], desc: `Loaded: ${name}`, props: {} } };
          scene.add(model);
          componentMeshesRef.current[ref] = model as unknown as THREE.Group;
        });
      } else if (file.name.endsWith('.obj')) {
        objLoader.load(url, (obj) => {
          scene.add(obj);
        });
      }
    });
  }, []);

  const setAnnotationUpdater = useCallback((fn: (camera: THREE.Camera) => void) => {
    annUpdateRef.current = fn;
  }, []);

  const getCamera = useCallback(() => cameraRef.current, []);

  // ── Raycaster click ───────────────────────────────────────────────
  const handleViewportClick = useCallback(
    (e: MouseEvent, viewportEl: HTMLElement) => {
      const scene = sceneRef.current;
      const camera = cameraRef.current;
      if (!scene || !camera) return;

      const rect = viewportEl.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);
      const meshes: THREE.Mesh[] = [];
      scene.traverse((obj) => { if (obj instanceof THREE.Mesh) meshes.push(obj); });
      const hits = raycaster.intersectObjects(meshes);
      if (hits.length) {
        let obj: THREE.Object3D | null = hits[0].object;
        while (obj && !obj.userData.refNum) obj = obj.parent;
        onSelect(obj?.userData?.refNum ?? null);
      } else {
        onSelect(null);
      }
    },
    [onSelect]
  );

  // ── Init ──────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d0c0a);
    scene.fog = new THREE.Fog(0x0d0c0a, 20, 60);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
    camera.position.set(6, 4, 8);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 1;
    controls.maxDistance = 30;
    controlsRef.current = controls;

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.4));
    const key = new THREE.DirectionalLight(0xfff4e0, 2.5);
    key.position.set(5, 8, 6);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xc8e8f0, 0.8);
    fill.position.set(-4, 3, -3);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0x4f98a3, 0.4);
    rim.position.set(0, -5, -8);
    scene.add(rim);

    // Grid
    const grid = new THREE.GridHelper(20, 40, 0x2a2926, 0x1e1d1b);
    grid.position.y = -2.8;
    scene.add(grid);

    // Build scene
    buildSensorModule(scene, '101', -2.0, 35, -1);
    buildSensorModule(scene, '102', 2.0, -35, 1);
    buildController(scene);
    buildReferenceLine(scene);
    buildTray(scene);
    buildPlants(scene);
    buildBeams(scene);

    // Resize
    const resizeObserver = new ResizeObserver(() => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    });
    resizeObserver.observe(mount);
    const w = mount.clientWidth;
    const h = mount.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();

    // Click handler
    const clickHandler = (e: MouseEvent) => handleViewportClick(e, mount);
    mount.addEventListener('click', clickHandler);

    // Render loop
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      annUpdateRef.current?.(camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(rafRef.current);
      resizeObserver.disconnect();
      mount.removeEventListener('click', clickHandler);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return {
    setHighlight,
    toggleWireframe,
    resetCamera,
    isolateComponent,
    showAll,
    toggleVisibility,
    focusComponent,
    exportGLB,
    loadFiles,
    setAnnotationUpdater,
    getCamera,
    componentMeshesRef,
  };
}
