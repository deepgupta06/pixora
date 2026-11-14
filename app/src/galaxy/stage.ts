import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const FOV = 50
const TAN = Math.tan(THREE.MathUtils.degToRad(FOV / 2))
const MOVE_MS = 900
/** On-screen radius, in pixels, at which a dot is replaced by the face itself. */
const THUMB_PX = 13
const MAX_THUMBS = 80
const MAX_CACHED = 600

export interface StageScene {
  /** Face ids, used to carry points over from the previous scene. */
  ids: number[]
  positions: Float32Array
  /** r, g, b per face. */
  colors: Float32Array
  /** Faces drawn as a see-through backdrop, such as unnamed ones. */
  faint: boolean[]
  pointSize: number
  /** Line segments: two x, y, z ends per line, with a colour per end. */
  links: Float32Array
  linkColors: Float32Array
  fitRadius: number
  faceUrl: (faceId: number) => string
  /** Look at the scene from the front instead of keeping the current angle. */
  front: boolean
}

export interface Projected {
  x: number
  y: number
  /** Pixels per world unit at this depth. */
  scale: number
  depth: number
  visible: boolean
}

interface Callbacks {
  onHover: (index: number | null) => void
  onPick: (index: number) => void
  /** Called after every frame, to place overlays. */
  onFrame: () => void
}

function disc(mask: boolean): THREE.CanvasTexture {
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = 64
  const ctx = canvas.getContext('2d')!
  if (mask) {
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, 64, 64)
  }
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.arc(32, 32, 30, 0, Math.PI * 2)
  ctx.fill()
  return new THREE.CanvasTexture(canvas)
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

/** The 3D scene: points, lines, close-up faces, the camera and picking. */
export class Stage {
  /** Where each point is right now, while points move between layouts. */
  current: Float32Array = new Float32Array(0)

  private mount: HTMLElement
  private callbacks: Callbacks
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(FOV, 1, 0.1, 5000)
  private controls: OrbitControls
  private points: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial[]>
  private lines: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
  private mask = disc(true)
  private observer: ResizeObserver
  private frameId = 0
  private tick = 0
  /** Size of the view, in CSS pixels. */
  width = 1
  height = 1
  private still = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  private ids: number[] = []
  private pointSize = 0.4
  private fitRadius = 1
  private faceUrl: (faceId: number) => string = () => ''
  private from: Float32Array = new Float32Array(0)
  private to: Float32Array = new Float32Array(0)
  private moveStart: number | null = null

  private cameraMove: {
    start: number
    fromPosition: THREE.Vector3
    toPosition: THREE.Vector3
    fromTarget: THREE.Vector3
    toTarget: THREE.Vector3
  } | null = null

  private pointer: { x: number; y: number } | null = null
  private pointerMoved = false
  private down: { x: number; y: number } | null = null
  private hovered: number | null = null

  private thumbs = new Map<number, THREE.Sprite>()
  private textures = new Map<number, THREE.Texture>()
  private requested = new Set<number>()
  private loader = new THREE.TextureLoader()

  private scratch = new THREE.Vector3()
  private projected: Projected = { x: 0, y: 0, scale: 0, depth: 0, visible: false }

  constructor(mount: HTMLElement, callbacks: Callbacks) {
    this.mount = mount
    this.callbacks = callbacks

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(this.renderer.domElement)

    this.camera.position.set(0.35, 0.3, 1).normalize().multiplyScalar(40)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.1

    const map = disc(false)
    this.points = new THREE.Points(new THREE.BufferGeometry(), [
      new THREE.PointsMaterial({ map, alphaTest: 0.5, vertexColors: true }),
      // Backdrop points never hide what is behind them.
      new THREE.PointsMaterial({ map, vertexColors: true, transparent: true, opacity: 0.3, depthWrite: false }),
    ])
    this.lines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7, depthWrite: false }),
    )
    // Points move every frame during a transition, so their bounds are not kept up to date.
    this.points.frustumCulled = false
    this.lines.frustumCulled = false
    this.scene.add(this.lines, this.points)

    const canvas = this.renderer.domElement
    canvas.addEventListener('pointermove', this.onPointerMove)
    canvas.addEventListener('pointerleave', this.onPointerLeave)
    canvas.addEventListener('pointerdown', this.onPointerDown)
    canvas.addEventListener('pointerup', this.onPointerUp)
    canvas.addEventListener('wheel', this.stopCameraMove, { passive: true })

    this.observer = new ResizeObserver(() => this.resize())
    this.observer.observe(mount)
    this.resize()
    this.frame()
  }

  setScene(next: StageScene): void {
    const before = new Map(this.ids.map((id, i) => [id, i]))
    const first = this.ids.length === 0
    const from = new Float32Array(next.positions.length)
    next.ids.forEach((id, i) => {
      const old = before.get(id)
      // Faces already on screen travel to their new place; new ones appear in place.
      const source = old === undefined ? next.positions : this.current
      const at = (old ?? i) * 3
      from.set(source.subarray(at, at + 3), i * 3)
    })
    this.ids = next.ids
    this.from = from
    this.to = next.positions
    this.current = this.still || first ? next.positions.slice() : from.slice()
    this.moveStart = this.still || first ? null : performance.now()
    this.pointSize = next.pointSize
    this.fitRadius = next.fitRadius
    this.faceUrl = next.faceUrl

    this.points.geometry.dispose()
    this.points.geometry = new THREE.BufferGeometry()
    this.points.geometry.setAttribute('position', new THREE.BufferAttribute(this.current, 3))
    this.points.geometry.setAttribute('color', new THREE.BufferAttribute(next.colors, 3))
    // Solid points are drawn first, then the backdrop over them.
    const solid = next.ids.map((_, i) => i).filter((i) => !next.faint[i])
    const faint = next.ids.map((_, i) => i).filter((i) => next.faint[i])
    this.points.geometry.setIndex([...solid, ...faint])
    this.points.geometry.addGroup(0, solid.length, 0)
    this.points.geometry.addGroup(solid.length, faint.length, 1)
    // A point's size on screen is set at a 90 degree view; this keeps it true to the positions.
    this.points.material[0].size = next.pointSize / TAN
    this.points.material[1].size = (next.pointSize / TAN) * 0.75

    this.lines.geometry.dispose()
    this.lines.geometry = new THREE.BufferGeometry()
    this.lines.geometry.setAttribute('position', new THREE.BufferAttribute(next.links, 3))
    this.lines.geometry.setAttribute('color', new THREE.BufferAttribute(next.linkColors, 3))

    this.clearThumbs()
    this.setHovered(null)
    this.fit(next.front, first)
  }

  /** Frame the whole scene. */
  fit(front = false, instant = false): void {
    this.lookAt(new THREE.Vector3(), this.fitRadius, 1.05, front, instant)
    this.controls.maxDistance = this.distanceFor(this.fitRadius) * 3
    this.controls.minDistance = this.pointSize * 2
  }

  /** Frame one group. */
  focus(centre: [number, number, number], radius: number): void {
    this.lookAt(new THREE.Vector3(...centre), Math.max(radius, 1), 1.5, false, false)
  }

  project(x: number, y: number, z: number): Projected {
    const out = this.projected
    const v = this.scratch.set(x, y, z).applyMatrix4(this.camera.matrixWorldInverse)
    out.depth = -v.z
    out.visible = out.depth > this.camera.near
    if (!out.visible) return out
    v.applyMatrix4(this.camera.projectionMatrix)
    out.x = ((v.x + 1) / 2) * this.width
    out.y = ((1 - v.y) / 2) * this.height
    out.scale = this.height / (2 * TAN * out.depth)
    return out
  }

  dispose(): void {
    cancelAnimationFrame(this.frameId)
    this.observer.disconnect()
    const canvas = this.renderer.domElement
    canvas.removeEventListener('pointermove', this.onPointerMove)
    canvas.removeEventListener('pointerleave', this.onPointerLeave)
    canvas.removeEventListener('pointerdown', this.onPointerDown)
    canvas.removeEventListener('pointerup', this.onPointerUp)
    canvas.removeEventListener('wheel', this.stopCameraMove)
    this.controls.dispose()
    this.clearThumbs()
    this.textures.forEach((t) => t.dispose())
    this.mask.dispose()
    this.points.geometry.dispose()
    this.points.material[0].map?.dispose()
    this.points.material.forEach((m) => m.dispose())
    this.lines.geometry.dispose()
    this.lines.material.dispose()
    this.renderer.dispose()
    canvas.remove()
  }

  private distanceFor(radius: number): number {
    const half = Math.min(Math.atan(TAN), Math.atan(TAN * this.camera.aspect))
    return radius / Math.sin(half)
  }

  private lookAt(target: THREE.Vector3, radius: number, margin: number, front: boolean, instant: boolean): void {
    const direction = front
      ? new THREE.Vector3(0, 0.12, 1)
      : this.camera.position.clone().sub(this.controls.target)
    direction.normalize()
    const position = target.clone().addScaledVector(direction, this.distanceFor(radius) * margin)
    if (instant || this.still) {
      this.cameraMove = null
      this.camera.position.copy(position)
      this.controls.target.copy(target)
      return
    }
    this.cameraMove = {
      start: performance.now(),
      fromPosition: this.camera.position.clone(),
      toPosition: position,
      fromTarget: this.controls.target.clone(),
      toTarget: target,
    }
  }

  private resize(): void {
    const width = this.mount.clientWidth
    const height = this.mount.clientHeight
    if (!width || !height) return
    this.width = width
    this.height = height
    this.renderer.setSize(width, height)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }

  private frame = (): void => {
    this.frameId = requestAnimationFrame(this.frame)
    const now = performance.now()

    if (this.moveStart !== null) {
      const t = Math.min((now - this.moveStart) / MOVE_MS, 1)
      const e = ease(t)
      for (let i = 0; i < this.current.length; i++) {
        this.current[i] = this.from[i] + (this.to[i] - this.from[i]) * e
      }
      this.points.geometry.attributes.position.needsUpdate = true
      if (t === 1) this.moveStart = null
    }
    // Lines join where groups will be, so they wait for the points to arrive.
    this.lines.visible = this.moveStart === null

    if (this.cameraMove) {
      const move = this.cameraMove
      const t = Math.min((now - move.start) / MOVE_MS, 1)
      this.camera.position.lerpVectors(move.fromPosition, move.toPosition, ease(t))
      this.controls.target.lerpVectors(move.fromTarget, move.toTarget, ease(t))
      if (t === 1) this.cameraMove = null
    }
    this.controls.update()
    this.camera.updateMatrixWorld()

    if (this.pointerMoved && !this.down) {
      this.pointerMoved = false
      this.setHovered(this.pointer ? this.pick(this.pointer.x, this.pointer.y) : null)
    }
    if (++this.tick % 8 === 0 && this.moveStart === null) this.refreshThumbs()

    this.renderer.render(this.scene, this.camera)
    this.callbacks.onFrame()
  }

  private pick(x: number, y: number): number | null {
    let best: number | null = null
    let bestDepth = Infinity
    for (let i = 0; i < this.ids.length; i++) {
      const p = this.project(this.current[i * 3], this.current[i * 3 + 1], this.current[i * 3 + 2])
      if (!p.visible || p.depth >= bestDepth) continue
      const reach = Math.max((this.pointSize * p.scale) / 2, 5)
      if ((p.x - x) ** 2 + (p.y - y) ** 2 <= reach * reach) {
        best = i
        bestDepth = p.depth
      }
    }
    return best
  }

  private setHovered(index: number | null): void {
    if (index === this.hovered) return
    this.hovered = index
    this.renderer.domElement.style.cursor = index === null ? 'grab' : 'pointer'
    this.callbacks.onHover(index)
  }

  private localPoint(e: PointerEvent): { x: number; y: number } {
    const box = this.renderer.domElement.getBoundingClientRect()
    return { x: e.clientX - box.left, y: e.clientY - box.top }
  }

  private onPointerMove = (e: PointerEvent): void => {
    this.pointer = this.localPoint(e)
    this.pointerMoved = true
  }

  private onPointerLeave = (): void => {
    this.pointer = null
    this.pointerMoved = true
  }

  private onPointerDown = (e: PointerEvent): void => {
    this.down = this.localPoint(e)
    this.cameraMove = null
  }

  private onPointerUp = (e: PointerEvent): void => {
    const down = this.down
    this.down = null
    if (!down || e.button !== 0) return
    const up = this.localPoint(e)
    // A drag rotates the view; only a press and release in one place opens a photo.
    if (Math.hypot(up.x - down.x, up.y - down.y) > 4) return
    const index = this.pick(up.x, up.y)
    if (index !== null) this.callbacks.onPick(index)
  }

  private stopCameraMove = (): void => {
    this.cameraMove = null
  }

  /** Swap the nearest dots for the faces they stand for. */
  private refreshThumbs(): void {
    const near: { index: number; depth: number }[] = []
    for (let i = 0; i < this.ids.length; i++) {
      const p = this.project(this.current[i * 3], this.current[i * 3 + 1], this.current[i * 3 + 2])
      const radius = (this.pointSize * p.scale) / 2
      if (!p.visible || radius < THUMB_PX) continue
      if (p.x < -radius || p.x > this.width + radius || p.y < -radius || p.y > this.height + radius) continue
      near.push({ index: i, depth: p.depth })
    }
    near.sort((a, b) => a.depth - b.depth)
    const wanted = new Set(near.slice(0, MAX_THUMBS).map((n) => n.index))

    for (const [index, sprite] of this.thumbs) {
      if (wanted.has(index)) continue
      this.scene.remove(sprite)
      sprite.material.dispose()
      this.thumbs.delete(index)
    }
    for (const index of wanted) {
      let sprite = this.thumbs.get(index)
      if (!sprite) {
        const id = this.ids[index]
        const texture = this.textures.get(id)
        if (!texture) {
          this.request(id)
          continue
        }
        sprite = new THREE.Sprite(
          new THREE.SpriteMaterial({ map: texture, alphaMap: this.mask, transparent: true, depthWrite: false }),
        )
        // Slightly smaller than the dot, which stays visible as a ring in the person's colour.
        sprite.scale.setScalar(this.pointSize * 0.84)
        this.thumbs.set(index, sprite)
        this.scene.add(sprite)
      }
      // Just in front of its dot, whichever way the camera faces.
      sprite.position
        .set(this.current[index * 3], this.current[index * 3 + 1], this.current[index * 3 + 2])
        .addScaledVector(this.scratch.copy(this.camera.position).sub(sprite.position).normalize(), 0.02)
    }

    if (this.textures.size > MAX_CACHED) {
      const used = new Set([...this.thumbs.keys()].map((index) => this.ids[index]))
      for (const [id, texture] of this.textures) {
        if (used.has(id)) continue
        texture.dispose()
        this.textures.delete(id)
        this.requested.delete(id)
      }
    }
  }

  private request(id: number): void {
    // A crop that fails to load stays requested, so its dot is simply kept.
    if (this.requested.has(id)) return
    this.requested.add(id)
    this.loader.load(this.faceUrl(id), (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace
      this.textures.set(id, texture)
    })
  }

  private clearThumbs(): void {
    for (const sprite of this.thumbs.values()) {
      this.scene.remove(sprite)
      sprite.material.dispose()
    }
    this.thumbs.clear()
  }
}
