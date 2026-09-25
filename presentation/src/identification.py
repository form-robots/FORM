"""Identification for the FORM video presentation."""

from paths import ASSET_ROOT

P = ASSET_ROOT / "identification"
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import identification_layout as flow
from equations import tex, arrow, TEXT_RECORDS, CONNECTORS, MATH_BOUNDS
from primitives import data, lerp_frame
from scipy.spatial import Delaunay, cKDTree
from framing import frame_method
from captions import caption_at

S = flow.S
BG, INK, BLUE, TEAL, ORANGE, MUTED = (
    flow.BG,
    flow.INK,
    flow.BLUE,
    flow.TEAL,
    flow.ORANGE,
    flow.MUTED,
)
GOLD, PURPLE, SLATE = (flow.VOLUME, flow.BASIS, flow.PARTICLES)
TIME = "#485862"
FPS = 25
DURATION = 28
LOOP = 4.0
SECOND_START = 3 * LOOP
THIRD_START = 5 * LOOP
REVEAL_SECONDS = 0.6
FAINT_OPACITY = 0.07
ACTION_SOURCE_DURATION = 16.0
T0 = 0.05
T1 = 2.0


def field(q):
    """psi=curl(y*g,-x*g,0), g=(1-|q|^2)^4_+; normalized coordinates."""
    r2 = np.sum(q * q, axis=-1)
    u = np.maximum(1 - r2, 0)
    grad = -8 * q * u[..., None] ** 3
    hess = (
        -8 * np.eye(3) * u[..., None, None] ** 3
        + 48 * q[..., :, None] * q[..., None, :] * u[..., None, None] ** 2
    )
    psi = np.stack(
        [
            q[..., 0] * grad[..., 2],
            q[..., 1] * grad[..., 2],
            -2 * u**4 - q[..., 0] * grad[..., 0] - q[..., 1] * grad[..., 1],
        ],
        axis=-1,
    )
    jac = np.zeros((*q.shape[:-1], 3, 3))
    jac[..., 0, :] = q[..., 0, None] * hess[..., 2, :]
    jac[..., 0, 0] += grad[..., 2]
    jac[..., 1, :] = q[..., 1, None] * hess[..., 2, :]
    jac[..., 1, 1] += grad[..., 2]
    jac[..., 2, :] = (
        -2 * grad - q[..., 0, None] * hess[..., 0, :] - q[..., 1, None] * hess[..., 1, :]
    )
    jac[..., 2, 0] -= grad[..., 0]
    jac[..., 2, 1] -= grad[..., 1]
    return (psi, jac)


@lru_cache(None)
def material_subvolume():
    """Interpolate a small reference-space subvolume through saved trajectories.

    This is a display illustration of quadrature, not a measured cell partition.
    Fixed barycentric weights follow material points; no fit or simulation runs.
    """
    hp = data()["hp"]
    seed = np.asarray(hp["seed"])
    tree = cKDTree(seed)
    center = seed[tree.query(seed.mean(0))[1]]
    corners = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    points = (
        np.array(
            [
                [x, y, z]
                for z in [-0.004, 0, 0.004]
                for y in [-0.004, 0, 0.004]
                for x in [-0.004, 0, 0.004]
            ]
        )
        + center
    )
    queries = np.concatenate(
        [
            center + corners * 0.008,
            points,
            (points[:, None, :] + corners[None, :, :] * 0.0018).reshape(-1, 3),
        ]
    )
    triangulation = Delaunay(seed)
    simplex = triangulation.find_simplex(queries)
    assert np.all(simplex >= 0), "Display subvolume must stay inside reconstructed material"
    transform = triangulation.transform[simplex]
    bary = np.einsum("nij,nj->ni", transform[:, :3], queries - transform[:, 3])
    weights = np.column_stack([bary, 1 - bary.sum(1)])
    indices = triangulation.simplices[simplex]
    trajectories = np.einsum("tqvc,qv->tqc", hp["pos"][:, indices, :], weights)
    return (hp["times"], trajectories)


def subvolume_at(t):
    times, trajectories = material_subvolume()
    current, _, _ = lerp_frame(trajectories, times, t)
    current = current - current[8:35].mean(0)
    return (current[:8], current[8:35], current[35:].reshape(27, 8, 3))


def constant_volume_cells(points, cells):
    """Display-only affine cells with unit determinant, as assumed in hardware.

    Retain the local deformation's shape/shear, remove interpolation-induced
    volume drift. Recorded particle trajectories and fitting data are untouched.
    """
    reference = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]]) * 0.0018
    fits = np.einsum("ij,njk->nik", np.linalg.pinv(reference), cells - points[:, None, :])
    determinant = np.linalg.det(fits)
    assert np.all(determinant > 0)
    fits = fits / np.cbrt(determinant)[:, None, None]
    return np.einsum("ij,njk->nik", reference, fits)


def moving_packet(c, points, source_time, color):
    pts = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    fraction = (source_time - T0) * 1.0 % 1
    distance = fraction * lengths.sum()
    for i, length in enumerate(lengths):
        if distance <= length or i == len(lengths) - 1:
            p = pts[i] + np.clip(distance / max(length, 1e-09), 0, 1) * (pts[i + 1] - pts[i])
            c.dot(p, 2.3, color, "white")
            return
        distance -= length


def operator_arrow(c, points, color, source_time, width=1.7):
    flow.route(c, points, color, width, arrowhead=True)
    moving_packet(c, points, source_time, color)


def project(points, center, scale=1.0):
    q = np.asarray(points) * scale
    return np.stack(
        [
            center[0] + q[..., 0] - 0.64 * q[..., 1],
            center[1] + 0.35 * q[..., 0] + 0.35 * q[..., 1] - q[..., 2],
        ],
        axis=-1,
    )


def cube(c, center, half, glass=False, particle=False, vertices=None):
    v = (
        np.array([[x, y, z] for x in [-half, half] for y in [-half, half] for z in [-half, half]])
        if vertices is None
        else np.asarray(vertices)
    )
    xy = project(v, center)
    faces = [[1, 5, 7, 3], [2, 3, 7, 6], [4, 5, 7, 6]]
    if glass:
        layer = Image.new("RGBA", c.im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for f, a in zip(faces, [38, 23, 48]):
            d.polygon([tuple(p * S) for p in xy[f]], fill=(181, 143, 66, a))
        c.im.paste(Image.alpha_composite(c.im.convert("RGBA"), layer).convert("RGB"))
        for i in range(8):
            for j in range(i + 1, 8):
                if i ^ j not in (1, 2, 4):
                    continue
                if i == 0:
                    for a in np.arange(0, 1, 0.25):
                        c.line(
                            [
                                xy[i] + a * (xy[j] - xy[i]),
                                xy[i] + min(1, a + 0.12) * (xy[j] - xy[i]),
                            ],
                            "#bdad8d",
                            0.65,
                        )
                else:
                    c.line([xy[i], xy[j]], GOLD, 0.85)
        if particle:
            c.dot(center, 3.4 if half > 10 else 2.9, SLATE, "#fffdf6")
    else:
        for f, col in zip(faces, ["#e2edf1", "#edf3f6", "#d5e4eb"]):
            pts = xy[f]
            c.poly(pts, col)
            c.line(np.vstack([pts, pts[0]]), "#b0c5ce", 0.8)
    return xy


def operator_scene(c, anchors, t):
    phase = float(np.clip((t - T0) / (T1 - T0), 0, 1))
    outer, points, cells = subvolume_at(t)
    c.rect((67, 576, 279, 705), BG, 7, edge="#b5c5ce")
    c.rect((191, 451, 274, 552), None, 6, edge="#96acb9")
    cloud = np.array([180.0, 627.0])
    scale = 3400.0
    cube(c, cloud, 22, vertices=outer * scale)
    selected = min(int(phase * len(points)), len(points) - 1)
    xy = project(points, cloud, scale)
    depth = points @ np.array([0.64, 1, 0.574])
    for i in np.argsort(depth):
        c.dot(xy[i], 2.3, SLATE, "#f4f6f7")
    cell = constant_volume_cells(points, cells)[selected]
    cube(c, xy[selected], 6.2, glass=True, particle=True, vertices=cell * scale)
    for i in np.argsort(depth):
        if depth[i] > depth[selected]:
            c.dot(xy[i], 2.3, SLATE, "#f4f6f7")
    tex(c, 10, (101, 628), height=78)
    tex(c, 11, (252, 632), height=23)
    operator_arrow(c, [(235, 576), (235, 553)], SLATE, t, 1.7)
    c.text(173, 671, "Sum over particles", 15, SLATE, True, "mt")
    c.text(173, 689, "and integrate over time", 15, TIME, True, "mt")
    zoom = np.array([356.0, 628.0])
    cube(c, zoom, 17, glass=True, particle=True, vertices=cell * (17 / 0.0018))
    vx, vy = anchors["volume"]["bottom"]
    operator_arrow(c, [(356, 583), (vx, 583), (vx, vy + 7)], GOLD, t)
    c.text(356, 672, "Particle volume", 15.5, GOLD, True, "mt")
    c.text(356, 693, "Constant volume", 13.5, MUTED, anchor="mt")
    wc = np.array([505.0, 626.0])
    cube(c, wc, 27)
    samples = np.array(
        [[x, y, z] for x in [-0.45, 0, 0.45] for y in [-0.35, 0.35] for z in [-0.35, 0.35]]
    )
    vec, _ = field(samples)
    weight = np.sin(np.pi * phase) ** 2
    for q, d in zip(samples, vec):
        a = project(q * 32, wc)
        fixed = project(q * 32 + d * 34, wc)
        c.arrow(a, fixed, "#a1c3bd", 1.0, 4)
        b = project(q * 32 + d * 34 * weight, wc)
        c.arrow(a, b, TEAL, 1.5, 4)
    wx, wy = anchors["weight"]["bottom"]
    operator_arrow(c, [(505, 572), (wx, 572), (wx, wy + 7)], TEAL, t, 1.8)
    c.text(505, 672, "Test field example", 15.5, TEAL, True, "mt")
    tex(c, 2, (505, 699), height=20)


@lru_cache(None)
def base():
    im = Image.open(P / "method02_flow_layout.png").convert("RGB")
    c = flow.Canvas(im)
    c.rect((627, 339, 881, 708), BG, 0)
    c.rect((626, 138, 882, 711), None, 10, edge="#ccd9e0")
    c.rect((578, 496, 640, 533), BG, 0)
    for x in (591, 626):
        c.line([(x, 496), (x, 533)], "#ccd9e0", 1)
    flow.identification_solve(c)
    c.rect((868, 259, 933, 269), BG, 0)
    for x in (882, 917):
        c.line([(x, 259), (x, 269)], "#ccd9e0", 1)
    flow.recovered_law(c)
    return im


def interblock_arrow(c):
    arrow(c, (578, 501), (641, 501), TEAL, 2.2, 7)


def action_panels(c, t):
    """Corresponding frozen red-plan replay and hardware, phase-aligned only."""
    folder = P / "shaping_pair"
    index = min(max(round(float(t) * FPS), 0), int(ACTION_SOURCE_DURATION * FPS) - 1)
    c.rect((933, 185, 1220, 439), "#eaf0f3", 7)
    c.text(1076, 200, "Plan with MPM", 22, INK, True, "mt")
    with Image.open(folder / "simulation_frames" / f"{index:04d}.png") as pic:
        flow.fitted(c, pic.crop((35, 0, 485, 340)), (938, 225, 230, 169))
    with Image.open(folder / "target_2d.png") as pic:
        flow.fitted(c, pic, (1170, 274, 44, 60))
    c.text(1192, 339, "Target", 14.5, MUTED, anchor="mt")
    c.text(1076, 396, "Plan actions to match the target", 15, INK, anchor="mt")
    c.text(1076, 418, "Keep the identified law fixed", 16.5, BLUE, True, "mt")
    c.rect((933, 492, 1220, 703), "#eaf0f3", 7)
    c.text(1076, 501, "Execute on real specimen", 20, INK, True, "mt")
    with Image.open(folder / "hardware_frames" / f"{index:04d}.png") as pic:
        flow.fitted(c, pic, (938, 529, 277, 168))


def reveal_progress(t, start):
    u = float(np.clip((t - start) / REVEAL_SECONDS, 0.0, 1.0))
    return u * u * (3 - 2 * u)


def action_time(t):
    return (
        float(np.clip((t - THIRD_START) / (DURATION - THIRD_START), 0.0, 1.0))
        * ACTION_SOURCE_DURATION
    )


def finish_frame(im, t, reveal=True):
    c = flow.ReviewCanvas(im)
    action_panels(c, action_time(t))
    if reveal:
        for box, start in [
            ((625, 137, 884, 713), SECOND_START),
            ((916, 137, 1238, 713), THIRD_START),
        ]:
            bounds = tuple((int(v * S) for v in box))
            tile = im.crop(bounds)
            opacity = FAINT_OPACITY + (1 - FAINT_OPACITY) * reveal_progress(t, start)
            im.paste(Image.blend(Image.new("RGB", tile.size, BG), tile, opacity), bounds[:2])
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    for start, end, color, width, at in [
        ((578, 501), (641, 501), TEAL, 2.2, SECOND_START),
        ((868, 264), (933, 264), BLUE, 2.1, THIRD_START),
    ]:
        alpha = reveal_progress(t, at) if reveal else 1.0
        if alpha <= 0:
            continue
        arrow(
            flow.Canvas(layer),
            start,
            end,
            (*ImageColor.getrgb(color), round(alpha * 255)),
            width,
            7,
        )
    im.paste(Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB"))
    return im


def presentation_frame(im, t):
    caption = caption_at("balance", t)
    return frame_method(im, 2, "Recover the material law. Plan the robot action.", caption)


def draw_frame(t, full=False, include_actions=True, reveal=True, presentation=True):
    source = round(min(T0 + 0.5 * (t % LOOP), T1), 8)
    TEXT_RECORDS.clear()
    CONNECTORS.clear()
    MATH_BOUNDS.clear()
    im = base().copy()
    c = flow.ReviewCanvas(im)
    c.rect((47, 180, 588, 707), BG, 0)
    anchors = flow.construction(c, source_time=source, operators=operator_scene, compact=True)
    flow.identification_solve(c)
    phase = (source - T0) / (T1 - T0)
    start = np.array([395.0, 326.0])
    knee = np.array([421.0, 304.0])
    end = np.array([456.0, 304.0])
    path_fraction = phase * 2
    marker = (
        start + (knee - start) * path_fraction
        if phase < 0.5
        else knee + (end - knee) * (path_fraction - 1)
    )
    c.dot(marker, 2.8, PURPLE, "white")
    nodes = [
        [(504, 318), (504, 333), (504, 348)],
        [(528, 314), (528, 325), (528, 337), (528, 352)],
        [(552, 324), (552, 342)],
    ]
    activation = min(int(phase * 6 % 3), 2)
    for point in nodes[activation]:
        location = (point[0], point[1] - 24)
        c.dot(location, 6.0, "#dfcdec")
        c.dot(location, 3.8, "#9060bf", "#ffffff")
    stress_x, stress_y = anchors["stress"]["top"]
    load_x, load_y = anchors["load"]["top"]
    for start, end, col in [
        ((225, 302), (250, 302), BLUE),
        ((320, 349), (320, 384), BLUE),
        ((490.5, 349), (490.5, 377), PURPLE),
        ((stress_x, 440), (stress_x, stress_y - 8), BLUE),
        ((load_x, 440), (load_x, load_y - 8), ORANGE),
    ]:
        moving_packet(c, [start, end], source, col)
    if include_actions:
        finish_frame(im, t, reveal)
    if presentation and include_actions:
        im = presentation_frame(im, t)
    return im if full else im.resize((1280, 720), Image.Resampling.LANCZOS)
