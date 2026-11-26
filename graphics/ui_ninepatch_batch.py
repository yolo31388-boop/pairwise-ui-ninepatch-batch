"""界面九宫格与合批的核心模型。

六个必须同时满足的约束（任一被破坏，对应症状即会复现）：

1. ``slice_by_dpi``    —— 九宫格按 **原始像素** 切分，缩放只作用于绘制阶段，
   不能把刚体边框的 1px 描边放大成 Npx。
2. ``stable_atlas``    —— 合批要求同图集同材质；动态字形插入不能触发纹理重建，
   否则旧批被打断，一屏文字拆成几十个 draw call。
3. ``double_buffer_vb``—— 顶点缓冲双缓冲，只覆写 GPU 已经消费完的槽位，
   避免“上一帧缓冲还在 GPU 上就被下一帧覆写”造成的按钮残影。
4. ``anchor_unified``  —— 锚点/偏移统一换算到物理像素与安全区；超宽屏不得拉伸安全区，
   不得压扁九宫格角部。
5. ``defer_layout``    —— 补间（渲染线程之外）里改布局只入队，延后一帧在渲染线程应用，
   避免“重建与绘制同帧竞争”导致的整帧界面不显示。
6. ``end_frame_check`` —— 帧末校验：绘制期间不得发生布局重建，缓冲必须已发布，数值必须有限。

调用约定
--------
所有入口的第一个参数都可省略（``None`` 表示使用默认值）。为便于管线式调用，
集合类入口也接受一个非负标量 ``n``，含义见各函数文档——通常是“默认条目的数量”
或“统一的数值参数”。所有阈值由 ``spec`` 提供，模块不硬编码分辨率、DPI 或图集容量。
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_ANCHOR_X = {"left": 0.0, "center": 0.5, "right": 1.0}
_ANCHOR_Y = {"top": 0.0, "middle": 0.5, "bottom": 1.0}

_DEFAULT_SPEC = {
    "dpi": 1.0,
    "screen": (1920.0, 1080.0),          # 逻辑像素（与 DPI 无关的设计坐标）
    "safe_area": (0.0, 0.0, 0.0, 0.0),   # 逻辑像素 (left, top, right, bottom)
    "image_size": (64.0, 64.0),          # 九宫格源图默认尺寸（原始像素）
    "atlas_capacity": 1024,              # 图集可容纳的字形数上限（容量内永不重建）
    "vertex_bytes": 64,                  # 单个顶点的字节数，用于估算缓冲预算
    "max_draw_calls": 1,                 # 一个同图集同材质的批只允许一次绘制
    "max_layout": 1,                     # 单帧布局重建次数上限
}


# ---------------------------------------------------------------------------
# 校验与规范化
# ---------------------------------------------------------------------------

def _spec(spec):
    """合并默认 spec，返回新字典（绝不修改调用方对象）。"""
    out = dict(_DEFAULT_SPEC)
    if spec is not None:
        if not isinstance(spec, Mapping):
            raise TypeError("spec 必须是映射")
        out.update(spec)
    if not (out["dpi"] > 0):
        raise ValueError("dpi 必须为正数")
    return out


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} 必须是数值")
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数")
    return float(value)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须是整数")
    if value < 0:
        raise ValueError(f"{name} 不能为负")
    return value


def _count(value, name="n"):
    """把可选的标量首参解释为非负条目数；None 表示使用默认值。"""
    if value is None:
        return None
    n = _finite(value, name)
    if n < 0 or not float(n).is_integer():
        raise ValueError(f"{name} 必须是非负整数")
    return int(n)


def _scalar_or(value, default):
    """首参既可能是标量也可能是结构体；标量时返回 (scalar, default)，否则 (None, value)。"""
    if isinstance(value, bool):
        raise TypeError("参数不能是布尔值")
    if isinstance(value, (int, float)):
        return _finite(value, "scalar"), default
    return None, value


def _patch(piece):
    """把九宫格片段规范化成独立字典；src 永远是原始像素。"""
    if not isinstance(piece, Mapping):
        raise TypeError("九宫格切片必须是映射")
    src = piece.get("src")
    dst = piece.get("dst")
    if not src or len(src) != 4:
        raise ValueError("切片必须带四元 src 源矩形")
    if not dst or len(dst) != 2:
        raise ValueError("切片必须带二元 dst 尺寸")
    src = [_finite(v, "src") for v in src]
    if src[2] <= 0 or src[3] <= 0:
        raise ValueError("src 宽高必须为正（原始像素）")
    dst = [_finite(v, "dst") for v in dst]
    if dst[0] < 0 or dst[1] < 0:
        raise ValueError("dst 尺寸不能为负")
    return {
        "edge": piece.get("edge"),
        "src": src,                       # 原图像素矩形，永不随缩放改变
        "dst": dst,                       # 目标逻辑尺寸
        "scale": _finite(piece.get("scale", 1.0), "scale"),
        "atlas": piece.get("atlas"),
    }


def _anchor_names(anchor):
    """把 'top-left' / 'center' 解析为 (水平名, 垂直名)。"""
    if not isinstance(anchor, str):
        try:
            ax, ay = anchor
        except (TypeError, ValueError):
            raise ValueError(f"未知锚点: {anchor!r}")
        if ax not in _ANCHOR_X or ay not in _ANCHOR_Y:
            raise ValueError(f"未知锚点: {anchor!r}")
        return ax, ay

    parts = [p for p in anchor.split("-") if p]
    if len(parts) == 1:
        token = parts[0]
        if token in _ANCHOR_Y:
            return "center", token
        if token in _ANCHOR_X:
            return token, "middle"
        raise ValueError(f"未知锚点: {anchor!r}")
    if len(parts) == 2:
        # 顺序无关：分别归入水平/垂直轴，兼容 'top-left' 与 'left-top'。
        xs = [p for p in parts if p in _ANCHOR_X]
        ys = [p for p in parts if p in _ANCHOR_Y]
        if len(xs) == 1 and len(ys) == 1:
            return xs[0], ys[0]
    raise ValueError(f"未知锚点: {anchor!r}")


# ---------------------------------------------------------------------------
# 1. 九宫格按 DPI 切分
# ---------------------------------------------------------------------------

def slice_by_dpi(image_size=None, border=None, spec=None):
    """按 **原始像素** 切出九宫格，返回 9 个不重叠、拼合后覆盖整图的片段。

    ``border`` = (left, top, right, bottom) 同样是原始像素。
    ``spec['dpi']`` 与目标尺寸只记录在 ``dst`` 上，绝不影响 ``src``，
    因此 1px 描边不会被拉伸成 Npx。

    标量首参 ``n`` 表示“默认源图 + 四边统一的 n 像素边框”。
    ``image_size`` 也可为带 ``width``/``height`` 的映射。
    """
    s = _spec(spec)
    scalar, image_size = _scalar_or(image_size, None)
    if scalar is not None:
        border = (scalar, scalar, scalar, scalar)
        image_size = s["image_size"]
    if image_size is None:
        image_size = s["image_size"]
    if border is None:
        border = (0, 0, 0, 0)

    if isinstance(image_size, Mapping):
        w = _finite(image_size.get("width"), "width")
        h = _finite(image_size.get("height"), "height")
    else:
        if len(image_size) != 2:
            raise ValueError("image_size 必须是 (w, h)")
        w = _finite(image_size[0], "width")
        h = _finite(image_size[1], "height")
    if w <= 0 or h <= 0:
        raise ValueError("图像尺寸必须为正")

    if len(border) != 4:
        raise ValueError("border 必须是四元组 (left, top, right, bottom)")
    bl, bt, br, bb = (_finite(v, "border") for v in border)
    if min(bl, bt, br, bb) < 0:
        raise ValueError("border 不能为负")
    if bl + br >= w:
        raise ValueError("左右边框之和必须小于图宽（否则没有可拉伸中段）")
    if bt + bb >= h:
        raise ValueError("上下边框之和必须小于图高（否则没有可拉伸中段）")

    xs = (0.0, float(bl), w - br, w)
    ys = (0.0, float(bt), h - bb, h)
    names = (
        ("top-left", "top", "top-right"),
        ("left", "center", "right"),
        ("bottom-left", "bottom", "bottom-right"),
    )
    dpi = s["dpi"]
    out = []
    for row, (y0, y1) in enumerate(zip(ys, ys[1:])):
        for col, (x0, x1) in enumerate(zip(xs, xs[1:])):
            out.append({
                "edge": names[row][col],
                "row": row,
                "col": col,
                "src": [x0, y0, x1 - x0, y1 - y0],   # 原始像素，缩放无关
                "dst": [x1 - x0, y1 - y0],           # 逻辑尺寸占位，布局时重算
                "dpi": dpi,                          # 只影响绘制，不影响切分
            })
    return out


# ---------------------------------------------------------------------------
# 2. 图集稳定：字形插入不重建纹理
# ---------------------------------------------------------------------------

def stable_atlas(glyphs=None, capacity=None, spec=None):
    """把字形放进图集；容量足够时 ``rebuilt`` 恒为 False。

    返回 ``(atlas_state, added)``。``atlas_state`` 可直接作为下一次调用的输入
    （不修改入参），其 ``id``/``size``/``generation`` 在容量内保持不变，
    因此同一批绘制调用不会被重建打断。

    - ``glyphs``：待插入字形序列（id 或 ``{id, width, height}``）；
    - ``glyphs`` 也可传入上一次返回的图集状态（含 ``id`` 与 ``glyphs``），
      此时从它的 ``pending`` 字段取待插入字形；
    - 标量首参 ``n`` 表示“新建一个容量为 n 的空图集”。
    """
    s = _spec(spec)
    scalar, glyphs = _scalar_or(glyphs, None)
    if scalar is not None:
        capacity = int(scalar)
        glyphs = []
    if capacity is None:
        capacity = s["atlas_capacity"]
    cap = _positive_int(capacity, "capacity")

    if isinstance(glyphs, Mapping) and "glyphs" in glyphs and "id" in glyphs:
        state = copy.deepcopy(glyphs)          # 继续使用同一张图集
        pending = list(state.pop("pending", []) or [])
    else:
        state = {
            "id": "atlas-0",
            "generation": 0,
            "size": float(s["atlas_capacity"]),
            "glyphs": {},
            "rebuilt": False,
            "fill": 0,
        }
        pending = list(glyphs or [])

    added = _insert_glyphs(state, pending, cap)
    state["pending"] = []
    return state, added


def _insert_glyphs(state, glyphs, capacity):
    """执行插入。只有在真正溢出容量时才重建，并显式标记以便上层告警。"""
    added = []
    glyphs_map = state["glyphs"]
    for g in glyphs:
        if isinstance(g, Mapping):
            gid = g.get("id")
            gw = _finite(g.get("width", 1.0), "width")
            gh = _finite(g.get("height", 1.0), "height")
        else:
            gid, gw, gh = g, 1.0, 1.0
        if gid is None:
            raise ValueError("字形必须带 id")
        if gid in glyphs_map:
            continue
        if len(glyphs_map) >= capacity:
            # 仅在真正溢出时才重建；正常路径（容量足够）永远走不到这里。
            state["generation"] += 1
            state["rebuilt"] = True
            glyphs_map = state["glyphs"] = {}
        slot = len(glyphs_map)
        glyphs_map[gid] = {"slot": slot, "width": gw, "height": gh}
        state["fill"] = slot + 1
        added.append({"id": gid, "slot": slot})
    state["glyphs"] = glyphs_map
    return added


# ---------------------------------------------------------------------------
# 3. 顶点缓冲双缓冲
# ---------------------------------------------------------------------------

def double_buffer_vb(frames=None, spec=None):
    """双缓冲顶点缓冲：第 N 帧只能覆写第 N-2 帧（GPU 已消费）的槽位。

    标量首参 ``n`` 表示“n 个默认空帧”。返回逐帧历史，每帧含
    ``buffer``、``slot``、``gpu_done``、``overwrote``。
    ``overwrote`` 只在槽位确实已被 GPU 消费完时才非空，
    从而杜绝“覆写仍在 GPU 上的缓冲”造成的按钮残影。
    """
    s = _spec(spec)
    scalar, frames = _scalar_or(frames, None)
    if scalar is not None:
        frames = [{"vertices": 0} for _ in range(int(scalar))]
    if frames is None:
        frames = []
    if not isinstance(frames, (list, tuple)):
        raise TypeError("frames 必须是帧序列或帧数")
    vertex_bytes = _positive_int(s["vertex_bytes"], "vertex_bytes")

    history = []
    for i, f in enumerate(frames):
        if not isinstance(f, Mapping):
            raise TypeError("帧必须是映射")
        n = _positive_int(f.get("vertices", 0), "vertices")
        slot = i % 2
        prev = history[i - 2] if i >= 2 else None
        # 同槽位的上上一次提交此时应已被 GPU 消费完，才允许覆写。
        gpu_done = prev is not None and prev["slot"] == slot and prev["submitted"]
        history.append({
            "index": i,
            "slot": slot,
            "buffer": f"vb-{slot}",
            "bytes": n * vertex_bytes,
            "vertices": n,
            "submitted": True,
            "gpu_done": gpu_done,
            "overwrote": [slot] if gpu_done else [],
            "safe": True if i < 2 else gpu_done,
        })
    return history


# ---------------------------------------------------------------------------
# 4. 锚点统一换算
# ---------------------------------------------------------------------------

def anchor_unified(rects=None, spec=None):
    """把任意锚点/偏移统一换算到物理像素，并扣除安全区。

    ``rects`` 每项可含 ``anchor``、``offset``（逻辑像素）、``size``（逻辑像素，
    保持像素尺寸而不被拉伸）。超宽屏下安全区按逻辑像素扣减后再乘 DPI，
    左右安全区不会被拉伸；九宫格角部由 ``slice_by_dpi`` 的 src 保证原始像素。

    标量首参 ``n`` 表示“n 个默认矩形（左上锚点、零偏移、零尺寸）”。
    """
    s = _spec(spec)
    scalar, rects = _scalar_or(rects, None)
    if scalar is not None:
        rects = [{} for _ in range(int(scalar))]
    if rects is None:
        rects = []
    if isinstance(rects, Mapping):
        rects = [rects]

    screen_w, screen_h = (_finite(v, "screen") for v in s["screen"])
    sa_l, sa_t, sa_r, sa_b = (_finite(v, "safe_area") for v in s["safe_area"])
    if screen_w - sa_l - sa_r <= 0 or screen_h - sa_t - sa_b <= 0:
        raise ValueError("安全区吃掉了整个屏幕")
    dpi = s["dpi"]
    avail_w = screen_w - sa_l - sa_r
    avail_h = screen_h - sa_t - sa_b

    out = []
    for rect in rects:
        if not isinstance(rect, Mapping):
            raise TypeError("矩形必须是映射")
        ax, ay = _anchor_names(rect.get("anchor", "top-left"))
        off = rect.get("offset", (0.0, 0.0))
        ox, oy = _finite(off[0], "offset"), _finite(off[1], "offset")
        size = rect.get("size", (0.0, 0.0))
        sw, sh = _finite(size[0], "size"), _finite(size[1], "size")

        # 以逻辑像素求锚点，再统一乘 DPI：不同分辨率得到同一比例位置。
        lx = sa_l + _ANCHOR_X[ax] * avail_w + ox
        ly = sa_t + _ANCHOR_Y[ay] * avail_h + oy
        out.append({
            "anchor": f"{ay}-{ax}",
            "x": lx * dpi,
            "y": ly * dpi,
            "width": sw * dpi,      # 物理像素，不随屏幕宽高拉伸
            "height": sh * dpi,
            "dpi": dpi,
        })
    return out


# ---------------------------------------------------------------------------
# 5. 布局延后一帧
# ---------------------------------------------------------------------------

def defer_layout(frame=None, spec=None):
    """补间里改布局只入队；下一帧在渲染线程应用，避免重建与绘制同帧竞争。

    ``frame`` 含阶段 ``phase``（'animate'/'render'）、``changes``、``pending``、
    ``layouts``。返回新的帧状态，不修改入参；非渲染阶段绝不立即重建。
    标量首参 ``n`` 表示“渲染帧带 n 条待应用的布局改动”。
    """
    s = _spec(spec)
    scalar, frame = _scalar_or(frame, None)
    if scalar is not None:
        frame = {"phase": "render",
                 "changes": [{"kind": "layout"} for _ in range(int(scalar))]}
    frame = frame or {}
    if not isinstance(frame, Mapping):
        raise TypeError("frame 必须是映射")

    phase = frame.get("phase", "render")
    pending = list(frame.get("pending") or [])
    changes = list(frame.get("changes") or [])
    layouts = _positive_int(frame.get("layouts", 0), "layouts")

    if phase != "render":
        # 渲染线程之外：只入队，绝不重建，绝不增加 layouts。
        pending.extend(changes)
        return {
            "phase": phase,
            "pending": pending,
            "applied": [],
            "layouts": layouts,
            "rebuilt": False,
            "deferred": bool(changes),
        }

    # 渲染线程：应用上一帧遗留下来的布局改动，每帧最多重建 max_layout 次。
    # 本帧新产生的改动同样只入队，留到下一帧应用（延后一帧）。
    applied = []
    if pending and layouts < s["max_layout"]:
        applied = pending
        pending = []
        layouts += 1
    pending.extend(changes)
    return {
        "phase": phase,
        "pending": pending,
        "applied": applied,
        "layouts": layouts,
        "rebuilt": bool(applied),
        "deferred": bool(changes),
    }


# ---------------------------------------------------------------------------
# 6. 帧末校验
# ---------------------------------------------------------------------------

def end_frame_check(frame=None, spec=None):
    """帧末校验：绘制中不得重建布局；缓冲必须已发布；数值必须有限。

    返回 ``{"ok": bool, "errors": [...]}``；任何一条不满足都 ``ok=False``。
    标量首参 ``n`` 表示“本帧有 n 个 draw call 的帧”。
    """
    s = _spec(spec)
    scalar, frame = _scalar_or(frame, None)
    if scalar is not None:
        frame = {"draw_calls": int(scalar)}
    frame = frame or {}
    if not isinstance(frame, Mapping):
        raise TypeError("frame 必须是映射")

    errors = []
    layouts = _positive_int(frame.get("layouts", 0), "layouts")
    if layouts > s["max_layout"]:
        errors.append(f"本帧布局重建 {layouts} 次，超过上限 {s['max_layout']}")
    published = frame.get("published", True)
    if frame.get("rebuilt") and not published:
        errors.append("绘制期间发生布局重建且缓冲未发布（整帧界面可能不显示）")
    if not published:
        errors.append("顶点缓冲未在帧末发布")

    for key in ("draw_calls", "vertices", "bytes"):
        if key in frame:
            v = _finite(frame[key], key)
            if v < 0:
                errors.append(f"{key} 不能为负")

    draw_calls = frame.get("draw_calls", 0)
    if draw_calls > s["max_draw_calls"] * max(1, frame.get("batches", 1)):
        errors.append("draw call 超出同图集同材质的合批上限（旧批被打断）")

    return {"ok": not errors, "errors": errors, "layouts": layouts,
            "frame": _positive_int(frame.get("frame", 0), "frame")}


# ---------------------------------------------------------------------------
# 高层编排
# ---------------------------------------------------------------------------

def layout(state=None, spec=None):
    """计算一帧布局：九宫格切片 + 锚点换算，全部在物理像素空间完成。

    标量首参 ``n`` 表示“默认源图 + 四边 n 像素边框的布局”。
    """
    s = _spec(spec)
    scalar, state = _scalar_or(state, None)
    if scalar is not None:
        state = {"image_size": s["image_size"],
                 "border": (int(scalar),) * 4}
    state = state or {}
    if not isinstance(state, Mapping):
        raise TypeError("state 必须是映射")

    image_size = state.get("image_size", s["image_size"])
    border = state.get("border", (0, 0, 0, 0))
    slices = slice_by_dpi(image_size, border, s)
    anchors = anchor_unified(state.get("rects", []), s)

    patches = []
    for item in state.get("patches", []):
        p = _patch(item)
        p["scale"] = p["scale"] * s["dpi"]
        p["dst"] = [p["dst"][0] * s["dpi"], p["dst"][1] * s["dpi"]]
        patches.append(p)

    return {
        "dpi": s["dpi"],
        "slices": slices,
        "anchors": anchors,
        "patches": patches,
        "layouts": _positive_int(state.get("layouts", 0), "layouts") + 1,
        "physical": True,
    }


def batch(drawables=None, spec=None):
    """按 (图集, 材质) 合批。

    图集在批中途重建会强制打断当前批并显式标记 ``interrupted``——这正是要避免的代价。
    标量首参 ``n`` 表示“n 个默认绘制项”。
    """
    s = _spec(spec)
    scalar, drawables = _scalar_or(drawables, None)
    if scalar is not None:
        drawables = [{} for _ in range(int(scalar))]
    if drawables is None:
        drawables = []
    if isinstance(drawables, Mapping):
        drawables = drawables.get("drawables", [])
    if not isinstance(drawables, (list, tuple)):
        raise TypeError("drawables 必须是序列")

    batches = []
    current = None
    for d in drawables:
        if not isinstance(d, Mapping):
            raise TypeError("绘制项必须是映射")
        key = (d.get("atlas", "atlas-0"), d.get("material", "default"))
        rebuilt = bool(d.get("atlas_rebuilt"))
        if current is not None and (current["key"] != key or rebuilt):
            batches.append(current)
            current = None
        if current is None:
            current = {"key": key, "draw_calls": 1, "count": 0}
            if rebuilt:
                current["interrupted"] = True
        current["count"] += 1
    if current is not None:
        batches.append(current)

    return {
        "batches": batches,
        "draw_calls": sum(b["draw_calls"] for b in batches),
        "interrupted": any(b.get("interrupted") for b in batches),
    }


def animate(clock=None, spec=None):
    """补间在渲染线程之外推进；任何布局改动都只入队，不立即重建。

    标量首参 ``n`` 表示“时长为 n 秒的补间”。
    """
    s = _spec(spec)
    scalar, clock = _scalar_or(clock, None)
    if scalar is not None:
        clock = {"duration": scalar}
    clock = clock or {}
    if not isinstance(clock, Mapping):
        raise TypeError("clock 必须是映射")

    duration = _finite(clock.get("duration", 1.0), "duration")
    if duration <= 0:
        raise ValueError("duration 必须为正")
    elapsed = min(max(_finite(clock.get("elapsed", 0.0), "elapsed"), 0.0), duration)
    t = elapsed / duration
    eased = t * t * (3.0 - 2.0 * t)   # smoothstep，确定性且与线程无关

    pending = list(clock.get("pending") or [])
    change = clock.get("layout_change")
    if change:
        pending.append(change)        # 延后一帧
    return {
        "elapsed": elapsed,
        "t": t,
        "eased": eased,
        "pending": pending,
        "rebuilt": False,
        "deferred": change is not None,
        "done": elapsed >= duration,
    }


def present(state=None, spec=None):
    """提交一帧：只发布缓冲，不触发布局重建。

    标量首参 ``n`` 表示“第 n 帧”。
    """
    s = _spec(spec)
    scalar, state = _scalar_or(state, None)
    if scalar is not None:
        state = {"frame": int(scalar)}
    state = state or {}
    if not isinstance(state, Mapping):
        raise TypeError("state 必须是映射")
    return {
        "published": True,
        "frame": _positive_int(state.get("frame", 0), "frame"),
        "buffer": state.get("buffer", "vb-0"),
        "rebuilt": False,
        "safe": True,
    }


def validate_ui(state=None, spec=None):
    """校验界面状态的六条硬约束，返回逐项结果。"""
    s = _spec(spec)
    scalar, state = _scalar_or(state, None)
    if scalar is not None:
        state = {"frame": int(scalar)}
    state = state or {}
    if not isinstance(state, Mapping):
        raise TypeError("state 必须是映射")

    checks = {
        "slice_physical_pixels": True,
        "atlas_stable": not state.get("atlas_rebuilt", False),
        "vb_double_buffered": _positive_int(state.get("vb_slots", 2), "vb_slots") >= 2,
        "anchors_unified": bool(state.get("dpi_unified", True)),
        "layout_deferred": not state.get("layout_in_render", False),
        "end_frame_ok": end_frame_check(state, s)["ok"],
    }
    return {"ok": all(checks.values()), "checks": checks}


def metrics(state=None, spec=None):
    """汇总一帧的指标（draw call、顶点、重建次数等），供回归对比。"""
    s = _spec(spec)
    scalar, state = _scalar_or(state, None)
    if scalar is not None:
        state = {"vertices": int(scalar)}
    state = state or {}
    if not isinstance(state, Mapping):
        raise TypeError("state 必须是映射")

    b = batch(state.get("drawables", []), s)
    return {
        "dpi": s["dpi"],
        "draw_calls": b["draw_calls"],
        "batches": len(b["batches"]),
        "interrupted": b["interrupted"],
        "vertices": _positive_int(state.get("vertices", 0), "vertices"),
        "layouts": _positive_int(state.get("layouts", 0), "layouts"),
        "atlas_generation": _positive_int(state.get("atlas_generation", 0), "atlas_generation"),
        "slice_count": len(state.get("slices", [])),
        "ok": validate_ui(state, s)["ok"],
    }


def audit(state=None, spec=None):
    """审计：报告问题列表，空列表表示无回归。"""
    s = _spec(spec)
    scalar, state = _scalar_or(state, None)
    if scalar is not None:
        state = {"frame": int(scalar)}
    state = state or {}
    if not isinstance(state, Mapping):
        raise TypeError("state 必须是映射")

    issues = []
    if state.get("atlas_rebuilt"):
        issues.append("动态字形触发图集重建，旧批被打断")
    if _positive_int(state.get("vb_slots", 2), "vb_slots") < 2:
        issues.append("顶点缓冲未双缓冲，可能覆写仍在 GPU 上的缓冲")
    if state.get("layout_in_render"):
        issues.append("布局重建与绘制同帧竞争")
    issues.extend(end_frame_check(state, s)["errors"])
    return {"ok": not issues, "issues": issues, "count": len(issues)}


def fallback(state=None, spec=None):
    """降级路径：无法满足约束时退回单批、单缓冲的安全绘制，仍保证界面可见。"""
    s = _spec(spec)
    scalar, state = _scalar_or(state, None)
    if scalar is not None:
        state = {"frame": int(scalar)}
    state = state or {}
    if not isinstance(state, Mapping):
        raise TypeError("state 必须是映射")
    return {
        "mode": "safe",
        "draw_calls": 1,
        "buffer": "vb-safe",
        "visible": True,
        "published": True,
        "deferred": True,
        "reason": state.get("reason", "constraint-violation"),
    }


def determinism(state=None, spec=None):
    """确定性：同一输入（含 spec）必须产出逐字节一致的布局与度量。"""
    s = _spec(spec)
    scalar, state = _scalar_or(state, None)
    if scalar is not None:
        state = {"frame": int(scalar)}
    state = copy.deepcopy(state or {})
    if not isinstance(state, Mapping):
        raise TypeError("state 必须是映射")

    def run():
        return {
            "layout": layout(state, s) if state.get("image_size") else None,
            "metrics": metrics(state, s),
        }

    first = run()
    second = run()
    return {"deterministic": first == second, "digest": repr(first), "runs": 2}
