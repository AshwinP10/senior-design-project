"""Record GIFs of real Box2D episodes (strike controller or a trained policy) for presentations.

Every frame is drawn from the simulator state after each 20 Hz control step. Optional overlays
show what the strike controller is planning: predicted puck path, interception point and aim.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts import scoring_geometry as sg
from scripts.strike_planner import predict

L, W = sg.LENGTH, sg.WIDTH
SS = 2                                   # supersampling factor for smooth edges
COL = {"bg": (255, 255, 255), "table": (240, 244, 248), "rail": (43, 47, 54), "center": (190, 198, 208),
       "goal": (34, 160, 90), "own_goal": (150, 156, 166), "obstacle": (74, 85, 104), "puck": (20, 20, 20),
       "paddle": (31, 111, 235), "trail": (160, 168, 178), "pred": (230, 140, 20), "aim": (34, 160, 90),
       "ink": (31, 31, 31), "muted": (107, 107, 107), "good": (34, 150, 80), "bad": (200, 55, 45)}


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Canvas:
    def __init__(self, width=900, band=52, margin=22):
        self.scale = (width - 2 * margin) / L
        self.width, self.band, self.margin = width, band, margin
        self.height = int(band + W * self.scale + 2 * margin)
        self.f_title, self.f_small, self.f_banner = _font(21 * SS), _font(15 * SS), _font(24 * SS)

    def px(self, x, y):
        s = SS
        return ((self.margin + (x + L / 2) * self.scale) * s, (self.band + self.margin + (W / 2 - y) * self.scale) * s)

    def circle(self, d, x, y, r, **kw):
        cx, cy = self.px(x, y)
        rr = r * self.scale * SS
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], **kw)

    def frame(self, puck, paddle, blocks, trail, title, status="", hit=False, banner=None,
              pred=None, aim=None, intercept=None):
        im = Image.new("RGB", (self.width * SS, self.height * SS), COL["bg"])
        d = ImageDraw.Draw(im)
        x0, y0 = self.px(-L / 2, W / 2)
        x1, y1 = self.px(L / 2, -W / 2)
        d.rectangle([x0, y0, x1, y1], fill=COL["table"])
        cx, _ = self.px(0, 0)
        d.line([cx, y0, cx, y1], fill=COL["center"], width=2 * SS)
        rail = 5 * SS
        d.rectangle([x0 - rail, y0 - rail, x1 + rail, y0], fill=COL["rail"])
        d.rectangle([x0 - rail, y1, x1 + rail, y1 + rail], fill=COL["rail"])
        for gx, colour in ((-L / 2, COL["goal"]), (L / 2, COL["own_goal"])):
            top, bottom = self.px(gx, sg.GOAL_WIDTH / 2), self.px(gx, -sg.GOAL_WIDTH / 2)
            xs = top[0] - rail if gx < 0 else top[0]
            d.rectangle([xs, y0, xs + rail, top[1]], fill=COL["rail"])
            d.rectangle([xs, bottom[1], xs + rail, y1], fill=COL["rail"])
            d.rectangle([xs, top[1], xs + rail, bottom[1]], fill=colour)
        gl = self.px(-L / 2, sg.GOAL_WIDTH / 2 + 0.06)
        d.text((gl[0] + 8 * SS, gl[1]), "GOAL", fill=COL["goal"], font=self.f_small)
        rl = self.px(L / 2 - 0.30, W / 2 - 0.03)
        d.text(rl, "robot side", fill=COL["muted"], font=self.f_small)
        for bx, by in blocks:
            a, b = self.px(bx - sg.BLOCK_W / 2, by + sg.BLOCK_W / 2), self.px(bx + sg.BLOCK_W / 2, by - sg.BLOCK_W / 2)
            d.rectangle([a, b], fill=COL["obstacle"])
        if pred is not None and len(pred) > 1:
            for p in pred[::4]:
                self.circle(d, p[0], p[1], 0.006, fill=COL["pred"])
        if aim is not None:
            pts = [self.px(*p) for p in aim]
            for a, b in zip(pts, pts[1:]):
                _dashed(d, a, b, COL["aim"], 3 * SS, 14 * SS)
        if intercept is not None:
            self.circle(d, intercept[0], intercept[1], sg.PUCK_R + 0.012, outline=COL["pred"], width=3 * SS)
        for i, p in enumerate(trail):
            shade = int(235 - 75 * (i + 1) / max(1, len(trail)))
            self.circle(d, p[0], p[1], 0.008, fill=(shade, shade, shade + 6))
        self.circle(d, paddle[0], paddle[1], 0.0508, fill=COL["paddle"])
        self.circle(d, puck[0], puck[1], sg.PUCK_R, fill=COL["puck"])
        d.text((self.margin * SS, 14 * SS), title, fill=COL["ink"], font=self.f_title)
        right = status + ("   paddle hit" if hit else "")
        tw = d.textlength(right, font=self.f_small)
        d.text((self.width * SS - self.margin * SS - tw, 18 * SS), right, fill=COL["muted"], font=self.f_small)
        if banner:
            text, good = banner
            tw = d.textlength(text, font=self.f_banner)
            bx = self.width * SS / 2
            by = (self.band + self.margin + 14) * SS
            d.rounded_rectangle([bx - tw / 2 - 16 * SS, by, bx + tw / 2 + 16 * SS, by + 40 * SS], 8 * SS,
                                fill=COL["good"] if good else COL["bad"])
            d.text((bx - tw / 2, by + 6 * SS), text, fill="white", font=self.f_banner)
        return im.resize((self.width, self.height), Image.LANCZOS)


def _dashed(d, a, b, colour, width, dash):
    a, b = np.array(a, float), np.array(b, float)
    length = np.linalg.norm(b - a)
    if length < 1:
        return
    u = (b - a) / length
    for s in np.arange(0, length, 2 * dash):
        p, q = a + u * s, a + u * min(s + dash, length)
        d.line([tuple(p), tuple(q)], fill=colour, width=width)


OUTCOME_TEXT = {"goal": ("GOAL", True), "timeout": ("TIME RAN OUT", False), "conceded": ("CONCEDED", False),
                "obstacle_collision": ("HIT OBSTACLE", False), "escaped": ("LEFT TABLE", False),
                "workspace_violation": ("OUT OF BOUNDS", False)}


def record(env, actor, reset_kwargs, title, planner=None, overlay=False, canvas=None):
    """Run one episode; actor(obs) -> action. Returns PIL frames and the final info dict."""
    canvas = canvas or Canvas()
    obs, _ = env.reset(**reset_kwargs)
    blocks = list(env.user_block_positions)
    frames, trail, info, t, done = [], [], {"hit": False}, 0, False

    def draw(banner=None):
        st = env.current_state
        puck = st["pucks"][0]["position"]
        paddle = st["paddles"]["paddle_ego"]["position"]
        pred = aim = intercept = None
        if overlay and planner is not None and planner.plan is not None and planner.struck_at is None:
            plan = planner.plan
            remaining = max(plan["T"] - planner.t, 0.0)
            _, ps, _ = predict(obs[4:6], obs[6:8], planner.damping, horizon=remaining + 0.02)
            pred = ps
            intercept = plan["S"]
            shot = plan["shot"]
            goal_pt = (sg.GOAL_X, shot.get("goal_y", 0.0))
            aim = [tuple(plan["S"]), tuple(shot["aim"])]
            if shot["kind"] == "bank":
                aim.append(goal_pt)
        frames.append(canvas.frame(puck, paddle, blocks, trail[-30:], title, f"t = {t * 0.05:.2f} s",
                                   info.get("hit", False), banner, pred, aim, intercept))

    draw()
    while not done:
        action = actor(obs)
        obs, _, terminated, truncated, info = env.step(action)
        t += 1
        trail.append(env.current_state["pucks"][0]["position"])
        done = terminated or truncated
        draw(OUTCOME_TEXT.get(info["outcome"]) if done else None)
    return frames, info


def save_gif(frames, path, ms=70, hold_ms=1400):
    durations = [ms] * (len(frames) - 1) + [hold_ms]
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=2)


def montage(clips, cols, path, ms=70, hold_frames=20):
    """Play several episodes side by side; each holds its last frame until the longest finishes."""
    n = max(len(c) for c in clips) + hold_frames
    w, h = clips[0][0].size
    rows = -(-len(clips) // cols)
    gap = 10
    out = []
    for i in range(n):
        im = Image.new("RGB", (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), (255, 255, 255))
        for k, clip in enumerate(clips):
            im.paste(clip[min(i, len(clip) - 1)], ((k % cols) * (w + gap), (k // cols) * (h + gap)))
        out.append(im)
    save_gif(out, path, ms=ms, hold_ms=ms)
