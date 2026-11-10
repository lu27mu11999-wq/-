"""Reproduce the 120-frame synthetic experiment from section 3.3."""
from collections import deque
from dataclasses import dataclass

WIDTH, HEIGHT = 320, 180
FRAMES = 120
SIZE = 24
FPS = 25
FACE_AREA = SIZE * SIZE
NON_TARGET_PIXELS = FRAMES * (WIDTH * HEIGHT - FACE_AREA)


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def area(self):
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    def intersection(self, other):
        return Box(max(self.left, other.left), max(self.top, other.top),
                   min(self.right, other.right), min(self.bottom, other.bottom))

    def expand(self, margin):
        return Box(self.left - margin, self.top - margin,
                   self.right + margin, self.bottom + margin).intersection(FRAME)


FRAME = Box(0, 0, WIDTH, HEIGHT)
EMPTY = Box(0, 0, 0, 0)


def evaluate(*, compensate=False, dynamic=False):
    detections = deque(maxlen=2)
    insufficient = longest = streak = extra = placeholders = 0
    for frame in range(FRAMES):
        left = 20 + frame
        truth = Box(left, 65, left + SIZE, 65 + SIZE)
        mask = EMPTY

        if 60 <= frame <= 62:
            if dynamic:
                mask = FRAME
                placeholders += 1
        elif 20 <= frame <= 22:
            if compensate and len(detections) == 2:
                (previous_frame, previous_x), (last_frame, last_x) = detections
                gap = frame - last_frame
                if gap <= 3:
                    displacement = (last_x - previous_x) * gap // (last_frame - previous_frame)
                    predicted = last_x + displacement
                    margin = 4 + gap if dynamic else 4
                    mask = Box(predicted, 65, predicted + SIZE, 65 + SIZE).expand(margin)
        else:
            low_confidence = frame % 10 == 5
            detected = left + (7 if low_confidence else 0)
            detections.append((frame, detected))
            margin = 8 if dynamic and low_confidence else 4
            mask = Box(detected, 65, detected + SIZE, 65 + SIZE).expand(margin)

        covered = mask.intersection(truth).area
        extra += mask.area - covered
        if covered * 100 < FACE_AREA * 95:
            insufficient += 1
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0

    return insufficient, longest, extra, placeholders


def main():
    cases = (
        ("Fixed", evaluate(), (18, 3, 51936, 0)),
        ("Fixed + prediction", evaluate(compensate=True), (15, 3, 53280, 0)),
        ("Dynamic", evaluate(compensate=True, dynamic=True), (0, 0, 231224, 3)),
    )
    for name, actual, expected in cases:
        insufficient, longest, extra, placeholders = actual
        print(f"{name}: insufficient={insufficient}/{FRAMES} ({insufficient / FRAMES:.2%}), "
              f"longest={longest} frames ({longest * 1000 // FPS} ms), "
              f"extra={extra}/{NON_TARGET_PIXELS} ({extra / NON_TARGET_PIXELS:.3%}), "
              f"placeholders={placeholders}")
        if actual != expected:
            raise SystemExit(f"{name}: expected {expected}, got {actual}")


if __name__ == "__main__":
    main()
