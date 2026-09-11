
"""

"""

import argparse
import queue
import sys
import threading
import time

import cv2
import numpy as np


PHONE_SCORE = 0.22        
BOOK_SCORE = 0.40
REMOTE_SCORE = 0.35       
PHONE_HOLD_MS = 3000     
BOOK_HOLD_MS = 1500
NEUTRAL_AFTER_MS = 4000
DETECT_EVERY = 2          


BG_NEUTRAL = (20, 16, 16)
BG_PHONE_A = (13, 13, 143)
BG_PHONE_B = (5, 5, 92)       
BG_BOOK = (38, 92, 15)
ACCENT_NEUTRAL = (49, 42, 42)
ACCENT_PHONE = (59, 59, 255)
ACCENT_BOOK = (113, 204, 46)
WHITE = (255, 255, 255)
DARK_GREEN = (15, 41, 6)

PAD = 24
HEADER = 96
FOOTER = 132
BORDER = 5

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX



class Speaker:
    """ """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.available = False
        self._queue = queue.Queue()
        if not enabled:
            return
        try:
            import pyttsx3  
        except ImportError:
            print("[info] pyttsx3 not installed - running without voice.")
            return
        self.available = True
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        if self.available and self.enabled:
            self._queue.put(text)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def _run(self):
        import pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        except Exception as exc:                       
            print(f"[info] voice engine unavailable ({exc}).")
            self.available = False
            return
        while True:
            text = self._queue.get()
            if text is None:
                break
            
            while not self._queue.empty():
                try:
                    text = self._queue.get_nowait()
                except queue.Empty:
                    break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass


class Detector:
    """
    """

    WANTED = {"cell phone", "book", "remote"}

    def __init__(self, weights="yolov8n.pt"):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit(
                "ultralytics is not installed.\n"
                "  pip install ultralytics opencv-python pyttsx3"
            )
        print(f"[info] loading {weights} (first run downloads the weights)...")
        self.model = YOLO(weights)
        self.names = self.model.names
        print("[info] model ready.")

    def detect(self, frame, floor=0.15):
        """"""
        results = self.model.predict(
            frame, conf=floor, verbose=False, max_det=20
        )
        out = []
        for box in results[0].boxes:
            name = self.names[int(box.cls[0])]
            if name not in self.WANTED:
                continue
            score = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append((name, score, (x1, y1, x2, y2)))
        return out



def centred_text(canvas, text, y, font, scale, colour, thickness=2):
    (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (canvas.shape[1] - w) // 2
    cv2.putText(canvas, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def pill(canvas, text, y, bg, fg):
    (w, h), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    x = (canvas.shape[1] - w) // 2
    cv2.rectangle(canvas, (x - 22, y - h - 10), (x + w + 22, y + 12), bg, -1)
    cv2.putText(canvas, text, (x, y), FONT, 0.7, fg, 2, cv2.LINE_AA)


def draw_box(frame, box, colour, label, score):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 4)
    tag = f"{label} {round(score * 100)}%"
    cv2.putText(frame, tag, (x1 + 6, max(28, y1 - 10)),
                FONT, 0.8, colour, 2, cv2.LINE_AA)


def compose(frame, state, pill_text, message, debug, show_debug, muted, name):
    """"""
    h, w = frame.shape[:2]
    canvas = np.zeros((h + HEADER + FOOTER + PAD * 2, w + PAD * 2, 3), np.uint8)

    if state == "phone":
      
        bg = BG_PHONE_A if int(time.time() * 2) % 2 == 0 else BG_PHONE_B
        accent, pill_bg, pill_fg = ACCENT_PHONE, ACCENT_PHONE, WHITE
    elif state == "book":
        bg = BG_BOOK
        accent, pill_bg, pill_fg = ACCENT_BOOK, ACCENT_BOOK, DARK_GREEN
    else:
        bg = BG_NEUTRAL
        accent, pill_bg, pill_fg = ACCENT_NEUTRAL, ACCENT_NEUTRAL, WHITE
    canvas[:] = bg

    centred_text(canvas, f"Study Guardian - {name} Edition", 46, FONT, 0.95, WHITE, 2)
    pill(canvas, pill_text, HEADER - 4, pill_bg, pill_fg)

    top, left = HEADER + PAD, PAD
    canvas[top:top + h, left:left + w] = frame
    cv2.rectangle(canvas, (left - BORDER, top - BORDER),
                  (left + w + BORDER, top + h + BORDER), accent, BORDER)

    base = top + h + 52
    centred_text(canvas, message, base, FONT, 1.0, WHITE, 2)
    if show_debug and debug:
        centred_text(canvas, debug, base + 34, FONT_S, 0.5, (150, 150, 150), 1)
    hint = "Phone = red  -  Book = green  -  [q] quit  [m] %s  [d] debug" % (
        "unmute" if muted else "mute"
    )
    centred_text(canvas, hint, base + 66, FONT_S, 0.48, (140, 140, 140), 1)
    return canvas



def main():
    ap = argparse.ArgumentParser(description="Study Guardian - webcam focus watchdog")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="YOLO weights; yolov8s.pt is slower but sharper")
    ap.add_argument("--name", default="Tan", help="name used in the messages")
    ap.add_argument("--no-voice", action="store_true", help="run silently")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    messages = {
        "phone": (
            "DISTRACTED",
            f"{args.name}, don't do this! You should focus and learn code",
            f"{args.name}, don't do this. You should focus and learn code.",
        ),
        "book": (
            "FOCUSED",
            f"Good boy {args.name}. You got this!",
            f"Good boy {args.name}. You got this.",
        ),
        "neutral": ("WAITING...", "Show me what you're doing...", None),
    }

    speaker = Speaker(enabled=not args.no_voice)
    detector = Detector(args.model)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        sys.exit(f"Couldn't open camera {args.camera}. Try --camera 1.")

    state = "neutral"
    last_phone = last_book = -1e9
    boxes = []
    debug = ""
    show_debug = True
    frame_no = 0
    window = "Study Guardian"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] lost the camera feed.")
                break
            frame = cv2.flip(frame, 1)          
            now = time.time() * 1000
            frame_no += 1

            if frame_no % DETECT_EVERY == 0:
                boxes, seen = [], []
                for name, score, box in detector.detect(frame):
                    seen.append(f"{name} {round(score * 100)}%")
                    if name == "cell phone" and score >= PHONE_SCORE:
                        last_phone = now
                        boxes.append((box, ACCENT_PHONE, "PHONE!", score))
                    elif name == "book" and score >= BOOK_SCORE:
                        last_book = now
                        boxes.append((box, ACCENT_BOOK, "BOOK", score))
                    elif name == "remote" and score >= REMOTE_SCORE:
                        last_phone = now
                        boxes.append((box, ACCENT_PHONE, "PHONE!", score))
                debug = "sees: " + ", ".join(seen[:4]) if seen else ""

            for box, colour, label, score in boxes:
                draw_box(frame, box, colour, label, score)

            
            if now - last_phone < PHONE_HOLD_MS:
                nxt = "phone"                                  
            elif now - last_book < BOOK_HOLD_MS:
                nxt = "book"
            elif now - max(last_phone, last_book) > NEUTRAL_AFTER_MS:
                nxt = "neutral"
            else:
                nxt = state

            if nxt != state:
                state = nxt
                if messages[state][2]:
                    speaker.say(messages[state][2])

            pill_text, message, _ = messages[state]
            cv2.imshow(window, compose(frame, state, pill_text, message, debug,
                                       show_debug, not speaker.enabled, args.name))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                speaker.toggle()
            if key == ord("d"):
                show_debug = not show_debug
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
