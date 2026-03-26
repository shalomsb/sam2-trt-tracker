"""Threaded video reader/writer — overlaps disk I/O with GPU compute."""

import queue
import threading

import cv2

# Reading a frame takes ~5ms, GPU processing takes ~50ms.
# Without threading: read(5ms) → process(50ms) → read(5ms) → process(50ms) = 55ms per frame.
# With threading: a background thread reads frames into a queue while the GPU processes.
# By the time the GPU finishes one frame, the next is already in the queue = ~50ms per frame.

class VideoReaderThread:
    def __init__(self, path: str, qsize: int = 3):
        # Open the video file. Returns a capture object we use to read frames one by one
        # (like open() for text files, but for video).
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        # Queue holds up to 3 frames. If full, the reader thread blocks until the main
        # thread consumes one. If empty, the main thread blocks until one is ready.
        self.q: queue.Queue = queue.Queue(maxsize=qsize)
        # Start reading immediately in the background.
        # daemon=True means the thread dies automatically when the program exits.
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ret, frame = self.cap.read()
            # Push the frame, or None if the video ended (signals the main thread to stop).
            self.q.put(frame if ret else None)
            if not ret:
                break

    # Blocks until a frame is available. Returns None when the video is over.
    def read(self):
        return self.q.get()

    @property
    def fps(self): return self.cap.get(cv2.CAP_PROP_FPS) or 30.0
    @property
    def frame_count(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    @property
    def width(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    @property
    def height(self): return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def release(self): self.cap.release()


# Same idea but reversed: the main thread drops finished frames into the queue
# and returns immediately, while a background thread encodes and writes them to disk.
# qsize=5 (bigger than reader's 3) because encoding is slower than decoding.

class VideoWriterThread:
    def __init__(self, path: str, fourcc, fps, size, qsize: int = 5):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.q: queue.Queue = queue.Queue(maxsize=qsize)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while True:
            frame = self.q.get()
            if frame is None:  # None = "no more frames, stop"
                break
            self.writer.write(frame)

    # Returns immediately — the actual write happens in the background thread.
    def write(self, frame): self.q.put(frame)

    def release(self):
        self.q.put(None)       # signal the writer thread to stop
        self._t.join()         # wait for it to finish writing all queued frames
        self.writer.release()  # close the output file (writes MP4 header/footer)
