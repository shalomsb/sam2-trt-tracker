#!/usr/bin/env python3
"""Click on the image to get pixel coordinates. Press q to quit."""
import cv2
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "frame0.png"
img = cv2.imread(path)
if img is None:
    print(f"Cannot open {path}")
    sys.exit(1)

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"{x},{y}")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, f"({x},{y})", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("Pick Point", img)

cv2.imshow("Pick Point", img)
cv2.setMouseCallback("Pick Point", on_click)
cv2.waitKey(0)
cv2.destroyAllWindows()
