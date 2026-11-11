"""Side-by-side crops of several pictures, labelled, for looking at differences.

    cropsheet.py out.png x y w h scale a.png [b.png ...]
"""
import sys
from PIL import Image, ImageDraw

out, x, y, w, h, scale = sys.argv[1], *map(int, sys.argv[2:7])
paths = sys.argv[7:]
tiles = []
for p in paths:
    im = Image.open(p).convert("RGB").crop((x, y, x + w, y + h)).resize((w * scale, h * scale), Image.NEAREST)
    ImageDraw.Draw(im).text((4, 4), p.replace("\\", "/").split("/")[-1], fill=(255, 255, 0))
    tiles.append(im)
cols = min(len(tiles), 3)
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w * scale, rows * h * scale))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * w * scale, (i // cols) * h * scale))
sheet.save(out)
