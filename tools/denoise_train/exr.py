"""Minimal OpenEXR reader for the training data: scanline files, ZIP / ZIPS / no
compression, half or float channels (what Blender and the render farm write). Reads
only the blocks a crop needs, and only the channels asked for.

    e = Exr(path)
    e.layers()                                   # {"interior": ["Combined", ...]}
    rgb = e.read("interior.Combined", y0, y1, x0, x1)   # float32 (rows, cols, 3)
"""
import struct, zlib
import numpy as np

COMPRESSION = ["none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab", "htj2k"]
LINES = {"none": 1, "zips": 1, "zip": 16}
DTYPE = {1: np.dtype("<f2"), 2: np.dtype("<f4"), 0: np.dtype("<u4")}


class Exr:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            magic, ver = struct.unpack("<ii", f.read(8))
            if magic != 20000630:
                raise ValueError(f"{path}: not an EXR")
            if ver & 0x200:
                raise ValueError(f"{path}: tiled EXR not supported")
            self.channels = []  # (name, type), sorted as stored
            comp = None
            dw = None

            def cstr():
                b = bytearray()
                while (c := f.read(1)) != b"\0":
                    b += c
                return b.decode()

            while name := cstr():
                typ = cstr()
                size = struct.unpack("<i", f.read(4))[0]
                data = f.read(size)
                if typ == "chlist":
                    i = 0
                    while data[i] != 0:
                        j = data.index(b"\0", i)
                        # name\0, pixel type (int32), pLinear + 3 reserved bytes, x/y sampling.
                        ptype, xs, ys = struct.unpack("<i4xii", data[j + 1 : j + 17])
                        if (xs, ys) != (1, 1):
                            raise ValueError(f"{path}: subsampled channels not supported")
                        self.channels.append((data[i:j].decode(), ptype))
                        i = j + 17
                elif name == "compression":
                    comp = COMPRESSION[data[0]]
                elif name == "dataWindow":
                    dw = struct.unpack("<4i", data)
            if comp not in LINES:
                raise ValueError(f"{path}: {comp} compression not supported")
            self.comp = comp
            self.x0, self.y0, x1, y1 = dw
            self.width, self.height = x1 - self.x0 + 1, y1 - self.y0 + 1
            self.lines = LINES[comp]
            n = (self.height + self.lines - 1) // self.lines
            self.offsets = np.frombuffer(f.read(8 * n), "<u8")
        # Bytes of one line of each channel, and where it starts within a line.
        self.line_bytes = [self.width * DTYPE[t].itemsize for _, t in self.channels]
        self.ch_start = np.cumsum([0] + self.line_bytes[:-1])
        self.stride = sum(self.line_bytes)
        self.index = {n: i for i, (n, _) in enumerate(self.channels)}

    def layers(self):
        out = {}
        for n, _ in self.channels:
            layer, _, ch = n.rpartition(".")
            layer, _, pas = layer.rpartition(".") if "." in layer else ("", "", layer)
            out.setdefault(layer, set()).add(pas)
        return {k: sorted(v) for k, v in out.items()}

    def _block(self, f, i):
        f.seek(int(self.offsets[i]))
        y, size = struct.unpack("<ii", f.read(8))
        data = f.read(size)
        rows = min(self.lines, self.height - (y - self.y0))
        raw_size = rows * self.stride
        if self.comp == "none" or size == raw_size:
            return np.frombuffer(data, np.uint8), rows
        t = np.frombuffer(zlib.decompress(data), np.uint8).astype(np.int32)
        # Predictor: each byte stored as the difference to the previous one (+128).
        t[1:] -= 128
        t = (np.cumsum(t) & 255).astype(np.uint8)
        # The bytes were split: all even ones first, then all odd ones.
        half = (len(t) + 1) // 2
        out = np.empty_like(t)
        out[0::2] = t[:half]
        out[1::2] = t[half:]
        return out, rows

    def read(self, pass_name, y0=0, y1=None, x0=0, x1=None, suffixes=("R", "G", "B")):
        """Channels `<pass_name>.R/G/B` (or other suffixes) for rows y0..y1 and
        columns x0..x1 (data-window coordinates from 0), as float32 (rows, cols, n)."""
        y1 = self.height if y1 is None else y1
        x1 = self.width if x1 is None else x1
        idx = []
        for s in suffixes:
            name = f"{pass_name}.{s}" if pass_name else s
            if name not in self.index:
                raise KeyError(f"{self.path}: no channel {name}")
            idx.append(self.index[name])
        out = np.empty((y1 - y0, x1 - x0, len(idx)), np.float32)
        with open(self.path, "rb") as f:
            for b in range(y0 // self.lines, (y1 - 1) // self.lines + 1):
                data, rows = self._block(f, b)
                block = data[: rows * self.stride].reshape(rows, self.stride)
                by = b * self.lines
                r0, r1 = max(y0, by), min(y1, by + rows)
                for k, ci in enumerate(idx):
                    dt = DTYPE[self.channels[ci][1]]
                    s = self.ch_start[ci]
                    line = block[r0 - by : r1 - by, s : s + self.line_bytes[ci]]
                    vals = np.ascontiguousarray(line).view(dt).reshape(r1 - r0, self.width)
                    out[r0 - y0 : r1 - y0, :, k] = vals[:, x0:x1]
        return out
