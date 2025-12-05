#!/usr/bin/env python3
"""
CUBE to ICC Converter
"""

import sys
import struct
import datetime
from pathlib import Path

# Check for numpy
try:
    import numpy as np
except ImportError:
    print("错误: 需要安装 numpy 库 (pip install numpy)")
    sys.exit(1)

# Check for PIL
try:
    from PIL import Image, ImageCms
except ImportError:
    print("错误: 需要安装 pillow 库 (pip install pillow)")
    sys.exit(1)

# --- 色彩数学核心 (RGB -> XYZ -> Lab) ---

class ColorMath:
    # D50 White Point for Lab conversion
    Xn, Yn, Zn = 0.9642, 1.0000, 0.8249

    # sRGB to XYZ (D50) Matrix
    M_RGB_XYZ = np.array([
        [0.4360747, 0.3850649, 0.1430804],
        [0.2225045, 0.7168786, 0.0606169],
        [0.0139322, 0.0971045, 0.7141733]
    ])

    @staticmethod
    def rgb_to_lab(rgb_block):
        """
        Convert sRGB (0-1) to Lab (D50)
        L: 0..100, a: -128..127, b: -128..127
        """
        # 1. Linearize sRGB
        mask = rgb_block <= 0.04045
        lin = np.empty_like(rgb_block)
        lin[mask] = rgb_block[mask] / 12.92
        lin[~mask] = ((rgb_block[~mask] + 0.055) / 1.055) ** 2.4

        # 2. To XYZ
        xyz = np.dot(lin, ColorMath.M_RGB_XYZ.T)

        # 3. To Lab
        # Scale relative to Tristimulus (D50)
        xyz[:, 0] /= ColorMath.Xn
        xyz[:, 1] /= ColorMath.Yn
        xyz[:, 2] /= ColorMath.Zn

        # F(t) function
        mask_xyz = xyz > 0.008856
        f_xyz = np.empty_like(xyz)
        f_xyz[mask_xyz] = np.cbrt(xyz[mask_xyz])
        f_xyz[~mask_xyz] = (7.787 * xyz[~mask_xyz]) + (16.0 / 116.0)

        # Calculate L, a, b
        lab = np.empty_like(xyz)
        lab[:, 0] = 116.0 * f_xyz[:, 1] - 16.0
        lab[:, 1] = 500.0 * (f_xyz[:, 0] - f_xyz[:, 1])
        lab[:, 2] = 200.0 * (f_xyz[:, 1] - f_xyz[:, 2])

        return lab

# --- 辅助工具 ---

def s15Fixed16Number(n): return int(round(n * 65536.0)) & 0xFFFFFFFF
def u16Fixed16Number(n): return int(round(n * 65536.0)) & 0xFFFFFFFF

# --- Trilinear Interpolator (Numpy) ---

class TrilinearInterpolator:
    def __init__(self, lut_data):
        # lut_data shape: (D, H, W, 3) -> (B, G, R, 3)
        self.lut = lut_data
        self.size = lut_data.shape[0]

    def __call__(self, coords):
        """
        coords: (N, 3) array of coordinates in range [0, size-1]
        Order: B, G, R (matching LUT structure)
        """
        # Clip coordinates
        coords = np.clip(coords, 0, self.size - 1.00001)
        
        # Get integer and fractional parts
        x0 = np.floor(coords).astype(int)
        x1 = x0 + 1
        
        # Weights
        wd = coords - x0
        
        # Extract indices for 8 corners
        # x0 shape: (N, 3) -> (z0, y0, x0)
        z0, y0, x0_ = x0[:, 0], x0[:, 1], x0[:, 2]
        z1, y1, x1_ = x1[:, 0], x1[:, 1], x1[:, 2]
        
        # Sample LUT
        # c000
        c000 = self.lut[z0, y0, x0_]
        c001 = self.lut[z0, y0, x1_]
        c010 = self.lut[z0, y1, x0_]
        c011 = self.lut[z0, y1, x1_]
        c100 = self.lut[z1, y0, x0_]
        c101 = self.lut[z1, y0, x1_]
        c110 = self.lut[z1, y1, x0_]
        c111 = self.lut[z1, y1, x1_]
        
        # Interpolate along X (last dim)
        wx = wd[:, 2][:, None]
        c00 = c000 * (1 - wx) + c001 * wx
        c01 = c010 * (1 - wx) + c011 * wx
        c10 = c100 * (1 - wx) + c101 * wx
        c11 = c110 * (1 - wx) + c111 * wx
        
        # Interpolate along Y
        wy = wd[:, 1][:, None]
        c0 = c00 * (1 - wy) + c01 * wy
        c1 = c10 * (1 - wy) + c11 * wy
        
        # Interpolate along Z
        wz = wd[:, 0][:, None]
        c = c0 * (1 - wz) + c1 * wz
        
        return c

# --- CUBE 读取器 ---

class CUBEReader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.size = 0
        self.title = Path(filepath).stem
        self.data = None

    def read(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except: return False

        data = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if parts[0] == 'LUT_3D_SIZE': self.size = int(parts[1])
            elif parts[0] == 'TITLE': self.title = " ".join(parts[1:]).strip('"')
            elif len(parts) == 3:
                try: data.append([float(p) for p in parts])
                except: pass
        
        if self.size == 0: self.size = round(len(data)**(1/3))
        
        try:
            # Reshape to (B, G, R, 3) - CUBE standard z,y,x
            self.data = np.array(data, dtype=np.float32).reshape(self.size, self.size, self.size, 3)
            return True
        except: return False

    def resample(self, target=33):
        if self.size == target: return self.data
        print(f"重采样: {self.size} -> {target}")
        idx = np.round(np.linspace(0, self.size-1, target)).astype(int)
        return self.data[np.ix_(idx, idx, idx)]

# --- ICC Parser (Base Profile Reader) ---

class ICCParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = None
        self.tags = []

    def parse(self):
        try:
            with open(self.filepath, 'rb') as f:
                self.header = bytearray(f.read(128))
                if len(self.header) < 128: return False
                
                # Read Tag Table
                tag_count_bytes = f.read(4)
                if len(tag_count_bytes) < 4: return False
                tag_count = struct.unpack('>I', tag_count_bytes)[0]
                
                # Read Tags
                for _ in range(tag_count):
                    entry = f.read(12)
                    if len(entry) < 12: break
                    sig, offset, size = struct.unpack('>4s2I', entry)
                    self.tags.append({'sig': sig, 'offset': offset, 'size': size})
                
                # Read Tag Data
                # We read the whole file content to extract data chunks easily
                f.seek(0)
                full_content = f.read()
                
                parsed_tags = []
                for t in self.tags:
                    data = full_content[t['offset'] : t['offset'] + t['size']]
                    parsed_tags.append((t['sig'], data))
                
                self.tags = parsed_tags
                return True
        except Exception as e:
            print(f"读取 ICC 失败: {e}")
            return False

# --- ICC Writer (Modified for Base Profile Injection) ---

class ICCWriter:
    def __init__(self, cube_data, title, base_profile_path=None, base_profile_data=None):
        self.lut = cube_data
        self.size = cube_data.shape[0]
        self.title = title
        self.base_profile_path = base_profile_path
        self.base_profile_data = base_profile_data # (header, tags) tuple

    def create_profile(self, output_path):
        print("正在构建色彩转换链路...")
        
        # Target Grid Size for ICC (33 is standard)
        grid_size = 33
        
        # 1. Generate Input Grid (Camera RGB Linear)
        # Shape: (33, 33, 33, 3) -> (B, G, R) order for iteration, but we need (R, G, B) for CMS
        # Let's create standard meshgrid
        x = np.linspace(0, 1, grid_size)
        # Meshgrid 'ij' indexing: z, y, x
        # Note: We need to be careful with channel order.
        # ICC CLUT input is usually RGB.
        # Let's generate (R, G, B) coordinates.
        rr, gg, bb = np.meshgrid(x, x, x, indexing='ij')
        # Flatten to (N, 3)
        # Order: We want to iterate R, then G, then B (or B, G, R depending on ICC packing)
        # Actually ICC CLUT is usually packed such that last component varies fastest.
        # But let's stick to a flat list of pixels for CMS.
        
        # Let's use (B, G, R) order to match our LUT logic, then flip for CMS if needed.
        # Wait, CUBE is B, G, R (Z, Y, X).
        # ICC A2B0 input is usually R, G, B.
        # Let's generate R, G, B grid.
        
        # Create (N, 3) array of RGB values
        # We need to match the order expected by the CLUT writing loop later.
        # The CLUT writing loop usually expects: R varies fastest? No, usually B varies fastest in memory.
        # Let's check _make_mft2 packing.
        # It takes `clut_u16` which is passed from `flat_lab`.
        # `flat_lab` comes from `lut_transposed`.
        # Previously: `lut_transposed = self.lut.transpose(2, 1, 0, 3)` -> (R, G, B)
        # So we need to generate a grid that corresponds to (R, G, B) structure.
        
        rr, gg, bb = np.meshgrid(x, x, x, indexing='ij') 
        # If we transpose to (2, 1, 0), we get R, G, B order where B varies fastest?
        # Let's just generate a flat list of all grid points in the order we want to write them.
        # We want to write a 3D table where index [r, g, b] maps to output.
        # So we need to evaluate the pipeline for every [r, g, b] combination.
        
        # Generate coordinates for (R, G, B)
        # R: 0..32, G: 0..32, B: 0..32
        # We want an array where the first dimension corresponds to R, second to G, third to B.
        grid_points = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
        for r in range(grid_size):
            for g in range(grid_size):
                for b in range(grid_size):
                    grid_points[r, g, b] = [x[r], x[g], x[b]]
                    
        # Flatten for processing
        flat_camera_rgb = grid_points.reshape(-1, 3)
        
        # 2. Camera RGB -> sRGB (using Base Profile)
        if self.base_profile_path:
            print("应用基础 ICC 转换 (Camera -> sRGB)...")
            # Create sRGB profile (built-in)
            srgb_profile = ImageCms.createProfile("sRGB")
            
            # Create Transform: Base -> sRGB
            # Intent: Relative Colorimetric (usually best for input -> display)
            try:
                transform = ImageCms.buildTransform(self.base_profile_path, srgb_profile, "RGB", "RGB", renderingIntent=1)
                
                # Convert to 8-bit for PIL (ImageCms works best with images)
                # Note: This might lose some precision, but it's standard for this flow.
                # Alternatively, use floating point if supported, but PIL CMS is often 8-bit.
                img_in = Image.fromarray((flat_camera_rgb * 255).astype(np.uint8).reshape(1, -1, 3))
                img_out = ImageCms.applyTransform(img_in, transform)
                
                # Back to float 0..1
                flat_srgb = np.array(img_out).reshape(-1, 3) / 255.0
                
            except Exception as e:
                print(f"⚠️ CMS 转换失败: {e}")
                print("回退到直接映射...")
                flat_srgb = flat_camera_rgb
        else:
            flat_srgb = flat_camera_rgb

        # 3. Apply LUT (sRGB -> LUT -> sRGB/Modified)
        print("应用 3D LUT...")
        # LUT expects B, G, R coordinates (0..Size-1)
        # Our flat_srgb is R, G, B.
        # So we need to feed [B, G, R] to interpolator.
        
        lut_coords = flat_srgb[:, ::-1] * (self.size - 1) # Flip to BGR and scale
        
        interpolator = TrilinearInterpolator(self.lut)
        flat_lut_out = interpolator(lut_coords) # Result is (N, 3) RGB
        
        # 4. Convert to Lab (D50)
        print("转换到 Lab (D50)...")
        flat_lut_out = np.clip(flat_lut_out, 0.0, 1.0)
        flat_lab = ColorMath.rgb_to_lab(flat_lut_out)
        
        # 5. Encode to ICC uint16
        lab_u16 = np.empty_like(flat_lab, dtype=np.uint16)
        lab_u16[:, 0] = np.clip(flat_lab[:, 0] * 655.35, 0, 65535).astype(np.uint16)
        lab_u16[:, 1] = np.clip((flat_lab[:, 1] + 128.0) * 256.0, 0, 65535).astype(np.uint16)
        lab_u16[:, 2] = np.clip((flat_lab[:, 2] + 128.0) * 256.0, 0, 65535).astype(np.uint16)
        
        # 6. Build Tags
        # Note: We pass grid_size to _make_mft2 because we generated a 33x33x33 grid
        a2b0_data = self._make_mft2(lab_u16, grid_size)
        
        final_tags = []
        
        if self.base_profile_data:
            base_header, base_tags = self.base_profile_data
            
            found_a2b0 = False
            for sig, data in base_tags:
                if sig == b'A2B0':
                    final_tags.append((sig, a2b0_data))
                    found_a2b0 = True
                elif sig == b'desc':
                     final_tags.append((sig, data))
                else:
                    final_tags.append((sig, data))
            
            if not found_a2b0:
                final_tags.append((b'A2B0', a2b0_data))
                
            self._write_file(output_path, final_tags, base_header)
            
        else:
            # Fallback
            rXYZ = (0.4360747, 0.2225045, 0.0139322)
            gXYZ = (0.3850649, 0.7168786, 0.0971045)
            bXYZ = (0.1430804, 0.0606169, 0.7141733)

            final_tags = [
                (b'desc', self._make_desc(f"{self.title} (Monitor)")),
                (b'cprt', b'text\x00\x00\x00\x00CUBE-ICC\x00'),
                (b'wtpt', struct.pack('>3i', s15Fixed16Number(0.9642), s15Fixed16Number(1.0), s15Fixed16Number(0.8249))),
                (b'rXYZ', self._make_xyz(rXYZ)),
                (b'gXYZ', self._make_xyz(gXYZ)),
                (b'bXYZ', self._make_xyz(bXYZ)),
                (b'rTRC', self._make_trc(2.2)),
                (b'gTRC', self._make_trc(2.2)),
                (b'bTRC', self._make_trc(2.2)),
                (b'A2B0', a2b0_data) 
            ]
            self._write_file(output_path, final_tags)
            
        return True

    def _make_xyz(self, xyz_tuple):
        return struct.pack('>3i', 
                           s15Fixed16Number(xyz_tuple[0]), 
                           s15Fixed16Number(xyz_tuple[1]), 
                           s15Fixed16Number(xyz_tuple[2]))

    def _make_trc(self, gamma):
        # 'curv' type
        # Count = 1 (gamma value), value is u8.8
        gamma_val = int(round(gamma * 256.0))
        return b'curv\x00\x00\x00\x00\x00\x00\x00\x01' + struct.pack('>H', gamma_val)

    def _make_mft2(self, clut_u16, grid_size):
        # mft2 (lut16) structure
        header = bytearray(b'mft2\x00\x00\x00\x00\x03\x03' + struct.pack('B', grid_size) + b'\x00')
        header += struct.pack('>9i', s15Fixed16Number(1),0,0,0,s15Fixed16Number(1),0,0,0,s15Fixed16Number(1)) # Matrix
        header += struct.pack('>2H', 256, 256) # Table entries
        
        # Input Table: Identity (Linear)
        # Since we handled the Camera -> sRGB conversion in the CLUT data itself,
        # the input to the A2B0 tag (which is Camera RGB) should be passed through linearly
        # to our CLUT which now encodes the full transform.
        ramp = np.linspace(0, 65535, 256, dtype=np.uint16).astype('>u2').tobytes()
        input_tbl = ramp * 3
        
        clut_bytes = clut_u16.astype('>u2').tobytes()
        
        # Output Table: Linear (Identity)
        output_tbl = ramp * 3
        
        return header + input_tbl + clut_bytes + output_tbl

    def _make_desc(self, text):
        data = text.encode('ascii', 'ignore') + b'\x00'
        return b'desc\x00\x00\x00\x00' + struct.pack('>I', len(data)) + data + b'\x00'*78

    def _write_file(self, path, tags, base_header=None):
        tag_count = len(tags)
        header_size = 128
        table_size = 4 + (tag_count * 12)
        offset = header_size + table_size
        
        table_bytes = struct.pack('>I', tag_count)
        data_bytes = bytearray()
        
        for sig, data in tags:
            pad = (4 - len(data) % 4) % 4
            data += b'\x00' * pad
            table_bytes += sig + struct.pack('>2I', offset, len(data))
            data_bytes += data
            offset += len(data)
            
        # Build Header
        size = header_size + table_size + len(data_bytes)
        
        head = bytearray(base_header)
        # Update size in header
        head[0:4] = struct.pack('>I', size)
        
        with open(path, 'wb') as f:
            f.write(head + table_bytes + data_bytes)

# --- Main ---

def main():
    if len(sys.argv) < 2:
        print("用法: python3 main.py input.cube")
        return
        
    f = sys.argv[1]
    print(f"处理: {f}")
    
    # Base Profile Path
    base_icc_path = '/Applications/Capture One.app/Contents/Frameworks/ImageProcessing.framework/Versions/A/Resources/Profiles/Input/FujiXT5-Generic.icm'
    base_profile_data = None
    
    if Path(base_icc_path).exists():
        print(f"读取基础 ICC: {base_icc_path}")
        parser = ICCParser(base_icc_path)
        if parser.parse():
            base_profile_data = (parser.header, parser.tags)
    else:
        print(f"⚠️ 未找到基础 ICC: {base_icc_path}")
        base_icc_path = None # Ensure it's None if not found
        exit(1)

    reader = CUBEReader(f)
    if reader.read():
        # Monitor LUTs are often 17, 33, or 65.
        # 33 is a good balance for ICC file size.
        if reader.size != 33: reader.data = reader.resample(33)
        
        out = Path(f).with_suffix('.icc')
        writer = ICCWriter(reader.data, reader.title, base_icc_path, base_profile_data)
        if writer.create_profile(str(out)):
            print(f"✅ 生成完毕: {out}")
            if base_profile_data:
                print("类型: 基于 FujiXT5-Generic 修改 (含 Camera->sRGB 转换)")
        else:
            print("❌ 生成失败")
            exit(1)

if __name__ == '__main__':
    main()