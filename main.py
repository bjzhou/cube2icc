#!/usr/bin/env python3
"""
CUBE to ICC Converter
"""

import sys
import struct
import argparse
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

# Check for colour-science
try:
    import colour
except ImportError:
    print("错误: 需要安装 colour-science 库 (pip install colour-science)")
    sys.exit(1)

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
                header = bytearray(f.read(128))
                if len(header) < 128: return False
                self.header = header
                
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

# --- ICC Writer (With Colour Science Support) ---

class ICCWriter:
    def __init__(self, cube_data, title, base_profile_path=None, base_profile_data=None, 
                 target_gamut='sRGB', target_curve='sRGB',
                 lut_output_gamut='sRGB', lut_output_curve='sRGB'):
        self.lut = cube_data
        self.size = cube_data.shape[0]
        self.title = title
        self.base_profile_path = base_profile_path
        self.base_profile_data = base_profile_data # (header, tags) tuple
        
        # Color Management Parameters
        self.target_gamut_name = target_gamut
        self.target_curve_name = target_curve
        self.lut_output_gamut_name = lut_output_gamut
        self.lut_output_curve_name = lut_output_curve

    def create_profile(self, output_path):
        print("正在构建色彩转换链路...")
        print(f"  Target Space: {self.target_gamut_name} / {self.target_curve_name}")
        print(f"  LUT Output Space: {self.lut_output_gamut_name} / {self.lut_output_curve_name}")
        
        # Target Grid Size for ICC (33 is standard)
        grid_size = 33
        
        # 1. Generate Input Grid (Camera RGB Linear)
        x = np.linspace(0, 1, grid_size)
        
        # Generate coordinates for (R, G, B)
        # We need an array where we can iterate through all combinations
        grid_points = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
        for r in range(grid_size):
            for g in range(grid_size):
                for b in range(grid_size):
                    grid_points[r, g, b] = [x[r], x[g], x[b]]
                    
        flat_camera_rgb = grid_points.reshape(-1, 3)
        
        # 2. Camera RGB -> sRGB (via ImageCms)
        # We start by converting the Camera RGB (defined by base profile) to sRGB.
        # This gives us a known starting point for colour-science.
        
        flat_srgb = flat_camera_rgb # Default fallback
        
        if self.base_profile_path:
            print("  Step: Camera -> sRGB (ImageCms)...")
            srgb_profile = ImageCms.createProfile("sRGB")
            try:
                # Intent: Relative Colorimetric
                transform = ImageCms.buildTransform(self.base_profile_path, srgb_profile, "RGB", "RGB", renderingIntent=1)
                
                # Input to ImageCms must be u8 usually for simple usage, or handle float bytes.
                # Standard ImageCms works well with PIL Images.
                img_in = Image.fromarray((flat_camera_rgb * 255).astype(np.uint8).reshape(1, -1, 3))
                img_out = ImageCms.applyTransform(img_in, transform)
                
                # Back to float 0..1 (Gamma Encoded sRGB)
                flat_srgb = np.array(img_out).reshape(-1, 3) / 255.0
                
            except Exception as e:
                print(f"⚠️ CMS 转换失败: {e}")
                print("回退到直接映射...")
                flat_srgb = flat_camera_rgb
        
        # 3. sRGB (Gamma) -> XYZ -> Target (Linear) -> Target (Log)
        print(f"  Step: sRGB -> {self.target_gamut_name} (Log: {self.target_curve_name})...")
        
        # sRGB (Gamma) -> sRGB (Linear)
        try:
             flat_srgb_linear = colour.cctf_decoding(flat_srgb, function='sRGB')
        except ValueError: # fallback for newer colour versions requiring different name
             flat_srgb_linear = colour.sRGB_to_XYZ(flat_srgb) # This goes to XYZ directly if needed, but let's stick to decoding
             # Actually sRGB decoding is standard.
             pass

        # sRGB (Linear) -> XYZ (D50 adapted)
        # Colour science sRGB -> XYZ is usually D65.
        # We need to be careful with white points.
        # ImageCms sRGB profile is likely D50 adapted (standard ICC PCS).
        # However, let's assume the sRGB values we got are standard D65 sRGB values.
        # So we convert sRGB(Linear) -> XYZ(D65) -> XYZ(Target WP) -> Target RGB(Linear)
        
        srgb_cs = colour.RGB_COLOURSPACES['sRGB']
        XYZ = colour.RGB_to_XYZ(flat_srgb_linear, 
                                srgb_cs.whitepoint,
                                srgb_cs.whitepoint,
                                srgb_cs.matrix_RGB_to_XYZ)
                                
        # XYZ -> Target RGB (Linear)
        # We need to know the target gamut whitepoint
        try:
            target_cs = colour.RGB_COLOURSPACES[self.target_gamut_name]
        except KeyError:
            print(f"❌ Error: Gamut '{self.target_gamut_name}' not found.")
            return False
            
        # If target uses different whitepoint, RGB_to_RGB handles chromatic adaptation
        flat_target_linear = colour.RGB_to_RGB(flat_srgb_linear,
                                               srgb_cs,
                                               target_cs)
                                               
        # Target RGB (Linear) -> Target RGB (Log/Encoded)
        # Note: colour.cctf_encoding expects linear input
        try:
            # First try specific name if it exists in cctf_encoding
            flat_target_log = colour.cctf_encoding(flat_target_linear, function=self.target_curve_name)
        except ValueError:
            # Fallback for some curves that might be named differently or strictly OETFs
            try: 
                 # Some curves are OETFs strictly
                 flat_target_log = colour.oetf(flat_target_linear, function=self.target_curve_name)
            except:
                 print(f"❌ Error: Curve '{self.target_curve_name}' not found.")
                 return False

        # flat_target_log = np.nan_to_num(flat_target_log)
        # flat_target_log = np.clip(flat_target_log, 0.0, 1.0)

        # 4. Apply LUT
        print("  Step: Applying 3D LUT...")
        # LUT expects B, G, R coordinates
        lut_coords = flat_target_log[:, ::-1] * (self.size - 1)
        interpolator = TrilinearInterpolator(self.lut)
        flat_lut_out = interpolator(lut_coords) # Result is (N, 3)

        flat_lut_out = self.apply_gamma_s_curve(flat_lut_out, gamma=2.2, contrast=1.0/1.3)

        # 5. LUT Output -> Lab (D50)
        # We need to interpret what the LUT output IS.
        # defined by lut_output_gamut and lut_output_curve.
        
        print(f"  Step: LUT Output ({self.lut_output_gamut_name}/{self.lut_output_curve_name}) -> Lab...")
        
        # LUT Out (Log/Gamma) -> Linear
        try:
            flat_out_linear = colour.cctf_decoding(flat_lut_out, function=self.lut_output_curve_name)
        except ValueError:
             try:
                 flat_out_linear = colour.eotf(flat_lut_out, function=self.lut_output_curve_name)
             except:
                 print(f"❌ Error: Curve '{self.lut_output_curve_name}' not found for decoding.")
                 return False
                 
        # Linear -> XYZ (D50)
        # We need final XYZ to be D50 for ICC Lab conversion.
        try:
            output_cs = colour.RGB_COLOURSPACES[self.lut_output_gamut_name]
        except KeyError:
             print(f"❌ Error: Gamut '{self.lut_output_gamut_name}' not found.")
             return False

        
        # Calculate Matrix RGB -> XYZ(D50)
        # If current WP is not D50, we adapt to D50.
        D50 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
        
        # We can use RGB_to_XYZ directly specifying illuminant if supported, 
        # but RGB_to_XYZ generally returns XYZ relative to the RGB space's whitepoint.
        # So we convert RGB -> XYZ(Native) -> XYZ(D50)
        
        XYZ_out = colour.RGB_to_XYZ(flat_out_linear,
                                    output_cs.whitepoint,
                                    output_cs.whitepoint,
                                    output_cs.matrix_RGB_to_XYZ)
                                    
        # Chromatic Adaptation XYZ(Native) -> XYZ(D50)
        if not np.allclose(output_cs.whitepoint, D50, atol=1e-3):
            # Convert xy to XYZ for adaptation function
            XYZ_current_wp = colour.xy_to_XYZ(output_cs.whitepoint)
            XYZ_D50_wp = colour.xy_to_XYZ(D50)
            # Using CAT02 by default usually
            XYZ_D50 = colour.adaptation.chromatic_adaptation_VonKries(XYZ_out, XYZ_current_wp, XYZ_D50_wp)
        else:
            XYZ_D50 = XYZ_out
            
        # XYZ(D50) -> Lab
        flat_lab = colour.XYZ_to_Lab(XYZ_D50, illuminant=D50)
        
        # 6. Encode to ICC uint16
        lab_u16 = np.empty_like(flat_lab, dtype=np.uint16)
        # Lab in ICC is: L scaled 0..100 -> 0..65535, a,b scaled -128..127
        # Wait, standard ICC Lab encoding (v2/v4 mft2) usually:
        # L: 0.0 -> 0x0000, 100.0 -> 0xFFFF (so * 655.35)
        # a: -128 -> 0x0000, 0 -> 0x8000, 127 -> 0xFFFF (offset 128, mul 256)
        
        lab_u16[:, 0] = np.clip(flat_lab[:, 0] * 655.35, 0, 65535).astype(np.uint16)
        lab_u16[:, 1] = np.clip((flat_lab[:, 1] + 128.0) * 256.0, 0, 65535).astype(np.uint16)
        lab_u16[:, 2] = np.clip((flat_lab[:, 2] + 128.0) * 256.0, 0, 65535).astype(np.uint16)
        
        # 7. Build Tags
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
            # Fallback (Generic Header)
            rXYZ = (0.4360747, 0.2225045, 0.0139322)
            gXYZ = (0.3850649, 0.7168786, 0.0971045)
            bXYZ = (0.1430804, 0.0606169, 0.7141733)

            final_tags = [
                (b'desc', self._make_desc(f"{self.title} (Target: {self.target_curve_name})")),
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
    
    def apply_gamma_s_curve(self, arr, gamma=2.2, contrast=1.0):
        """
        模拟标准曲线：先 Gamma, 再 S 曲线，再反 Gamma (为了保持线性工作流)
        contrast > 1.0 : 增加对比 (模拟 Adobe Standard / Film Standard)
        contrast < 1.0 : 降低对比 (模拟 Inverse Curve)
        """
        # 1. 转换到感知域 (Gamma 编码)
        # 加上极小值防止 0 的导数问题
        enc = np.power(np.maximum(arr, 0), 1/gamma)
        
        # 2. 应用 S 曲线 (以 0.5 为轴心)
        # 使用 arctan 模拟平滑的 S 曲线 (Sigmoid)
        # 公式: 归一化(arctan((x-0.5)*contrast))
        
        x = enc - 0.5
        
        # 这种 S 曲线算法是可逆的，且非常接近胶片响应
        # 预计算归一化因子
        norm_factor = np.arctan(0.5 * contrast) * 2.0
        
        # 应用
        if contrast != 1.0:
            y = np.arctan(x * contrast) * 2.0 / norm_factor
            y = y * 0.5 + 0.5
        else:
            y = enc
            
        # Clip 保护
        y = np.clip(y, 0, 1)
        
        # 3. 转换回线性域
        return np.power(y, gamma)

    def _make_xyz(self, xyz_tuple):
        return struct.pack('>3i', 
                           s15Fixed16Number(xyz_tuple[0]), 
                           s15Fixed16Number(xyz_tuple[1]), 
                           s15Fixed16Number(xyz_tuple[2]))

    def _make_trc(self, gamma):
        gamma_val = int(round(gamma * 256.0))
        return b'curv\x00\x00\x00\x00\x00\x00\x00\x01' + struct.pack('>H', gamma_val)

    def _make_mft2(self, clut_u16, grid_size):
        # mft2 (lut16) structure
        header = bytearray(b'mft2\x00\x00\x00\x00\x03\x03' + struct.pack('B', grid_size) + b'\x00')
        header += struct.pack('>9i', s15Fixed16Number(1),0,0,0,s15Fixed16Number(1),0,0,0,s15Fixed16Number(1)) # Matrix
        header += struct.pack('>2H', 256, 256) # Table entries
        
        # Input Table: Identity
        ramp = np.linspace(0, 65535, 256, dtype=np.uint16).astype('>u2').tobytes()
        input_tbl = ramp * 3
        
        clut_bytes = clut_u16.astype('>u2').tobytes()
        
        # Output Table: Identity
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
        
        if base_header:
            head = bytearray(base_header)
        else:
             # Minimal valid header if no base provided
             head = bytearray(128)
             head[0:4] = struct.pack('>I', size)
             head[12:16] = b'mntr' # Class: Monitor
             head[16:20] = b'RGB ' # Data Space
             head[20:24] = b'XYZ ' # PCS
             head[64:68] = struct.pack('>I', 0x61637370) # 'acsp' signature
        
        # Update size
        head[0:4] = struct.pack('>I', size)
        
        with open(path, 'wb') as f:
            f.write(head + table_bytes + data_bytes)

# --- Presets ---

PRESETS = {
    'F-Log2': {'gamut': 'F-Gamut', 'curve': 'F-Log2'},
    'F-Log':  {'gamut': 'F-Gamut', 'curve': 'F-Log'},
    'V-Log':  {'gamut': 'V-Gamut', 'curve': 'V-Log'},
    'S-Log3': {'gamut': 'S-Gamut3.Cine', 'curve': 'S-Log3'}, # Common pairing, though S-Gamut3 also exists
    'LogC3':  {'gamut': 'Alexa Wide Gamut', 'curve': 'LogC3'},
    'LogC4':  {'gamut': 'Alexa Wide Gamut 4', 'curve': 'LogC4'},
}

def main():
    parser = argparse.ArgumentParser(description="Convert CUBE LUT to ICC Profile with Color Space Control")
    parser.add_argument("input_cube", help="Input .cube file")
    
    # Target (Inputs to LUT)
    parser.add_argument("--target-gamut", default="sRGB", help="Target Colour Gamut (e.g. F-Gamut, S-Gamut3, sRGB)")
    parser.add_argument("--target-curve", default="sRGB", help="Target OETF/Curve (e.g. F-Log2, S-Log3, sRGB)")
    
    # LUT Output
    parser.add_argument("--lut-output-gamut", default="sRGB", help="LUT Output Gamut (default: sRGB)")
    parser.add_argument("--lut-output-curve", default="sRGB", help="LUT Output Gamma/Curve (default: sRGB)")
    
    # Preset
    parser.add_argument("--preset", choices=PRESETS.keys(), help="Apply preset for Target Gamut/Curve")
    
    # Base Profile
    default_base_icc = '/Applications/Capture One.app/Contents/Frameworks/ImageProcessing.framework/Versions/A/Resources/Profiles/Input/SonyA7CM2-ProStandard.icm'
    # default_base_icc = '/Applications/Capture One.app/Contents/Frameworks/ImageProcessing.framework/Versions/A/Resources/Profiles/Input/FujiXT5-Generic.icm'
    parser.add_argument("--base-icc", default=default_base_icc, help=f"Base ICC Profile path (default: {default_base_icc})")

    args = parser.parse_args()
    
    # Apply Preset override
    if args.preset:
        p = PRESETS[args.preset]
        args.target_gamut = p['gamut']
        args.target_curve = p['curve']
        print(f"应用预设 [{args.preset}]: Gamut={args.target_gamut}, Curve={args.target_curve}")

    f = args.input_cube
    print(f"处理: {f}")
    
    # Base Profile Path
    base_icc_path = args.base_icc
    base_profile_data = None
    
    if Path(base_icc_path).exists():
        print(f"读取基础 ICC: {base_icc_path}")
        parser = ICCParser(base_icc_path)
        if parser.parse():
            base_profile_data = (parser.header, parser.tags)
    else:
        print(f"⚠️ 未找到基础 ICC: {base_icc_path}")
        exit(1)

    reader = CUBEReader(f)
    if reader.read():
        if reader.size != 33: reader.data = reader.resample(33)
        
        # Construct Output Filename: {CameraModel}-{LUTName}.icc
        # Example: SonyA7CM2-ProStandard.icm + FujiNC.cube -> SonyA7CM2-FujiNC.icc
        
        base_stem = Path(base_icc_path).stem
        # Heuristic: Take part before the last dash as the model name
        # If no dash, use the whole stem
        if '-' in base_stem:
            camera_model = base_stem.rpartition('-')[0]
        else:
            camera_model = base_stem
            
        lut_name = Path(f).stem
        out_filename = f"{camera_model}-{lut_name}.icc"
        out = Path(f).parent / out_filename
        writer = ICCWriter(reader.data, reader.title, base_icc_path, base_profile_data,
                           target_gamut=args.target_gamut,
                           target_curve=args.target_curve,
                           lut_output_gamut=args.lut_output_gamut,
                           lut_output_curve=args.lut_output_curve)
                           
        if writer.create_profile(str(out)):
            print(f"✅ 生成完毕: {out}")
        else:
            print("❌ 生成失败")
            exit(1)

if __name__ == '__main__':
    main()