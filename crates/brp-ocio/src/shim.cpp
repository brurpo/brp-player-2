// C entry points over OpenColorIO for Rust (DESIGN.md §7d): load configs, describe
// them as JSON, and make GPU shaders (Vulkan GLSL + LUT textures) or CPU processors for
// a conversion, within one config or across two (through their interchange roles).

#include <OpenColorIO/OpenColorIO.h>

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace OCIO = OCIO_NAMESPACE;

static char *dup(const std::string &s) {
    char *p = static_cast<char *>(std::malloc(s.size() + 1));
    std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

static void fail(char **err, const char *what) {
    if (err) *err = dup(what);
}

static std::string json(const char *s) {
    std::string o = "\"";
    for (const char *c = s ? s : ""; *c; ++c) {
        switch (*c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(*c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof b, "\\u%04x", *c);
                    o += b;
                } else {
                    o += *c;
                }
        }
    }
    return o + "\"";
}

struct BrpConfig {
    OCIO::ConstConfigRcPtr cfg;
};

struct BrpShader {
    OCIO::GpuShaderDescRcPtr desc;
};

struct BrpCpu {
    OCIO::ConstCPUProcessorRcPtr cpu;
};

static const char *encoding_of(const OCIO::ConstColorSpaceRcPtr &cs) {
    const char *e = cs->getEncoding();
    return e ? e : "";
}

/// The processor for src config/space -> dst config and either a colour space
/// (`dst_cs`) or a display/view; `to_xyz`: then on to the destination's display-referred
/// CIE-XYZ D65 (HDR screens: we encode for the swapchain ourselves).
static OCIO::ConstProcessorRcPtr make_whole(BrpConfig *src, const char *src_cs, BrpConfig *dst, const char *display, const char *view, const char *dst_cs, int to_xyz) {
    OCIO::ConstProcessorRcPtr p;
    bool same = src == dst || src->cfg == dst->cfg;
    if (dst_cs && *dst_cs) {
        p = same ? src->cfg->getProcessor(src_cs, dst_cs) : OCIO::Config::GetProcessorFromConfigs(src->cfg, src_cs, dst->cfg, dst_cs);
    } else {
        p = same ? src->cfg->getProcessor(src_cs, display, view, OCIO::TRANSFORM_DIR_FORWARD)
                 : OCIO::Config::GetProcessorFromConfigs(src->cfg, src_cs, dst->cfg, display, view, OCIO::TRANSFORM_DIR_FORWARD);
    }
    if (!to_xyz) return p;
    // The view's display colour space (or the target space) -> display-referred XYZ.
    std::string from = (dst_cs && *dst_cs) ? std::string(dst_cs) : std::string(dst->cfg->getDisplayViewColorSpaceName(display, view));
    if (from == "<USE_DISPLAY_NAME>") from = display;
    auto tail = dst->cfg->getProcessor(from.c_str(), OCIO::ROLE_INTERCHANGE_DISPLAY);
    auto group = OCIO::GroupTransform::Create();
    group->appendTransform(p->createGroupTransform());
    group->appendTransform(tail->createGroupTransform());
    return dst->cfg->getProcessor(group);
}

// ---- the effects' working space (DESIGN.md §7d) ----------------------------------
//
// With effects, the conversion is split at the player's working space: linear light,
// Rec.2020 primaries, 1.0 = SDR white. Part 1 takes the input there, part 2 from there
// to the target. Both halves meet at one of the input config's interchange colour
// spaces, the route OCIO itself takes, so part2(part1(x)) is the whole conversion:
// - display-referred inputs (Rec.1886, sRGB, PQ, HLG displays): display-referred
//   CIE-XYZ D65 (`cie_xyz_d65_interchange`), then XYZ -> Rec.2020: no tone curve, no
//   view. HDR displays' XYZ has 1.0 = 100 nits: scaled by 100/203 (1.0 = SDR white).
// - scene-referred inputs (log, ACEScg, linear): ACES2065-1 (`aces_interchange`), then
//   AP0 -> Rec.2020 exactly as OCIO's studio config converts it.

static void mul3v(const double *m, const double *v, double *o) {
    for (int i = 0; i < 3; ++i) o[i] = m[i * 3] * v[0] + m[i * 3 + 1] * v[1] + m[i * 3 + 2] * v[2];
}

static bool inv3(const double *m, double *o) {
    double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
    double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (det == 0.0) return false;
    double r[9] = {(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det, (f * g - d * i) / det, (a * i - c * g) / det,
                   (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det};
    std::memcpy(o, r, sizeof r);
    return true;
}

/// Linear Rec.2020 -> CIE-XYZ D65 (from the primaries; row-major).
static void rec2020_to_xyz(double *o) {
    const double p[6] = {0.708, 0.292, 0.170, 0.797, 0.131, 0.046};
    const double wx = 0.3127, wy = 0.3290;
    double m[9];
    for (int k = 0; k < 3; ++k) {
        double x = p[2 * k], y = p[2 * k + 1];
        m[k] = x / y;
        m[3 + k] = 1.0;
        m[6 + k] = (1.0 - x - y) / y;
    }
    double mi[9], s[3];
    inv3(m, mi);
    const double w[3] = {wx / wy, 1.0, (1.0 - wx - wy) / wy};
    mul3v(mi, w, s);
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) o[i * 3 + k] = m[i * 3 + k] * s[k];
}

/// ACES2065-1 -> linear Rec.2020, from OCIO's built-in studio config (unit vectors
/// through its own conversion, so it is exactly what that config does).
static bool ap0_to_2020(double *o) {
    static double m[9];
    static int state = 0;  // 0 not tried, 1 ok, 2 failed
    if (state == 0) {
        try {
            auto cfg = OCIO::Config::CreateFromFile("ocio://studio-config-latest");
            auto cpu = cfg->getProcessor("ACES2065-1", "Linear Rec.2020")->getDefaultCPUProcessor();
            for (int k = 0; k < 3; ++k) {
                float v[3] = {0, 0, 0};
                v[k] = 1.0f;
                cpu->applyRGB(v);
                for (int i = 0; i < 3; ++i) m[i * 3 + k] = v[i];
            }
            state = 1;
        } catch (...) {
            state = 2;
        }
    }
    if (state != 1) return false;
    std::memcpy(o, m, sizeof m);
    return true;
}

static OCIO::MatrixTransformRcPtr matrix(const double *m3) {
    double m44[16] = {m3[0], m3[1], m3[2], 0, m3[3], m3[4], m3[5], 0, m3[6], m3[7], m3[8], 0, 0, 0, 0, 1};
    auto t = OCIO::MatrixTransform::Create();
    t->setMatrix(m44);
    return t;
}

/// Where the input meets the working space: the interchange colour space of `src` the
/// conversion goes through, and its matrix to linear Rec.2020 (1.0 = SDR white).
static std::string split_hub(BrpConfig *src, const char *src_cs, double *to_work) {
    auto cs = src->cfg->getColorSpace(src_cs);
    if (!cs) throw OCIO::Exception((std::string("no colour space ") + src_cs).c_str());
    if (cs->isData()) throw OCIO::Exception("a data colour space holds no light to work on");
    bool display = cs->getReferenceSpaceType() == OCIO::REFERENCE_SPACE_DISPLAY;
    const char *role = display ? OCIO::ROLE_INTERCHANGE_DISPLAY : OCIO::ROLE_INTERCHANGE_SCENE;
    auto hub = src->cfg->getColorSpace(role);
    if (!hub) throw OCIO::Exception((std::string("the input's config has no ") + role + " role").c_str());
    if (display) {
        double m[9];
        rec2020_to_xyz(m);
        inv3(m, to_work);
        // HDR displays: XYZ 1.0 = 100 nits; the working space has 1.0 = 203 nits.
        if (std::string(encoding_of(cs)) == "hdr-video")
            for (int i = 0; i < 9; ++i) to_work[i] *= 100.0 / 203.0;
    } else if (!ap0_to_2020(to_work)) {
        throw OCIO::Exception("OCIO's built-in config is not available");
    }
    return hub->getName();
}

/// `part` 0: the whole conversion; 1: input -> working space; 2: working space ->
/// target (+ XYZ).
static OCIO::ConstProcessorRcPtr make_processor(BrpConfig *src, const char *src_cs, BrpConfig *dst, const char *display, const char *view, const char *dst_cs, int to_xyz,
                                                int part) {
    if (part == 0) return make_whole(src, src_cs, dst, display, view, dst_cs, to_xyz);
    double m[9];
    std::string hub = split_hub(src, src_cs, m);
    auto group = OCIO::GroupTransform::Create();
    if (part == 1) {
        group->appendTransform(src->cfg->getProcessor(src_cs, hub.c_str())->createGroupTransform());
        group->appendTransform(matrix(m));
    } else {
        double mi[9];
        if (!inv3(m, mi)) throw OCIO::Exception("singular working-space matrix");
        group->appendTransform(matrix(mi));
        group->appendTransform(make_whole(src, hub.c_str(), dst, display, view, dst_cs, to_xyz)->createGroupTransform());
    }
    return src->cfg->getProcessor(group);
}

extern "C" {

void brp_ocio_free(char *s) { std::free(s); }

const char *brp_ocio_version() { return OCIO::GetVersion(); }

/// A config file, or a built-in one ("ocio://studio-config-latest").
BrpConfig *brp_ocio_load(const char *path, char **err) {
    try {
        auto cfg = OCIO::Config::CreateFromFile(path);
        cfg->validate();
        return new BrpConfig{cfg};
    } catch (const std::exception &e) {
        fail(err, e.what());
        return nullptr;
    }
}

void brp_ocio_release(BrpConfig *c) { delete c; }

/// Everything the interface lists, as JSON.
char *brp_ocio_describe(BrpConfig *c) {
    const auto &cfg = c->cfg;
    std::ostringstream o;
    o << "{\"name\":" << json(cfg->getName()) << ",\"description\":" << json(cfg->getDescription());
    o << ",\"colorspaces\":[";
    int n = cfg->getNumColorSpaces(OCIO::SEARCH_REFERENCE_SPACE_ALL, OCIO::COLORSPACE_ACTIVE);
    for (int i = 0; i < n; ++i) {
        const char *name = cfg->getColorSpaceNameByIndex(OCIO::SEARCH_REFERENCE_SPACE_ALL, OCIO::COLORSPACE_ACTIVE, i);
        auto cs = cfg->getColorSpace(name);
        if (!cs) continue;
        if (i) o << ",";
        o << "{\"name\":" << json(name) << ",\"family\":" << json(cs->getFamily()) << ",\"encoding\":" << json(encoding_of(cs))
          << ",\"data\":" << (cs->isData() ? "true" : "false")
          << ",\"display\":" << (cs->getReferenceSpaceType() == OCIO::REFERENCE_SPACE_DISPLAY ? "true" : "false") << ",\"aliases\":[";
        for (size_t a = 0; a < cs->getNumAliases(); ++a) o << (a ? "," : "") << json(cs->getAlias(a));
        o << "]}";
    }
    o << "],\"displays\":[";
    for (int d = 0; d < cfg->getNumDisplays(); ++d) {
        const char *disp = cfg->getDisplay(d);
        o << (d ? "," : "") << "{\"name\":" << json(disp) << ",\"default_view\":" << json(cfg->getDefaultView(disp)) << ",\"views\":[";
        for (int v = 0; v < cfg->getNumViews(disp); ++v) {
            const char *view = cfg->getView(disp, v);
            std::string space = cfg->getDisplayViewColorSpaceName(disp, view);
            if (space == "<USE_DISPLAY_NAME>") space = disp;
            auto cs = cfg->getColorSpace(space.c_str());
            o << (v ? "," : "") << "{\"name\":" << json(view) << ",\"colorspace\":" << json(space.c_str())
              << ",\"encoding\":" << json(cs ? encoding_of(cs) : "") << "}";
        }
        o << "]}";
    }
    o << "],\"default_display\":" << json(cfg->getDefaultDisplay()) << ",\"roles\":{";
    for (int r = 0; r < cfg->getNumRoles(); ++r) {
        o << (r ? "," : "") << json(cfg->getRoleName(r)) << ":" << json(cfg->getRoleColorSpace(r));
    }
    o << "}}";
    return dup(o.str());
}

/// GPU shader for a conversion (see `make_processor`): Vulkan GLSL with `OCIOMain`,
/// its textures in descriptor set `set` from binding `binding_start`.
BrpShader *brp_ocio_shader(BrpConfig *src, const char *src_cs, BrpConfig *dst, const char *display, const char *view, const char *dst_cs, int to_xyz,
                           int part, unsigned set, unsigned binding_start, char **err) {
    try {
        auto proc = make_processor(src, src_cs, dst, display, view, dst_cs, to_xyz, part);
        auto gpu = proc->getOptimizedGPUProcessor(OCIO::OPTIMIZATION_DEFAULT);
        auto desc = OCIO::GpuShaderDesc::CreateShaderDesc();
        desc->setLanguage(OCIO::GPU_LANGUAGE_GLSL_VK_4_6);
        desc->setFunctionName("OCIOMain");
        desc->setResourcePrefix("ocio_");
        desc->setDescriptorSetIndex(set, binding_start);
        gpu->extractGpuShaderInfo(desc);
        return new BrpShader{desc};
    } catch (const std::exception &e) {
        fail(err, e.what());
        return nullptr;
    }
}

void brp_ocio_shader_release(BrpShader *s) { delete s; }

const char *brp_ocio_shader_text(BrpShader *s) { return s->desc->getShaderText(); }
unsigned brp_ocio_shader_uniforms(BrpShader *s) { return s->desc->getNumUniforms(); }
unsigned brp_ocio_shader_textures(BrpShader *s) { return s->desc->getNumTextures(); }
unsigned brp_ocio_shader_textures3d(BrpShader *s) { return s->desc->getNum3DTextures(); }

/// 1D/2D texture `i`: sizes, 1 or 3 channels, 1 or 2 dimensions, linear filtering,
/// binding, values (width * height * channels floats).
void brp_ocio_shader_texture(BrpShader *s, unsigned i, const char **name, unsigned *w, unsigned *h, unsigned *channels, unsigned *dims, int *linear,
                             unsigned *binding, const float **values) {
    const char *sampler = nullptr;
    OCIO::GpuShaderDesc::TextureType ch;
    OCIO::GpuShaderDesc::TextureDimensions d;
    OCIO::Interpolation interp;
    s->desc->getTexture(i, *name, sampler, *w, *h, ch, d, interp);
    *channels = ch == OCIO::GpuShaderDesc::TEXTURE_RED_CHANNEL ? 1 : 3;
    *dims = d == OCIO::GpuShaderDesc::TEXTURE_1D ? 1 : 2;
    *linear = interp != OCIO::INTERP_NEAREST;
    *binding = s->desc->getTextureShaderBindingIndex(i);
    s->desc->getTextureValues(i, *values);
}

/// 3D texture `i` (RGB, edge^3 * 3 floats, red fastest).
void brp_ocio_shader_texture3d(BrpShader *s, unsigned i, const char **name, unsigned *edge, int *linear, unsigned *binding, const float **values) {
    const char *sampler = nullptr;
    OCIO::Interpolation interp;
    s->desc->get3DTexture(i, *name, sampler, *edge, interp);
    *linear = interp != OCIO::INTERP_NEAREST;
    *binding = s->desc->get3DTextureShaderBindingIndex(i);
    s->desc->get3DTextureValues(i, *values);
}

/// CPU processor for the same conversion (checks, references).
BrpCpu *brp_ocio_cpu(BrpConfig *src, const char *src_cs, BrpConfig *dst, const char *display, const char *view, const char *dst_cs, int to_xyz, int part, char **err) {
    try {
        auto proc = make_processor(src, src_cs, dst, display, view, dst_cs, to_xyz, part);
        return new BrpCpu{proc->getOptimizedCPUProcessor(OCIO::BIT_DEPTH_F32, OCIO::BIT_DEPTH_F32, OCIO::OPTIMIZATION_DEFAULT)};
    } catch (const std::exception &e) {
        fail(err, e.what());
        return nullptr;
    }
}

void brp_ocio_cpu_release(BrpCpu *c) { delete c; }

/// `n` packed RGB pixels in place.
void brp_ocio_cpu_apply(BrpCpu *c, float *rgb, long long n) {
    OCIO::PackedImageDesc img(rgb, n, 1, 3);
    c->cpu->apply(img);
}

}  // extern "C"
