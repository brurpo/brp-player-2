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
static OCIO::ConstProcessorRcPtr make_processor(BrpConfig *src, const char *src_cs, BrpConfig *dst, const char *display, const char *view, const char *dst_cs, int to_xyz) {
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
                           unsigned set, unsigned binding_start, char **err) {
    try {
        auto proc = make_processor(src, src_cs, dst, display, view, dst_cs, to_xyz);
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
BrpCpu *brp_ocio_cpu(BrpConfig *src, const char *src_cs, BrpConfig *dst, const char *display, const char *view, const char *dst_cs, int to_xyz, char **err) {
    try {
        auto proc = make_processor(src, src_cs, dst, display, view, dst_cs, to_xyz);
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
