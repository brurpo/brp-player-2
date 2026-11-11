//! The Color panel and the Settings section for OpenColorIO (DESIGN.md §7d).
//!
//! Each choice (input, display, view, output) always offers "(Auto)" (from the file,
//! the screen, the export format) and "(Default)" (the one set in Settings), whatever
//! is chosen now, then the configs' entries as nested menus: config (OCIO's own, Maya,
//! Blender), then the config's own families ("Input/ARRI", "Utility"…), then the
//! entries. The branch holding the current choice is marked with •.

use crate::color::Pick;
use crate::player::Player;
use crate::ui::{on_off_switch, section, UiState};
use brp_ocio::auto::{self, Media, OutputTarget, Screen};
use brp_ocio::{Entry, Library};
use egui::{Color32, RichText, Vec2};

const PICK_W: f32 = 300.0;
const LABEL_W: f32 = 64.0;
/// Most a tree shows before it scrolls.
const TREE_H: f32 = 360.0;

/// An entry of a menu tree: the value, its label, and the submenus it sits in.
struct Item<T> {
    value: T,
    /// On the picker's button ("ACEScg (maya)").
    label: String,
    /// In the tree, under its config ("ACEScg").
    short: String,
    path: Vec<String>,
}

/// The tree at `depth` of `items`: a folding branch per group (in order of first
/// appearance; the one holding `cur` starts open), then the entries that end here.
/// Sets `out` on a click.
fn tree_level<T: Clone + PartialEq>(ui: &mut egui::Ui, id: &str, items: &[&Item<T>], depth: usize, cur: Option<&T>, out: &mut Option<T>) {
    let mut branches: Vec<&str> = Vec::new();
    for it in items {
        if let Some(b) = it.path.get(depth) {
            if !branches.contains(&b.as_str()) {
                branches.push(b);
            }
        }
    }
    for b in branches {
        let sub: Vec<&Item<T>> = items.iter().copied().filter(|it| it.path.get(depth).map(String::as_str) == Some(b)).collect();
        let here = cur.is_some_and(|c| sub.iter().any(|it| &it.value == c));
        let text = if here { RichText::new(format!("• {b}")).strong() } else { RichText::new(b) };
        let path_id: Vec<&str> = sub[0].path[..=depth].iter().map(String::as_str).collect();
        egui::CollapsingHeader::new(text).id_salt((id, path_id)).default_open(here).show(ui, |ui| {
            tree_level(ui, id, &sub, depth + 1, cur, out);
        });
    }
    for it in items.iter().filter(|it| it.path.len() == depth) {
        let on = cur == Some(&it.value);
        if ui.selectable_label(on, &it.short).clicked() {
            *out = Some(it.value.clone());
        }
    }
}

/// A picker: a button showing `selected`; open (`s.ocio_open == id`), the choices
/// below it, in the panel (it grows): `top` (Auto, Default…) then the tree.
#[allow(clippy::too_many_arguments)]
fn picker<T: Clone + PartialEq>(ui: &mut egui::Ui, open: &mut Option<String>, id: &str, label: &str, tip: &str, selected: &str, top: &mut dyn FnMut(&mut egui::Ui) -> bool, items: &[Item<T>], cur: Option<&T>, out: &mut Option<T>) {
    let is_open = open.as_deref() == Some(id);
    ui.horizontal(|ui| {
        ui.add_sized(Vec2::new(LABEL_W, 20.0), egui::Label::new(RichText::new(label).color(Color32::from_gray(185)))).on_hover_text(tip);
        let r = egui::Frame::new()
            .fill(if is_open { Color32::from_gray(52) } else { Color32::from_gray(38) })
            .stroke(egui::Stroke::new(1.0, Color32::from_gray(if is_open { 90 } else { 60 })))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::symmetric(8, 4))
            .show(ui, |ui| {
                ui.set_width(PICK_W - 16.0);
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new(RichText::new(selected).color(Color32::from_gray(225))).truncate());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (rr, _) = ui.allocate_exact_size(Vec2::new(10.0, 10.0), egui::Sense::hover());
                        let c = rr.center();
                        let pts = if is_open { vec![c + egui::vec2(-3.5, 1.5), c + egui::vec2(0.0, -2.0), c + egui::vec2(3.5, 1.5)] } else { vec![c + egui::vec2(-3.5, -1.5), c + egui::vec2(0.0, 2.0), c + egui::vec2(3.5, -1.5)] };
                        ui.painter().line(pts, egui::Stroke::new(1.4, Color32::from_gray(200)));
                    });
                });
            })
            .response
            .interact(egui::Sense::click())
            .on_hover_cursor(egui::CursorIcon::PointingHand);
        if r.clicked() {
            *open = if is_open { None } else { Some(id.to_string()) };
        }
    });
    if !is_open {
        return;
    }
    ui.indent((id, "tree"), |ui| {
        egui::Frame::new().fill(Color32::from_gray(28)).corner_radius(6.0).inner_margin(egui::Margin::same(6)).show(ui, |ui| {
            ui.set_width(PICK_W + LABEL_W - 24.0);
            if top(ui) {
                *open = None;
            }
            ui.separator();
            egui::ScrollArea::vertical().id_salt((id, "scroll")).max_height(TREE_H).show(ui, |ui| {
                let refs: Vec<&Item<T>> = items.iter().collect();
                tree_level(ui, id, &refs, 0, cur, out);
            });
        });
    });
    if out.is_some() {
        *open = None;
    }
}

/// One choice of the panel: (Auto), (Default), then the tree. True if changed.
#[allow(clippy::too_many_arguments)]
fn pick_tree<T: Clone + PartialEq>(ui: &mut egui::Ui, open: &mut Option<String>, id: &str, label: &str, tip: &str, pick: &mut Pick<T>, auto_text: &str, default_text: &str, items: &[Item<T>], l: crate::i18n::Lang) -> bool {
    let auto_label = format!("({}) {auto_text}", l.t("ocio_auto"));
    let default_label = format!("({}) {default_text}", l.t("ocio_default"));
    let selected = match &*pick {
        Pick::Auto => auto_label.clone(),
        Pick::Default => default_label.clone(),
        Pick::Manual(t) => items.iter().find(|it| &it.value == t).map(|it| it.label.clone()).unwrap_or_default(),
    };
    let cur = match &*pick {
        Pick::Manual(t) => Some(t.clone()),
        _ => None,
    };
    let mut new: Option<Pick<T>> = None;
    let mut leaf: Option<T> = None;
    let (is_auto, is_default) = (matches!(pick, Pick::Auto), matches!(pick, Pick::Default));
    picker(
        ui,
        open,
        id,
        label,
        tip,
        &selected,
        &mut |ui| {
            if ui.selectable_label(is_auto, &auto_label).clicked() {
                new = Some(Pick::Auto);
            }
            if ui.selectable_label(is_default, &default_label).clicked() {
                new = Some(Pick::Default);
            }
            new.is_some()
        },
        items,
        cur.as_ref(),
        &mut leaf,
    );
    if let Some(t) = leaf {
        new = Some(Pick::Manual(t));
    }
    match new {
        Some(n) if n != *pick => {
            *pick = n;
            true
        }
        _ => false,
    }
}

/// A Settings default: none (Auto), or an entry of the tree. True if changed.
#[allow(clippy::too_many_arguments)]
fn default_tree(ui: &mut egui::Ui, open: &mut Option<String>, id: &str, label: &str, cur: &mut Option<String>, none: &str, items: &[Item<String>]) -> bool {
    let selected = cur.as_ref().and_then(|c| items.iter().find(|it| &it.value == c).map(|it| it.label.clone())).unwrap_or(none.to_string());
    let mut cleared = false;
    let mut leaf: Option<String> = None;
    let is_none = cur.is_none();
    let now = cur.clone();
    picker(
        ui,
        open,
        id,
        label,
        "",
        &selected,
        &mut |ui| {
            if ui.selectable_label(is_none, none).clicked() {
                cleared = true;
            }
            cleared
        },
        items,
        now.as_ref(),
        &mut leaf,
    );
    if cleared && cur.is_some() {
        *cur = None;
        return true;
    }
    match leaf {
        Some(v) if cur.as_ref() != Some(&v) => {
            *cur = Some(v);
            true
        }
        _ => false,
    }
}

/// Colour spaces: config / family path / name.
fn spaces(lib: &Library) -> Vec<Item<Entry>> {
    let mut v = Vec::new();
    for (src, c) in lib.sources() {
        for sp in &c.spaces {
            let mut path = vec![lib.group_name(src)];
            path.extend(sp.family.split('/').map(str::trim).filter(|f| !f.is_empty()).map(str::to_string));
            v.push(Item { value: Entry::new(src, sp.name.clone()), label: sp.name.clone(), short: sp.name.clone(), path });
        }
    }
    v
}

/// Displays: config / name.
fn displays(lib: &Library) -> Vec<Item<Entry>> {
    let mut v = Vec::new();
    for (src, c) in lib.sources() {
        for d in &c.displays {
            v.push(Item { value: Entry::new(src, d.name.clone()), label: d.name.clone(), short: d.name.clone(), path: vec![lib.group_name(src)] });
        }
    }
    v
}

/// Output targets: config / Displays / name (through the view), and config / family /
/// name for colour spaces that are not a display's (linear, log, textures…).
fn outputs(lib: &Library, l: crate::i18n::Lang) -> Vec<Item<OutputTarget>> {
    let mut v = Vec::new();
    for (src, c) in lib.sources() {
        for d in &c.displays {
            let e = Entry::new(src, d.name.clone());
            let path = vec![lib.group_name(src), l.t("ocio_displays").to_string(), d.name.clone()];
            // The window's view, or one of the display's own (AgX, Filmic, ACES…).
            v.push(Item { value: OutputTarget::Display(e.clone()), label: d.name.clone(), short: l.t("ocio_same_view").to_string(), path: path.clone() });
            for view in &d.views {
                v.push(Item { value: OutputTarget::DisplayView(e.clone(), view.name.clone()), label: view.name.clone(), short: view.name.clone(), path: path.clone() });
            }
        }
        for sp in c.spaces.iter().filter(|s| !s.display && !s.data) {
            let mut path = vec![lib.group_name(src)];
            path.extend(sp.family.split('/').map(str::trim).filter(|f| !f.is_empty()).map(str::to_string));
            v.push(Item { value: OutputTarget::Space(Entry::new(src, sp.name.clone())), label: sp.name.clone(), short: sp.name.clone(), path });
        }
    }
    v
}

/// Every view of every display: config / display / view; the value is the pair.
fn all_views(lib: &Library) -> Vec<Item<(Entry, String)>> {
    let mut v = Vec::new();
    for (src, c) in lib.sources() {
        for d in &c.displays {
            let e = Entry::new(src, d.name.clone());
            for view in &d.views {
                v.push(Item { value: (e.clone(), view.name.clone()), label: format!("{} · {}", view.name, e.label()), short: view.name.clone(), path: vec![lib.group_name(src), d.name.clone()] });
            }
        }
    }
    v
}

fn output_label(t: &OutputTarget, l: crate::i18n::Lang) -> String {
    match t {
        OutputTarget::Display(e) => format!("{} · {}", l.t("ocio_display"), e.label()),
        OutputTarget::DisplayView(e, v) => format!("{} · {} · {v}", l.t("ocio_display"), e.label()),
        OutputTarget::Space(e) => e.label(),
    }
}

/// Selected-text labels carry the config suffix ("ACEScg (maya)").
fn with_suffix<T>(items: Vec<Item<T>>, label: impl Fn(&T) -> String) -> Vec<Item<T>> {
    items.into_iter().map(|it| Item { label: label(&it.value), ..it }).collect()
}

fn media(p: &Player) -> Media {
    p.source.as_ref().map(|s| crate::color::media_of(&s.info)).unwrap_or(Media::SdrVideo)
}

pub fn color_contents(ui: &mut egui::Ui, p: &mut Player, s: &mut UiState, actions: &mut Vec<crate::ui::Action>) {
    let l = s.lang;
    ui.horizontal(|ui| {
        let mut on = s.ocio.enabled;
        if on_off_switch(ui, &mut on).on_hover_text(l.t("ocio_tip")).clicked() {
            s.ocio.enabled = on;
            s.settings.ocio_enabled = on;
            s.settings_dirty = true;
        }
        ui.label(RichText::new("OpenColorIO").strong().color(Color32::WHITE));
        ui.label(RichText::new(brp_ocio::version()).small().weak());
    });
    ui.add_space(4.0);
    if !s.ocio.enabled {
        ui.label(RichText::new(l.t("ocio_off")).weak());
        return;
    }
    let Some(lib) = s.ocio.library() else {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.label(RichText::new(l.t("ocio_loading")).weak());
        });
        ui.ctx().request_repaint();
        return;
    };
    // A config file of one's own, besides OCIO's, Maya's and Blender's.
    ui.horizontal(|ui| {
        ui.add_sized(Vec2::new(LABEL_W, 20.0), egui::Label::new(RichText::new(l.t("ocio_custom")).color(Color32::from_gray(185)))).on_hover_text(l.t("ocio_custom_tip"));
        match s.settings.ocio_custom.clone() {
            Some(path) => {
                let name = lib.custom_name.clone().unwrap_or_else(|| std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
                ui.label(RichText::new(name).color(Color32::from_gray(225))).on_hover_text(&path);
                if ui.small_button(l.t("ocio_remove")).clicked() {
                    s.settings.ocio_custom = None;
                    s.settings_dirty = true;
                    s.ocio.set_custom(None);
                }
            }
            None => {
                ui.label(RichText::new(format!("— {}", l.t("ocio_none"))).weak());
            }
        }
        if ui.small_button(l.t("ocio_load")).clicked() {
            actions.push(crate::ui::Action::OcioLoadConfig);
        }
    });
    if let Some(e) = &lib.custom_error {
        ui.label(RichText::new(e).small().color(Color32::from_rgb(0xf0, 0x7a, 0x6a)));
    }
    ui.add_space(4.0);
    let m = media(p);
    let screen = if s.screen_hdr { Screen::Hdr } else { Screen::Sdr };
    let def = |o: &Option<String>| o.as_deref().and_then(Entry::from_id).map(|e| e.label());
    let none = format!("— {}", l.t("ocio_none"));
    let r = s.ocio.resolve(&lib, &s.settings, m, screen);
    let auto_in = auto::input(&lib, m).map(|e| e.label()).unwrap_or_default();
    let auto_disp = auto::display(&lib, screen);
    let shown_disp = r.as_ref().map(|r| r.display.clone()).or(auto_disp.clone());
    let auto_view = shown_disp.as_ref().and_then(|d| auto::view(&lib, d, m)).unwrap_or_default();
    let view_items = all_views(&lib);
    let space_items = with_suffix(spaces(&lib), |e| e.label());
    let display_items = with_suffix(displays(&lib), |e| e.label());
    let output_items = with_suffix(outputs(&lib, l), |t| output_label(t, l));

    let mut changed = false;
    let def_in = def(&s.settings.ocio_input).unwrap_or(none.clone());
    let def_disp = def(&s.settings.ocio_display).unwrap_or(none.clone());
    let def_view = s.settings.ocio_view.clone().unwrap_or(none.clone());
    let def_out = s.settings.ocio_output.as_deref().and_then(crate::color::output_from_id).map(|t| output_label(&t, l)).unwrap_or(none.clone());
    let auto_disp_label = auto_disp.map(|e| e.label()).unwrap_or_default();
    let open = &mut s.ocio_open;
    ui.spacing_mut().item_spacing.y = 6.0;
    changed |= pick_tree(ui, open, "in", l.t("ocio_input"), l.t("ocio_input_tip"), &mut s.ocio.input, &auto_in, &def_in, &space_items, l);
    changed |= pick_tree(ui, open, "disp", l.t("ocio_display"), l.t("ocio_display_tip"), &mut s.ocio.display, &auto_disp_label, &def_disp, &display_items, l);
    // A view is picked with its display (AgX under Blender › sRGB sets both).
    let mut vpick: Pick<(Entry, String)> = match (&s.ocio.view, &r) {
        (Pick::Manual(v), Some(r)) => Pick::Manual((r.display.clone(), v.clone())),
        (Pick::Default, _) => Pick::Default,
        _ => Pick::Auto,
    };
    if pick_tree(ui, open, "view", l.t("ocio_view"), l.t("ocio_view_tip"), &mut vpick, &auto_view, &def_view, &view_items, l) {
        changed = true;
        match vpick {
            Pick::Manual((d, v)) => {
                s.ocio.display = Pick::Manual(d);
                s.ocio.view = Pick::Manual(v);
            }
            Pick::Default => s.ocio.view = Pick::Default,
            Pick::Auto => s.ocio.view = Pick::Auto,
        }
    }
    changed |= pick_tree(ui, open, "out", l.t("ocio_output"), l.t("ocio_output_tip"), &mut s.ocio.output, l.t("ocio_by_format"), &def_out, &output_items, l);
    if changed {
        ui.ctx().request_repaint();
    }
    ui.add_space(6.0);
    match (&r, &s.ocio.error) {
        (_, Some(e)) => {
            ui.label(RichText::new(e).small().color(Color32::from_rgb(0xf0, 0x7a, 0x6a)));
        }
        (Some(r), None) => {
            ui.label(RichText::new(format!("{}  ->  {} / {}", r.input.label(), r.display.label(), r.view)).small().weak());
        }
        (None, None) => {}
    }
    if s.ocio.busy() {
        ui.ctx().request_repaint();
    }
}

/// Settings: the defaults the panel's "(Default)" uses.
pub fn settings_section(ui: &mut egui::Ui, s: &mut UiState) {
    let l = s.lang;
    section(ui, l.t("sec_color"));
    ui.label(RichText::new(l.t("ocio_defaults_tip")).small().weak());
    let Some(lib) = s.ocio.library() else {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.label(RichText::new(l.t("ocio_loading")).weak());
        });
        ui.ctx().request_repaint();
        return;
    };
    let none = format!("— {} ({})", l.t("ocio_none"), l.t("ocio_auto"));
    let ids = |v: Vec<Item<Entry>>| -> Vec<Item<String>> { v.into_iter().map(|it| Item { label: it.value.label(), value: it.value.id(), short: it.short, path: it.path }).collect() };
    let space_items = ids(spaces(&lib));
    let display_items = ids(displays(&lib));
    // Views under their config and display; picking one sets the default display too
    // ("<display id>|<view>").
    let view_items: Vec<Item<String>> = all_views(&lib).into_iter().map(|it| Item { value: format!("{}|{}", it.value.0.id(), it.value.1), label: it.label, short: it.short, path: it.path }).collect();
    let output_items: Vec<Item<String>> = outputs(&lib, l).into_iter().map(|it| Item { label: output_label(&it.value, l), value: crate::color::output_id(&it.value), short: it.short, path: it.path }).collect();
    let mut dirty = false;
    let open = &mut s.ocio_open;
    let st = &mut s.settings;
    dirty |= default_tree(ui, open, "def_in", l.t("ocio_input"), &mut st.ocio_input, &none, &space_items);
    dirty |= default_tree(ui, open, "def_disp", l.t("ocio_display"), &mut st.ocio_display, &none, &display_items);
    let mut pair = match (&st.ocio_display, &st.ocio_view) {
        (Some(d), Some(v)) => Some(format!("{d}|{v}")),
        _ => None,
    };
    if default_tree(ui, open, "def_view", l.t("ocio_view"), &mut pair, &none, &view_items) {
        dirty = true;
        match pair.as_deref().and_then(|p| p.rsplit_once('|')) {
            Some((d, v)) => {
                st.ocio_display = Some(d.to_string());
                st.ocio_view = Some(v.to_string());
            }
            None => st.ocio_view = None,
        }
    }
    dirty |= default_tree(ui, open, "def_out", l.t("ocio_output"), &mut st.ocio_output, &none, &output_items);
    if dirty {
        s.settings_dirty = true;
    }
}
