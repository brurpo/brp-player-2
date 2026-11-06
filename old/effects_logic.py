# effects_logic.py
import os
import re
import json
import shutil
import tempfile
import uuid
import sys
import subprocess
import copy
from PyQt6.QtWidgets import (QSlider, QLabel, QCheckBox, QHBoxLayout, QWidget,
                             QFileDialog, QFormLayout, QListWidgetItem, QMessageBox, QSpinBox, QComboBox, QLineEdit,
                             QPushButton, QColorDialog, QGroupBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette, QDoubleValidator, QIntValidator, QKeySequence, QAction
import traceback
from ui_effects_dialog import CollapsibleBox

# --- CENTRAL CONFIGURATION FOR BUILT-IN FILTERS (Translated) ---
BUILTIN_FILTERS_CONFIG = {
    "atadenoise": {
        "display": "Builtin-Temporal-Denoiser",
        "params": {
            "0a": {"label": "Threshold A", "desc": "Threshold A for luma plane. (cat: Luma)", "type": "float", "min": 0.0, "max": 0.3, "default": 0.02},
            "0b": {"label": "Threshold B", "desc": "Threshold B for luma plane.", "type": "float", "min": 0.0, "max": 5.0, "default": 0.04},
            "1a": {"label": "Threshold A", "desc": "Threshold A for chroma planes. (cat: Chroma)", "type": "float", "min": 0.0, "max": 0.3, "default": 0.02},
            "1b": {"label": "Threshold B", "desc": "Threshold B for chroma planes.", "type": "float", "min": 0.0, "max": 5.0, "default": 0.04},
            "s":  {"label": "Frame Count", "desc": "Number of frames to average (must be odd). (cat: General)", "type": "int_spinbox", "min": 5, "max": 129, "step": 2, "default": 9}
        }
    },
    "hqdn3d": {
        "display": "Builtin-Temporal-Denoiser3D",
        "params": {
            "luma_spatial":   {"label": "Luma Spatial Strength", "desc": "Spatial luma strength.", "type": "float", "min": 0.0, "max": 32.0, "default": 4.0},
            "chroma_spatial": {"label": "Chroma Spatial Strength", "desc": "Spatial chroma strength.", "type": "float", "min": 0.0, "max": 32.0, "default": 3.0},
            "luma_tmp":       {"label": "Temporal Strength (Luma)", "desc": "Temporal luma strength.", "type": "float", "min": 0.0, "max": 32.0, "default": 6.0},
            "chroma_tmp":     {"label": "Temporal Strength (Chroma)", "desc": "Temporal chroma strength.", "type": "float", "min": 0.0, "max": 32.0, "default": 4.5}
        }
    },
    "deband": {
        "display": "Builtin-Deband",
        "is_libplacebo_option": True,
        "params": {
            "deband_iterations": {"label": "Iterations", "desc": "Number of deband iterations.", "type": "int", "min": 1, "max": 16, "default": 1},
            "deband_threshold":  {"label": "Threshold", "desc": "Deband threshold for all components.", "type": "float", "min": 0.0, "max": 64.0, "default": 4.0},
            "deband_radius":     {"label": "Radius", "desc": "Deband radius.", "type": "float", "min": 0.0, "max": 64.0, "default": 16.0},
            "deband_grain":      {"label": "Grain", "desc": "Amount of grain to add.", "type": "float", "min": 0.0, "max": 64.0, "default": 6.0}
        }
    },
    "fftdnoiz": {
        "display": "Builtin-FFT-Denoiser",
        "params": {
            "sigma":   {"label": "Strength (Sigma)", "desc": "Denoising strength.", "type": "float", "min": 0.0, "max": 30.0, "default": 1.0},
            "amount":  {"label": "Amount", "desc": "Amount of noise to reduce.", "type": "float", "min": 0.0, "max": 1.0, "default": 1.0},
            "block":   {"label": "Block Size", "desc": "Size of the block in pixels.", "type": "int_spinbox", "min": 8, "max": 256, "step": 8, "default": 32},
            "overlap": {"label": "Overlap", "desc": "Block overlap percentage.", "type": "float", "min": 0.2, "max": 0.8, "default": 0.5},
            "prev":    {"label": "Use Previous Frame", "desc": "Uses the previous frame for temporal denoising.", "type": "uint", "default": 1},
            "next":    {"label": "Use Next Frame", "desc": "Uses the next frame for temporal denoising.", "type": "uint", "default": 1},
            "method":  {"label": "Denoise Method", "desc": "Denoising method to use.", "type": "choice", "options": ["wiener", "hard"], "default": "wiener"}
        }
    },
    "deshake": {
        "display": "Builtin-Deshake",
        "params": {
            "rx":        {"label": "Max Horizontal Mov.", "desc": "Maximum movement in pixels on the X-axis.", "type": "int", "min": 0, "max": 64, "default": 16},
            "ry":        {"label": "Max Vertical Mov.", "desc": "Maximum movement in pixels on the Y-axis.", "type": "int", "min": 0, "max": 64, "default": 16},
            "blocksize": {"label": "Block Size", "desc": "Block size for motion search.", "type": "int_spinbox", "min": 4, "max": 128, "step": 4, "default": 8},
            "contrast":  {"label": "Contrast Threshold", "desc": "Contrast threshold for blocks.", "type": "float", "min": 1.0, "max": 255.0, "default": 125.0},
            "edge":      {"label": "Edge Fill", "desc": "How to generate pixels to fill blank edges.", "type": "choice", "options": ["blank", "original", "clamp", "mirror"], "default": "mirror"}
        }
    },

}

class EffectsManager:
    UNDO_LIMIT = 50

    def __init__(self, main_app, dialog, active_shader_config):
        self.main_app = main_app
        self.player = main_app.player
        self.dialog = dialog
        self.current_shader_index = -1
        self.app_data_dir = self._get_app_data_dir()
        self.shader_dir = os.path.join(self.app_data_dir, "Shaders")
        os.makedirs(self.shader_dir, exist_ok=True)
        self.active_shaders = active_shader_config

        self.undo_stack = []
        self.redo_stack = []
        self._save_state(is_initial_state=True)

        self.connect_signals()
        self.setup_shortcuts()
        self.discover_shaders()
        self._update_shader_list_widget()

        # --- CORREÇÃO AQUI ---
        # Se a janela for aberta com efeitos já existentes, seleciona o primeiro
        # e mostra os seus parâmetros.
        if self.dialog.shader_list.count() > 0:
            if not self.dialog.shader_list.currentItem():
                self.dialog.shader_list.setCurrentRow(0)
            self.populate_effect_params_ui(self.dialog.shader_list.currentItem(), None)
        # --- FIM DA CORREÇÃO ---

        self.main_app.apply_all_effects()

    def _snapshot_state(self):
        snapshot = []
        for effect in self.active_shaders:
            effect_copy = {}
            for key, value in effect.items():
                if key == "params":
                    params_copy = {}
                    for p_name, p_data in value.items():
                        p_data_copy = p_data.copy()
                        p_data_copy.pop("widget", None)
                        params_copy[p_name] = p_data_copy
                    effect_copy[key] = params_copy
                elif key != "widget":
                    effect_copy[key] = copy.deepcopy(value)
            snapshot.append(effect_copy)
        return snapshot

    def _save_state(self, is_initial_state=False):
        if not is_initial_state and self.undo_stack:
            last_state = self.undo_stack[-1]
            current_state_snapshot = self._snapshot_state()
            if last_state == current_state_snapshot:
                return

        new_state = self._snapshot_state()
        self.undo_stack.append(new_state)

        if len(self.undo_stack) > self.UNDO_LIMIT:
            self.undo_stack.pop(0)

        if not is_initial_state:
            self.redo_stack.clear()

    def _apply_state(self, state):
        previous_row = self.dialog.shader_list.currentRow()

        self.active_shaders.clear()
        self.active_shaders.extend(copy.deepcopy(state))

        self._update_shader_list_widget()

        if 0 <= previous_row < len(self.active_shaders):
            self.dialog.shader_list.setCurrentRow(previous_row)
        elif len(self.active_shaders) > 0:
            self.dialog.shader_list.setCurrentRow(0)

        self.main_app.apply_all_effects()

        new_current_item = self.dialog.shader_list.currentItem()
        self.populate_effect_params_ui(new_current_item, None)

    def undo(self):
        if len(self.undo_stack) > 1:
            current_state = self.undo_stack.pop()
            self.redo_stack.append(current_state)

            previous_state = self.undo_stack[-1]
            self._apply_state(previous_state)

    def redo(self):
        if self.redo_stack:
            state_to_restore = self.redo_stack.pop()
            self.undo_stack.append(state_to_restore)
            self._apply_state(state_to_restore)

    def setup_shortcuts(self):
        undo_action = QAction(self.dialog)
        undo_action.setShortcut(QKeySequence("Ctrl+Z"))
        undo_action.triggered.connect(self.undo)
        self.dialog.addAction(undo_action)

        redo_action = QAction(self.dialog)
        redo_action.setShortcut(QKeySequence("Ctrl+Y"))
        redo_action.triggered.connect(self.redo)
        self.dialog.addAction(redo_action)

    def _get_app_data_dir(self):
        path = os.path.join(os.getenv('LOCALAPPDATA'), "BRP_Player")
        os.makedirs(path, exist_ok=True)
        return path

    def _get_savable_config(self):
        savable_list = []
        for effect_data in self.active_shaders:
            clean_data = effect_data.copy()
            if not clean_data.get("is_builtin"):
                clean_data["path"] = clean_data.get("original_path")
                clean_data.pop("runtime_path", None)
                clean_data.pop("original_path", None)
                if not clean_data.get("path") or not os.path.exists(clean_data["path"]):
                    continue
            if "params" in clean_data:
                new_params = {}
                for name, param in clean_data["params"].items():
                    param_copy = param.copy()
                    param_copy.pop("widget", None)
                    if param_copy.get("type") == "color" and "r_param_name" in param_copy:
                        r_name, g_name, b_name = param_copy["r_param_name"], param_copy["g_param_name"], param_copy["b_param_name"]
                        r_val, g_val, b_val = param_copy["r_float_value"], param_copy["g_float_value"], param_copy["b_float_value"]
                        base_param_data = {k: v for k, v in param_copy.items() if k not in ['value', 'r_param_name', 'g_param_name', 'b_param_name', 'r_float_value', 'g_float_value', 'b_float_value']}
                        base_param_data['type'] = 'float'
                        new_params[r_name] = {**base_param_data, "value": r_val}
                        new_params[g_name] = {**base_param_data, "value": g_val}
                        new_params[b_name] = {**base_param_data, "value": b_val}
                    else:
                        new_params[name] = param_copy
                clean_data["params"] = new_params
            savable_list.append(clean_data)
        return savable_list

    def connect_signals(self):
        self.dialog.add_shader_button.clicked.connect(self.add_effect_from_combo)
        self.dialog.remove_shader_button.clicked.connect(self.remove_selected_effect)
        self.dialog.shader_list.currentItemChanged.connect(self.populate_effect_params_ui)
        self.dialog.shader_list.model().rowsMoved.connect(self.on_effect_reordered)
        self.dialog.load_preset_button.clicked.connect(self.load_preset)
        self.dialog.save_preset_button.clicked.connect(self.save_preset)
        self.dialog.clear_button.clicked.connect(self.clear_effects)
        self.dialog.delete_requested.connect(self.remove_selected_effect)
        self.dialog.open_shader_folder_button.clicked.connect(self.open_shader_folder)
        self.dialog.files_dropped.connect(self.handle_files_dropped)

    def open_shader_folder(self):
        path = self.shader_dir
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])

    def handle_files_dropped(self, file_paths):
        refresh_needed = False
        for path in file_paths:
            if not path.lower().endswith('.glsl'):
                continue
            filename = os.path.basename(path)
            dest_path = os.path.join(self.shader_dir, filename)
            try:
                if os.path.exists(dest_path):
                    reply = QMessageBox.question(self.dialog, 'Overwrite Shader',
                                                 f"The shader '{filename}' already exists. Overwrite?",
                                                 QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                                 QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.No:
                        continue
                shutil.copy2(path, dest_path)
                refresh_needed = True
            except Exception as e:
                QMessageBox.critical(self.dialog, "Copy Error", f"Could not copy shader '{filename}'.\n\nError: {e}")
        if refresh_needed:
            self.discover_shaders()
            QMessageBox.information(self.dialog, "Success", "Shaders imported.")
            self._save_state()

    def discover_shaders(self):
        self.dialog.shader_combo.clear()
        found_shaders = set()
        if self.shader_dir and os.path.exists(self.shader_dir):
            for file in os.listdir(self.shader_dir):
                if file.lower().endswith('.glsl'):
                    found_shaders.add(os.path.join(self.shader_dir, file))
        base_path = self.main_app.base_path
        if os.path.exists(base_path):
            for file in os.listdir(base_path):
                if file.lower().endswith('.glsl'):
                    found_shaders.add(os.path.join(base_path, file))
        for name, config in BUILTIN_FILTERS_CONFIG.items():
            self.dialog.shader_combo.addItem(config["display"], f"builtin:{name}")
        if found_shaders:
            self.dialog.shader_combo.insertSeparator(self.dialog.shader_combo.count())
        for shader_path in sorted(list(found_shaders)):
            self.dialog.shader_combo.addItem(os.path.basename(shader_path), shader_path)
        self.dialog.shader_combo.insertSeparator(self.dialog.shader_combo.count())
        self.dialog.shader_combo.addItem("Load Custom Shader...", "custom")

    def _reconstruct_color_params_from_save(self, config_list):
        color_re = re.compile(r'^(.*)(R|G|B)$', re.IGNORECASE)
        for effect_data in config_list:
            if effect_data.get("is_builtin"):
                continue
            raw_params = effect_data.get("params", {})
            if not raw_params:
                continue
            new_params = {}
            param_names_to_process = list(raw_params.keys())
            processed_components = set()
            for name in param_names_to_process:
                if name in processed_components:
                    continue
                param_copy = raw_params[name].copy()
                match = color_re.match(name)
                if match and param_copy.get("type") == "float":
                    prefix = match.group(1)
                    r_name, g_name, b_name = f"{prefix}R", f"{prefix}G", f"{prefix}B"
                    if r_name in raw_params and g_name in raw_params and b_name in raw_params:
                        r_data, g_data, b_data = raw_params[r_name], raw_params[g_name], raw_params[b_name]
                        try:
                            r_val_float = float(r_data.get("value", 0.0))
                            g_val_float = float(g_data.get("value", 0.0))
                            b_val_float = float(b_data.get("value", 0.0))
                            r_val_int, g_val_int, b_val_int = int(r_val_float * 255), int(g_val_float * 255), int(b_val_float * 255)
                            hex_color = f"#{r_val_int:02x}{g_val_int:02x}{b_val_int:02x}"
                            color_name = prefix + "Color"
                            base_label = r_data.get("label", color_name)
                            clean_label = re.sub(r'(\s?ColorR|\s?R)$', '', base_label, flags=re.IGNORECASE).strip()
                            new_params[color_name] = { "label": clean_label, "desc": r_data.get("desc", ""), "type": "color", "default": hex_color, "value": hex_color, "r_param_name": r_name, "g_param_name": g_name, "b_param_name": b_name, "r_float_value": r_val_float, "g_float_value": g_val_float, "b_float_value": b_val_float, "category": r_data.get("category") }
                            processed_components.update([r_name, g_name, b_name])
                        except (ValueError, TypeError) as e:
                            new_params[name] = param_copy
                    else:
                        new_params[name] = param_copy
                else:
                    new_params[name] = param_copy
            effect_data["params"] = new_params

    def load_preset(self):
        filepath, _ = QFileDialog.getOpenFileName(self.dialog, "Load Effects Preset", self.shader_dir, "JSON Files (*.json)")
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_from_file = json.load(f)
            self._reconstruct_color_params_from_save(config_from_file)
            self.active_shaders.clear()
            self.active_shaders.extend(config_from_file)
            self._update_shader_list_widget()
            if self.dialog.shader_list.count() > 0:
                self.dialog.shader_list.setCurrentRow(0)

            self.populate_effect_params_ui(self.dialog.shader_list.currentItem(), None)

            self.main_app.apply_all_effects()
            self._save_state()
        except (json.JSONDecodeError, IOError, KeyError) as e:
            QMessageBox.critical(self.dialog, "Load Error", f"Could not load or parse the preset file:\n{e}")

    def save_preset(self):
        filepath, _ = QFileDialog.getSaveFileName(self.dialog, "Save Effects Preset", self.shader_dir, "JSON Files (*.json)")
        if not filepath: return
        if not filepath.lower().endswith('.json'):
            filepath += '.json'
        savable_config = self._get_savable_config()
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(savable_config, f, indent=4)
        except IOError as e:
            QMessageBox.critical(self.dialog, "Save Error", f"Could not save the preset file:\n{e}")

    def clear_effects(self):
        reply = QMessageBox.question(self.dialog, 'Clear Filters', "Are you sure you want to remove all filters from the list?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if not self.active_shaders: return
            for effect_data in self.active_shaders:
                self._cleanup_temp_file(effect_data)
            self.active_shaders.clear()
            self._update_shader_list_widget()
            self.main_app.apply_all_effects()
            self.populate_effect_params_ui(None, None)
            self._save_state()

    def on_effect_reordered(self, parent, start, end, dest, row):
        moved_item = self.active_shaders.pop(start)
        self.active_shaders.insert(row, moved_item)
        self.main_app.apply_all_effects()
        self.dialog.shader_list.setCurrentRow(row)
        self._save_state()

    def add_effect_from_combo(self):
        path_or_id = self.dialog.shader_combo.currentData()
        effect_data = None
        if path_or_id.startswith("builtin:"):
            filter_name = path_or_id.split(":")[1]
            if any(s.get("path") == path_or_id for s in self.active_shaders):
                QMessageBox.warning(self.dialog, "Duplicate Filter", f"The filter '{BUILTIN_FILTERS_CONFIG[filter_name]['display']}' is already in the list.")
                return
            config = BUILTIN_FILTERS_CONFIG[filter_name]
            params = {p_name: {**p_data, "value": p_data["default"]} for p_name, p_data in config["params"].items()}
            effect_data = {"path": path_or_id, "display_name": config["display"], "params": params, "enabled": True, "is_builtin": True, "is_libplacebo_option": config.get("is_libplacebo_option", False)}
        elif path_or_id == "custom":
            filepath, _ = QFileDialog.getOpenFileName(self.dialog, "Select GLSL Shader", "", "GLSL Shaders (*.glsl)")
            if not filepath: return
            try:
                dest_path = os.path.join(self.shader_dir, os.path.basename(filepath))
                shutil.copy2(filepath, dest_path)
                path_or_id = dest_path
                self.discover_shaders()
                index = self.dialog.shader_combo.findData(path_or_id)
                if index != -1: self.dialog.shader_combo.setCurrentIndex(index)
            except Exception as e:
                QMessageBox.warning(self.dialog, "Error", f"Could not copy the shader to the Shaders folder.\n\n{e}")
                return
        if path_or_id and not path_or_id.startswith("builtin:") and path_or_id != "custom":
            original_shader_path = path_or_id
            temp_dir = self.main_app.get_temp_dir_for_shaders()
            unique_suffix = uuid.uuid4().hex[:8]
            base, ext = os.path.splitext(os.path.basename(original_shader_path))
            temp_filename = f"{base}_{unique_suffix}{ext}"
            runtime_path = os.path.join(temp_dir, temp_filename)
            shutil.copy2(original_shader_path, runtime_path)
            base_display_name = os.path.basename(original_shader_path)
            final_display_name = base_display_name
            count = 2
            while any(s.get("display_name") == final_display_name for s in self.active_shaders):
                final_display_name = f"{base_display_name} ({count})"
                count += 1
            params = self._parse_shader_params(original_shader_path)
            effect_data = {"path": original_shader_path, "original_path": original_shader_path, "runtime_path": runtime_path, "display_name": final_display_name, "params": params, "enabled": True, "is_builtin": False}
        if effect_data:
            self.active_shaders.append(effect_data)
            self._update_shader_list_widget()
            self.dialog.shader_list.setCurrentRow(len(self.active_shaders) - 1)

            self.populate_effect_params_ui(self.dialog.shader_list.currentItem(), None)

            self.main_app.apply_all_effects()
            self._save_state()

    def _parse_shader_params(self, shader_path):
        final_params = {}
        try:
            with open(shader_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            raw_params_ordered = []
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("//!PARAM"):
                    param_details = {"name": line.split(" ")[1]}
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if next_line.startswith("//!"):
                            tag_parts = next_line[3:].split(" ", 1)
                            if len(tag_parts) == 2:
                                param_details[tag_parts[0].lower()] = tag_parts[1]
                        elif next_line and not next_line.startswith("//"):
                            param_details["default"] = next_line.strip()
                            break
                        j += 1
                    raw_params_ordered.append(param_details)
                    i = j
                i += 1
            current_category = None
            for param_details in raw_params_ordered:
                full_desc = param_details.get("desc", "")
                cat_match = re.search(r'\(cat:\s*([^)]+)\)', full_desc)
                if cat_match:
                    current_category = cat_match.group(1).strip()
                param_details["category"] = current_category
                param_details["tooltip"] = re.sub(r'\s*\([^)]*\)', '', full_desc).strip()
            color_re = re.compile(r'^(.*)(r|g|b)$', re.IGNORECASE)
            processed_components = set()
            for param_details in raw_params_ordered:
                name = param_details["name"]
                if name in processed_components: continue
                try:
                    param_type = param_details.get("type")
                    caster = float if param_type in ['float', 'DEFINE'] else int
                    full_param = { "label": param_details.get("label", name), "desc": param_details.get("tooltip", ""), "category": param_details.get("category"), "type": param_type, "min": caster(param_details.get("minimum", 0)), "max": caster(param_details.get("maximum", 1)), "default": caster(param_details.get("default", 0)), "value": caster(param_details.get("default", 0)) }
                except (ValueError, TypeError, KeyError) as e:
                    continue
                match = color_re.match(name)
                if match and full_param["type"] in ["float", "DEFINE"]:
                    prefix = match.group(1)
                    r_name, g_name, b_name = f"{prefix}R", f"{prefix}G", f"{prefix}B"
                    all_raw_names = [p["name"] for p in raw_params_ordered]
                    if r_name in all_raw_names and g_name in all_raw_names and b_name in all_raw_names:
                        r_val_float = float(next(p['default'] for p in raw_params_ordered if p['name'] == r_name))
                        g_val_float = float(next(p['default'] for p in raw_params_ordered if p['name'] == g_name))
                        b_val_float = float(next(p['default'] for p in raw_params_ordered if p['name'] == b_name))
                        hex_color = f"#{int(r_val_float*255):02x}{int(g_val_float*255):02x}{int(b_val_float*255):02x}"
                        base_label = full_param["label"]
                        clean_label = re.sub(r'(\s?ColorR|\s?R)$', '', base_label, flags=re.IGNORECASE).strip()
                        color_name = prefix + "Color"
                        final_params[color_name] = { "label": clean_label, "desc": full_param["desc"], "type": "color", "default": hex_color, "value": hex_color, "r_param_name": r_name, "g_param_name": g_name, "b_param_name": b_name, "r_float_value": r_val_float, "g_float_value": g_val_float, "b_float_value": b_val_float, "category": full_param["category"] }
                        processed_components.update([r_name, g_name, b_name])
                    else:
                        final_params[name] = full_param
                else:
                    final_params[name] = full_param
        except Exception as e:
            traceback.print_exc()
        return final_params

    def _update_shader_list_widget(self):
        try:
            self.dialog.shader_list.currentItemChanged.disconnect()
        except TypeError: pass
        current_row = self.dialog.shader_list.currentRow()
        self.dialog.shader_list.clear()
        for effect_data in self.active_shaders:
            item_widget = QWidget()
            layout = QHBoxLayout(item_widget)
            layout.setContentsMargins(5, 5, 5, 5)
            checkbox = QCheckBox()
            checkbox.setChecked(effect_data.get("enabled", True))
            checkbox.stateChanged.connect(lambda state, s=effect_data: self._toggle_effect(s, state))
            label = QLabel(effect_data.get("display_name", "Unknown Effect"))
            layout.addWidget(checkbox)
            layout.addWidget(label)
            layout.addStretch()
            list_item = QListWidgetItem()
            list_item.setSizeHint(item_widget.sizeHint())
            self.dialog.shader_list.addItem(list_item)
            self.dialog.shader_list.setItemWidget(list_item, item_widget)
        if 0 <= current_row < self.dialog.shader_list.count():
            self.dialog.shader_list.setCurrentRow(current_row)
        else:
            if self.dialog.shader_list.count() > 0:
                self.dialog.shader_list.setCurrentRow(self.dialog.shader_list.count() - 1)
        self.dialog.shader_list.currentItemChanged.connect(self.populate_effect_params_ui)

    def _cleanup_temp_file(self, effect_data):
        if effect_data.get("is_builtin"): return
        try:
            runtime_path = effect_data.get("runtime_path")
            if runtime_path and os.path.exists(runtime_path):
                os.remove(runtime_path)
        except OSError as e:
            print(f"Could not remove temporary shader {runtime_path}: {e}")

    def _toggle_effect(self, effect_data, state):
        effect_data["enabled"] = (state == 2)
        self.main_app.apply_all_effects()
        self._save_state()

    def remove_selected_effect(self):
        selected_items = self.dialog.shader_list.selectedItems()
        if not selected_items:
            return
        row = self.dialog.shader_list.row(selected_items[0])
        try:
            self.dialog.shader_list.currentItemChanged.disconnect(self.populate_effect_params_ui)
        except TypeError:
            pass
        effect_to_remove = self.active_shaders.pop(row)
        self._cleanup_temp_file(effect_to_remove)
        self.dialog.shader_list.takeItem(row)
        self.main_app.apply_all_effects()
        self._save_state()
        self.dialog.shader_list.currentItemChanged.connect(self.populate_effect_params_ui)
        new_current_item = self.dialog.shader_list.currentItem()
        self.populate_effect_params_ui(new_current_item, None)

    def populate_effect_params_ui(self, current_item, previous_item):
        for i in reversed(range(self.dialog.params_layout.count())):
            widget_to_remove = self.dialog.params_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
        self.dialog.recreate_params_container()
        if not current_item:
            self.dialog.no_params_label.show()
            self.current_shader_index = -1
            return
        self.dialog.no_params_label.hide()
        row = self.dialog.shader_list.row(current_item)
        if row < 0 or row >= len(self.active_shaders): return
        self.current_shader_index = row
        effect_data = self.active_shaders[row]
        params = effect_data.get("params", {})
        if not params:
            self.dialog.no_params_label.show()
            return
        category_boxes = {}
        uncategorized_layout = self.dialog.params_widget_layout
        for name, param in params.items():
            param_category = param.get("category")
            target_layout_or_box = uncategorized_layout
            if param_category:
                if param_category not in category_boxes:
                    box = CollapsibleBox(title=param_category)
                    self.dialog.params_layout.addWidget(box)
                    category_boxes[param_category] = box
                target_layout_or_box = category_boxes[param_category]
            self.create_param_widget(name, param, target_layout_or_box, effect_data)

    def create_param_widget(self, name, param, layout_or_box, effect_data):
        param_type = param["type"]
        row_label = QLabel(param["label"])
        row_label.setToolTip(param["desc"])
        add_row_method = layout_or_box.addRow if isinstance(layout_or_box, (QFormLayout, CollapsibleBox)) else layout_or_box.layout().addRow
        if param_type == "color":
            color_button = QPushButton()
            color_button.setFixedSize(100, 28)
            param['widget'] = color_button
            current_color = str(param.get("value", param["default"]))
            color_button.setStyleSheet(f"background-color: {current_color}; border-radius: 5px;")
            color_button.clicked.connect(lambda ch, s_data=effect_data, p_name=name: self._open_color_picker(s_data, p_name))
            add_row_method(row_label, color_button)
        elif param_type in ["float", "int", "DEFINE"]:
            param_widget = QWidget()
            param_layout = QHBoxLayout(param_widget)
            param_layout.setContentsMargins(0,0,0,0)
            param_layout.setSpacing(10)
            slider = QSlider(Qt.Orientation.Horizontal)
            line_edit = QLineEdit()
            line_edit.setMaximumWidth(70)
            current_value = param.get("value", param["default"])
            if param_type in ["float", "DEFINE"]:
                slider.setRange(0, 1000)
                slider_value = int(((float(current_value) - param["min"]) / (param["max"] - param["min"])) * 1000) if (param["max"] - param["min"]) != 0 else 0
                line_edit.setValidator(QDoubleValidator(param["min"], param["max"], 3))
                line_edit.setText(f"{float(current_value):.3f}")
            else:
                slider.setRange(int(param["min"]), int(param["max"]))
                slider_value = int(current_value)
                line_edit.setValidator(QIntValidator(param["min"], param["max"]))
                line_edit.setText(str(current_value))
            slider.setValue(slider_value)
            slider.valueChanged.connect(lambda val, le=line_edit, s_data=effect_data, p_name=name: self._update_effect_param_from_slider(val, le, s_data, p_name))
            slider.sliderReleased.connect(self._save_state)
            line_edit.textChanged.connect(lambda text, sl=slider, s_data=effect_data, p_name=name: self._update_effect_param_from_line_edit_live(text, sl, s_data, p_name))
            line_edit.editingFinished.connect(lambda sl=slider, le=line_edit, s_data=effect_data, p_name=name: self._validate_and_save_param_from_line_edit(sl, le, s_data, p_name))
            param_layout.addWidget(slider)
            param_layout.addWidget(line_edit)
            add_row_method(row_label, param_widget)
        else:
            widget = None
            if param_type == "string":
                widget = QLineEdit()
                widget.setText(str(param.get("value", param["default"])))
                widget.editingFinished.connect(lambda le=widget, s_data=effect_data, p_name=name: self._update_effect_param_string(le, s_data, p_name))
            elif param_type == "choice":
                widget = QComboBox()
                widget.addItems(param["options"])
                widget.setCurrentText(param.get("value", param["default"]))
                widget.currentTextChanged.connect(lambda text, s_data=effect_data, p_name=name: self._update_effect_param_choice(text, s_data, p_name))
            elif param_type == "int_spinbox":
                widget = QSpinBox()
                widget.setRange(param["min"], param["max"])
                widget.setSingleStep(param.get("step", 1))
                widget.setValue(int(param.get("value", param["default"])))
                widget.valueChanged.connect(lambda val, s_data=effect_data, p_name=name: self._update_effect_param_spinbox(val, s_data, p_name))
            elif param_type == "uint":
                widget = QCheckBox()
                widget.setChecked(bool(param.get("value", param["default"])))
                widget.stateChanged.connect(lambda state, s_data=effect_data, p_name=name: self._update_effect_param_checkbox(state, s_data, p_name))
            if widget:
                add_row_method(row_label, widget)

    def _update_effect_param_from_slider(self, slider_value, line_edit, effect_data, param_name):
        param_data = effect_data["params"][param_name]
        new_value = 0
        if param_data["type"] in ["float", "DEFINE"]:
            if (param_data["max"] - param_data["min"]) == 0: return
            new_value = param_data["min"] + (slider_value / 1000.0) * (param_data["max"] - param_data["min"])
            line_edit.blockSignals(True)
            line_edit.setText(f"{new_value:.3f}")
            line_edit.blockSignals(False)
        else:
            new_value = slider_value
            line_edit.blockSignals(True)
            line_edit.setText(f"{new_value}")
            line_edit.blockSignals(False)
        effect_data["params"][param_name]["value"] = new_value
        self.main_app.apply_all_effects()

    def _update_effect_param_from_line_edit_live(self, text, slider, effect_data, param_name):
        param_data = effect_data["params"][param_name]
        try:
            text_value_str = text.replace(',', '.')
            new_value = float(text_value_str) if param_data["type"] in ["float", "DEFINE"] else int(text_value_str)
        except ValueError:
            return
        new_value = max(param_data["min"], min(new_value, param_data["max"]))
        effect_data["params"][param_name]["value"] = new_value
        slider.blockSignals(True)
        if param_data["type"] in ["float", "DEFINE"]:
            if (param_data["max"] - param_data["min"]) == 0: return
            slider_pos = int(((new_value - param_data["min"]) / (param_data["max"] - param_data["min"])) * 1000)
            slider.setValue(slider_pos)
        else:
            slider.setValue(int(new_value))
        slider.blockSignals(False)
        self.main_app.apply_all_effects()

    def _validate_and_save_param_from_line_edit(self, slider, line_edit, effect_data, param_name):
        param_data = effect_data["params"][param_name]
        current_value = param_data["value"]
        line_edit.blockSignals(True)
        if param_data["type"] in ["float", "DEFINE"]:
            line_edit.setText(f"{float(current_value):.3f}")
        else:
            line_edit.setText(str(int(current_value)))
        line_edit.blockSignals(False)
        self._save_state()

    def _open_color_picker(self, effect_data, param_name):
        param = effect_data["params"][param_name]
        initial_color = QColor(param.get("value", param["default"]))
        color_dialog = QColorDialog(self.dialog)
        color_dialog.setCurrentColor(initial_color)
        color_dialog.currentColorChanged.connect(lambda color: self._update_shader_color_live(color, effect_data, param_name))
        color_dialog.accepted.connect(lambda: self._finalize_shader_color_update(color_dialog.currentColor(), effect_data, param_name))
        color_dialog.setOption(QColorDialog.ColorDialogOption.NoButtons, True)
        color_dialog.finished.connect(self._save_state)
        color_dialog.show()

    def _update_shader_color_live(self, color, effect_data, param_name):
        param = effect_data["params"][param_name]
        new_color_hex = color.name()
        param["value"] = new_color_hex
        if 'widget' in param and param['widget']:
            param['widget'].setStyleSheet(f"background-color: {new_color_hex}; border-radius: 5px;")
        if "r_param_name" in param:
            param['r_float_value'] = color.red() / 255.0
            param['g_float_value'] = color.green() / 255.0
            param['b_float_value'] = color.blue() / 255.0
        self.main_app.apply_all_effects()

    def _finalize_shader_color_update(self, color, effect_data, param_name):
        pass

    def _update_effect_param_string(self, line_edit, effect_data, param_name):
        effect_data["params"][param_name]["value"] = line_edit.text()
        self.main_app.apply_all_effects()
        self._save_state()

    def _update_effect_param_choice(self, text, effect_data, param_name):
        effect_data["params"][param_name]["value"] = text
        self.main_app.apply_all_effects()
        self._save_state()

    def _update_effect_param_spinbox(self, value, effect_data, param_name):
        effect_data["params"][param_name]["value"] = value
        self.main_app.apply_all_effects()
        self._save_state()

    def _update_effect_param_checkbox(self, state, effect_data, param_name):
        new_value = 1 if state == 2 else 0
        effect_data["params"][param_name]["value"] = new_value
        self.main_app.apply_all_effects()
        self._save_state()
