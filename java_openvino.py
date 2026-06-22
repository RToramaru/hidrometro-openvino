import cv2
import numpy as np
import openvino as ov
import math
import time
import threading
import os
import sys
from tkinter import filedialog, messagebox
import customtkinter as ctkinter
from PIL import Image

# =========================================================
# LÓGICA DE DETECÇÃO DE CAMINHOS PARA EXECUTÁVEL ÚNICO (.ONEFILE)
# =========================================================
if getattr(sys, 'frozen', False):
    # O PyInstaller salva o caminho da pasta temporária em sys._MEIPASS
    DIRETORIO_BASE = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
    # O cache do OpenVINO vai para AppData/Local do utilizador logado
    DIRETORIO_CACHE = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'ov_cache_dir')
else:
    # Se estiver a rodar como script Python normal
    DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
    DIRETORIO_CACHE = os.path.join(DIRETORIO_BASE, 'ov_cache_dir')

# Garante que a pasta de cache do OpenVINO existe silenciosamente no sistema
os.makedirs(DIRETORIO_CACHE, exist_ok=True)

# Definição dinâmica do caminho dos modelos baseada no diretório de execução
PATH_PESOS = os.path.join(DIRETORIO_BASE, "pesos")

MODELOS = {
    "visor": {"xml": os.path.join(PATH_PESOS, "visor_openvino.xml"), "onnx": os.path.join(PATH_PESOS, "visor.onnx")},
    "yolo": {"xml": os.path.join(PATH_PESOS, "ponteiro_openvino.xml"),
             "onnx": os.path.join(PATH_PESOS, "ponteiro.onnx")},
    "ocr": {"xml": os.path.join(PATH_PESOS, "ocr_openvino.xml"), "onnx": os.path.join(PATH_PESOS, "ocr.onnx")},
    "yolo_pose": {"xml": os.path.join(PATH_PESOS, "posicao_openvino.xml"),
                  "onnx": os.path.join(PATH_PESOS, "posicao.onnx")}
}

SEG_SIZE = 512
YOLO_SIZE = 640
YOLO_POSE_SIZE = 320

THRESHOLD_UNET = 0.5
CONF_THRESHOLD_YOLO = 0.4
CONF_THRESHOLD_POSE = 0.25
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

ctkinter.set_appearance_mode("Dark")
ctkinter.set_default_color_theme("blue")


# =========================================================
# FUNÇÃO AUXILIAR PARA DETECTAR CÂMERAS DISPONÍVEIS
# =========================================================
def listar_cameras_disponiveis():
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras


# =========================================================
# GESTOR DE ENGINE DE INFERÊNCIA HÍBRIDO (CUDA / OPENVINO)
# =========================================================
class HybridInferenceEngine:
    def __init__(self):
        self.backend = None
        self.core = ov.Core()

        # Define o local do Cache dinamicamente sem poluir a pasta da aplicação
        self.core.set_property({'CACHE_DIR': DIRETORIO_CACHE})

        # 1.ª Opção: Verifica se o OpenVINO tem uma GPU disponível (ex: Intel Graphics)
        if any("GPU" in dev for dev in self.core.available_devices):
            self.backend = "OPENVINO_GPU"
            print("[Engine] GPU Intel ativa via OpenVINO.")

        # Se não encontrou GPU via OpenVINO, tenta os provedores do ONNX Runtime
        else:
            try:
                import onnxruntime as ort
                provedores = ort.get_available_providers()

                # 2.ª Opção: Verifica se tem GPU NVIDIA (CUDA)
                if "CUDAExecutionProvider" in provedores:
                    self.backend = "CUDA"
                    self.ort = ort
                    print("[Engine] CUDA ativa via ONNX Runtime.")

                # 3.ª Opção: Verifica se tem GPU AMD ou outra compatível com DirectML (Windows)
                elif "DmlExecutionProvider" in provedores:
                    self.backend = "DIRECTML"
                    self.ort = ort
                    print("[Engine] GPU AMD/Genérica ativa via ONNX Runtime (DirectML).")

            except ImportError:
                pass

        # 4.ª Opção: Se nenhuma GPU foi detetada acima, faz o Fallback para CPU via OpenVINO
        if self.backend is None:
            self.backend = "OPENVINO_CPU"
            print("[Engine] Nenhuma GPU detetada. Usando CPU via OpenVINO.")

        print(f"--> [BACKEND GLOBAL SELECIONADO]: {self.backend}\n")

    def load_model(self, model_keys, target_forced_backend=None):
        backend_atual = target_forced_backend if target_forced_backend else self.backend
        engine = HybridInferenceEngineInstance()
        engine.backend = backend_atual

        # Configura a sessão ONNX para DirectML ou CUDA
        if backend_atual in ["DIRECTML", "CUDA"]:
            provedor = 'DmlExecutionProvider' if backend_atual == "DIRECTML" else 'CUDAExecutionProvider'
            engine.ort_session = self.ort.InferenceSession(
                model_keys["onnx"], providers=[provedor, 'CPUExecutionProvider']
            )
            engine.input_name = engine.ort_session.get_inputs()[0].name
        else:
            # Configura para OpenVINO (Seja GPU ou CPU)
            device = "GPU" if backend_atual == "OPENVINO_GPU" else "CPU"
            try:
                model = self.core.read_model(model_keys["xml"])
                engine.ov_compiled_model = self.core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
            except Exception:
                engine.backend = "OPENVINO_CPU"
                model = self.core.read_model(model_keys["xml"])
                engine.ov_compiled_model = self.core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": "0"})
            engine.ov_infer_request = engine.ov_compiled_model.create_infer_request()
        return engine


class HybridInferenceEngineInstance:
    def __init__(self):
        self.backend = None
        self.ort_session = None
        self.ov_compiled_model = None
        self.ov_infer_request = None
        self.input_name = None

    def infer(self, tensor_input):
        if self.backend in ["CUDA", "DIRECTML"]:
            return self.ort_session.run(None, {self.input_name: tensor_input})[0]
        else:
            self.ov_infer_request.start_async([tensor_input])
            self.ov_infer_request.wait()
            return self.ov_infer_request.get_output_tensor().data


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2;
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh


# =========================================================
# COMPONENTE DE CADA INSTÂNCIA DE VÍDEO/CÂMERA
# =========================================================
class VideoInstanceFrame(ctkinter.CTkFrame):
    def __init__(self, master, app_master, video_source, titulo_nome):
        super().__init__(master, fg_color="#2b2b2b", border_color="#3a3a3a", border_width=1)
        self.app_master = app_master
        self.video_source = video_source

        self.running = True
        self.paused = False
        self.recording = False
        self.video_writer = None

        self.top_info = ctkinter.CTkFrame(self, fg_color="transparent")
        self.top_info.pack(fill="x", padx=10, pady=(5, 2))

        self.lbl_nome = ctkinter.CTkLabel(self.top_info, text=titulo_nome, font=("Arial", 11, "bold"),
                                          text_color="#ffffff", anchor="w")
        self.lbl_nome.pack(side="left", fill="x", expand=True)

        self.btn_fechar = ctkinter.CTkButton(self.top_info, text="✕ Close", width=55, height=20, fg_color="#c62828",
                                             hover_color="#b71c1c", font=("Arial", 10, "bold"),
                                             command=self.fechar_instancia)
        self.btn_fechar.pack(side="right")

        self.video_canvas = ctkinter.CTkLabel(self, text="", fg_color="#000000", width=280, height=240)
        self.video_canvas.pack(padx=10, pady=2)

        self.telemetria_frame = ctkinter.CTkFrame(self, fg_color="#182515", border_color="#2a4025", border_width=1)
        self.telemetria_frame.pack(fill="x", padx=10, pady=4)

        self.lbl_ocr = ctkinter.CTkLabel(self.telemetria_frame, text="OCR: ...", font=("Consolas", 12, "bold"),
                                         text_color="#00ff00", anchor="w")
        self.lbl_ocr.pack(fill="x", padx=10, pady=(4, 1))

        self.lbl_ponteiros = ctkinter.CTkLabel(self.telemetria_frame, text="Ponteiros: ...",
                                               font=("Consolas", 12, "bold"), text_color="#00ff00", anchor="w")
        self.lbl_ponteiros.pack(fill="x", padx=10, pady=(1, 4))

        self.control_bar = ctkinter.CTkFrame(self, fg_color="transparent")
        self.control_bar.pack(fill="x", padx=10, pady=(2, 8))

        self.btn_pausa = ctkinter.CTkButton(self.control_bar, text="⏸ Pausar", width=80, height=25, fg_color="#4a4a4a",
                                            hover_color="#5a5a5a", font=("Arial", 11), command=self.alternar_pausa)
        self.btn_pausa.pack(side="left", padx=(0, 5))

        self.btn_gravar = ctkinter.CTkButton(self.control_bar, text="🟠 Gravar", width=80, height=25, fg_color="#e64a19",
                                             hover_color="#bf360c", font=("Arial", 11), command=self.alternar_gravacao)
        self.btn_gravar.pack(side="left", padx=5)

        self.thread = threading.Thread(target=self.loop_pipeline, daemon=True)
        self.thread.start()

    def alternar_pausa(self):
        self.paused = not self.paused
        self.btn_pausa.configure(text="▶ Retomar" if self.paused else "⏸ Pausar")

    def alternar_gravacao(self):
        if not self.recording:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            # Salva gravações no diretório base atual do programa
            filename = os.path.join(DIRETORIO_BASE, f"gravacao_{timestamp}_{np.random.randint(100, 999)}.mp4")
            self.output_filename = filename
            self.recording = True
            self.btn_gravar.configure(text="🔴 Parar", fg_color="#b71c1c", hover_color="#7f0000")
        else:
            self.recording = False
            self.btn_gravar.configure(text="🟠 Gravar", fg_color="#e64a19", hover_color="#bf360c")
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                messagebox.showinfo("Gravação Concluída", f"Vídeo salvo com sucesso em:\n{self.output_filename}")

    def fechar_instancia(self):
        self.running = False
        if self.video_writer:
            self.video_writer.release()
        self.app_master.remover_instancia_grid(self)

    def loop_pipeline(self):
        cap = cv2.VideoCapture(self.video_source)

        calibrado = False
        rotation_step = None
        bbox_visor_estatico = None
        caixas_ponteiros_estaticas = []
        params_letterbox_ponteiros = []
        map_x, map_y = None, None

        texto_ocr_cache = "..."
        ocr_inicial_realizado = False
        ocr_no_zero_realizado = False

        p_superior_estatico = 0
        p_superior_anterior = 0
        p_inferior_anterior = 0.0

        buffer_ocr_inp = np.zeros((1, 1, 32, 128), dtype=np.float32)

        while self.running:
            if self.paused:
                time.sleep(0.03)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            if self.recording:
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0 or math.isnan(fps): fps = 30.0
                    self.video_writer = cv2.VideoWriter(self.output_filename, fourcc, fps,
                                                        (frame.shape[1], frame.shape[0]))
                self.video_writer.write(frame)

            h_orig, w_orig = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not calibrado:
                seg_inp = cv2.resize(frame_rgb, (SEG_SIZE, SEG_SIZE)).transpose(2, 0, 1)[None, ...].astype(np.float32,
                                                                                                           copy=False) / 255.0
                seg_inp = np.ascontiguousarray(seg_inp)

                pred_seg = self.app_master.model_visor.infer(seg_inp)
                mask_prob = 1 / (1 + np.exp(-pred_seg))
                mask_v = (np.squeeze(mask_prob) > THRESHOLD_UNET).astype(np.uint8)
                mask_v = cv2.resize(mask_v, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                contours_v, _ = cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_v) == 0: continue
                maior_cnt = max(contours_v, key=cv2.contourArea)
                x_b, y_b, w_b, h_b = cv2.boundingRect(maior_cnt)
                centro_x_visor = x_b + (w_b / 2)
                centro_y_visor = y_b + (h_b / 2)

                inp_img, ratio, dw, dh = letterbox(frame_rgb, new_shape=(YOLO_SIZE, YOLO_SIZE))
                yolo_inp = inp_img.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                yolo_inp = np.ascontiguousarray(yolo_inp)

                saida_bruta_yolo = self.app_master.model_yolo.infer(yolo_inp)
                saida_bruta_yolo = np.squeeze(saida_bruta_yolo)
                if len(saida_bruta_yolo.shape) == 1:
                    saida_bruta_yolo = np.expand_dims(saida_bruta_yolo, axis=0)

                caixas_p = []
                for pred in saida_bruta_yolo:
                    if len(pred) < 5 or pred[4] < CONF_THRESHOLD_YOLO: continue
                    caixas_p.append([
                        int(max(0, (pred[0] - dw) / ratio)), int(max(0, (pred[1] - dh) / ratio)),
                        int(min(w_orig, (pred[2] - dw) / ratio)), int(min(h_orig, (pred[3] - dh) / ratio))
                    ])

                if len(caixas_p) == 0: continue
                centro_x_ponteiros = np.mean([(b[0] + b[2]) / 2 for b in caixas_p])
                centro_y_ponteiros = np.mean([(b[1] + b[3]) / 2 for b in caixas_p])

                frame_calib_rgb = frame_rgb.copy()
                mask_calib = mask_v.copy()
                if h_b > w_b:
                    if centro_x_ponteiros > centro_x_visor:
                        frame_calib_rgb = cv2.rotate(frame_calib_rgb, cv2.ROTATE_90_CLOCKWISE)
                        mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_90_CLOCKWISE)
                        rotation_step = cv2.ROTATE_90_CLOCKWISE
                    elif centro_x_ponteiros < centro_x_visor:
                        frame_calib_rgb = cv2.rotate(frame_calib_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        rotation_step = cv2.ROTATE_90_COUNTERCLOCKWISE
                else:
                    if centro_y_visor > centro_y_ponteiros:
                        frame_calib_rgb = cv2.rotate(frame_calib_rgb, cv2.ROTATE_180)
                        mask_calib = cv2.rotate(mask_calib, cv2.ROTATE_180)
                        rotation_step = cv2.ROTATE_180

                h_c, w_c = frame_calib_rgb.shape[:2]
                contours_c, _ = cv2.findContours(mask_calib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                maior_cnt_c = max(contours_c, key=cv2.contourArea)
                rect = cv2.minAreaRect(maior_cnt_c)
                (cx_c, cy_c), (l_rect, a_rect), angulo = rect
                if l_rect < a_rect:
                    angulo = angulo - 90 if angulo > 0 else angulo + 90
                else:
                    if angulo > 45:
                        angulo -= 90
                    elif angulo < -45:
                        angulo += 90

                if abs(angulo) > 0.1:
                    M_rot = cv2.getRotationMatrix2D((cx_c, cy_c), angulo, 1.0)
                    mask_calib = cv2.warpAffine(mask_calib, M_rot, (w_c, h_c), flags=cv2.INTER_NEAREST)
                    contours_c, _ = cv2.findContours(mask_calib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    maior_cnt_c = max(contours_c, key=cv2.contourArea)

                rect = cv2.minAreaRect(maior_cnt_c)
                box = cv2.boxPoints(rect).astype(np.intp)
                pts_origem = np.zeros((4, 2), dtype="float32")
                s = box.sum(axis=1)
                pts_origem[0] = box[np.argmin(s)]
                pts_origem[2] = box[np.argmax(s)]
                diff = np.diff(box, axis=1)
                pts_origem[1] = box[np.argmin(diff)]
                pts_origem[3] = box[np.argmax(diff)]

                dim1, dim2 = rect[1]
                largura_visor = int(max(dim1, dim2))
                altura_visor = int(min(dim1, dim2))
                margem_w, margem_h = int(largura_visor * 1.2), int(altura_visor * 4.5)

                pts_destino = np.array([
                    [margem_w, margem_h], [margem_w + largura_visor, margem_h],
                    [margem_w + largura_visor, margem_h + altura_visor], [margem_w, margem_h + altura_visor]
                ], dtype="float32")

                Matriz_H = cv2.getPerspectiveTransform(pts_origem, pts_destino)
                dimensoes_finais = (largura_visor + margem_w * 2, altura_visor + margem_h * 2)

                H_inv = np.linalg.inv(Matriz_H)
                grid_y, grid_x = np.indices((dimensoes_finais[1], dimensoes_finais[0]), dtype=np.float32)

                w_div = H_inv[2, 0] * grid_x + H_inv[2, 1] * grid_y + H_inv[2, 2]
                map_x = ((H_inv[0, 0] * grid_x + H_inv[0, 1] * grid_y + H_inv[0, 2]) / w_div).astype(np.float32)
                map_y = ((H_inv[1, 0] * grid_x + H_inv[1, 1] * grid_y + H_inv[1, 2]) / w_div).astype(np.float32)

                frame_piloto_est_rgb = cv2.remap(frame_calib_rgb, map_x, map_y, cv2.INTER_LINEAR)
                h_est, w_est = frame_piloto_est_rgb.shape[:2]

                seg_inp_est = cv2.resize(frame_piloto_est_rgb, (SEG_SIZE, SEG_SIZE)).transpose(2, 0, 1)[
                                  None, ...].astype(np.float32, copy=False) / 255.0
                seg_inp_est = np.ascontiguousarray(seg_inp_est)

                pred_seg_est = self.app_master.model_visor.infer(seg_inp_est)
                mask_v_est = (np.squeeze(1 / (1 + np.exp(-pred_seg_est))) > THRESHOLD_UNET).astype(np.uint8)
                mask_v_est = cv2.resize(mask_v_est, (w_est, h_est), interpolation=cv2.INTER_NEAREST)
                contours_est, _ = cv2.findContours(mask_v_est, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_est) > 0:
                    bbox_visor_estatico = cv2.boundingRect(max(contours_est, key=cv2.contourArea))

                inp_img_est, ratio_est, dw_est, dh_est = letterbox(frame_piloto_est_rgb,
                                                                   new_shape=(YOLO_SIZE, YOLO_SIZE))
                yolo_inp_est = inp_img_est.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                yolo_inp_est = np.ascontiguousarray(yolo_inp_est)
                saida_bruta_yolo_est = self.app_master.model_yolo.infer(yolo_inp_est)

                saida_bruta_yolo_est = np.squeeze(saida_bruta_yolo_est)
                if len(saida_bruta_yolo_est.shape) == 1:
                    saida_bruta_yolo_est = np.expand_dims(saida_bruta_yolo_est, axis=0)

                ponteiros_validos = []
                for pred in saida_bruta_yolo_est:
                    if len(pred) < 5 or pred[4] < CONF_THRESHOLD_YOLO: continue
                    ponteiros_validos.append([
                        int(max(0, (pred[0] - dw_est) / ratio_est)), int(max(0, (pred[1] - dh_est) / ratio_est)),
                        int(min(w_est, (pred[2] - dw_est) / ratio_est)), int(min(h_est, (pred[3] - dw_est) / ratio_est))
                    ])
                caixas_ponteiros_estaticas = sorted(ponteiros_validos, key=lambda p: p[1])

                for ponteiro in caixas_ponteiros_estaticas:
                    x1, y1, x2, y2 = ponteiro
                    w_p_crop = x2 - x1;
                    h_p_crop = y2 - y1
                    r_p = min(YOLO_POSE_SIZE / h_p_crop, YOLO_POSE_SIZE / w_p_crop)
                    new_unpad_w, new_unpad_h = int(round(w_p_crop * r_p)), int(round(h_p_crop * r_p))
                    dw_p = (YOLO_POSE_SIZE - new_unpad_w) / 2
                    dh_p = (YOLO_POSE_SIZE - new_unpad_h) / 2
                    params_letterbox_ponteiros.append((r_p, dw_p, dh_p, new_unpad_w, new_unpad_h))

                if len(caixas_ponteiros_estaticas) > 0:
                    x1, y1, x2, y2 = caixas_ponteiros_estaticas[0]
                    crop_p_rgb = frame_piloto_est_rgb[y1:y2, x1:x2]
                    if crop_p_rgb.size > 0:
                        r_p, dw_p, dh_p, _, _ = params_letterbox_ponteiros[0]
                        top, left = int(round(dh_p - 0.1)), int(round(dw_p - 0.1))
                        crop_padded = np.full((YOLO_POSE_SIZE, YOLO_POSE_SIZE, 3), 114, dtype=np.uint8)
                        crop_resized = cv2.resize(crop_p_rgb,
                                                  (crop_padded.shape[1] - left * 2, crop_padded.shape[0] - top * 2),
                                                  interpolation=cv2.INTER_LINEAR)
                        crop_padded[top:top + crop_resized.shape[0], left:left + crop_resized.shape[1]] = crop_resized

                        tensor_single = crop_padded.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                        preds_pose_single = self.app_master.model_pose.infer(tensor_single)

                        single_pred = np.squeeze(preds_pose_single)
                        if len(single_pred.shape) == 1: single_pred = np.expand_dims(single_pred, axis=0)
                        preds_pose = np.transpose(single_pred) if single_pred.shape[0] < single_pred.shape[
                            1] else single_pred

                        if preds_pose.shape[0] > 0 and np.max(preds_pose[:, 4]) >= CONF_THRESHOLD_POSE:
                            melhor_pred = preds_pose[np.argmax(preds_pose[:, 4])]
                            kpts = melhor_pred[5:]
                            if len(kpts) >= 6 and kpts[2] > 0.35 and kpts[5] > 0.35:
                                angulo = math.degrees(math.atan2((kpts[4] - dh_p) / r_p - (kpts[1] - dh_p) / r_p,
                                                                 (kpts[3] - dw_p) / r_p - (
                                                                             kpts[0] - dw_p) / r_p)) + 90.0
                                if angulo < 0: angulo += 360.0
                                val = round((angulo % 360.0) / 36.0, 1)
                                if val >= 10.0: val = 0.0
                                p_superior_estatico = int(round(val))

                calibrado = True

            if rotation_step is not None:
                frame_rgb = cv2.rotate(frame_rgb, rotation_step)

            frame_estabilizado_rgb = cv2.remap(frame_rgb, map_x, map_y, cv2.INTER_LINEAR)

            if len(caixas_ponteiros_estaticas) > 1:
                x1, y1, x2, y2 = caixas_ponteiros_estaticas[1]
                crop_p_rgb = frame_estabilizado_rgb[y1:y2, x1:x2]
                if crop_p_rgb.size > 0:
                    r_p, dw_p, dh_p, _, _ = params_letterbox_ponteiros[1]
                    top, left = int(round(dh_p - 0.1)), int(round(dw_p - 0.1))
                    crop_padded = np.full((YOLO_POSE_SIZE, YOLO_POSE_SIZE, 3), 114, dtype=np.uint8)
                    crop_resized = cv2.resize(crop_p_rgb,
                                              (crop_padded.shape[1] - left * 2, crop_padded.shape[0] - top * 2),
                                              interpolation=cv2.INTER_LINEAR)
                    crop_padded[top:top + crop_resized.shape[0], left:left + crop_resized.shape[1]] = crop_resized

                    tensor_inf = crop_padded.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                    preds_pose_inf = self.app_master.model_pose.infer(tensor_inf)

                    single_pred = np.squeeze(preds_pose_inf)
                    if len(single_pred.shape) == 1: single_pred = np.expand_dims(single_pred, axis=0)
                    preds_pose = np.transpose(single_pred) if single_pred.shape[0] < single_pred.shape[
                        1] else single_pred

                    melhor_pred = None
                    if preds_pose.shape[0] > 0 and np.max(preds_pose[:, 4]) >= CONF_THRESHOLD_POSE:
                        melhor_pred = preds_pose[np.argmax(preds_pose[:, 4])]

                    if melhor_pred is not None:
                        kpts = melhor_pred[5:]
                        if len(kpts) >= 6 and kpts[2] > 0.35 and kpts[5] > 0.35:
                            angulo_ajustated = math.degrees(math.atan2((kpts[4] - dh_p) / r_p - (kpts[1] - dh_p) / r_p,
                                                                       (kpts[3] - dw_p) / r_p - (
                                                                                   kpts[0] - dw_p) / r_p)) + 90.0
                            if angulo_ajustated < 0: angulo_ajustated += 360.0
                            valor_original = round((angulo_ajustated % 360.0) / 36.0, 1)
                            if valor_original >= 10.0: valor_original = 0.0

                            decimais = int(round((valor_original - int(valor_original)) * 10))
                            if decimais % 2 != 0:
                                valor_original = round(valor_original - 0.1, 1)
                                if valor_original < 0.0: valor_original = 0.0
                            p_inferior_cache = valor_original
                        else:
                            p_inferior_cache = 0.0
                    else:
                        p_inferior_cache = 0.0
                else:
                    p_inferior_cache = 0.0
            else:
                p_inferior_cache = 0.0

            completou_volta = (p_inferior_cache == 0.0) or (p_inferior_anterior > 5.0 and p_inferior_cache < 1.0)
            p_inferior_anterior = p_inferior_cache

            if completou_volta and len(caixas_ponteiros_estaticas) > 0:
                x1, y1, x2, y2 = caixas_ponteiros_estaticas[0]
                crop_p_rgb = frame_estabilizado_rgb[y1:y2, x1:x2]
                if crop_p_rgb.size > 0:
                    r_p, dw_p, dh_p, _, _ = params_letterbox_ponteiros[0]
                    top, left = int(round(dh_p - 0.1)), int(round(dw_p - 0.1))
                    crop_padded = np.full((YOLO_POSE_SIZE, YOLO_POSE_SIZE, 3), 114, dtype=np.uint8)
                    crop_resized = cv2.resize(crop_p_rgb,
                                              (crop_padded.shape[1] - left * 2, crop_padded.shape[0] - top * 2),
                                              interpolation=cv2.INTER_LINEAR)
                    crop_padded[top:top + crop_resized.shape[0], left:left + crop_resized.shape[1]] = crop_resized

                    tensor_sup = crop_padded.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                    preds_pose_sup = self.app_master.model_pose.infer(tensor_sup)

                    single_pred = np.squeeze(preds_pose_sup)
                    if len(single_pred.shape) == 1: single_pred = np.expand_dims(single_pred, axis=0)
                    preds_pose = np.transpose(single_pred) if single_pred.shape[0] < single_pred.shape[
                        1] else single_pred

                    melhor_pred = None
                    if preds_pose.shape[0] > 0 and np.max(preds_pose[:, 4]) >= CONF_THRESHOLD_POSE:
                        melhor_pred = preds_pose[np.argmax(preds_pose[:, 4])]

                    if melhor_pred is not None:
                        kpts = melhor_pred[5:]
                        if len(kpts) >= 6 and kpts[2] > 0.35 and kpts[5] > 0.35:
                            angulo_ajustated = math.degrees(math.atan2((kpts[4] - dh_p) / r_p - (kpts[1] - dh_p) / r_p,
                                                                       (kpts[3] - dw_p) / r_p - (
                                                                                   kpts[0] - dw_p) / r_p)) + 90.0
                            if angulo_ajustated < 0: angulo_ajustated += 360.0
                            valor_original = round((angulo_ajustated % 360.0) / 36.0, 1)
                            if valor_original >= 10.0: valor_original = 0.0

                            p_superior_estatico = int(round(valor_original))
                            p_superior_cache = p_superior_estatico
                        else:
                            p_superior_cache = p_superior_estatico
                    else:
                        p_superior_cache = p_superior_estatico
                else:
                    p_superior_cache = p_superior_estatico
            else:
                p_superior_cache = p_superior_estatico

            deve_processar_ocr = False
            ponteiro_superior_mudou = (p_superior_cache != p_superior_anterior)
            if not ocr_inicial_realizado:
                deve_processar_ocr = True
            elif (completou_volta or ponteiro_superior_mudou) and not ocr_no_zero_realizado:
                deve_processar_ocr = True

            p_superior_anterior = p_superior_cache

            if not completou_volta and not ponteiro_superior_mudou:
                ocr_no_zero_realizado = False

            if deve_processar_ocr and bbox_visor_estatico is not None:
                x_v, y_v, ww_v, hh_v = bbox_visor_estatico
                crop_visor_rgb = frame_estabilizado_rgb[y_v:y_v + hh_v, x_v:x_v + ww_v]
                if crop_visor_rgb.size > 0:
                    crop_gray = cv2.cvtColor(crop_visor_rgb, cv2.COLOR_RGB2GRAY)
                    crop_resized = cv2.resize(crop_gray, (128, 32), interpolation=cv2.INTER_LINEAR)
                    buffer_ocr_inp[0, 0, :, :] = crop_resized / 255.0

                    pred_ocr = self.app_master.model_ocr.infer(buffer_ocr_inp)
                    pred_argmax = np.argmax(pred_ocr, axis=2).flatten()

                    texto_temp = ""
                    last = -1
                    for p in pred_argmax:
                        if p != last and p != 0 and (p - 1) < len(CHARS): texto_temp += CHARS[p - 1]
                        last = p
                    if len(texto_temp) > 6: texto_temp = texto_temp[:6]

                    if texto_temp.strip() and texto_temp != "...":
                        texto_ocr_cache = texto_temp
                        if not ocr_inicial_realizado:
                            ocr_inicial_realizado = True
                        else:
                            ocr_no_zero_realizado = True

            self.atualizar_ui_elementos(frame_estabilizado_rgb, texto_ocr_cache, p_superior_cache, p_inferior_cache)

        cap.release()

    def atualizar_ui_elementos(self, frame_rgb, ocr, p_sup, p_inf):
        frame_resized = cv2.resize(frame_rgb, (280, 240), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(frame_resized)

        img_ctk = ctkinter.CTkImage(light_image=img_pil, dark_image=img_pil, size=(280, 240))
        self.after(0, self._set_labels_and_image, img_ctk, ocr, p_sup, p_inf)

    def _set_labels_and_image(self, img_ctk, ocr, p_sup, p_inf):
        if not self.running: return
        self.video_canvas.configure(image=img_ctk)
        self.video_canvas.image = img_ctk
        self.lbl_ocr.configure(text=f"OCR: {ocr}")
        self.lbl_ponteiros.configure(text=f"Ponteiros: {p_sup} | {p_inf:.1f}")


# =========================================================
# INTERFACE GRÁFICA PRINCIPAL MÚLTIPLAS INSTÂNCIAS
# =========================================================
class AppMedidorMultiplo(ctkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Monitoramento Multicâmera Otimizado")
        self.geometry("1280x760")
        self.configure(fg_color="#1e1e1e")

        self.instancias_ativas = []

        self.master_engine = HybridInferenceEngine()
        print("A carregar pipeline de modelos compartilhados...")
        self.model_visor = self.master_engine.load_model(MODELOS["visor"])
        self.model_yolo = self.master_engine.load_model(MODELOS["yolo"])
        self.model_ocr = self.master_engine.load_model(MODELOS["ocr"], target_forced_backend="OPENVINO_CPU")
        self.model_pose = self.master_engine.load_model(MODELOS["yolo_pose"])

        self.construir_interface_principal()

    def construir_interface_principal(self):
        self.top_bar = ctkinter.CTkFrame(self, height=60, fg_color="#2b2b2b", corner_radius=0)
        self.top_bar.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_camera = ctkinter.CTkButton(self.top_bar, text="📷 Câmera Local", font=("Arial", 13, "bold"),
                                             fg_color="#007acc", hover_color="#005999",
                                             command=self.abrir_janela_selecao_camera)
        self.btn_camera.pack(side="left", padx=15, pady=12)

        self.btn_arquivo = ctkinter.CTkButton(self.top_bar, text="📁 Adicionar Vídeo", font=("Arial", 13, "bold"),
                                              fg_color="#007acc", hover_color="#005999",
                                              command=self.adicionar_arquivo_video)
        self.btn_arquivo.pack(side="left", padx=5, pady=12)

        self.grid_scroll = ctkinter.CTkScrollableFrame(self, fg_color="transparent")
        self.grid_scroll.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        self.grid_scroll.grid_columnconfigure((0, 1, 2, 3), weight=1, pad=10)

    def adicionar_arquivo_video(self):
        caminho = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mkv")])
        if caminho:
            nome_limpo = caminho.split('/')[-1]
            self.criar_nova_instancia_stream(caminho, f"🎥 {nome_limpo}")

    def abrir_janela_selecao_camera(self):
        janela_cam = ctkinter.CTkToplevel(self)
        janela_cam.title("Selecionar Câmera Disponível")
        janela_cam.geometry("320x200")
        janela_cam.resizable(False, False)
        janela_cam.grab_set()

        lbl = ctkinter.CTkLabel(janela_cam, text="Buscando dispositivos de vídeo...", font=("Arial", 12))
        lbl.pack(pady=15)

        lista_indices = listar_cameras_disponiveis()

        if not lista_indices:
            lbl.configure(text="Nenhuma câmera física encontrada!", text_color="#c62828")
            btn_ok = ctkinter.CTkButton(janela_cam, text="Fechar", width=100, command=janela_cam.destroy)
            btn_ok.pack(pady=10)
            return

        lbl.configure(text="Selecione a câmera desejada:")

        opcoes_texto = [f"Dispositivo de Vídeo {i}" for i in lista_indices]
        combo_cams = ctkinter.CTkComboBox(janela_cam, values=opcoes_texto, width=200)
        combo_cams.pack(pady=10)

        def confirmar_selecao():
            idx_selecionado = lista_indices[opcoes_texto.index(combo_cams.get())]
            janela_cam.destroy()
            self.criar_nova_instancia_stream(idx_selecionado, f"📷 Live Cam #{idx_selecionado}")

        btn_conf = ctkinter.CTkButton(janela_cam, text="Conectar", fg_color="#2e7d32", hover_color="#1b5e20",
                                      command=confirmar_selecao)
        btn_conf.pack(pady=15)

    def criar_nova_instancia_stream(self, fonte, titulo):
        nova_instancia = VideoInstanceFrame(self.grid_scroll, self, fonte, titulo)
        self.instancias_ativas.append(nova_instancia)
        self.redesenhar_grid_layouts()

    def remover_instancia_grid(self, instancia):
        if instancia in self.instancias_ativas:
            self.instancias_ativas.remove(instancia)
            instancia.destroy()
            self.redesenhar_grid_layouts()

    def redesenhar_grid_layouts(self):
        for inst in self.instancias_ativas:
            inst.grid_forget()

        max_colunas = 3
        for idx, inst in enumerate(self.instancias_ativas):
            linha = idx // max_colunas
            coluna = idx % max_colunas
            inst.grid(row=linha, column=coluna, padx=10, pady=10, sticky="nsew")


if __name__ == "__main__":
    app = AppMedidorMultiplo()
    app.mainloop()