import cv2
import numpy as np
import openvino as ov
import math
import time
import threading
import tkinter as tk
from tkinter import filedialog
import customtkinter as ctkinter
from PIL import Image, ImageTk

# =========================================================
# CONFIGURAÇÕES GERAIS E CAMINHOS DOS MODELOS
# =========================================================
MODELOS = {
    "visor": {"xml": "pesos/visor_openvino.xml", "onnx": "pesos/visor.onnx"},
    "yolo": {"xml": "pesos/ponteiro_openvino.xml", "onnx": "pesos/ponteiro.onnx"},
    "ocr": {"xml": "pesos/ocr_openvino.xml", "onnx": "pesos/ocr.onnx"},
    "yolo_pose": {"xml": "pesos/posicao_openvino.xml", "onnx": "pesos/posicao.onnx"}
}

SEG_SIZE = 512
YOLO_SIZE = 640
YOLO_POSE_SIZE = 320

THRESHOLD_UNET = 0.5
CONF_THRESHOLD_YOLO = 0.4
CONF_THRESHOLD_POSE = 0.25
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Configuração de Aparência do CustomTkinter
ctkinter.set_appearance_mode("Dark")
ctkinter.set_default_color_theme("blue")


# =========================================================
# GESTOR DE ENGINE DE INFERÊNCIA HÍBRIDO (CUDA / OPENVINO)
# =========================================================
class HybridInferenceEngine:
    def __init__(self):
        self.backend = "CPU"
        self.ort_session = None
        self.ov_compiled_model = None
        self.ov_infer_request = None
        self.core = ov.Core()
        self.core.set_property({'CACHE_DIR': './ov_cache_dir'})

        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                self.backend = "CUDA"
                self.ort = ort
                print("[Engine] CUDA (NVIDIA) detetado com sucesso via ONNX Runtime.")
        except ImportError:
            pass

        if self.backend == "CPU":
            ov_devices = self.core.available_devices
            if any("GPU" in dev for dev in ov_devices):
                self.backend = "OPENVINO_GPU"
            else:
                self.backend = "OPENVINO_CPU"

        print(f"--> [BACKEND SELECIONADO]: {self.backend}\n")

    def load_model(self, model_keys, target_forced_backend=None):
        backend_atual = target_forced_backend if target_forced_backend else self.backend
        engine = HybridInferenceEngineInstance()
        engine.backend = backend_atual

        if backend_atual == "CUDA":
            onnx_path = model_keys["onnx"]
            engine.ort_session = self.ort.InferenceSession(
                onnx_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            engine.input_name = engine.ort_session.get_inputs()[0].name
        else:
            xml_path = model_keys["xml"]
            device = "GPU" if backend_atual == "OPENVINO_GPU" else "CPU"
            config = {"PERFORMANCE_HINT": "LATENCY"} if device == "GPU" else {"INFERENCE_NUM_THREADS": "0"}

            try:
                model = self.core.read_model(xml_path)
                engine.ov_model = model
                engine.ov_compiled_model = self.core.compile_model(model, device, config)
            except Exception:
                engine.backend = "OPENVINO_CPU"
                model = self.core.read_model(xml_path)
                engine.ov_model = model
                engine.ov_compiled_model = self.core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": "0"})

            engine.ov_infer_request = engine.ov_compiled_model.create_infer_request()

        return engine


class HybridInferenceEngineInstance:
    def __init__(self):
        self.backend = None
        self.ort_session = None
        self.ov_model = None
        self.ov_compiled_model = None
        self.ov_infer_request = None
        self.input_name = None

    def infer(self, tensor_input):
        if self.backend == "CUDA":
            outputs = self.ort_session.run(None, {self.input_name: tensor_input})
            return outputs[0]
        else:
            self.ov_infer_request.start_async([tensor_input])
            self.ov_infer_request.wait()
            return self.ov_infer_request.get_output_tensor().data


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, dw, dh


# =========================================================
# INTERFACE GRÁFICA PRINCIPAL (LEVE)
# =========================================================
class AppMedidor(ctkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Monitoramento Otimizado")
        self.geometry("1100x720")
        self.configure(fg_color="#1e1e1e")

        # Controle de Estado da Thread de Vídeo
        self.cap = None
        self.running = False
        self.paused = False
        self.video_source = None
        self.video_thread = None

        # Instancia as Engines uma única vez para economizar memória
        self.master_engine = HybridInferenceEngine()
        print("A carregar pipeline de modelos...")
        self.model_visor = self.master_engine.load_model(MODELOS["visor"])
        self.model_yolo = self.master_engine.load_model(MODELOS["yolo"])
        self.model_ocr = self.master_engine.load_model(MODELOS["ocr"], target_forced_backend="OPENVINO_CPU")
        self.model_pose = self.master_engine.load_model(MODELOS["yolo_pose"])

        self.construir_interface()

    def construir_interface(self):
        # --- Top Menu Bar ---
        self.top_bar = ctkinter.CTkFrame(self, height=60, fg_color="#2b2b2b", corner_radius=0)
        self.top_bar.pack(side="top", fill="x", padx=10, pady=10)

        self.btn_camera = ctkinter.CTkButton(self.top_bar, text="📷 Câmera Local", font=("Arial", 13, "bold"), fg_color="#007acc", hover_color="#005999", command=self.iniciar_camera)
        self.btn_camera.pack(side="left", padx=15, pady=12)

        self.btn_arquivo = ctkinter.CTkButton(self.top_bar, text="📁 Arquivo de Vídeo", font=("Arial", 13, "bold"), fg_color="#007acc", hover_color="#005999", command=self.abrir_arquivo)
        self.btn_arquivo.pack(side="left", padx=5, pady=12)

        # --- Área Central ---
        self.main_container = ctkinter.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(side="top", fill="both", expand=True, padx=10, pady=5)

        # Container Esquerdo (Vídeo + Status)
        self.left_container = ctkinter.CTkFrame(self.main_container, width=360, fg_color="#2b2b2b", border_color="#3a3a3a", border_width=1)
        self.left_container.pack(side="left", fill="both", expand=False, padx=(0, 10))

        # Cabeçalho do arquivo
        self.lbl_nome_arquivo = ctkinter.CTkLabel(self.left_container, text="Nenhum arquivo carregado", font=("Arial", 12, "bold"), text_color="#ffffff", anchor="w")
        self.lbl_nome_arquivo.pack(fill="x", padx=15, pady=(10, 5))

        # Viewport de Imagem
        self.video_canvas = ctkinter.CTkLabel(self.left_container, text="", fg_color="#000000", width=330, height=330)
        self.video_canvas.pack(padx=15, pady=5)

        # Painel de Telemetria (Verde Escuro)
        self.telemetria_frame = ctkinter.CTkFrame(self.left_container, fg_color="#182515", border_color="#2a4025", border_width=1)
        self.telemetria_frame.pack(fill="x", padx=15, pady=10)

        self.lbl_ocr = ctkinter.CTkLabel(self.telemetria_frame, text="OCR: ...", font=("Consolas", 14, "bold"), text_color="#00ff00", anchor="w")
        self.lbl_ocr.pack(fill="x", padx=15, pady=(8, 2))

        self.lbl_ponteiros = ctkinter.CTkLabel(self.telemetria_frame, text="Ponteiros: ...", font=("Consolas", 14, "bold"), text_color="#00ff00", anchor="w")
        self.lbl_ponteiros.pack(fill="x", padx=15, pady=(2, 8))

        # Botões de Controle Inferiores
        self.control_bar = ctkinter.CTkFrame(self.left_container, fg_color="transparent")
        self.control_bar.pack(fill="x", padx=15, pady=(5, 15))

        self.btn_pausa = ctkinter.CTkButton(self.control_bar, text="▶ Retomar", width=80, fg_color="#4a4a4a", hover_color="#5a5a5a", command=self.alternar_pausa)
        self.btn_pausa.pack(side="left", padx=(0, 5))

        self.btn_gravar = ctkinter.CTkButton(self.control_bar, text="🟠 Gravar", width=80, fg_color="#e64a19", hover_color="#bf360c")
        self.btn_gravar.pack(side="left", padx=5)

        self.chk_onnx = ctkinter.CTkCheckBox(self.control_bar, text="Processamento ONNX", font=("Arial", 11), fg_color="#007acc")
        self.chk_onnx.select()
        self.chk_onnx.pack(side="left", padx=10)

        # Painel Direito Espaçador
        self.right_container = ctkinter.CTkFrame(self.main_container, fg_color="#1e1e1e", border_color="#2b2b2b", border_width=1)
        self.right_container.pack(side="right", fill="both", expand=True)

    def abrir_arquivo(self):
        caminho = filedialog.askopenfilename(filetypes=[("Vídeos", "*.mp4 *.avi *.mkv")])
        if caminho:
            self.lbl_nome_arquivo.configure(text=f"🎥 {caminho.split('/')[-1]}  •  Processando")
            self.iniciar_processamento(caminho)

    def iniciar_camera(self):
        self.lbl_nome_arquivo.configure(text="🎥 Câmera Local Ativa  •  Live")
        self.iniciar_processamento(0)

    def iniciar_processamento(self, fonte):
        self.parar_processamento()
        self.running = True
        self.paused = False
        self.btn_pausa.configure(text="⏸ Pausar")
        self.video_source = fonte
        self.video_thread = threading.Thread(target=self.loop_video_pipeline, daemon=True)
        self.video_thread.start()

    def alternar_pausa(self):
        if self.running:
            self.paused = not self.paused
            if self.paused:
                self.btn_pausa.configure(text="▶ Retomar")
                self.lbl_nome_arquivo.configure(text=self.lbl_nome_arquivo.cget("text").split("•")[0] + " • Pausado")
            else:
                self.btn_pausa.configure(text="⏸ Pausar")
                self.lbl_nome_arquivo.configure(text=self.lbl_nome_arquivo.cget("text").split("•")[0] + " • Processando")

    def parar_processamento(self):
        self.running = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    # =========================================================
    # PIPELINE INTEGRADO FIXO EM BATCH 1 (EVITA ERRO DE RESHAPE)
    # =========================================================
    def loop_video_pipeline(self):
        self.cap = cv2.VideoCapture(self.video_source)
        
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
        p_superior_cache = 0
        p_inferior_cache = 0.0
        p_inferior_anterior = 0.0

        buffer_ocr_inp = np.zeros((1, 1, 32, 128), dtype=np.float32)

        while self.running:
            if self.paused:
                time.sleep(0.03)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            h_orig, w_orig = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- ETAPA DE CALIBRAÇÃO (APENAS FRAME 1) ---
            if not calibrado:
                seg_inp = cv2.resize(frame_rgb, (SEG_SIZE, SEG_SIZE)).transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                seg_inp = np.ascontiguousarray(seg_inp)

                pred_seg = self.model_visor.infer(seg_inp)
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

                saida_bruta_yolo = self.model_yolo.infer(yolo_inp)
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
                    if angulo > 45: angulo -= 90
                    elif angulo < -45: angulo += 90

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

                seg_inp_est = cv2.resize(frame_piloto_est_rgb, (SEG_SIZE, SEG_SIZE)).transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                seg_inp_est = np.ascontiguousarray(seg_inp_est)

                pred_seg_est = self.model_visor.infer(seg_inp_est)
                mask_v_est = (np.squeeze(1 / (1 + np.exp(-pred_seg_est))) > THRESHOLD_UNET).astype(np.uint8)
                mask_v_est = cv2.resize(mask_v_est, (w_est, h_est), interpolation=cv2.INTER_NEAREST)
                contours_est, _ = cv2.findContours(mask_v_est, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours_est) > 0:
                    bbox_visor_estatico = cv2.boundingRect(max(contours_est, key=cv2.contourArea))

                inp_img_est, ratio_est, dw_est, dh_est = letterbox(frame_piloto_est_rgb, new_shape=(YOLO_SIZE, YOLO_SIZE))
                yolo_inp_est = inp_img_est.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                yolo_inp_est = np.ascontiguousarray(yolo_inp_est)
                saida_bruta_yolo_est = self.model_yolo.infer(yolo_inp_est)

                saida_bruta_yolo_est = np.squeeze(saida_bruta_yolo_est)
                if len(saida_bruta_yolo_est.shape) == 1:
                    saida_bruta_yolo_est = np.expand_dims(saida_bruta_yolo_est, axis=0)

                ponteiros_validos = []
                for pred in saida_bruta_yolo_est:
                    if len(pred) < 5 or pred[4] < CONF_THRESHOLD_YOLO: continue
                    ponteiros_validos.append([
                        int(max(0, (pred[0] - dw_est) / ratio_est)), int(max(0, (pred[1] - dh_est) / ratio_est)),
                        int(min(w_est, (pred[2] - dw_est) / ratio_est)), int(min(h_est, (pred[3] - dh_est) / ratio_est))
                    ])
                caixas_ponteiros_estaticas = sorted(ponteiros_validos, key=lambda p: p[1])

                for ponteiro in caixas_ponteiros_estaticas:
                    x1, y1, x2, y2 = ponteiro
                    w_p_crop = x2 - x1; h_p_crop = y2 - y1
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
                        crop_resized = cv2.resize(crop_p_rgb, (crop_padded.shape[1] - left * 2, crop_padded.shape[0] - top * 2), interpolation=cv2.INTER_LINEAR)
                        crop_padded[top:top + crop_resized.shape[0], left:left + crop_resized.shape[1]] = crop_resized
                        
                        tensor_single = crop_padded.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                        preds_pose_single = self.model_pose.infer(tensor_single)
                        
                        single_pred = np.squeeze(preds_pose_single)
                        if len(single_pred.shape) == 1: single_pred = np.expand_dims(single_pred, axis=0)
                        preds_pose = np.transpose(single_pred) if single_pred.shape[0] < single_pred.shape[1] else single_pred
                        
                        if preds_pose.shape[0] > 0 and np.max(preds_pose[:, 4]) >= CONF_THRESHOLD_POSE:
                            melhor_pred = preds_pose[np.argmax(preds_pose[:, 4])]
                            kpts = melhor_pred[5:]
                            if len(kpts) >= 6 and kpts[2] > 0.35 and kpts[5] > 0.35:
                                angulo = math.degrees(math.atan2((kpts[4] - dh_p) / r_p - (kpts[1] - dh_p) / r_p, (kpts[3] - dw_p) / r_p - (kpts[0] - dw_p) / r_p)) + 90.0
                                if angulo < 0: angulo += 360.0
                                val = round((angulo % 360.0) / 36.0, 1)
                                if val >= 10.0 or val == 10: val = 0.0
                                p_superior_estatico = int(round(val))

                calibrado = True

            # --- PROCESSAMENTO EM TEMPO REAL ---
            if rotation_step is not None:
                frame_rgb = cv2.rotate(frame_rgb, rotation_step)

            frame_estabilizado_rgb = cv2.remap(frame_rgb, map_x, map_y, cv2.INTER_LINEAR)

            # 1. Processar Ponteiro Inferior (Índice 1) Se existir
            if len(caixas_ponteiros_estaticas) > 1:
                x1, y1, x2, y2 = caixas_ponteiros_estaticas[1]
                crop_p_rgb = frame_estabilizado_rgb[y1:y2, x1:x2]
                if crop_p_rgb.size > 0:
                    r_p, dw_p, dh_p, _, _ = params_letterbox_ponteiros[1]
                    top, left = int(round(dh_p - 0.1)), int(round(dw_p - 0.1))
                    crop_padded = np.full((YOLO_POSE_SIZE, YOLO_POSE_SIZE, 3), 114, dtype=np.uint8)
                    crop_resized = cv2.resize(crop_p_rgb, (crop_padded.shape[1] - left * 2, crop_padded.shape[0] - top * 2), interpolation=cv2.INTER_LINEAR)
                    crop_padded[top:top + crop_resized.shape[0], left:left + crop_resized.shape[1]] = crop_resized

                    tensor_inf = crop_padded.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                    preds_pose_inf = self.model_pose.infer(tensor_inf)

                    single_pred = np.squeeze(preds_pose_inf)
                    if len(single_pred.shape) == 1: single_pred = np.expand_dims(single_pred, axis=0)
                    preds_pose = np.transpose(single_pred) if single_pred.shape[0] < single_pred.shape[1] else single_pred

                    melhor_pred = None
                    if preds_pose.shape[0] > 0 and np.max(preds_pose[:, 4]) >= CONF_THRESHOLD_POSE:
                        melhor_pred = preds_pose[np.argmax(preds_pose[:, 4])]

                    if melhor_pred is not None:
                        kpts = melhor_pred[5:]
                        if len(kpts) >= 6 and kpts[2] > 0.35 and kpts[5] > 0.35:
                            angulo_ajustated = math.degrees(math.atan2((kpts[4] - dh_p) / r_p - (kpts[1] - dh_p) / r_p, (kpts[3] - dw_p) / r_p - (kpts[0] - dw_p) / r_p)) + 90.0
                            if angulo_ajustated < 0: angulo_ajustated += 360.0
                            valor_original = round((angulo_ajustated % 360.0) / 36.0, 1)
                            if valor_original >= 10.0 or valor_original == 10: valor_original = 0.0

                            # REGRA DE ARREDONDAMENTO ORIGINAL RESTAURADA
                            decimais = int(round((valor_original - int(valor_original)) * 10))
                            if decimais % 2 != 0:
                                valor_original = round(valor_original - 0.1, 1)
                                if valor_original < 0.0: valor_original = 0.0
                            p_inferior_cache = valor_original
                        else: p_inferior_cache = 0.0
                    else: p_inferior_cache = 0.0
                else: p_inferior_cache = 0.0
            else: p_inferior_cache = 0.0

            # REGRA DE SALTO/ZERAGEM DO PONTEIRO INFERIOR
            completou_volta = (p_inferior_cache == 0.0) or (p_inferior_anterior > 5.0 and p_inferior_cache < 1.0)
            p_inferior_anterior = p_inferior_cache

            # 2. Processar Ponteiro Superior (Apenas se completou a volta)
            if completou_volta and len(caixas_ponteiros_estaticas) > 0:
                x1, y1, x2, y2 = caixas_ponteiros_estaticas[0]
                crop_p_rgb = frame_estabilizado_rgb[y1:y2, x1:x2]
                if crop_p_rgb.size > 0:
                    r_p, dw_p, dh_p, _, _ = params_letterbox_ponteiros[0]
                    top, left = int(round(dh_p - 0.1)), int(round(dw_p - 0.1))
                    crop_padded = np.full((YOLO_POSE_SIZE, YOLO_POSE_SIZE, 3), 114, dtype=np.uint8)
                    crop_resized = cv2.resize(crop_p_rgb, (crop_padded.shape[1] - left * 2, crop_padded.shape[0] - top * 2), interpolation=cv2.INTER_LINEAR)
                    crop_padded[top:top + crop_resized.shape[0], left:left + crop_resized.shape[1]] = crop_resized

                    tensor_sup = crop_padded.transpose(2, 0, 1)[None, ...].astype(np.float32, copy=False) / 255.0
                    preds_pose_sup = self.model_pose.infer(tensor_sup)

                    single_pred = np.squeeze(preds_pose_sup)
                    if len(single_pred.shape) == 1: single_pred = np.expand_dims(single_pred, axis=0)
                    preds_pose = np.transpose(single_pred) if single_pred.shape[0] < single_pred.shape[1] else single_pred

                    melhor_pred = None
                    if preds_pose.shape[0] > 0 and np.max(preds_pose[:, 4]) >= CONF_THRESHOLD_POSE:
                        melhor_pred = preds_pose[np.argmax(preds_pose[:, 4])]

                    if melhor_pred is not None:
                        kpts = melhor_pred[5:]
                        if len(kpts) >= 6 and kpts[2] > 0.35 and kpts[5] > 0.35:
                            angulo_ajustated = math.degrees(math.atan2((kpts[4] - dh_p) / r_p - (kpts[1] - dh_p) / r_p, (kpts[3] - dw_p) / r_p - (kpts[0] - dw_p) / r_p)) + 90.0
                            if angulo_ajustated < 0: angulo_ajustated += 360.0
                            valor_original = round((angulo_ajustated % 360.0) / 36.0, 1)
                            if valor_original >= 10.0 or valor_original == 10: valor_original = 0.0

                            p_superior_estatico = int(round(valor_original))
                            p_superior_cache = p_superior_estatico
                        else: p_superior_cache = p_superior_estatico
                    else: p_superior_cache = p_superior_estatico
                else: p_superior_cache = p_superior_estatico
            else:
                p_superior_cache = p_superior_estatico

            # --- CONTROLE DO OCR ---
            deve_processar_ocr = False
            if not ocr_inicial_realizado: deve_processar_ocr = True
            elif p_superior_cache == 0 and p_inferior_cache == 0.0 and not ocr_no_zero_realizado: deve_processar_ocr = True

            if p_superior_cache != 0 or p_inferior_cache != 0.0: ocr_no_zero_realizado = False

            if deve_processar_ocr and bbox_visor_estatico is not None:
                x_v, y_v, ww_v, hh_v = bbox_visor_estatico
                crop_visor_rgb = frame_estabilizado_rgb[y_v:y_v + hh_v, x_v:x_v + ww_v]
                if crop_visor_rgb.size > 0:
                    crop_gray = cv2.cvtColor(crop_visor_rgb, cv2.COLOR_RGB2GRAY)
                    crop_resized = cv2.resize(crop_gray, (128, 32), interpolation=cv2.INTER_LINEAR)
                    buffer_ocr_inp[0, 0, :, :] = crop_resized / 255.0

                    pred_ocr = self.model_ocr.infer(buffer_ocr_inp)
                    pred_argmax = np.argmax(pred_ocr, axis=2).flatten()

                    texto_temp = ""
                    last = -1
                    for p in pred_argmax:
                        if p != last and p != 0 and (p - 1) < len(CHARS): texto_temp += CHARS[p - 1]
                        last = p
                    if len(texto_temp) > 6: texto_temp = texto_temp[:6]

                    if texto_temp.strip() and texto_temp != "...":
                        texto_ocr_cache = texto_temp
                        if not ocr_inicial_realizado: ocr_inicial_realizado = True
                        elif p_superior_cache == 0 and p_inferior_cache == 0.0: ocr_no_zero_realizado = True

            # --- ATUALIZAR INTERFACE (SEGURA COM THREADS) ---
            self.atualizar_ui_elementos(frame_estabilizado_rgb, texto_ocr_cache, p_superior_cache, p_inferior_cache)

        if self.cap:
            self.cap.release()

    def atualizar_ui_elementos(self, frame_rgb, ocr, p_sup, p_inf):
        frame_resized = cv2.resize(frame_rgb, (330, 330), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.after(0, self._set_labels_and_image, img_tk, ocr, p_sup, p_inf)

    def _set_labels_and_image(self, img_tk, ocr, p_sup, p_inf):
        self.video_canvas.configure(image=img_tk)
        self.video_canvas.image = img_tk  
        self.lbl_ocr.configure(text=f"OCR: {ocr}")
        self.lbl_ponteiros.configure(text=f"Ponteiros: {p_sup} | {p_inf:.1f}")


if __name__ == "__main__":
    app = AppMedidor()
    app.mainloop()