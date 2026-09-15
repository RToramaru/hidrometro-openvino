import socket
import time
import threading
import uvicorn
from fastapi import FastAPI, HTTPException

app = FastAPI(title="API de Telemetria de Relógios")

# Referência global para a aplicação gráfica Tkinter
app_instance = None
porta_ativa = 8000


def set_app_instance(instance):
    global app_instance
    app_instance = instance


def encontrar_porta_disponivel(porta_inicial=8000, max_tentativas=50):
    """Verifica se a porta está livre; se não estiver, incrementa até achar uma disponível."""
    for porta in range(porta_inicial, porta_inicial + max_tentativas):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", porta))
                return porta
            except OSError:
                continue
    raise RuntimeError("Não foi possível encontrar nenhuma porta livre para a API.")


def formatar_dados_instancia(inst):
    ocr_str = getattr(inst, 'texto_ocr_atual', '000000')
    p_sup = getattr(inst, 'p_superior_atual', 0)
    p_inf = getattr(inst, 'p_inferior_atual', 0.0)

    p_inf = int(p_inf * 10)  # Converte para inteiro multiplicando por 10
    valor_relogio = f"{ocr_str}.{p_sup}{p_inf}"

    return {
        "indiceCamera": inst.video_source,
        "valorRelogio": valor_relogio,
        "ocr": ocr_str,
        "ponteiroSuperior": p_sup,
        "ponteiroInferior": float(p_inf),
        "timestamp": time.time()
    }


@app.get("/relogios")
def listar_todos_relogios():
    if app_instance is None:
        raise HTTPException(status_code=503, detail="Aplicação Tkinter não inicializada.")

    relogios = []
    for inst in app_instance.instancias_ativas:
        if inst.running:
            relogios.append(formatar_dados_instancia(inst))

    return {
        "quantidade": len(relogios),
        "relogios": relogios
    }


@app.get("/relogios/{indice_camera}")
def obter_relogio_por_indice(indice_camera: str):
    if app_instance is None:
        raise HTTPException(status_code=503, detail="Aplicação Tkinter não inicializada.")

    if indice_camera.isdigit():
        indice_busca = int(indice_camera)
    else:
        indice_busca = indice_camera

    for inst in app_instance.instancias_ativas:
        if inst.running and inst.video_source == indice_busca:
            return formatar_dados_instancia(inst)

    raise HTTPException(
        status_code=404,
        detail=f"Câmera com o índice/fonte '{indice_camera}' não foi encontrada ou não está ativa."
    )


def iniciar_servidor_api(host="0.0.0.0", porta_inicial=8000):
    """Encontra uma porta livre e inicia o servidor FastAPI em background."""
    global porta_ativa
    porta_ativa = encontrar_porta_disponivel(porta_inicial)

    config = uvicorn.Config(
        app=app,
        host=host,
        port=porta_ativa,
        log_level="error",
        use_colors=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    url_local = f"http://localhost:{porta_ativa}/relogios"
    print(f"[API] Servidor rodando com sucesso em: {url_local}")
    return url_local