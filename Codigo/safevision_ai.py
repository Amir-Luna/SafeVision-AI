"""SafeVision AI: verificación de EPP con Teachable Machine, Keras y registro CSV.

Ejecución normal (modelos reales):
    python safevision_ai_final.py

Prueba de la interfaz sin modelos:
    python safevision_ai_final.py --prueba --sin-camara --sin-voz

Prueba por consola sin hardware:
    python safevision_ai_final.py --prueba --consola --sin-voz --actividad corte
"""
from __future__ import annotations

import os

# Debe configurarse antes de importar TensorFlow (se importa sólo en modo real).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import csv
import importlib
import math
from pathlib import Path
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


# ========================= CONFIGURACIÓN =========================
MODO_PRUEBA = False
# Herramienta de depuración. En la demostración final puede cambiarse a False.
MOSTRAR_REGIONES = True
CONFIDENCE_THRESHOLD = 0.80
STABILITY_TIME = 0.70
# La evaluación completa debe mantenerse sin cambios antes de guardarse como resultado final.
FINAL_RESULT_STABILITY_TIME = 2.0
SPEECH_COOLDOWN = 2.5
# Velocidad algo más natural para español.
VOICE_RATE = 150
# Preferencias de voz. Se prioriza español latinoamericano si está instalado.
VOICE_PREFERRED_LOCALES = ("es-bo", "es-mx", "es-us", "es-419", "es-es", "es")
VOICE_PREFERRED_NAMES = ("sabina", "helena", "pablo", "laura", "jorge", "alvaro", "spanish", "español")
CAMERA_INDEX = 0
IMG_SIZE = (224, 224)
# Si se interrumpen las observaciones, se exige estabilidad desde cero.
MAX_OBSERVATION_GAP = 2.0
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "modelos"
DATA_DIR = BASE_DIR / "datos"
REGISTROS_REALES_CSV = DATA_DIR / "registros_safevision.csv"
REGISTROS_PRUEBA_CSV = DATA_DIR / "registros_safevision_prueba.csv"


@dataclass(frozen=True)
class RegionConfig:
    x1: float
    y1: float
    x2: float
    y2: float
    nombre: str


# Parámetros iniciales: calibrar según cámara, distancia y encuadre del dataset.
# Los datasets de cada modelo deben utilizar un encuadre visual similar
# a la región que recibirá el modelo durante la inferencia.
# Ejemplo: guantes = torso + brazos + ambas manos, tanto en fotos como en "manos".
# x crece hacia la derecha; y hacia abajo. Las regiones pueden superponerse.
CONFIG_REGIONES = {
    "cabeza": RegionConfig(x1=0.25, y1=0.00, x2=0.75, y2=0.45, nombre="CABEZA"),
    "manos": RegionConfig(x1=0.10, y1=0.30, x2=0.90, y2=0.80, nombre="MANOS"),
    "vestimenta": RegionConfig(x1=0.20, y1=0.28, x2=0.80, y2=1.00, nombre="VESTIMENTA"),
}


@dataclass(frozen=True)
class UsuarioSesion:
    nombre_completo: str
    ci: str


@dataclass(frozen=True)
class ConfigModelo:
    nombre: str
    clase_presente: str
    clase_ausente: str
    frase_faltante: str
    carpeta: str
    region: str

    @property
    def ruta_modelo(self):
        # Archivo Keras exportado directamente desde Teachable Machine.
        return MODELS_DIR / self.carpeta / "keras_model.h5"

    @property
    def ruta_etiquetas(self):
        return MODELS_DIR / self.carpeta / "labels.txt"


CONFIG_MODELOS = {
    "mascara_soldadura": ConfigModelo(
        "Máscara de soldadura", "CON_MASCARA", "SIN_MASCARA",
        "la máscara de soldadura", "mascara_soldadura", region="cabeza"),
    "guantes_soldadura": ConfigModelo(
        "Guantes de soldadura", "CON_GUANTES", "SIN_GUANTES",
        "los guantes de soldadura", "guantes_soldadura", region="manos"),
    "gafas_corte": ConfigModelo(
        "Gafas de seguridad", "CON_GAFAS", "SIN_GAFAS",
        "las gafas de seguridad", "gafas_corte", region="cabeza"),
    "guantes_corte": ConfigModelo(
        "Guantes de corte", "CON_GUANTES", "SIN_GUANTES",
        "los guantes de corte", "guantes_corte", region="manos"),
    "vestimenta": ConfigModelo(
        "Vestimenta adecuada (camisa y pantalón jean)",
        "VESTIMENTA_ADECUADA", "VESTIMENTA_INADECUADA",
        "la vestimenta adecuada", "vestimenta", region="vestimenta"),
}

ACTIVIDADES = {
    "soldadura": ("mascara_soldadura", "guantes_soldadura", "vestimenta"),
    "corte": ("gafas_corte", "guantes_corte", "vestimenta"),
}

# True = presente, False = ausente, None = incierto (50 %).
# Para elegir una confianza concreta: (True, 0.57) o (False, 0.94).
# Son datos manuales; NO se generan números aleatorios.
RESULTADOS_PRUEBA = {
    "mascara_soldadura": True,
    "guantes_soldadura": False,
    "gafas_corte": True,
    "guantes_corte": False,
    "vestimenta": True,
}


class ErrorSafeVision(RuntimeError):
    """Error comprensible para el usuario."""


def importar(nombre, paquete=None):
    try:
        return importlib.import_module(nombre)
    except Exception as exc:
        raise ErrorSafeVision(
            f"No se pudo importar {nombre}. Instala/verifica {paquete or nombre}. "
            f"Detalle: {exc}") from exc


def ids_actividad(actividad):
    if actividad not in ACTIVIDADES:
        raise ErrorSafeVision(f"Actividad desconocida: {actividad}.")
    ids = ACTIVIDADES[actividad]
    if not ids or len(set(ids)) != len(ids):
        raise ErrorSafeVision(f"Lista de EPP vacía o duplicada en {actividad}.")
    for identificador in ids:
        if identificador not in CONFIG_MODELOS:
            raise ErrorSafeVision(f"Falta CONFIG_MODELOS[{identificador!r}].")
    return ids


# ============================= REGISTRO CSV =============================
CAMPOS_REGISTRO = (
    "fecha_hora", "nombre_completo", "ci", "trabajo", "resultado", "faltantes"
)


def validar_usuario(nombre_completo, ci):
    nombre = " ".join(str(nombre_completo or "").strip().split())
    ci_limpio = str(ci or "").strip()
    if len(nombre) < 3:
        raise ErrorSafeVision("El nombre completo no puede estar vacío.")
    if not ci_limpio:
        raise ErrorSafeVision("El número de carnet de identidad no puede estar vacío.")
    if len(ci_limpio) > 32 or any(c in ci_limpio for c in "\r\n\t"):
        raise ErrorSafeVision("El carnet de identidad contiene caracteres no válidos.")
    return UsuarioSesion(nombre, ci_limpio)


def registrar_usuario_consola(nombre_inicial=None, ci_inicial=None):
    """Registro temporal por consola; la GUI definitiva se implementará después."""
    nombre = nombre_inicial
    ci = ci_inicial
    while nombre is None or not str(nombre).strip():
        try:
            nombre = input("Nombre completo: ").strip()
        except EOFError as exc:
            raise ErrorSafeVision(
                "No hay entrada para el nombre. Usa --nombre y --ci al ejecutar el programa."
            ) from exc
        if not nombre:
            print("El nombre no puede estar vacío.")
            nombre = None
    while ci is None or not str(ci).strip():
        try:
            ci = input("Carnet de identidad (CI): ").strip()
        except EOFError as exc:
            raise ErrorSafeVision(
                "No hay entrada para el CI. Usa --nombre y --ci al ejecutar el programa."
            ) from exc
        if not ci:
            print("El CI no puede estar vacío.")
            ci = None
    return validar_usuario(nombre, ci)


class RegistroCSV:
    """Almacena verificaciones en un CSV estructurado y compatible con futuras migraciones."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        self._asegurar_encabezado()

    def _asegurar_encabezado(self):
        if not self.ruta.exists() or self.ruta.stat().st_size == 0:
            try:
                with self.ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
                    csv.DictWriter(archivo, fieldnames=CAMPOS_REGISTRO).writeheader()
            except OSError as exc:
                raise ErrorSafeVision(f"No se pudo crear el archivo de registros: {self.ruta}") from exc
            return
        try:
            with self.ruta.open("r", encoding="utf-8-sig", newline="") as archivo:
                lector = csv.reader(archivo)
                encabezado = next(lector, None)
        except OSError as exc:
            raise ErrorSafeVision(f"No se pudo leer el archivo de registros: {self.ruta}") from exc
        if tuple(encabezado or ()) != CAMPOS_REGISTRO:
            raise ErrorSafeVision(
                f"El CSV existente tiene un encabezado incompatible: {self.ruta}. "
                f"Se espera: {', '.join(CAMPOS_REGISTRO)}"
            )

    def guardar(self, usuario, evaluacion):
        if evaluacion.estado not in ("EPP COMPLETO", "EPP INCOMPLETO"):
            return False
        faltantes = "; ".join(CONFIG_MODELOS[i].nombre for i in evaluacion.faltantes)
        fila = {
            "fecha_hora": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z"),
            "nombre_completo": usuario.nombre_completo,
            "ci": usuario.ci,
            "trabajo": evaluacion.actividad.capitalize(),
            "resultado": evaluacion.estado,
            "faltantes": faltantes,
        }
        try:
            with self.ruta.open("a", encoding="utf-8", newline="") as archivo:
                csv.DictWriter(archivo, fieldnames=CAMPOS_REGISTRO).writerow(fila)
        except OSError as exc:
            raise ErrorSafeVision(f"No se pudo guardar el registro en: {self.ruta}") from exc
        return True


class SesionRegistro:
    """Garantiza una sola fila CSV por cada verificación iniciada por el usuario."""

    def __init__(self, usuario, registro_csv):
        self.usuario = usuario
        self.registro_csv = registro_csv
        self.resultado_guardado = False

    def iniciar_verificacion(self):
        self.resultado_guardado = False

    def guardar_resultado_final(self, evaluacion):
        if self.resultado_guardado:
            return False
        if evaluacion.estado not in ("EPP COMPLETO", "EPP INCOMPLETO"):
            return False
        guardado = self.registro_csv.guardar(self.usuario, evaluacion)
        if guardado:
            self.resultado_guardado = True
        return guardado


# ============================= MODELOS =============================
def cargar_etiquetas(ruta):
    """Acepta líneas '0 CON_MASCARA' o 'CON_MASCARA', sin cambiar su significado."""
    try:
        lineas = [s.strip() for s in Path(ruta).read_text(encoding="utf-8-sig").splitlines()
                  if s.strip()]
    except OSError as exc:
        raise ErrorSafeVision(f"No se pudo leer labels.txt. Ruta esperada: {ruta}") from exc
    if not lineas:
        raise ErrorSafeVision(f"labels.txt está vacío: {ruta}")
    numeradas = [re.fullmatch(r"(\d+)\s+(.+)", s) for s in lineas]
    if any(numeradas):
        if not all(numeradas):
            raise ErrorSafeVision(f"Etiquetas con índices mezcladas con etiquetas sin índice: {ruta}")
        pares = [(int(m.group(1)), m.group(2).strip()) for m in numeradas]
        if sorted(i for i, _ in pares) != list(range(len(pares))):
            raise ErrorSafeVision(f"Los índices de labels.txt deben ser únicos y comenzar en 0: {ruta}")
        etiquetas = [nombre for _, nombre in sorted(pares)]
    else:
        etiquetas = lineas
    if len(set(etiquetas)) != len(etiquetas):
        raise ErrorSafeVision(f"Hay etiquetas duplicadas: {ruta}")
    return etiquetas


@dataclass(frozen=True)
class Prediccion:
    clase: str
    confianza: float


class ModeloEPP:
    """Un modelo Keras .h5 por elemento; las salidas siguen el orden de labels.txt."""

    def __init__(self, identificador, config):
        self.identificador = identificador
        self.config = config
        self.ruta_modelo = config.ruta_modelo
        self.ruta_etiquetas = config.ruta_etiquetas
        if not self.ruta_modelo.is_file():
            raise ErrorSafeVision(
                f"No se encontró el modelo Keras de {config.nombre.lower()}.\n"
                f"Ruta esperada: {self.ruta_modelo}\n"
                "Coloca aquí el archivo keras_model.h5 exportado desde Teachable Machine.")
        self.etiquetas = cargar_etiquetas(self.ruta_etiquetas)
        esperadas = {config.clase_presente, config.clase_ausente}
        if len(esperadas) != 2 or len(self.etiquetas) != 2 or set(self.etiquetas) != esperadas:
            raise ErrorSafeVision(
                f"Etiquetas incorrectas en {config.nombre}: {self.etiquetas}. "
                f"Se esperan exactamente {sorted(esperadas)}.")
        self.tf = importar("tensorflow")
        try:
            self.modelo = self.tf.keras.models.load_model(str(self.ruta_modelo), compile=False)
            self._validar_estructura()
        except (OSError, ValueError, TypeError, ImportError, KeyError, self.tf.errors.OpError) as exc:
            detalle = str(exc)
            ayuda = ""
            if "DepthwiseConv2D" in detalle and "groups" in detalle:
                ayuda = (
                    "\nLa configuración de DepthwiseConv2D de esta exportación no es "
                    "compatible con el cargador instalado. Prueba un entorno separado "
                    "con Python 3.11 y tensorflow==2.15.1 (Keras 2.15). "
                    "No es necesario modificar el archivo .h5.")
            raise ErrorSafeVision(
                f"No se pudo cargar el modelo Keras de {config.nombre.lower()}.\n"
                f"Ruta: {self.ruta_modelo}\n"
                f"TensorFlow: {self.tf.__version__}\nDetalle: {detalle}{ayuda}") from exc
        print(f"Modelo Keras: {identificador} | archivo: {self.ruta_modelo.name} | etiquetas: {self.etiquetas}")

    def _validar_estructura(self):
        if len(self.modelo.inputs) != 1 or len(self.modelo.outputs) != 1:
            raise ValueError("Se requiere un modelo Keras con una entrada y una salida.")
        entrada = self.modelo.inputs[0]
        if self.tf.as_dtype(entrada.dtype) != self.tf.float32:
            raise ValueError("La entrada del modelo debe ser float32.")
        forma = tuple(entrada.shape)
        if len(forma) != 4 or any(a not in (None, b) for a, b in zip(forma, (1, 224, 224, 3))):
            raise ValueError(f"Entrada incompatible: {forma}; se espera (1,224,224,3).")
        salida = tuple(self.modelo.outputs[0].shape)
        if len(salida) != 2 or salida[0] not in (None, 1) or salida[1] != 2:
            raise ValueError(f"Salida incompatible: {salida}; se esperan dos clases, (1,2).")

    def predecir(self, tensor):
        np = importar("numpy")
        try:
            # Llamada directa: un solo batch por frame, sin el bucle de predict().
            resultado = self.modelo(tensor, training=False)
            if not self.tf.is_tensor(resultado):
                raise ValueError("El modelo debe devolver un único tensor de probabilidades.")
            valores = np.asarray(resultado.numpy())
            if valores.shape != (1, 2):
                raise ValueError(f"Salida {valores.shape}; se espera (1,2).")
            if (not np.isfinite(valores).all() or (valores < 0).any() or (valores > 1).any()
                    or not np.isclose(valores.sum(), 1.0, atol=0.01)):
                raise ValueError("La salida no contiene dos probabilidades válidas cuya suma sea 1.")
            indice = int(np.argmax(valores[0]))
            return Prediccion(self.etiquetas[indice], float(valores[0, indice]))
        except (ValueError, TypeError, self.tf.errors.OpError) as exc:
            raise ErrorSafeVision(f"Error de inferencia en {self.config.nombre}: {exc}") from exc


class ModeloSimulado:
    def __init__(self, identificador, config, resultados):
        self.identificador = identificador
        self.config = config
        self.resultados = resultados

    def predecir(self, tensor=None):
        if self.identificador not in self.resultados:
            raise ErrorSafeVision(f"Falta resultado de prueba para {self.identificador}.")
        valor = self.resultados[self.identificador]
        confianza = 0.96
        if isinstance(valor, tuple) and len(valor) == 2:
            valor, confianza = valor
        if valor is None:
            valor, confianza = True, 0.50
        if type(valor) is not bool:
            raise ErrorSafeVision(f"Resultado de prueba inválido para {self.identificador}.")
        try:
            confianza = float(confianza)
            if not math.isfinite(confianza) or not 0.5 <= confianza <= 1:
                raise ValueError()
        except (TypeError, ValueError) as exc:
            raise ErrorSafeVision("La confianza de la clase ganadora binaria debe estar entre 0.5 y 1.") from exc
        clase = self.config.clase_presente if valor else self.config.clase_ausente
        return Prediccion(clase, confianza)


# ========================= REGIONES Y PREPROCESAMIENTO =========================
def validar_region(region_id):
    if not isinstance(region_id, str) or not region_id:
        raise ErrorSafeVision("El identificador de región debe ser un texto no vacío.")
    region = CONFIG_REGIONES.get(region_id)
    if not isinstance(region, RegionConfig):
        raise ErrorSafeVision(
            f"No se pudo obtener la región {region_id!r}. "
            "Debe existir como RegionConfig en CONFIG_REGIONES.")
    valores = (region.x1, region.y1, region.x2, region.y2)
    if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in valores)
            or not (0 <= region.x1 < region.x2 <= 1 and 0 <= region.y1 < region.y2 <= 1)):
        raise ErrorSafeVision(
            f"Región {region_id!r} mal configurada: {valores}. "
            "Comprueba CONFIG_REGIONES: se requiere 0 <= x1 < x2 <= 1 y 0 <= y1 < y2 <= 1.")
    if not isinstance(region.nombre, str) or not region.nombre.strip():
        raise ErrorSafeVision(f"La región {region_id!r} debe tener un nombre visible.")
    return region


def regiones_de_modelos(identificadores):
    """Regiones únicas de los modelos activos, conservando el orden."""
    regiones = []
    for identificador in identificadores:
        if identificador not in CONFIG_MODELOS:
            raise ErrorSafeVision(f"Falta CONFIG_MODELOS[{identificador!r}].")
        region_id = CONFIG_MODELOS[identificador].region
        validar_region(region_id)
        if region_id not in regiones:
            regiones.append(region_id)
    return tuple(regiones)


def coordenadas_region(frame, region_id):
    """Límites en píxeles: x2/y2 son exclusivos, como en los slices de NumPy."""
    region = validar_region(region_id)
    if (frame is None or getattr(frame, "ndim", None) != 3
            or frame.shape[2] != 3 or frame.size == 0):
        raise ErrorSafeVision(
            f"No se pudo obtener la región {region_id!r}: el frame BGR está vacío o es inválido.")
    alto, ancho = frame.shape[:2]
    x1, y1 = int(region.x1 * ancho), int(region.y1 * alto)
    x2, y2 = int(region.x2 * ancho), int(region.y2 * alto)
    if x1 >= x2 or y1 >= y2:
        raise ErrorSafeVision(
            f"No se pudo obtener la región {region_id!r}: el recorte queda vacío "
            f"con la resolución {ancho}x{alto}. Comprueba CONFIG_REGIONES.")
    return x1, y1, x2, y2


def recortar_region(frame, region_id):
    """Recorta una región BGR sin modificar el frame completo."""
    x1, y1, x2, y2 = coordenadas_region(frame, region_id)
    recorte = frame[y1:y2, x1:x2]
    if recorte.size == 0:
        raise ErrorSafeVision(
            f"No se pudo obtener la región {region_id!r}. Comprueba CONFIG_REGIONES.")
    return recorte.copy()


def dibujar_regiones(frame, region_ids):
    """Devuelve el frame COMPLETO anotado; nunca dibuja en el original."""
    cv2 = importar("cv2", "opencv-python")
    copia = frame.copy()
    for region_id in dict.fromkeys(region_ids):
        x1, y1, x2, y2 = coordenadas_region(frame, region_id)
        cv2.rectangle(copia, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
        # Desplazamientos de texto, no límites de las regiones.
        origen = (x1 + 4, min(y1 + 20, copia.shape[0] - 1))
        cv2.putText(copia, CONFIG_REGIONES[region_id].nombre, origen,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return copia


def preprocesar_imagen(frame):
    """Mismo preprocesamiento de Teachable Machine, aplicado a un recorte BGR."""
    cv2 = importar("cv2", "opencv-python")
    np = importar("numpy")
    if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ErrorSafeVision("La cámara no proporcionó una imagen BGR válida.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    imagen = np.asarray(imagen, dtype=np.float32)
    imagen = (imagen / 127.5) - 1.0
    return np.expand_dims(imagen, axis=0)


# ============================ ESTABILIZACIÓN ============================
@dataclass(frozen=True)
class ResultadoElemento:
    identificador: str
    nombre: str
    clase_detectada: Optional[str]
    confianza: float
    valido: bool
    estable: bool
    presente: Optional[bool]
    detalle: str = ""


class Estabilizador:
    def __init__(self, umbral=CONFIDENCE_THRESHOLD, duracion=STABILITY_TIME):
        self.umbral = umbral
        self.duracion = duracion
        self.reiniciar()

    def reiniciar(self):
        self.candidata = None
        self.inicio_candidata = None
        self.prediccion_estable = None
        self.confianza = 0.0
        self.ultima_observacion = None
        self.estable = False

    def actualizar(self, prediccion, ahora):
        if (self.ultima_observacion is not None and
                (ahora < self.ultima_observacion or
                 ahora - self.ultima_observacion > MAX_OBSERVATION_GAP)):
            self.reiniciar()
        self.ultima_observacion = ahora
        self.confianza = prediccion.confianza
        valido = math.isfinite(self.confianza) and self.umbral <= self.confianza <= 1
        if not valido:
            self.reiniciar()
            self.confianza = prediccion.confianza
            return False, False
        if prediccion.clase != self.candidata:
            self.candidata = prediccion.clase
            self.inicio_candidata = ahora
            self.prediccion_estable = None  # No arrastrar el resultado anterior.
            self.estable = False
        self.estable = ahora - self.inicio_candidata >= self.duracion
        self.prediccion_estable = self.candidata if self.estable else None
        return True, self.estable


class EstabilizadorEvaluacion:
    """Estabiliza la combinación final de EPP antes de cerrar una verificación."""

    def __init__(self, duracion=FINAL_RESULT_STABILITY_TIME):
        self.duracion = duracion
        self.reiniciar()

    def reiniciar(self):
        self.clave_candidata = None
        self.inicio_candidata = None

    def actualizar(self, evaluacion, ahora=None):
        ahora = time.monotonic() if ahora is None else ahora
        if evaluacion.estado not in ("EPP COMPLETO", "EPP INCOMPLETO"):
            self.reiniciar()
            return False

        clave = (evaluacion.estado, evaluacion.faltantes)
        if clave != self.clave_candidata:
            self.clave_candidata = clave
            self.inicio_candidata = ahora
            return False

        if self.inicio_candidata is None:
            self.inicio_candidata = ahora
            return False

        return ahora - self.inicio_candidata >= self.duracion


# ============================== INFERENCIA ==============================
class SistemaEPP:
    """La futura GUI sólo necesita seleccionar_actividad() y procesar_frame()."""

    def __init__(self, actividad, modo_prueba=MODO_PRUEBA, resultados_prueba=None):
        self.modo_prueba = modo_prueba
        self.resultados_prueba = dict(RESULTADOS_PRUEBA if resultados_prueba is None else resultados_prueba)
        self.modelos = {}
        self.estabilizadores = {}
        self.seleccionar_actividad(actividad)

    def seleccionar_actividad(self, actividad):
        ids = ids_actividad(actividad)
        if not self.modo_prueba:
            regiones_de_modelos(ids)  # Validar sólo las regiones necesarias antes de cargar H5.
        modelos = {}
        for identificador in ids:
            config = CONFIG_MODELOS[identificador]
            if identificador in self.modelos:
                modelos[identificador] = self.modelos[identificador]
            elif self.modo_prueba:
                modelos[identificador] = ModeloSimulado(identificador, config, self.resultados_prueba)
            else:
                modelos[identificador] = ModeloEPP(identificador, config)
        # Se conservan sólo los tres modelos activos; vestimenta puede reutilizarse.
        self.actividad = actividad
        self.modelos = modelos
        self.estabilizadores = {i: Estabilizador() for i in ids}

    def invalidar(self, motivo):
        for estabilizador in self.estabilizadores.values():
            estabilizador.reiniciar()
        return {i: ResultadoElemento(i, CONFIG_MODELOS[i].nombre, None, 0.0,
                                    False, False, None, motivo) for i in self.modelos}

    def procesar_frame(self, frame=None, ahora=None):
        # Caché local: nunca reutilizar una imagen ni un tensor del frame anterior.
        tensores_regiones = {}
        if not self.modo_prueba:
            tf = next(iter(self.modelos.values())).tf
            try:
                for region_id in regiones_de_modelos(self.modelos):
                    recorte = recortar_region(frame, region_id)
                    imagen = preprocesar_imagen(recorte)
                    tensores_regiones[region_id] = tf.convert_to_tensor(imagen, dtype=tf.float32)
            except ErrorSafeVision as exc:
                self.invalidar(str(exc))
                raise
        ahora = time.monotonic() if ahora is None else ahora
        resultados = {}
        for identificador, modelo in self.modelos.items():
            config = CONFIG_MODELOS[identificador]
            try:
                tensor = None if self.modo_prueba else tensores_regiones[config.region]
                prediccion = modelo.predecir(tensor)
                if prediccion.clase not in (config.clase_presente, config.clase_ausente):
                    raise ErrorSafeVision(f"Clase inesperada: {prediccion.clase}")
                valido, estable = self.estabilizadores[identificador].actualizar(prediccion, ahora)
                presente = prediccion.clase == config.clase_presente if valido and estable else None
                resultados[identificador] = ResultadoElemento(
                    identificador, config.nombre, prediccion.clase, prediccion.confianza,
                    valido, estable, presente)
            except ErrorSafeVision as exc:
                self.estabilizadores[identificador].reiniciar()
                resultados[identificador] = ResultadoElemento(
                    identificador, config.nombre, None, 0, False, False, None, str(exc))
        return resultados


# ============================= LÓGICA DEL EPP =============================
@dataclass(frozen=True)
class EvaluacionEPP:
    actividad: str
    estado: str
    faltantes: tuple[str, ...]
    no_verificados: tuple[str, ...]

    @property
    def clave_voz(self):
        return (self.actividad, self.estado, self.faltantes)


def evaluar_epp(actividad, resultados):
    pendientes, faltantes = [], []
    for identificador in ids_actividad(actividad):
        resultado = resultados.get(identificador)
        config = CONFIG_MODELOS[identificador]
        if (resultado is None or resultado.identificador != identificador or not resultado.valido
                or not resultado.estable or not math.isfinite(resultado.confianza)
                or not CONFIDENCE_THRESHOLD <= resultado.confianza <= 1
                or resultado.presente is None
                or resultado.clase_detectada not in (config.clase_presente, config.clase_ausente)
                or resultado.presente != (resultado.clase_detectada == config.clase_presente)):
            pendientes.append(identificador)
        elif not resultado.presente:
            faltantes.append(identificador)
    if pendientes:
        # La incertidumbre tiene prioridad; no declarar incompleto ni completo.
        return EvaluacionEPP(actividad, "VERIFICANDO", (), tuple(pendientes))
    estado = "EPP INCOMPLETO" if faltantes else "EPP COMPLETO"
    return EvaluacionEPP(actividad, estado, tuple(faltantes), ())


def unir_lista(elementos):
    return elementos[0] if len(elementos) == 1 else ", ".join(elementos[:-1]) + " y " + elementos[-1]


def frase_evaluacion(evaluacion):
    """Texto pensado para voz; evita pronunciar la sigla EPP de forma poco natural."""
    if evaluacion.estado == "VERIFICANDO":
        return None
    if not evaluacion.faltantes:
        return f"Equipo de protección personal completo para {evaluacion.actividad}."
    nombres = [CONFIG_MODELOS[i].frase_faltante for i in evaluacion.faltantes]
    plural = len(nombres) > 1 or nombres[0].startswith(("los ", "las "))
    verbo = "Faltan" if plural else "Falta"
    return (f"Equipo de protección personal incompleto para {evaluacion.actividad}. "
            f"{verbo} {unir_lista(nombres)}.")


def texto_resultados(evaluacion, resultados):
    lineas = [f"Actividad: {evaluacion.actividad.upper()}"]
    for i in ids_actividad(evaluacion.actividad):
        r = resultados.get(i)
        if r is None:
            lineas.append(f"{CONFIG_MODELOS[i].nombre}: SIN RESULTADO")
            continue
        estado = "INCIERTO" if not r.valido else "ANALIZANDO"
        if r.valido and r.estable:
            estado = "OK" if r.presente else "FALTA"
        lineas.append(f"{r.nombre}: {estado} | {r.clase_detectada or '--'} | {r.confianza:.1%}")
        if r.detalle:
            lineas.append(f"  {r.detalle}")
    lineas.append(evaluacion.estado + (" EPP..." if evaluacion.estado == "VERIFICANDO" else ""))
    if evaluacion.no_verificados:
        lineas.append("No se pudo verificar: " + ", ".join(CONFIG_MODELOS[i].nombre for i in evaluacion.no_verificados))
    elif evaluacion.faltantes:
        lineas.append("Faltantes: " + ", ".join(CONFIG_MODELOS[i].nombre for i in evaluacion.faltantes))
    return "\n".join(lineas)


# ============================= SISTEMA DE VOZ =============================
class SistemaVoz:
    """
    Sistema de voz en un único hilo.

    Cada verificación tiene una generación propia. Esto permite volver a
    anunciar exactamente el mismo resultado al pulsar REPETIR VERIFICACIÓN
    sin confundirlo con el anuncio de la verificación anterior.
    """

    def __init__(self, habilitado=True):
        self.habilitado = habilitado
        self.cola = queue.Queue(maxsize=1)
        self.cerrado = threading.Event()
        self.lock = threading.Lock()
        self.clave_actual = None
        self.ultima_anunciada = None
        self.ultimo_anuncio = float("-inf")
        self.generacion = 0
        self.hilo = None
        if habilitado:
            self.hilo = threading.Thread(
                target=self._worker, name="SafeVisionVoz", daemon=True)
            self.hilo.start()

    def _vaciar_cola(self):
        while True:
            try:
                self.cola.get_nowait()
                self.cola.task_done()
            except queue.Empty:
                return

    def cancelar_pendientes(self):
        with self.lock:
            self.clave_actual = None
            self._vaciar_cola()

    def actualizar(self, evaluacion):
        if not self.habilitado or self.cerrado.is_set():
            return
        frase = frase_evaluacion(evaluacion)
        if frase is None:
            self.cancelar_pendientes()
            return

        # La generación forma parte de la clave. Dos verificaciones diferentes
        # pueden tener el mismo resultado y aun así se anuncian por separado.
        clave = (self.generacion,) + tuple(evaluacion.clave_voz)
        with self.lock:
            if clave == self.clave_actual:
                return
            self.clave_actual = clave
            self._vaciar_cola()
            if clave != self.ultima_anunciada:
                self.cola.put_nowait((clave, frase))

    def reiniciar_anuncios(self):
        """Inicia una nueva generación de anuncios para una nueva verificación."""
        with self.lock:
            self.generacion += 1
            self.clave_actual = None
            self.ultima_anunciada = None
            self.ultimo_anuncio = float("-inf")
            self._vaciar_cola()

    @staticmethod
    def _texto_voz(voz):
        partes = [str(getattr(voz, "name", "")),
                  str(getattr(voz, "id", "")),
                  str(getattr(voz, "languages", ""))]
        return " ".join(partes).lower()

    @classmethod
    def _seleccionar_voz_espanol(cls, engine):
        """Devuelve la mejor voz española instalada, priorizando Latinoamérica."""
        voces = list(engine.getProperty("voices") or [])
        mejor = None
        mejor_puntuacion = -1

        for voz in voces:
            info = cls._texto_voz(voz)
            puntuacion = 0

            # Prioridad de locale: Bolivia > México/LatAm > España > español genérico.
            for indice, locale in enumerate(VOICE_PREFERRED_LOCALES):
                if locale in info:
                    puntuacion = max(puntuacion, 120 - indice * 10)

            if "spanish" in info or "español" in info:
                puntuacion = max(puntuacion, 60)

            for nombre in VOICE_PREFERRED_NAMES:
                if nombre in info:
                    puntuacion += 15

            if puntuacion > mejor_puntuacion:
                mejor = voz
                mejor_puntuacion = puntuacion

        # Si ninguna voz parece española, no forzar la primera voz inglesa.
        if mejor is None or mejor_puntuacion <= 0:
            return None
        return mejor

    @classmethod
    def listar_voces_instaladas(cls):
        """Imprime las voces que pyttsx3 puede usar en este equipo."""
        pyttsx3 = importar("pyttsx3")
        engine = pyttsx3.init("sapi5") if os.name == "nt" else pyttsx3.init()
        try:
            voces = list(engine.getProperty("voices") or [])
            if not voces:
                print("No se encontraron voces instaladas.")
                return
            print("Voces disponibles para SafeVision AI:")
            for indice, voz in enumerate(voces, 1):
                print(f"{indice}. Nombre: {getattr(voz, 'name', '')}")
                print(f"   ID: {getattr(voz, 'id', '')}")
                print(f"   Idiomas: {getattr(voz, 'languages', '')}")
        finally:
            try:
                engine.stop()
            except Exception:
                pass

    def _worker(self):
        engine, com = None, None
        try:
            if os.name == "nt":
                com = importar("pythoncom", "pywin32")
                com.CoInitialize()
            pyttsx3 = importar("pyttsx3")
            engine = pyttsx3.init("sapi5") if os.name == "nt" else pyttsx3.init()
            engine.setProperty("rate", VOICE_RATE)
            engine.setProperty("volume", 1.0)

            voz_es = self._seleccionar_voz_espanol(engine)
            if voz_es is not None:
                engine.setProperty("voice", voz_es.id)
                print(f"VOZ SELECCIONADA: {getattr(voz_es, 'name', voz_es.id)}")
            else:
                print(
                    "AVISO: no se encontró una voz española instalada. "
                    "pyttsx3 usará la voz predeterminada de Windows, por lo que el español "
                    "puede sonar con acento inglés. Ejecuta --listar-voces para comprobarlas.")

            while not self.cerrado.is_set():
                try:
                    clave, frase = self.cola.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    self._pronunciar_si_vigente(engine, clave, frase)
                finally:
                    self.cola.task_done()
        except Exception as exc:
            self.habilitado = False
            self.cancelar_pendientes()
            print(f"AVISO: voz desactivada; la verificación continúa. Detalle: {exc}")
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            if com is not None:
                com.CoUninitialize()

    def _pronunciar_si_vigente(self, engine, clave, frase):
        while not self.cerrado.is_set():
            with self.lock:
                if clave != self.clave_actual or clave == self.ultima_anunciada:
                    return
                listo = time.monotonic() - self.ultimo_anuncio >= SPEECH_COOLDOWN
                if listo:
                    self.ultima_anunciada = clave
                    self.ultimo_anuncio = time.monotonic()
            if listo:
                print(f"VOZ: {frase}")
                engine.say(frase)
                engine.runAndWait()
                return
            self.cerrado.wait(0.05)

    def cerrar(self):
        self.cerrado.set()
        self.cancelar_pendientes()
        if self.hilo is not None:
            self.hilo.join(timeout=1.5)


# ================================= CÁMARA =================================
class Camara:
    def __init__(self, indice=CAMERA_INDEX):
        self.cv2 = importar("cv2", "opencv-python")
        self.captura = None
        backends = [self.cv2.CAP_DSHOW, self.cv2.CAP_ANY] if os.name == "nt" else [self.cv2.CAP_ANY]
        for backend in backends:
            self.captura = self.cv2.VideoCapture(indice, backend)
            if self.captura.isOpened():
                break
            self.captura.release()
        if self.captura is None or not self.captura.isOpened():
            raise ErrorSafeVision(
                f"Cámara {indice} no disponible. Comprueba conexión, permisos y otras aplicaciones. "
                "Para probar sin cámara usa --consola o --sin-camara en modo prueba.")
        self.captura.set(self.cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.captura.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def leer(self):
        correcto, frame = self.captura.read()
        if not correcto or frame is None or frame.size == 0:
            raise ErrorSafeVision("Se perdió la imagen de la cámara. Verificación suspendida.")
        return frame

    def cerrar(self):
        if self.captura is not None:
            self.captura.release()


# =========================== INTERFAZ FINAL ===========================
class AppSafeVision:
    ESTADO_REGISTRO = "REGISTRO"
    ESTADO_SELECCION = "SELECCION"
    ESTADO_PREPARADO = "PREPARADO"
    ESTADO_VERIFICANDO = "VERIFICANDO"
    ESTADO_RESULTADO = "RESULTADO"

    def __init__(self, window, modo_prueba=False, camara_indice=CAMERA_INDEX,
                 habilitar_camara=True, habilitar_voz=True,
                 nombre_inicial=None, ci_inicial=None):
        self.tk = importar("tkinter", "Tkinter de la instalación de Python")
        self.messagebox = importar("tkinter.messagebox", "Tkinter de la instalación de Python")
        self.Image = importar("PIL.Image", "Pillow")
        self.ImageTk = importar("PIL.ImageTk", "Pillow")

        self.window = window
        self.modo_prueba = modo_prueba
        self.camara_indice = camara_indice
        self.habilitar_camara = habilitar_camara
        self.voz = SistemaVoz(habilitar_voz)
        self.ruta_csv = REGISTROS_PRUEBA_CSV if modo_prueba else REGISTROS_REALES_CSV
        self.registro_csv = RegistroCSV(self.ruta_csv)

        self.usuario = None
        self.sesion_registro = None
        self.sistema = None
        self.camara = None
        self.actividad = None
        self.estado = self.ESTADO_REGISTRO
        self.estabilizador_evaluacion = EstabilizadorEvaluacion()
        self.resultados_actuales = {}
        self.evaluacion_final = None
        self.cerrada = False
        self.after_id = None
        self.photo = None
        self.labels_epp = {}
        self.nombre_inicial = nombre_inicial or ""
        self.ci_inicial = ci_inicial or ""

        self.window.title("SafeVision AI — Verificación de EPP")
        self.window.geometry("1020x760")
        self.window.minsize(900, 680)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.bind("<Escape>", lambda _event: self.on_closing())

        self.contenedor = self.tk.Frame(self.window, padx=20, pady=18)
        self.contenedor.pack(fill="both", expand=True)

        self.mostrar_registro()
        self.after_id = self.window.after(30, self.update)

    # ---------- Utilidades de pantallas ----------
    def limpiar_pantalla(self):
        for widget in self.contenedor.winfo_children():
            widget.destroy()
        self.labels_epp = {}

    def crear_encabezado(self, subtitulo=None):
        self.tk.Label(
            self.contenedor,
            text="SafeVision AI",
            font=("Arial", 24, "bold")
        ).pack(pady=(4, 2))
        if subtitulo:
            self.tk.Label(
                self.contenedor,
                text=subtitulo,
                font=("Arial", 11)
            ).pack(pady=(0, 18))

    def _cerrar_camara(self):
        if self.camara is not None:
            self.camara.cerrar()
            self.camara = None

    def _abrir_camara(self):
        if not self.habilitar_camara:
            return
        if self.camara is None:
            self.camara = Camara(self.camara_indice)

    # ---------- Pantalla 1: registro ----------
    def mostrar_registro(self):
        self.estado = self.ESTADO_REGISTRO
        self._cerrar_camara()
        self.actividad = None
        self.sistema = None
        self.resultados_actuales = {}
        self.evaluacion_final = None
        self.voz.cancelar_pendientes()
        self.limpiar_pantalla()
        self.crear_encabezado("Registro de usuario")

        tarjeta = self.tk.Frame(self.contenedor, bd=1, relief="solid", padx=30, pady=25)
        tarjeta.pack(pady=35)

        self.tk.Label(tarjeta, text="Nombre completo", font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 5))
        self.entry_nombre = self.tk.Entry(tarjeta, width=42, font=("Arial", 12))
        self.entry_nombre.grid(row=1, column=0, pady=(0, 18), ipady=6)
        self.entry_nombre.insert(0, self.nombre_inicial)

        self.tk.Label(tarjeta, text="Carnet de identidad (CI)", font=("Arial", 11, "bold")).grid(
            row=2, column=0, sticky="w", pady=(0, 5))
        self.entry_ci = self.tk.Entry(tarjeta, width=42, font=("Arial", 12))
        self.entry_ci.grid(row=3, column=0, pady=(0, 18), ipady=6)
        self.entry_ci.insert(0, self.ci_inicial)

        self.lbl_error_registro = self.tk.Label(tarjeta, text="", fg="firebrick", font=("Arial", 10))
        self.lbl_error_registro.grid(row=4, column=0, pady=(0, 12))

        self.tk.Button(
            tarjeta,
            text="CONTINUAR",
            width=24,
            font=("Arial", 11, "bold"),
            command=self.confirmar_usuario
        ).grid(row=5, column=0, pady=4, ipady=7)

        self.tk.Label(
            self.contenedor,
            text="Los datos se almacenan localmente junto con el resultado de cada verificación.",
            font=("Arial", 9)
        ).pack(pady=8)

    def confirmar_usuario(self):
        try:
            usuario = validar_usuario(self.entry_nombre.get(), self.entry_ci.get())
        except ErrorSafeVision as exc:
            self.lbl_error_registro.config(text=str(exc))
            return
        self.usuario = usuario
        self.sesion_registro = SesionRegistro(usuario, self.registro_csv)
        self.mostrar_seleccion_actividad()

    # ---------- Pantalla 2: selección ----------
    def mostrar_seleccion_actividad(self):
        self.estado = self.ESTADO_SELECCION
        self._cerrar_camara()
        self.voz.cancelar_pendientes()
        self.limpiar_pantalla()
        self.crear_encabezado("Seleccione la actividad que realizará")

        if self.usuario is not None:
            self.tk.Label(
                self.contenedor,
                text=f"Usuario: {self.usuario.nombre_completo}   |   CI: {self.usuario.ci}",
                font=("Arial", 11, "bold")
            ).pack(pady=(0, 20))

        botones = self.tk.Frame(self.contenedor)
        botones.pack(pady=35)

        self.tk.Button(
            botones,
            text="SOLDADURA",
            width=24,
            height=3,
            font=("Arial", 13, "bold"),
            command=lambda: self.seleccionar_actividad("soldadura")
        ).grid(row=0, column=0, padx=18, pady=10)

        self.tk.Button(
            botones,
            text="CORTE",
            width=24,
            height=3,
            font=("Arial", 13, "bold"),
            command=lambda: self.seleccionar_actividad("corte")
        ).grid(row=0, column=1, padx=18, pady=10)

        self.lbl_carga = self.tk.Label(self.contenedor, text="", font=("Arial", 10))
        self.lbl_carga.pack(pady=12)

        self.tk.Button(
            self.contenedor,
            text="CAMBIAR USUARIO",
            width=22,
            command=self.finalizar_sesion
        ).pack(pady=18)

    def seleccionar_actividad(self, actividad):
        self.lbl_carga.config(text=f"Cargando modelos para {actividad}...")
        self.window.update_idletasks()
        try:
            if self.sistema is None:
                self.sistema = SistemaEPP(actividad, modo_prueba=self.modo_prueba)
            else:
                self.sistema.seleccionar_actividad(actividad)
            self.actividad = actividad
            self._abrir_camara()
            self.mostrar_verificacion()
        except ErrorSafeVision as exc:
            self.lbl_carga.config(text="")
            self._cerrar_camara()
            self.messagebox.showerror("SafeVision AI", str(exc))

    # ---------- Pantalla 3: verificación / resultado ----------
    def mostrar_verificacion(self):
        self.estado = self.ESTADO_PREPARADO
        self.evaluacion_final = None
        self.resultados_actuales = self.sistema.invalidar("Esperando inicio de verificación")
        self.estabilizador_evaluacion.reiniciar()
        self.voz.cancelar_pendientes()
        self.limpiar_pantalla()

        superior = self.tk.Frame(self.contenedor)
        superior.pack(fill="x", pady=(0, 8))
        self.tk.Label(superior, text="SafeVision AI", font=("Arial", 20, "bold")).pack(side="left")
        self.tk.Label(
            superior,
            text=self.actividad.upper(),
            font=("Arial", 15, "bold")
        ).pack(side="right")

        self.tk.Label(
            self.contenedor,
            text=f"Usuario: {self.usuario.nombre_completo}   |   CI: {self.usuario.ci}",
            font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(0, 10))

        cuerpo = self.tk.Frame(self.contenedor)
        cuerpo.pack(fill="both", expand=True)

        izquierda = self.tk.Frame(cuerpo)
        izquierda.pack(side="left", fill="both", expand=True, padx=(0, 14))
        derecha = self.tk.Frame(cuerpo, bd=1, relief="solid", padx=18, pady=16, width=300)
        derecha.pack(side="right", fill="y")
        derecha.pack_propagate(False)

        self.canvas = self.tk.Canvas(izquierda, width=640, height=480, bg="black", highlightthickness=0)
        self.canvas.pack(pady=(0, 10))
        self.canvas_image = self.canvas.create_image(0, 0, anchor=self.tk.NW)
        if self.camara is None:
            self.canvas.create_text(320, 240, text="Simulación sin cámara", fill="white", font=("Arial", 14))

        self.tk.Label(derecha, text="Estado del EPP", font=("Arial", 14, "bold")).pack(pady=(0, 14))
        for identificador in ids_actividad(self.actividad):
            fila = self.tk.Frame(derecha)
            fila.pack(fill="x", pady=8)
            self.tk.Label(
                fila,
                text=CONFIG_MODELOS[identificador].nombre,
                justify="left",
                anchor="w",
                wraplength=180,
                font=("Arial", 10, "bold")
            ).pack(side="left", fill="x", expand=True)
            lbl = self.tk.Label(fila, text="ESPERANDO", font=("Arial", 9, "bold"), fg="gray35")
            lbl.pack(side="right", padx=(8, 0))
            self.labels_epp[identificador] = lbl

        self.lbl_estado_general = self.tk.Label(
            derecha,
            text="Listo para iniciar",
            wraplength=250,
            justify="center",
            font=("Arial", 13, "bold")
        )
        self.lbl_estado_general.pack(fill="x", pady=(24, 10))

        self.lbl_detalle = self.tk.Label(
            derecha,
            text="Colóquese frente a la cámara y pulse INICIAR VERIFICACIÓN.",
            wraplength=250,
            justify="left",
            font=("Arial", 10)
        )
        self.lbl_detalle.pack(fill="x", pady=(0, 12))

        self.lbl_registro = self.tk.Label(
            derecha,
            text="",
            wraplength=250,
            justify="center",
            font=("Arial", 9, "bold")
        )
        self.lbl_registro.pack(fill="x", pady=8)

        controles = self.tk.Frame(self.contenedor)
        controles.pack(fill="x", pady=(12, 4))

        self.btn_principal = self.tk.Button(
            controles,
            text="INICIAR VERIFICACIÓN",
            width=26,
            font=("Arial", 11, "bold"),
            command=self.accion_principal
        )
        self.btn_principal.pack(side="left", padx=6, ipady=6)

        self.btn_cambiar = self.tk.Button(
            controles,
            text="CAMBIAR TRABAJO",
            width=20,
            command=self.cambiar_trabajo
        )
        self.btn_cambiar.pack(side="left", padx=6, ipady=6)

        self.tk.Button(
            controles,
            text="FINALIZAR SESIÓN",
            width=20,
            command=self.finalizar_sesion
        ).pack(side="right", padx=6, ipady=6)

        self.tk.Label(
            self.contenedor,
            text="SafeVision AI es una herramienta de apoyo y no sustituye la supervisión ni los procedimientos de seguridad.",
            font=("Arial", 9)
        ).pack(pady=(8, 0))

        self._actualizar_estado_elementos(self.resultados_actuales)

    def accion_principal(self):
        if self.estado == self.ESTADO_PREPARADO:
            self.iniciar_verificacion()
        elif self.estado == self.ESTADO_RESULTADO:
            self.iniciar_verificacion()

    def iniciar_verificacion(self):
        if self.sistema is None or self.usuario is None:
            return
        self.estado = self.ESTADO_VERIFICANDO
        self.resultados_actuales = self.sistema.invalidar("Nueva verificación")
        self.evaluacion_final = None
        self.estabilizador_evaluacion.reiniciar()
        self.sesion_registro.iniciar_verificacion()
        self.voz.reiniciar_anuncios()
        self.btn_principal.config(text="VERIFICANDO...", state="disabled")
        self.btn_cambiar.config(state="disabled")
        self.lbl_estado_general.config(text="VERIFICANDO EPP...")
        self.lbl_detalle.config(
            text=(f"Mantenga la posición hasta que el resultado permanezca estable durante "
                  f"{FINAL_RESULT_STABILITY_TIME:.1f} s."))
        self.lbl_registro.config(text="")
        self._actualizar_estado_elementos(self.resultados_actuales)

    def _actualizar_estado_elementos(self, resultados):
        for identificador in ids_actividad(self.actividad):
            r = resultados.get(identificador)
            lbl = self.labels_epp.get(identificador)
            if lbl is None:
                continue
            if self.estado == self.ESTADO_PREPARADO:
                texto, color = "ESPERANDO", "gray35"
            elif r is None or not r.valido:
                texto, color = "INCIERTO", "darkorange3"
            elif not r.estable:
                texto, color = "ANALIZANDO", "steelblue4"
            elif r.presente:
                texto, color = "OK", "forestgreen"
            else:
                texto, color = "FALTA", "firebrick"
            lbl.config(text=texto, fg=color)

    def _mostrar_evaluacion_provisional(self, evaluacion):
        if evaluacion.estado == "VERIFICANDO":
            self.lbl_estado_general.config(text="VERIFICANDO EPP...")
            pendientes = [CONFIG_MODELOS[i].nombre for i in evaluacion.no_verificados]
            if pendientes:
                self.lbl_detalle.config(text="Esperando estabilidad de: " + ", ".join(pendientes))
        elif evaluacion.estado == "EPP COMPLETO":
            self.lbl_estado_general.config(text="Resultado candidato: EPP COMPLETO")
            self.lbl_detalle.config(text="Confirmando estabilidad del resultado...")
        else:
            faltantes = [CONFIG_MODELOS[i].nombre for i in evaluacion.faltantes]
            self.lbl_estado_general.config(text="Resultado candidato: EPP INCOMPLETO")
            self.lbl_detalle.config(text="Confirmando: faltan " + ", ".join(faltantes))

    def finalizar_verificacion(self, evaluacion):
        if self.estado != self.ESTADO_VERIFICANDO:
            return
        self.estado = self.ESTADO_RESULTADO
        self.evaluacion_final = evaluacion
        guardado = self.sesion_registro.guardar_resultado_final(evaluacion)
        self.voz.actualizar(evaluacion)

        self.btn_principal.config(text="REPETIR VERIFICACIÓN", state="normal")
        self.btn_cambiar.config(state="normal")

        if evaluacion.estado == "EPP COMPLETO":
            self.lbl_estado_general.config(text="EPP COMPLETO", fg="forestgreen")
            self.lbl_detalle.config(
                text="Los elementos de EPP configurados para esta actividad fueron verificados como presentes.")
        else:
            faltantes = [CONFIG_MODELOS[i].nombre for i in evaluacion.faltantes]
            self.lbl_estado_general.config(text="EPP INCOMPLETO", fg="firebrick")
            self.lbl_detalle.config(text="Elementos faltantes: " + ", ".join(faltantes))

        if guardado:
            self.lbl_registro.config(text="Resultado registrado correctamente en CSV.", fg="forestgreen")
            print(f"REGISTRO CSV guardado: {self.registro_csv.ruta}")
        else:
            self.lbl_registro.config(text="El resultado no generó un nuevo registro.", fg="gray35")

    def cambiar_trabajo(self):
        if self.estado == self.ESTADO_VERIFICANDO:
            return
        if self.sistema is not None:
            self.sistema.invalidar("Cambio de actividad")
        self.estabilizador_evaluacion.reiniciar()
        self.voz.cancelar_pendientes()
        self._cerrar_camara()
        self.mostrar_seleccion_actividad()

    def finalizar_sesion(self):
        if self.sistema is not None:
            self.sistema.invalidar("Fin de sesión")
        self.estabilizador_evaluacion.reiniciar()
        self.voz.cancelar_pendientes()
        self._cerrar_camara()
        self.usuario = None
        self.sesion_registro = None
        self.sistema = None
        self.actividad = None
        self.nombre_inicial = ""
        self.ci_inicial = ""
        self.mostrar_registro()

    # ---------- Bucle de cámara/inferencia ----------
    def update(self):
        if self.cerrada:
            return
        self.after_id = None
        try:
            frame = self.camara.leer() if self.camara is not None else None

            if self.estado == self.ESTADO_VERIFICANDO and self.sistema is not None:
                self.resultados_actuales = self.sistema.procesar_frame(frame)
                evaluacion = evaluar_epp(self.actividad, self.resultados_actuales)
                self._actualizar_estado_elementos(self.resultados_actuales)
                self._mostrar_evaluacion_provisional(evaluacion)
                if self.estabilizador_evaluacion.actualizar(evaluacion):
                    self.finalizar_verificacion(evaluacion)

            if frame is not None and hasattr(self, "canvas") and self.canvas.winfo_exists():
                cv2 = self.camara.cv2
                frame_visual = frame
                if MOSTRAR_REGIONES and self.sistema is not None and self.actividad is not None:
                    frame_visual = dibujar_regiones(frame, regiones_de_modelos(self.sistema.modelos))
                rgb = cv2.cvtColor(frame_visual, cv2.COLOR_BGR2RGB)
                imagen = self.Image.fromarray(rgb).resize((640, 480), self.Image.Resampling.LANCZOS)
                self.photo = self.ImageTk.PhotoImage(image=imagen)
                self.canvas.itemconfig(self.canvas_image, image=self.photo)

        except ErrorSafeVision as exc:
            if self.estado == self.ESTADO_VERIFICANDO:
                self.estado = self.ESTADO_PREPARADO
                if hasattr(self, "btn_principal") and self.btn_principal.winfo_exists():
                    self.btn_principal.config(text="INICIAR VERIFICACIÓN", state="normal")
                if hasattr(self, "btn_cambiar") and self.btn_cambiar.winfo_exists():
                    self.btn_cambiar.config(state="normal")
            self.messagebox.showerror("SafeVision AI", str(exc))
            self._cerrar_camara()

        self.after_id = self.window.after(30, self.update)

    def on_closing(self):
        if self.cerrada:
            return
        self.cerrada = True
        if self.after_id is not None:
            try:
                self.window.after_cancel(self.after_id)
            except Exception:
                pass
        self._cerrar_camara()
        self.voz.cerrar()
        self.window.destroy()


# ================================= CONSOLA DE PRUEBA =================================
def seleccionar_actividad_consola():
    opciones = list(ACTIVIDADES)
    print("Seleccione actividad:")
    for indice, actividad in enumerate(opciones, 1):
        print(f"{indice}. {actividad.capitalize()}")
    while True:
        try:
            seleccion = input("Opción: ").strip()
        except EOFError as exc:
            raise ErrorSafeVision("No hay entrada de terminal. Usa --actividad soldadura o --actividad corte.") from exc
        if seleccion.isdigit() and 1 <= int(seleccion) <= len(opciones):
            return opciones[int(seleccion) - 1]
        print("Opción inválida. Vuelve a intentarlo.")


def ejecutar_consola(sistema, voz, sesion_registro, ciclos=100):
    print("MODO PRUEBA — SIMULACIÓN SIN CÁMARA; no son detecciones reales.")
    estabilizador_final = EstabilizadorEvaluacion()
    sesion_registro.iniciar_verificacion()
    voz.reiniciar_anuncios()
    anterior = None
    for _ in range(ciclos):
        resultados = sistema.procesar_frame()
        evaluacion = evaluar_epp(sistema.actividad, resultados)
        texto = texto_resultados(evaluacion, resultados)
        if texto != anterior:
            print("\n" + texto)
            anterior = texto
        if estabilizador_final.actualizar(evaluacion):
            if sesion_registro.guardar_resultado_final(evaluacion):
                print(f"REGISTRO CSV guardado: {sesion_registro.registro_csv.ruta}")
            voz.actualizar(evaluacion)
            print("\nRESULTADO FINAL CONFIRMADO")
            print(texto_resultados(evaluacion, resultados))
            return True
        time.sleep(0.05)
    print("\nNo se obtuvo un resultado final dentro del número de ciclos indicado.")
    return False


# ================================= MAIN =================================
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actividad", choices=list(ACTIVIDADES), help="Sólo para pruebas por consola")
    parser.add_argument("--nombre", help="Prellenar nombre o usarlo en consola")
    parser.add_argument("--ci", help="Prellenar CI o usarlo en consola")

    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--real", action="store_true", help="Usar modelos Keras .h5 reales")
    modo.add_argument("--prueba", action="store_true", help="Usar resultados simulados")

    parser.add_argument("--sin-camara", action="store_true", help="Sólo permitido en modo prueba")
    parser.add_argument("--sin-voz", action="store_true")
    parser.add_argument("--listar-voces", action="store_true", help="Mostrar las voces disponibles en pyttsx3 y salir")
    parser.add_argument("--consola", action="store_true", help="Simulación por consola sin GUI ni cámara")
    parser.add_argument("--ciclos", type=int, default=100, help="Iteraciones en prueba por consola")
    parser.add_argument("--camara", type=int, default=CAMERA_INDEX)
    args = parser.parse_args(argv)

    if args.listar_voces:
        SistemaVoz.listar_voces_instaladas()
        return 0

    prueba = True if args.prueba else False if args.real else MODO_PRUEBA
    if args.consola and not prueba:
        parser.error("--consola se reserva para el modo prueba. Usa --prueba --consola.")
    if args.sin_camara and not prueba:
        parser.error("--sin-camara sólo se permite en modo prueba.")
    if args.ciclos <= 0:
        parser.error("--ciclos debe ser mayor que cero.")

    voz = None
    root = None
    try:
        if args.consola:
            usuario = registrar_usuario_consola(args.nombre, args.ci)
            actividad = args.actividad or seleccionar_actividad_consola()
            sistema = SistemaEPP(actividad, modo_prueba=True)
            registro_csv = RegistroCSV(REGISTROS_PRUEBA_CSV)
            sesion_registro = SesionRegistro(usuario, registro_csv)
            voz = SistemaVoz(not args.sin_voz)
            print(
                f"SafeVision AI | {actividad.upper()} | SIMULACIÓN | "
                f"Usuario: {usuario.nombre_completo} | CI: {usuario.ci}")
            ejecutar_consola(sistema, voz, sesion_registro, args.ciclos)
            return 0

        tk = importar("tkinter", "Tkinter de la instalación de Python")
        try:
            root = tk.Tk()
        except Exception as exc:
            raise ErrorSafeVision(f"No se pudo abrir Tkinter: {exc}.") from exc

        app = AppSafeVision(
            root,
            modo_prueba=prueba,
            camara_indice=args.camara,
            habilitar_camara=not args.sin_camara,
            habilitar_voz=not args.sin_voz,
            nombre_inicial=args.nombre,
            ci_inicial=args.ci
        )

        def error_callback(tipo, valor, traza):
            print(f"ERROR de aplicación: {tipo.__name__}: {valor}")
            app.on_closing()

        root.report_callback_exception = error_callback
        root.mainloop()
        return 0

    except ErrorSafeVision as exc:
        print(f"\nERROR:\n{exc}")
        return 1
    except KeyboardInterrupt:
        print("\nCerrando aplicación...")
        return 0
    finally:
        if voz is not None:
            voz.cerrar()
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
