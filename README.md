[README.md](https://github.com/user-attachments/files/33600826/README.md)
# SafeVision AI — Instalación y primera ejecución en Windows

Guía para ejecutar **`safevision_ai_final_csv_corregido.py`** desde **Símbolo del sistema (CMD)**. Incluye la creación del entorno virtual, la instalación de librerías y la preparación de los modelos Keras de Teachable Machine.

## 1. Requisitos previos

- Windows de 64 bits y **Python 3.11 de 64 bits**. Puedes obtener el instalador de Windows en la [página oficial de Python 3.11.9](https://www.python.org/downloads/release/python-3119/). Conserva los componentes **pip**, **Python Launcher** y **Tcl/Tk** durante la instalación.
- Una carpeta con el archivo `safevision_ai_final_csv_corregido.py` y tus modelos entrenados, si vas a realizar verificaciones reales.
- Conexión a Internet para instalar las librerías y una cámara disponible para el modo real.

Esta guía utiliza **TensorFlow 2.15.1 y Keras 2.15**, siguiendo la recomendación de compatibilidad del propio código para las exportaciones `.h5`. El entorno se configura para trabajar con CPU en Windows; no necesitas instalar CUDA.

Abre **CMD** y comprueba que Python 3.11 esté disponible:

```bat
py -3.11 --version
```

Debe mostrar `Python 3.11.x`. Si acabas de instalar Python, cierra y vuelve a abrir CMD antes de continuar.

## 2. Entrar en la carpeta del proyecto

Reemplaza la ruta del ejemplo por la carpeta donde guardaste el archivo Python. Conserva las comillas si la ruta contiene espacios:

```bat
cd /d "C:\ruta\SafeVision_AI"
dir safevision_ai_final_csv_corregido.py
```

El segundo comando debe mostrar el archivo. Si no aparece, verifica la ruta antes de continuar. También puedes abrir la carpeta en el Explorador de archivos, escribir `cmd` en su barra de direcciones y presionar **Enter** para abrir CMD directamente allí.

## 3. Crear y activar el entorno virtual

Ejecuta estos comandos dentro de la carpeta del proyecto:

```bat
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python --version
python -c "import sys; print(sys.executable)"
```

La consola normalmente mostrará `(.venv)` al inicio. La versión debe ser `3.11.x` y la ruta del ejecutable debe terminar en `.venv\Scripts\python.exe`.

El entorno virtual separa las librerías de SafeVision AI de las de otros proyectos. **Solo se crea la primera vez; se activa cada vez que abres una nueva ventana de CMD.**

## 4. Instalar las librerías

Con el entorno virtual activo, ejecuta:

```bat
python -m pip install --upgrade pip
python -m pip install "tensorflow==2.15.1" "keras==2.15.0" "numpy==1.26.4" "opencv-python==4.11.0.86" Pillow pyttsx3 pywin32
python -m pip check
```

Espera a que termine cada comando. La instalación de TensorFlow puede tardar varios minutos. Si `pip check` no encuentra conflictos, mostrará `No broken requirements found.`

| Librería | Uso en el proyecto |
|---|---|
| TensorFlow / Keras | Cargar y ejecutar los clasificadores de imágenes `.h5`. |
| NumPy | Preparar los datos de imagen para los modelos. |
| OpenCV (`opencv-python`) | Capturar video, recortar regiones y procesar imágenes. |
| Pillow | Mostrar imágenes en la interfaz. |
| pyttsx3 | Reproducir las alertas por voz. |
| pywin32 | Proporcionar `pythoncom`, utilizado por la voz en Windows. |
| Tkinter | Crear la interfaz; se incluye con la instalación de Python que incorpora Tcl/Tk. |

**No ejecutes `pip install tkinter`.** Tampoco necesitas instalar por separado `csv`, `os`, `threading`, `pathlib` ni los demás módulos de la biblioteca estándar.

Para comprobar que las dependencias principales se pueden importar:

```bat
python -c "import tensorflow as tf; import keras; import numpy; import cv2; from PIL import Image, ImageTk; import tkinter; import pyttsx3; import pythoncom; print('Importaciones correctas'); print('TensorFlow:', tf.__version__); print('Keras:', keras.__version__)"
```

Esta comprobación valida las importaciones; la carga de tus modelos se comprobará al iniciar una verificación real.

## 5. Colocar los modelos entrenados

Dentro de la misma carpeta que contiene el archivo Python, crea estas subcarpetas. Puedes usar CMD:

```bat
mkdir modelos\mascara_soldadura
mkdir modelos\guantes_soldadura
mkdir modelos\gafas_corte
mkdir modelos\guantes_corte
mkdir modelos\vestimenta
```

Si alguna carpeta ya existe, conserva su contenido y continúa. En **cada carpeta** coloca los dos archivos de su respectiva exportación de Teachable Machine: **`keras_model.h5`** y **`labels.txt`**.

| Carpeta dentro del proyecto | Clases que espera el código |
|---|---|
| `modelos\mascara_soldadura` | `CON_MASCARA` / `SIN_MASCARA` |
| `modelos\guantes_soldadura` | `CON_GUANTES` / `SIN_GUANTES` |
| `modelos\gafas_corte` | `CON_GAFAS` / `SIN_GAFAS` |
| `modelos\guantes_corte` | `CON_GUANTES` / `SIN_GUANTES` |
| `modelos\vestimenta` | `VESTIMENTA_ADECUADA` / `VESTIMENTA_INADECUADA` |

Por ejemplo, el modelo de gafas debe quedar en `modelos\gafas_corte\keras_model.h5`, junto a `modelos\gafas_corte\labels.txt`.

Conserva el `labels.txt` que corresponde a cada modelo y el orden de sus clases. Los nombres de las clases deben coincidir con la tabla. Si tu entrenamiento utiliza otros nombres, revisa la configuración del proyecto o la exportación; no intercambies etiquetas para ocultar un error. El archivo exportado puede incluir índices, como `0 CON_GAFAS`, que el programa admite.

Las librerías **no descargan ni entrenan estos modelos**: debes colocar tus propias exportaciones. Para habilitar corte y soldadura se necesitan los cinco modelos; el de vestimenta es compartido.

## 6. Abrir el proyecto por primera vez

Con el entorno virtual activo y los modelos en sus carpetas:

```bat
python safevision_ai_final_csv_corregido.py
```

Esta versión inicia en **modo real** de forma predeterminada.

1. Ingresa el nombre y el CI en la interfaz.
2. Selecciona **corte** o **soldadura**.
3. Inicia la verificación y colócate dentro de las regiones de cabeza, manos y vestimenta que muestra la cámara.
4. Espera a que el sistema estabilice el resultado y lo registre.

La carpeta `datos` se crea automáticamente junto al script. Las verificaciones reales se guardan en **`datos\registros_safevision.csv`**, con las columnas `fecha_hora`, `nombre_completo`, `ci`, `trabajo`, `resultado` y `faltantes`.

### Prueba opcional sin modelos ni cámara

Puedes comprobar la interfaz antes de disponer de los modelos usando:

```bat
python safevision_ai_final_csv_corregido.py --prueba --sin-camara --sin-voz
```

Este modo utiliza **resultados simulados**, sin realizar inferencia sobre imágenes. Sus registros se guardan por separado en `datos\registros_safevision_prueba.csv`.

## 7. Volver a abrir el proyecto otro día

Abre una nueva ventana de CMD y ejecuta, sustituyendo la ruta por la de tu proyecto:

```bat
cd /d "C:\ruta\SafeVision_AI"
call .venv\Scripts\activate.bat
python safevision_ai_final_csv_corregido.py
```

**No necesitas volver a crear el entorno ni instalar las librerías** si sigues usando la misma carpeta y el mismo entorno.

Para terminar, cierra la ventana de SafeVision AI y luego ejecuta en CMD:

```bat
deactivate
```

## 8. Comandos útiles

Ejecuta estos comandos desde la carpeta del proyecto y con el entorno activado.

**Probar otra cámara** — la cámara predeterminada es la de índice `0`:

```bat
python safevision_ai_final_csv_corregido.py --camara 1
```

**Abrir el modo real sin voz**:

```bat
python safevision_ai_final_csv_corregido.py --sin-voz
```

**Consultar las voces instaladas**:

```bat
python safevision_ai_final_csv_corregido.py --listar-voces
```

**Consultar todas las opciones del programa**:

```bat
python safevision_ai_final_csv_corregido.py --help
```

**Guardar las versiones instaladas**, una vez que hayas comprobado el funcionamiento con tus modelos:

```bat
python -m pip freeze > requirements_instalados.txt
```

Este archivo permite reinstalar las mismas versiones en otro entorno de Python 3.11 para Windows. Después de crear y activar ese nuevo entorno, puedes ejecutar:

```bat
python -m pip install -r requirements_instalados.txt
```

## 9. Solución de problemas frecuentes

| Problema | Qué revisar |
|---|---|
| `py` no se reconoce o Python 3.11 no está disponible | Instala Python 3.11 x64 con Python Launcher y abre de nuevo CMD. Usa `py -0p` para consultar las instalaciones disponibles. |
| No se encuentra el archivo `.py` | Comprueba la carpeta actual y ejecuta `dir safevision_ai_final_csv_corregido.py`. |
| `No matching distribution found` al instalar TensorFlow | Comprueba que el entorno utilice Python 3.11 de 64 bits. Un entorno creado con otra versión debe reemplazarse por uno nuevo creado explícitamente con `py -3.11 -m venv`. |
| `ModuleNotFoundError` | Activa `.venv`, comprueba la ruta de `sys.executable` e instala las librerías con `python -m pip` dentro de ese entorno. |
| No se encuentra `tkinter` | Modifica la instalación de Python e incluye Tcl/Tk. Puedes comprobarlo con `python -m tkinter`; cierra su ventana de prueba al terminar. |
| Error al cargar `.h5`, por ejemplo `DepthwiseConv2D` y `groups` | Comprueba que estés usando TensorFlow 2.15.1 y Keras 2.15.0 en el entorno indicado. El código recomienda esta combinación; no modifiques el `.h5` para suprimir el error. |
| Modelo o etiquetas no encontrados | Revisa los nombres y ubicaciones del paso 5. Cada modelo debe tener su propio `labels.txt`. |
| La cámara no abre | Cierra otras aplicaciones que estén usándola, permite el acceso de aplicaciones de escritorio a la cámara en Windows y prueba `--camara 1`. |
| No se escucha la voz | Revisa volumen y salida de audio; consulta `--listar-voces`. Para comprobar el resto de la aplicación, utiliza `--sin-voz`. |
| Error de DLL al importar TensorFlow u OpenCV | Revisa la instalación del Microsoft Visual C++ Redistributable x64 siguiendo la [guía oficial de TensorFlow para Windows](https://www.tensorflow.org/install/pip#windows-native). |
| El CSV está abierto en Excel o no se puede escribir | Cierra el archivo en Excel y comprueba los permisos de escritura. El programa intenta guardar en un CSV con sufijo `_recuperacion` si el archivo principal está bloqueado; revisa el mensaje mostrado. |

## Referencias de la guía

- Código revisado: `safevision_ai_final_csv_corregido.py`.
- [Entornos virtuales y activación en CMD — documentación de Python](https://docs.python.org/3.11/library/venv.html).
- [TensorFlow 2.15.1 — paquetes disponibles para Python 3.11 y Windows x64](https://pypi.org/project/tensorflow/2.15.1/).
- [Instalación de TensorFlow y requisitos de Windows](https://www.tensorflow.org/install/pip).

