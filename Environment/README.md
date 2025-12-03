# ⚙️ Configuración de entorno para ejecución de modelos

## 🐍 Entorno Python 3.12 + PyTorch 2.9.0 (AMD ROCm 7.1.1)

Este entorno permite ejecutar proyectos Python en **GPU AMD Radeon y CPUs Ryzen AI** mediante la nueva versión oficial de **PyTorch 2.9.0** con soporte **ROCm 7.1.1 / HIP SDK**.

La instalación es completamente automatizada mediante `pip`. Incluye librerías esenciales para **Deep Learning**, **detección de objetos (YOLOv5/v8/v11, DETR)** y herramientas de **visualización y análisis** (TensorBoard, scikit-learn, OpenCV, etc.).

---
## 📋 Requisitos previos

- **Windows 11** actualizado (recomendado: últimas actualizaciones de Windows Update).
- **Python 3.12** instalado y accesible desde terminal / PyCharm.  
  ↳ Verificar: `python --version`
- **pip** actualizado.  
  ↳ `python -m pip install --upgrade pip`
- **Controlador AMD compatible con ROCm 7.1.1** ↳ Requiere **AMD Software: Adrenalin Edition 24.40.0.0** (o superior compatible).  
  ↳ Consultar notas de lanzamiento oficiales ([ver enlace](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-1-1.html)).
- **GPU/CPU compatibles** ↳ Radeon compatibles (serie 7000/8000/9000 según lista oficial) **o** CPU **Ryzen AI** (aceleración vía ROCm).
- **(Opcional)** Git Bash para comandos de terminal en Windows ([descargar](https://gitforwindows.org/)).  
  *(Puedes usar PowerShell o la terminal integrada de PyCharm si prefieres).*
- **Ruta al repositorio del proyecto** https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection.git
## 🚀 Crear y activar el entorno virtual

En Git Bash:
```bash
# Clonar repositorio
git clone https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection.git
#Dirigirse a ruta del repositorio (modificar)
cd ../Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection

# Crear entorno virtual
python3.12 -m venv .venv


# Activar entorno virtual
source .venv/Scripts/activate  # En Git Bash Windows
# o en Linux/WSL:
# source .venv/bin/activate

# Actualizar pip
pip install --upgrade pip

Nota: si se trabaja con Pycharm, basta con seleccionar el intérprete una vez instalado python 3.12, de esta forma, se creará el entorno virtual dentro de la carpeta del repositorio.
``` 

## 📦 Instalación de dependencias

La instalación del entorno es completamente automática gracias al archivo Environment/requirements.txt, que orquesta la instalación del **SDK de ROCm 7.1.1** seguido de los binarios de PyTorch.

---

### ⚙️ Instalación completa

Ejecutar dentro del entorno virtual activado en la ruta principal del repositorio:

```bash
pip install -r Environment/requirements.txt --no-cache-dir
```

## 🧪 Verificación de instalación

Una vez completada la instalación, se recomienda verificar que **PyTorch**, **TorchVision**, **TorchAudio** y **TensorBoard** funcionen correctamente dentro del entorno virtual.

Ejecutar los siguientes comandos en la terminal:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, '| HIP:', torch.version.hip)"
python -c "import torchvision, torchaudio; print('TorchVision:', torchvision.__version__, '| TorchAudio:', torchaudio.__version__)"
python -c "import tensorboard; print('TensorBoard OK')"
```

Se debe verificar las versiones instaladas, donde deberías ver `2.9.0+rocmsdk...` para PyTorch, la versión ROCm en uso, y confirmación de TensorBoard.

⚠️Por último, ejecutar el script de test `check_environment.py` en la terminal mediante el siguiente comando:

```bash
python Environment/check_environment.py  
```
Este código permite verificar el uso de CPU/GPU a la hora de utilizar Pytorch.

Además, comprobar la compatibilidad con Tensorboard mediante el siguiente script:

```bash
python Environment/test_tensorboard_amd.py  
```

## 🗑️ Eliminar entorno virtual

Tras usar el ambiente, ejecutar el siguiente comando para eliminarlo/salir:

```bash
deactivate
rm -rf .venv
# En PowerShell: Remove-Item -Recurse -Force .venv
```

Nota: esto permite realizar una limpieza del ambiente sin necesidad de eliminar todo el repositorio.
