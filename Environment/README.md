# ⚙️ Configuración de entorno para ejecución de modelos  
## 🐍 Entorno Python 3.11 + PyTorch 2.7.0 (ROCm 6.4)

Este entorno permite ejecutar proyectos Python en GPU AMD usando **PyTorch 2.7.0** con soporte **ROCm/HIP SDK 6.4**, gestionando dependencias con `pip` y ejecutándose en **Git Bash** sobre Windows o WSL.  
Incluye librerías esenciales para **Deep Learning**, **CNNs** y análisis/visualización de resultados (TensorBoard, scikit-learn, OpenCV, etc.).

---

## 📋 Requisitos previos

- **Git Bash** instalado ([descargar aquí](https://gitforwindows.org/)).
- **Python 3.11** instalado y accesible desde Git Bash (`python3.11 --version`).
- **pip** actualizado (`python.exe -m pip install --upgrade pip`) (`pip --version`)
- **ROCm/HIP SDK 6.4** configurado en el sistema (Linux/WSL recomendado).  
  **Nota:** Solo realizar este último paso en caso de contar con GPU AMD Radeon
- Link repositorio: https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection.git
## 🚀 Crear y activar el entorno virtual

En Git Bash:
```bash
# Clonar repositorio
git clone https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection.git
#Dirigirse a ruta del repositorio (modificar)
cd ../Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection

# Crear entorno virtual
python3.11 -m venv .venv


# Activar entorno virtual
source .venv/Scripts/activate  # En Git Bash Windows
# o en Linux/WSL:
# source .venv/bin/activate

# Actualizar pip
pip install --upgrade pip

Nota: si se trabaja con Pycharm, basta con seleccionar el intérprete una vez instalado python 3.11, de esta forma, se creará el entorno virtual dentro de la carpeta del repositorio.
``` 

## 📦 Instalación de dependencias
### 1️⃣ Instalar PyTorch con soporte HIP SDK (ROCm)

⚠️**Precaución: build no oficial**⚠️

Descargar los 3 archivos en formato wheel (.whl) de releases:

```bash
torch-2.7.0a0+rocm_git3f903c3-cp311-cp311-win_amd64.whl
torchaudio-2.7.0a0+52638ef-cp311-cp311-win_amd64.whl
torchvision-0.22.0+9eb57cd-cp311-cp311-win_amd64.whl
```

[Descargar aquí](https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection/releases).

Luego, copiar y pegar en la carpeta `pytorch-wheels` en la siguiente ruta relativa del repositorio: 

`..\Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection\Environment\pytorch-wheels`

Ejecutar cada línea de código en el orden mostrado a continuación:

```bash
pip install Environment/pytorch-wheels/torch-2.7.0a0+rocm_git3f903c3-cp311-cp311-win_amd64.whl
pip install Environment/pytorch-wheels/torchaudio-2.7.0a0+52638ef-cp311-cp311-win_amd64.whl
pip install Environment/pytorch-wheels/torchvision-0.22.0+9eb57cd-cp311-cp311-win_amd64.whl
```

⚠️En caso de error debido al largo de la ruta (Long Path), habilitar long path en windows mediante el siguiente [Link](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry)
### 2️⃣ Instalar librerías adicionales (Deep Learning, análisis, CNN):

```bash
pip install -r Environment/environment.txt
``` 

## 🧪 Verificación de instalación 

Ejecutar dentro del entorno virtual en el siguiente orden:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, ' HIP:', torch.version.hip)"
python -c "import torchvision, torchaudio; print('vision', torchvision.__version__, 'audio', torchaudio.__version__)"
python -c "import tensorboard; print('tensorboard OK')"
```
Se debe verificar las versiones instaladas, donde deberías ver 2.7.0 para PyTorch, la versión ROCm en uso, y confirmación de TensorBoard.

⚠️Por último, ejecutar el script de test `test_rocm_pytorch.py` en la terminal mediante el siguiente comando:

```bash
python Environment/test_rocm_pytorch.py  
```
Este código permite verificar el uso de CPU/GPU a la hora de utilizar Pytorch.

## 🗑️ Eliminar entorno virtual

Tras usar el ambiente, ejecutar el siguiente comando para eliminarlo/salir:

```bash
deactivate
rm -rf .venv
```

