# ⚙️ Configuración de entorno para ejecución de modelos  
## 🐍 Entorno Python 3.11 + PyTorch 2.7.0 (ROCm 6.4)

Este entorno permite ejecutar proyectos Python en GPU AMD usando **PyTorch 2.7.0** con soporte **ROCm/HIP SDK 6.4**, gestionando dependencias con `pip` y ejecutándose en **Git Bash** sobre Windows o WSL.  
Incluye librerías esenciales para **Deep Learning**, **CNNs** y análisis/visualización de resultados (TensorBoard, scikit-learn, OpenCV, etc.).

---

## 📋 Requisitos previos

- **Git Bash** instalado ([descargar aquí](https://gitforwindows.org/)).
- **Python 3.11** instalado y accesible desde Git Bash (`python3.11 --version`).
- **pip** actualizado (`pip --version`).
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

**Precaución: build no oficial**

Ejecutar cada línea de código en el orden mostrado a continuación:

```bash
pip install Environment/pytorch-wheels/torch-*.whl
pip install Environment/pytorch-wheels/torchvision-*.whl
pip install Environment/pytorch-wheels/torchaudio-*.whl
```

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
Se debe verificar las versiones instaladas, correspondientes a los nombres de los wheels.

## 🗑️ Eliminar entorno virtual

Tras usar el ambiente, ejecutar el siguiente comando para eliminarlo/salir:

```bash
deactivate
rm -rf .venv
```

