
# ⚙️ Configuración de entorno para ejecución de modelos  
## 🐍 Entorno Python 3.11 + PyTorch 2.7.0 (ROCm 6.4)

Este entorno permite ejecutar proyectos Python en GPU AMD usando **PyTorch 2.7.0** con soporte **ROCm/HIP SDK 6.4**, gestionando dependencias con `pip` y ejecutándose en **Git Bash** sobre Windows o WSL.  
Incluye librerías esenciales para **Deep Learning**, **CNNs** y análisis/visualización de resultados (TensorBoard, scikit-learn, OpenCV, etc.).


## 📋 Requisitos previos

- **Git Bash** instalado ([descargar aquí](https://gitforwindows.org/)).
- **Python 3.11** instalado y accesible desde Git Bash (`python3.11 --version`).
- **pip** actualizado (`pip --version`).
- **ROCm/HIP SDK 6.4** configurado en el sistema (en Linux/WSL).  
  Verificar con:
  ```bash
  hipcc --version

## 🚀 Crear y activar el entorno virtual

En Git Bash:

```bash
# Clonar repositorio
git clone https://tu-repo.git
cd tu-repo

# Crear entorno virtual
python3.11 -m venv .venv

# Activar entorno virtual
source .venv/Scripts/activate  # En Git Bash Windows
# o en Linux/WSL:
# source .venv/bin/activate

# Actualizar pip
pip install --upgrade pip
```

---

## 📦 Instalación de dependencias

### 1. Instalar PyTorch con soporte ROCm:

```bash
bash install_pytorch.sh
```

*(el script ya contiene los flags para ROCm 6.4).*

### 2. Instalar librerías adicionales (Deep Learning, análisis, CNN):

```bash
pip install -r environment.txt \
  --extra-index-url https://download.pytorch.org/whl/rocm6.4
```

---

## 📑 Archivos incluidos

* **`environment.txt`**: lista de librerías Python adicionales (ej. numpy, pandas, matplotlib, scikit-learn, tensorboard, opencv, etc.).
* **`install_pytorch.sh`**: script para instalar PyTorch 2.7.0 + ROCm 6.4.
* **`src/`**: carpeta para tu código fuente y notebooks.

---

## 🧪 Verificación de instalación

Ejecutar dentro del entorno virtual:

```bash
python -c "import torch; print('PyTorch:', torch.__version__, ' HIP:', torch.version.hip)"
python -c "import torchvision, torchaudio; print('vision', torchvision.__version__, 'audio', torchaudio.__version__)"
python -c "import tensorboard; print('tensorboard OK')"
```

Deberías ver `2.7.0` para PyTorch, la versión ROCm en uso, y confirmación de TensorBoard.

---

## 🗑️ Eliminar entorno virtual

```bash
deactivate
rm -rf .venv
```

---

## 💡 Notas

* En Windows puro sin WSL, PyTorch ROCm no está soportado oficialmente; se recomienda usar WSL2 con ROCm configurado o un sistema Linux nativo.
* Ejecuta siempre `source .venv/Scripts/activate` (Windows) o `source .venv/bin/activate` (Linux/WSL) antes de trabajar en este proyecto.

```

---

¿Quieres que también te ponga aquí, en formato Markdown, el contenido sugerido para `install_pytorch.sh` y `environment.txt`? (para que copies y pegues todo de una vez)
```
