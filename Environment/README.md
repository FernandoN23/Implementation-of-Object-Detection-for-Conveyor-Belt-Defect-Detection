# Configuración de entorno para ejecución de los modelos
## 🐍 Entorno Python 3.11 + PyTorch 2.7.0 (ROCm 6.4)

Este entorno permite ejecutar proyectos Python en GPU AMD usando **PyTorch 2.7.0** con soporte **ROCm/HIP SDK 6.4**, gestionando dependencias con `pip` y ejecutándose en **Git Bash** sobre Windows o WSL.

---

## 📋 Requisitos previos

- **Git Bash** instalado (https://gitforwindows.org/).
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

## 📦 Instalación de dependencias

1.Instalar PyTorch con soporte ROCm:

```bash
bash install_pytorch.sh
```
2.Instalar librerías adicionales:

```bash
pip install -r environment.txt
```

## 📑 Archivos incluidos

- environment.txt: lista de librerías Python adicionales (ej. numpy, pandas, matplotlib, etc.).

- install_pytorch.sh: script para instalar PyTorch 2.7.0 + ROCm 6.4.

- src/: carpeta para tu código fuente.

## 🧪 Verificación

```bash
python -c "import torch; print(torch.__version__); print(torch.version.hip)"
```

