# Configuración de Entorno: PyTorch 2.9.1 + AMD ROCm 7.2.0

Este manifiesto documenta la configuración del entorno para la ejecución de algoritmos de detección de objetos (YOLO, SSD, DETR, DINO) utilizando aceleración por hardware de AMD en Windows (ROCm).

---

## 1. Requisitos Previos

* **Sistema Operativo:** Windows 11.
* **Intérprete:** Python 3.12 (`python --version`).
* **Gestor de Paquetes:** pip actualizado (`python -m pip install --upgrade pip`).
* **Controladores Gráficos:** AMD Software: Adrenalin Edition 26.1.1 (o superior).
* **Hardware:** GPU AMD Radeon compatible (serie 7000/8000/9000) o CPU Ryzen AI.
* **Entorno de Desarrollo:** PyCharm (con terminal integrada configurada en PowerShell).

## 2. Inicialización del Entorno (PowerShell)

Ejecute los siguientes comandos en la terminal integrada de PyCharm  o en una sesión independiente de PowerShell:

```powershell
# 1. Clonar el repositorio
git clone [https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection.git](https://github.com/FernandoN23/Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection.git)

# 2. Acceder al directorio
cd Implementation-of-Object-Recognition-Algorithms-for-Conveyor-Belt-Defect-Detection

# 3. Crear entorno virtual
python -m venv .venv

# 4. Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# 5. Actualizar gestor de paquetes base
python -m pip install --upgrade pip

```

## 3. Instalación de Dependencias

Con el entorno virtual activo `(.venv)`, inicie el despliegue del SDK de ROCm y PyTorch de manera secuencial.

```powershell
pip install -r Environment/requirements.txt --no-cache-dir
```

## 4. Validación de Despliegue

Ejecute el siguiente protocolo para auditar la integridad de la instalación y la comunicación con el hardware.

### Fase 1: Protocolo de Hardware AMD (Obligatorio)

Compruebe el soporte a nivel de plataforma:

**1. Importación base:**

```powershell
python -c "import torch"; if ($?) { echo 'Success' } else { echo 'Failure' }
```

*Resultado esperado:* `Success`

**2. Disponibilidad del backend computacional:**

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

*Resultado esperado:* `True`

**3. Reconocimiento de dispositivo:**

```powershell
python -c "import torch; print(f'device name [0]:', torch.cuda.get_device_name(0))"
```

*Resultado esperado:* `device name [0]: AMD Radeon(TM) 8060s Graphics`

**4. Telemetría de variables de entorno:**

```powershell
python -m torch.utils.collect_env
```
Resultado obtenido a fecha 09/03/2026:

```
PyTorch version: 2.9.1+rocmsdk20260116
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 7.2.26024-f6f897bd3d

OS: Microsoft Windows 11 Pro (10.0.26200 64 bits)
GCC version: (MinGW-W64 x86_64-ucrt-posix-seh, built by Brecht Sanders, r8) 13.2.0
Clang version: Could not collect
CMake version: version 3.29.2
Libc version: N/A

Python version: 3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)] (64-bit runtime)
Python platform: Windows-11-10.0.26200-SP0
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: 
GPU models and configuration: AMD Radeon(TM) 8060S Graphics (gfx1151)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
Is XPU available: False
HIP runtime version: 7.2.26024
MIOpen runtime version: 3.5.1
Is XNNPACK available: True

CPU:
Name: AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S      
Manufacturer: AuthenticAMD
Family: 107
Architecture: 9
ProcessorType: 3
DeviceID: CPU0
CurrentClockSpeed: 3000
MaxClockSpeed: 3000
L2CacheSize: 16384
L2CacheSpeed: None
Revision: 28672

Versions of relevant libraries:
[pip3] numpy==1.26.4
[pip3] torch==2.9.1+rocmsdk20260116
[pip3] torch-tb-profiler==0.4.3
[pip3] torchaudio==2.9.1+rocmsdk20260116
[pip3] torchvision==0.24.1+rocmsdk20260116
[conda] Could not collect
```

Nota: de existir algún cambio en el equipo actual de la memoria, esta información puede discrepar.
### Fase 2: Diagnóstico de Proyecto (Opcional)

Audite la carga de tensores y el sistema de métricas mediante los scripts locales:

**Prueba de memoria e iteración (GEMM):**

```powershell
python Environment/check_environment.py
```
*Resultado esperado:* ejecutar el script correctamente y retornar la terminal.
## 5. Purga de Entorno

Para desactivar y eliminar los artefactos del entorno virtual en el repositorio local (PowerShell):

```powershell
deactivate
Remove-Item -Recurse -Force .venv
```
