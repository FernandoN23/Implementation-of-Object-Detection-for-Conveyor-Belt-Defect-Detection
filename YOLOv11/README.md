# YOLOv11

---

## 🔧 Entrenamiento — YOLOv11

### Descripción General
El proceso de entrenamiento del modelo **YOLOv11** fue diseñado bajo una estructura modular, garantizando legibilidad, escalabilidad y control experimental por variantes (n, s, m, l, xl).  
El flujo integra los componentes principales del proyecto: *modelos, configuraciones, utilidades, métricas y gestión de checkpoints.*

---

### 🧩 Estructura del Flujo de Entrenamiento

```text
YOLOv11/
│
├── configs/
│   ├── dataset.yaml          ← rutas y clases del dataset
│   ├── yolo11.yaml           ← definición estructural del modelo
│   ├── model_variants.yaml   ← parámetros de escalado (depth/width)
│   ├── train.yaml            ← hiperparámetros de entrenamiento
│   ├── valid.yaml            ← validación y métricas
│   └── parser.yaml           ← rutas y opciones globales
│
├── models/                   ← Backbone, Neck y Head del detector
├── utility/                  ← funciones de soporte (losses, logs, métricas, etc.)
├── train.py                  ← script principal de entrenamiento
└── runs/, logs/, checkpoints/ ← resultados, registros y pesos


```

⚙️ Componentes Principales
1. Parser YAML (parser_yaml.py)

    Lee las configuraciones del modelo, dataset y entrenamiento desde archivos .yaml.
Permite parametrizar rutas, hiperparámetros y variantes sin modificar el código fuente.

2. Modelo YOLOv11 (yolo11.py)

    Integra los tres módulos principales:

    Backbone → extracción jerárquica de características (bloques Conv y C3k2).

    Neck → fusión FPN + PAN para combinación multi-escala.

    Head → predicción de cajas, clases y confianza por nivel de resolución.

3. Carga de Datos (data_loader.py)

    Implementa CustomDataset y DataLoader para imágenes y etiquetas YOLO.
Soporta lectura de data.yaml, cacheo en RAM y collate_fn para lotes variables.

4. Función de Pérdida (losses.py)

    Define la clase YoloLoss, que calcula:

    Pérdida de cajas (λ_box)

    Pérdida de objeto (λ_obj)

    Pérdida de clasificación (λ_cls)
Usando funciones SmoothL1, BCE y MSE respectivamente.

5. Registro y Visualización

    logger.py → registra mensajes de entrenamiento y validación.

    visualization.py → activa TensorBoard con subcarpetas por variante (n, s, m...).
Permite monitorear loss, mAP, precision y recall en tiempo real.

6. Gestión de Pesos (weights.py)

    Guarda y carga checkpoints:

    save_checkpoint() → crea epoch_xx.pt y latest.pt.

    load_checkpoint() → permite reanudar entrenamientos interrumpidos.

7. Métricas y Resultados (metrics.py)

    Calcula y guarda métricas clave post-entrenamiento:

    Precision, Recall, AP, mAP, F-beta, IoU.

    Genera gráficos .png y resúmenes .txt por prueba (test_000X).

🚀 Flujo del Script train.py

1. Inicialización

2. Lectura de configuración desde parser.yaml y train.yaml.

3. Selección de variante del modelo (YOLOv11-n, YOLOv11-s, etc.).

4. Configuración del entorno TensorBoard.

5. Carga de Datos

6. Creación de train_loader con create_dataloader().

7. Verificación de estructura y clases del dataset.

8. Construcción del Modelo

9. Instanciación de YOLOv11(...).

10. Envío a dispositivo (cuda o cpu).

11. Inicialización de optimizador (AdamW, SGD, etc.).

12. Bucle de Entrenamiento

13. Forward → cálculo de pérdidas con YoloLoss.

14. Backward → actualización de pesos.

15. Registro de métricas en TensorBoard y logs/.

16. Guardado de checkpoints cada N épocas (weights/).

17. Validación

18. Evaluación periódica según validate_every en train.yaml.

19. Cálculo de métricas (metrics.py) y guardado de resultados.

20. Finalización

21. Cierre de TensorBoard y logger.

22. Limpieza opcional de checkpoints con clean_checkpoints.py.

📊 Salidas del Entrenamiento

- Checkpoints: YOLOv11/checkpoints/{variant}/yolo11_epoch_X.pt

- Logs: YOLOv11/logs/{variant}/train_yolo11.log

- TensorBoard: YOLOv11/runs/{variant}/

- Métricas finales: YOLOv11/metrics/{variant}/test_XXXX/