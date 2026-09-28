# Detección de Activos en Locaciones Petroleras - Drone AI Services

## Parte 1: Análisis, Diseño y Evaluación del Modelo

### 1. Análisis previo al entrenamiento

Antes de definir la arquitectura o los hiperparámetros, analicé el dataset original y las notas proporcionadas, llegando a las siguientes conclusiones críticas que definieron el diseño:

* **Escala:** La mediana de los Skids ronda los 120px sobre imágenes recortadas a 640x640. Si el modelo achicaba las imágenes antes de procesarlas, los activos se convertirían en ruido irreconocible. **Decisión:** Forzar el entrenamiento y la inferencia a tamaño nativo (`imgsz=640`).

* **Desbalance de clases:** Con 1583 cajas de Skids y solo 34 de Baños, el dataset es altamente asimétrico. Cualquier intento de forzar al modelo a optimizar la clase `Bano` mediante manipulación de pesos corría el riesgo de degradar el rendimiento en `Skid`. Un enfoque que se consideró como próximo paso es la utilización de la función *Focal Loss* para corregir los pesos en escenarios de desbalance como este.

* **Data Augmentation:** Las imágenes provienen de vuelos reales con alturas variables (18m-75m) y ángulos de cámara (cenital a 20°). **Decisión:** Se decidió no aplicar aumentación de datos manual debido a la variabilidad natural inherente de las tomas descripta en el ReadMe provisto por DAIS y para evaluar el rendimiento base de la arquitectura (apoyándose únicamente en el *pipeline* por defecto de YOLO).

* **Entrenamiento y protección de datos:** Debido a la limitación de hardware personal, no me es posible realizar un entrenamiento de modelos de manera local garantizando una completa protección de datos, al estar alojados únicamente en mi entorno. Decidí crear un cuaderno en Google Colab, cargando el archivo .zip del dataset en mi unidad privada de Google Drive y entrenar con el hardware proporcionado por la compañía. Revisé no exponer los datos a ningún entorno público el cual puedan utilizar para entrenar sus modelos y eliminé todos los remanentes del proyecto posterior a la finalización del mismo, según lo indicado.

### 2. De COCO a YOLO: Transformación y Elección de Arquitectura

**El punto de partida:** El dataset original fue provisto en formato COCO estándar (Anotaciones en un archivo `_annotations.coco.json` por partición, con *bounding boxes* en píxeles absolutos). Decidí migrar la estructura y utilizar la arquitectura **YOLO** con un modelo pre-entrenado para conseguir un entrenamiento ágil y eficiente.

**La transformación de los datos:** Para compatibilizar los datos con YOLO, utilicé como guía la [documentación oficial de Ultralytics](https://docs.ultralytics.com/guides/coco-to-yolo). Previo a realizar la transformación, se modificó la estructura de carpetas del dataset original para ajustarse al árbol de directorios esperado por la función `convert_coco()`.

### 3. Arquitectura y Parámetros

* **Modelo Base:** `YOLOv8s` (Small).

* **Por qué:** La versión "Small" logra procesar imágenes a < 5ms por recorte en GPU (tiempo de sobra para aplicaciones en tiempo real), mientras que la **Transferencia de Aprendizaje** (pesos pre-entrenados en COCO) compensa la falta de datos en clases minoritarias, al contar con una red que ya posee capacidades primarias de extracción de características visuales.

* **Hiperparámetros:** El conjunto de parámetros utilizado fue una prueba base orientada a agilizar el entrenamiento en búsqueda de evaluar el comportamiento general de la arquitectura YOLO. Los resultados demostraron un rendimiento aceptable rápidamente, por lo que decidí enfocar el análisis en los resultados de dicho modelo, ya que la exploración exhaustiva de modelos con mejor rendimiento empírico no era el enfoque principal pedido en el Challenge.

### 4. Métricas de Evaluación: ¿Por qué estas y no otras?

Se descartó el mAP (*Mean Average Precision*) general como métrica principal, ya que el bajo rendimiento inherente de la clase `Bano` arrastra el promedio y oculta el éxito en las clases mayoritarias. Además, se descartó el *Accuracy* global por no ser apto para datasets desbalanceados. Utilizar el *Accuracy* como métrica principal podría llevar a pensar erróneamente que el modelo presenta un buen rendimiento global (por ejemplo, identificando únicamente Skids y Volquetes, pero ignorando Baños), ya que el desbalance sesgaría fuertemente los resultados.

Se priorizaron **métricas clase por clase**, evaluadas sobre el conjunto de *Test* (datos nunca antes vistos por el modelo):

**Resultados en el conjunto de Test (Umbral de Confianza óptimo = 0.511)**

| Clase | F1-Score | Precisión | Recall | mAP50 | 
 | ----- | ----- | ----- | ----- | ----- | 
| **Skid** | 0.903 | 0.899 | 0.907 | 0.940 | 
| **Volquete** | 0.780 | 0.823 | 0.742 | 0.770 | 
| **Bano** | 0.632 | 0.766 | 0.538 | 0.690 | 

### 5. ¿Qué haría distinto con más tiempo o datos?

1. **Imágenes Georreferenciadas y Overlap:** En vuelos de dron sobre locaciones repetitivas, el mismo skid aparece en múltiples recortes adyacentes. Con más tiempo, integraría telemetría o metadata espacial para aplicar seguimiento *inter-frame* y evitar contar el mismo equipo dos veces en recortes diferentes, ya que el objetivo final del cliente es contabilizar con exactitud los activos físicos que posee.

2. **Focal Loss:** Probaría utilizar explícitamente la función de pérdida *Focal Loss* (como se mencionó anteriormente) para ajustar dinámicamente los pesos de las clases Skids y Baños. Esto ayudaría a mejorar el rendimiento en el reconocimiento de la clase minoritaria, aun si penaliza levemente la métrica general de los Skids.

### 6. Zonas Grises: ¿Qué no verifiqué y podría no ser cierto?

* **Análisis en detalle del dataset:** Por falta de tiempo no hice una inspección visual en detalle del conjunto de datos. Revisé parcialmente el conjunto de testeo para entender que características presentan los activos que busca identificar el cliente y entender mejor el problema a resolver pero no verifiqué al completo el conjunto de datos. Esto con un conjunto de datos limpio no es un problema pero de no ser el caso podría acarrear problemas en el entrenamiento del modelo posteriormente.

* **Análisis visual exhaustivo de la predicción:** Por experiencia en proyectos de visión computacional (ej. detección de arbolado), considero que una inspección visual cualitativa sobre la predicción del modelo permite entender mejor en qué casos contextuales tiene mayor dificultad. Realicé un análisis visual rápido sobre un batch de 16 imágenes de *Test*, observando un desempeño generalmente correcto, con falencias esporádicas en activos muy lejanos y un caso aislado de confusión entre un Baño y un Volquete. Al no inspeccionar un volumen mayor de imágenes, las conclusiones sobre estos falsos positivos carecen de rigor estadístico visual.

* **Robustez frente a cambios ambientales drásticos:** Las sombras proyectadas (que son características visuales fuertes) cambian diametralmente en invierno vs. verano en la Patagonia. Al no tener la fecha y hora exacta de cada vuelo, no puedo garantizar que el F1-Score se mantenga estable bajo condiciones de iluminación extrema. En caso de detectarse dificultades en producción, se podría recurrir a Aumentación de Datos en iluminación para mitigar el problema, aunque la variabilidad natural del dataset actual puede ser suficiente.

## Parte 2: \[Título a definir\]

\[Aquí va el contenido de la parte 2...\]