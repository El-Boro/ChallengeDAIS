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

**¿Por qué utilicé estas métricas?**

*   **F1-Score:** Se utilizó como métrica principal para hallar el punto de máxima eficiencia operativa (Umbral de 0.511), equilibrando el costo logístico y el riesgo financiero. Al ser la media armónica entre precisión y recall, penaliza los extremos y resulta especialmente útil en escenarios con alto desbalance de clases (como se validó en mi proyecto final con asimetrías del 90/10 entre las dos clases a clasificar).
*   **Precisión:** Evalúa la confiabilidad de las detecciones del modelo. Es importante que el valor sea alto para confiar en la correcta detección del activo que identificó el modelo.
*   **Recall:** Cuantifica la capacidad del modelo para encontrar todos los activos físicos (minimizando los Falsos Negativos). Mantener un recall alto previene la omisión de equipos en el campo, evitando fallas en la contabilidad general, siendo lo que busca el cliente.
*   **mAP50:** Dado que el cliente necesita contabilizar activos, no se busca una precisión milimétrica en los bordes de la *bounding box*. Confirmar la presencia y ubicación general del objeto es suficiente para resolver el problema de negocio.

### 5. ¿Qué haría distinto con más tiempo o datos?

1. **Imágenes Georreferenciadas y Overlap:** En vuelos de dron sobre locaciones repetitivas, el mismo skid aparece en múltiples recortes adyacentes (por ejemplo las imágenes img_00055.jpg e img_00061.jpg). Con más tiempo, integraría telemetría o metadata espacial para aplicar seguimiento *inter-frame* y evitar contar el mismo equipo dos veces en recortes diferentes, ya que el objetivo final del cliente es contabilizar con exactitud los activos físicos que posee.

2. **Focal Loss:** Probaría utilizar explícitamente la función de pérdida *Focal Loss* (como se mencionó anteriormente) para ajustar dinámicamente los pesos de las clases Skids y Baños. Esto ayudaría a mejorar el rendimiento en el reconocimiento de la clase minoritaria, aun si penaliza levemente la métrica general de los Skids.

### 6. Zonas Grises: ¿Qué no verifiqué y podría no ser cierto?

* **Análisis en detalle del dataset:** Por falta de tiempo no hice una inspección visual en detalle del conjunto de datos. Revisé parcialmente el conjunto de testeo para entender que características presentan los activos que busca identificar el cliente y entender mejor el problema a resolver pero no verifiqué al completo el conjunto de datos. Esto con un conjunto de datos limpio no es un problema pero de no ser el caso podría acarrear problemas en el entrenamiento del modelo posteriormente.

* **Análisis visual exhaustivo de la predicción:** Por experiencia en proyectos de visión computacional (ej. Proyecto final de detección de arbolado), considero que una inspección visual cualitativa sobre la predicción del modelo permite entender mejor en qué casos contextuales tiene mayor dificultad. Realicé un análisis visual rápido sobre un batch de 16 imágenes de *Test*, observando un desempeño generalmente correcto, con falencias esporádicas en activos muy lejanos y un caso aislado de confusión entre un Baño y un Volquete. Al no inspeccionar un volumen mayor de imágenes, las conclusiones sobre estos falsos positivos carecen de rigor estadístico visual.

* **Robustez frente a cambios ambientales drásticos:** Las sombras proyectadas (que son características visuales fuertes) cambian diametralmente en invierno vs. verano en la Patagonia. Al no tener la fecha y hora exacta de cada vuelo, no puedo garantizar que el F1-Score se mantenga estable bajo condiciones de iluminación extrema. En caso de detectarse dificultades en producción, se podría recurrir a Aumentación de Datos en iluminación para mitigar el problema, aunque la variabilidad natural del dataset actual puede ser suficiente.

# Parte 2: Análisis de Estimación de Nivel en Bidones

## 1. Viabilidad a partir de las imágenes actuales

Si bien los bidones cuentan con un enrejado metálico exterior que, a simple vista, actúa como una guía visual para calcular el nivel de forma aproximada, mi conclusión es que estimar el llenado de forma automatizada y confiable con *este* set de imágenes sigue siendo **muy poco viable**, salvo en excepciones.

Las razones principales que dificultan aprovechar ese enrejado son:

* **Ángulo de captura:** Las imágenes varían entre un ángulo cenital (90° sobre la vertical) y 20° sobre la horizontal. En tomas verticales, solo se observa el techo del bidón, ocultando por completo el enrejado lateral y la línea de líquido. En esos casos es imposible saber el nivel de líquido.
* **Resolución efectiva:** Un skid completo mide en mediana 120 px. El maxibidón ocupa una fracción de eso. Intentar distinguir la línea del líquido (que suele ser translúcido) contrastándola con los barrotes de la jaula en una resolución tan baja introduce un margen de error altísimo.
* **Factores ópticos:** El plástico blanco del bidón permite reconocer el nivel de líquido en condiciones de luz pero se necesita un dataset específio orientado a este problema puntual con las condiciones de vuelo adaptadas a resolver este problema de negocio.

Solo se podría intentar adivinar el nivel en tomas capturadas a la altura mínima, con ángulo oblicuo y donde la iluminación no sature el plástico pero sí permita aprovechar su característica traslúcida.

## 2. Estrategia y etiquetado

Habría que comunicarse con el cliente para entender conceptualmente que tan fino debe ser el análisis de nivel de líquido. Un análisis fino considero que no sería posible pero un enfoque más generalista sería más viable aunque no descarto que pueda tratarse de un problema sin solución para los requerimientos que tenga el cliente.

Gracias a la existencia del enrejado metálico, buscaría utilizarlo como guía sencilla sobre el nivel de líquido sin intentar enfocar en la cantidad de litros de líquido, que considero que no es posible.

La estrategia sería:
* **Qué etiquetaría:** Usaría los propios cuadrantes de la jaula metálica para definir los niveles. Las clases serían, por ejemplo: `Nivel_Vacio`, `Nivel_Menos_Mitad`, `Nivel_Mas_Mitad`, `Nivel_Lleno`. Este enfoque considero que acepta un margen de error mayor para permitirle al modelo abordar el problema.
* **Variable:** Trataría al tanque entero como un bounding box y lo clasificaría directamente en una de esas 4 categorías. 
* **Cantidad de ejemplos:** Considero necesario un dataset entero enfocado en los maxibidones con el ángulo e iluminación correcto para abordar el problema. De igual manera entiendo que esto no debe ser posible en todos los casos y desconozco el tiempo meteorologíco habitual en la zona.
* **Aumentación de datos:** Para ayudar al modelo, y en caso de no ser posible conseguir características de vuelo idóneas, lo que podría ayudar es la aumentación de datos buscando generar un contraste entre el líquido y el plástico del bidón. Para probar esto elegí editar manualmente con la herramienta de software libre GIMP la imagen img_00055.jpg del conjunto Test, bajando el brillo a -127 y aumentando el contraste a 52. Si bien la imagen presenta mayor cantidad de ruido, se logra identificar de mejor manera el nivel de líquido.

## Uso de IA

Se utilizó herramientas de IA a la hora de estructurar el dataset en formato COCO al formato esperado por la función convert_coco() según se indica en la documentación. También se utilizó en otras secciones de programación para generar el código Python que necesitaba.
Además de esto, sirvió de base para generar el archivo README en formato markdown para estructurar el mismo y verificar errores en redacción.
