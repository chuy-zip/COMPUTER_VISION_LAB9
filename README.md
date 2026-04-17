# Laboratorio 9 -- CC3182 Visión por Computadora

**Integrantes:**
- Sergio Orellana 221122
- Rodrigo Mansilla 22611
- Ricardo Chuy 221007

## Resultado:

![Alt Text](./video/demo.gif)

Tambien se puede ver el archivo en la carpeta /video y luego al seleccionar el archivo llamado Ejecucion.mp4

## Créditos y Fuentes

### Dataset de Cartas Pokémon TCG
El modelo utilizado en Task 3 fue entrenado con el dataset **pokemon-cards-63wlp v5**, publicado por Aaron en Roboflow Universe.  
🔗 https://universe.roboflow.com/aaron-qwuzu/pokemon-cards-63wlp

### Video de prueba
El video `cards.mp4` es un youtube short del canal de YouTube **@ShortPocketMonster**.  
🔗 https://www.youtube.com/shorts/BcpV2dgmznk

### Librerías utilizadas

| Librería | Uso |
|----------|-----|
| `ultralytics` | Carga del modelo YOLO |
| `opencv-python` | Captura de video y visualización de resultados |

**Instalación:**

```bash
pip install ultralytics opencv-python
```

> **Nota:** Si se tiene `opencv-python-headless` instalado (lo instala `roboflow` como dependencia), se debe desinstalar primero para evitar conflictos. Roboflow no es necesario ya que la extracción de pesos se hizo en colab. Otra altentativa es usar venv con todas las librerias indicadas en este readme. Esta librerias tambien son necesarias:
> ```bash
> pip uninstall opencv-python-headless
> pip install opencv-python
> ```
