import cv2,sys
import numpy as np




#################Funciones##############################################
# Detectar y limpiar texto en los ejes
# Definir una función para calcular la intersección de dos líneas
def line_intersection(line1, line2):
    x1, y1, x2, y2, _ = line1
    x3, y3, x4, y4, _ = line2

    # Calcular determinantes para encontrar el punto de intersección
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  # Las líneas son paralelas y no se intersectan

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    return int(px), int(py)

# Modificar la función de filtrado para devolver las líneas seleccionadas (horizontal y vertical)
def remove_duplicate_lines(lines, proximity_threshold=10, orientation='horizontal'):
    # Ordenar las líneas por posición (y para horizontales, x para verticales)
    lines = sorted(lines, key=lambda line: line[1] if orientation == 'horizontal' else line[0])
    unique_lines = []

    for line in lines:
        x1, y1, x2, y2, length = line
        duplicate_found = False

        # Comparar con las líneas ya seleccionadas
        for unique_line in unique_lines:
            ux1, uy1, ux2, uy2, ulength = unique_line

            if orientation == 'horizontal':
                # Para líneas horizontales, compara la coordenada Y
                if abs(y1 - uy1) < proximity_threshold and abs(y2 - uy2) < proximity_threshold:
                    duplicate_found = True
                    break
            elif orientation == 'vertical':
                # Para líneas verticales, compara la coordenada X
                if abs(x1 - ux1) < proximity_threshold and abs(x2 - ux2) < proximity_threshold:
                    duplicate_found = True
                    break

        # Agregar la línea si no es duplicada
        if not duplicate_found:
            unique_lines.append(line)

    return unique_lines

def filter_lines(lines, width, height):
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        # Filtrar líneas horizontales y verticales
        if abs(dx) > abs(dy) * 5:  # Línea horizontal
            horizontal_lines.append((x1, y1, x2, y2, length))
        elif abs(dy) > abs(dx) * 5:  # Línea vertical
            vertical_lines.append((x1, y1, x2, y2, length))

    # Eliminar líneas solapadas en cada orientación
    horizontal_lines = remove_duplicate_lines(horizontal_lines, orientation='horizontal')
    vertical_lines = remove_duplicate_lines(vertical_lines, orientation='vertical')

    # Ordenar por longitud y seleccionar las dos más largas para cada eje
    horizontal_lines = sorted(horizontal_lines, key=lambda line: line[4], reverse=True)[:2]
    vertical_lines = sorted(vertical_lines, key=lambda line: line[4], reverse=True)[:2]

    # Seleccionar la línea horizontal más baja de las dos más largas
    if len(horizontal_lines) == 2:
        horizontal_line = max(horizontal_lines, key=lambda line: line[1])
        parallel_horizontal_line = min(horizontal_lines, key=lambda line: line[1])
    elif horizontal_lines:
        horizontal_line = horizontal_lines[0]
        parallel_horizontal_line = None
    else:
        horizontal_line = None
        parallel_horizontal_line = None

    # Seleccionar la línea vertical más a la izquierda de las dos más largas
    if len(vertical_lines) == 2:
        vertical_line = min(vertical_lines, key=lambda line: line[0])
        parallel_vertical_line = max(vertical_lines, key=lambda line: line[0])
    elif vertical_lines:
        vertical_line = vertical_lines[0]
        parallel_vertical_line = None
    else:
        vertical_line = None
        parallel_vertical_line = None

    return vertical_line, parallel_vertical_line, horizontal_line, parallel_horizontal_line

def draw_lines(image, lines, vertical_lines=None, horizontal_lines=None, intersection_points=[]):
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if vertical_lines:
        for vline in vertical_lines:
            x1, y1, x2, y2, _ = vline
            if vline == vertical_lines[0]:  # Eje Y
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            else:  # Línea paralela al eje Y
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    if horizontal_lines:
        for hline in horizontal_lines:
            x1, y1, x2, y2, _ = hline
            if hline == horizontal_lines[0]:  # Eje X
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:  # Línea paralela al eje X
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    for i, point in enumerate(intersection_points):
        px, py = point
        if i == 0:
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)  # Origen de coordenadas en rojo
        else:
            cv2.circle(image, (px, py), 5, (0, 165, 255), -1)  # Otros puntos en naranja
    return image


#########################################################################

##########################––––––––––MODELO---------------##################################





##########################––––––––––MODELO---------------##################################
