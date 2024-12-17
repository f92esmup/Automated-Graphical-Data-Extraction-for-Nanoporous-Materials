# Import necessary libraries
import numpy as np
import scipy
import cv2
import BB_Inference as BB_Inference
import matplotlib.pyplot as plt
import os
import csv

# For line interpolation:
from bresenham import bresenham

##############################################################
def process_image(inference,image_path):
    """
    Procesa una imagen para obtener la serie de datos de línea.
    """
    # Carga la imagen en formato BGR
    image = cv2.imread(image_path)
    
    # Obtiene la serie de datos de la línea usando la instancia de inferencia
    line_dataseries = inference.get_dataseries(image)
    
    # Convierte y limpia los puntos si es necesario
    line_dataseries = convert_points(line_dataseries)
    
    line_dataseries = elimanate_duplicates(line_dataseries)
    
    return line_dataseries

def save_and_plot_data(line_dataseries, image_path,plot=False, save_path='./data/Line_output/'):
    """
    Guarda la serie de datos en un archivo CSV y genera un gráfico.
    """
    # Obtiene el nombre del archivo sin la extensión
    name = image_path.split('/')[-1].split('.')[0]
    
    # Guarda la serie de datos en un archivo CSV
    get_csv_data(line_dataseries,save_path, name)
    
    # Genera el gráfico de la serie de datos
    if plot:
        plot_line(line_dataseries)

def draw_lines_on_image(image_path, line_dataseries, output_dir=None):
 
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}.")
        return

    # Dibujar cada línea en la imagen con un color diferente
    for line in line_dataseries:
        # Generar un color aleatorio para cada línea
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        
        for i in range(len(line) - 1):
            # Extraer los puntos consecutivos (x1, y1) y (x2, y2)
            (x1, y1) = line[i]
            (x2, y2) = line[i + 1]
            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    # Guardar o mostrar la imagen
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"lines_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, image)
        print(f"Imagen con líneas guardada en: {output_path}")
    else:
        cv2.imshow("Imagen con Líneas", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


##############################################################

# Functions for extracting data from images
def get_images_path(dir='./'):
    # Lista todos los archivos en el directorio
    files = os.listdir(dir)
    
    # Filtra solo los archivos (excluye directorios)
    # List comprehension to filter the list 'archivos' [nueva_lista] = [expresión for elemento in secuencia if condición ]
    # 'f' iterates over each element in 'archivos'
    # 'os.path.join(directorio, f)' constructs the full path of each file
    # 'os.path.isfile()' checks if the constructed path is a regular file
    # Only files are included in the new 'archivos' list
    #archivos = [f for f in archivos if os.path.isfile(os.path.join(directorio, f))]


    # Filtra solo los archivos que son imágenes (extensiones .png, .jpg, .jpeg, .gif)
    # os.path.splitext() divides the name into a root and an extension like ('image1', '.png') which is a tuple.
    # 'ext' is the extension part of the tuple
    # 'ext.lower()' converts the extension to lowercase PNG -> png.
    # 'os.path.join(directorio, file)' constructs the full path of each file.

    image_extension = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif','.tiff', '.webp')

    files = [os.path.join(dir,f) for f in files if os.path.splitext(f)[1].lower() in image_extension]

    return files

def get_csv_data(Line_dataseries,save_path,name="data"):
    
    # Create a CSV file to store the data of ONE chart
    with open(save_path + name + ".csv", mode='w', newline="") as datos_csv:
        # Create a writer object
        archivo = csv.writer(datos_csv) #Can change the delimiter with 'delimiter='; default is ','
        # Write the header. For each chart, we have the columns 'LineID', 'X', and 'Y'
            #LineID: The ID of the line in the chart
            #X: The X-coordinate of the point
            #Y: The Y-coordinate of the point
        archivo.writerow(['LineID','X', 'Y'])
        # Write the data
        for i,line in enumerate(Line_dataseries):
            for pt in line:
                archivo.writerow([i, pt[0], pt[1]])


    return


###############-------------------##################
def get_xrange(bin_line_mask): # bin_line_mask is a binary image
    # Apply a median filter to the sum of the binary line mask along the columns
    smooth_signal = scipy.signal.medfilt(bin_line_mask.sum(axis=0), kernel_size=5)
    # Find the non-zero elements in the smoothed signal
    x_range = np.nonzero(smooth_signal)
    # If there are non-zero elements, get the first and last indices
    if len(x_range) and len(x_range[0]):
        x_range = x_range[0][[0, -1]]
    else:
        x_range = None
    return x_range

def get_kp(line_img, interval=10, x_range=None, get_num_lines=False, get_center=True):
    # Get the height and width of the image
    im_h, im_w = line_img.shape[:2]
    kps = []
    # If x_range is not provided, set it to the full width of the image
    if x_range is None:
        x_range = (0, im_w)
    num_comps = []
    # Iterate over the x_range with the given interval
    for x in range(x_range[0], x_range[1], interval):
        fg_y = []
        fg_y_center = []
        # Find the y-coordinates where the pixel value is 255
        all_y_points = np.where(line_img[:, x] == 255)[0]
        if all_y_points.size != 0:
            fg_y.append(all_y_points[0])
            y = all_y_points[0]
            n_comps = 1
            # Iterate over the y-coordinates to find connected components
            for idx in range(1, len(all_y_points)):
                y_next = all_y_points[idx]
                if abs(y_next - y) > 2:
                    n_comps += 1
                    if fg_y[-1] != y:
                        fg_y_center.append(round(y + fg_y[-1]) // 2)
                        fg_y.append(y)
                    else:
                        fg_y_center.append(y)
                    fg_y.append(y_next)
                y = y_next
            if fg_y[-1] != y:
                fg_y_center.append(round(y + fg_y[-1]) // 2)
                fg_y.append(y)
            else:
                fg_y_center.append(y)
            num_comps.append(n_comps)
        # If there are foreground points and only one component, add keypoints
        if (fg_y or fg_y_center) and (n_comps == 1):
            if get_center:
                kps.extend([{'x': float(x), 'y': y} for y in fg_y_center])
            else:
                kps.extend([{'x': float(x), 'y': y} for y in fg_y])
    res = kps
    # If get_num_lines is True, return keypoints and the 85th percentile of num_comps
    if get_num_lines:
        res = kps, int(np.percentile(num_comps, 85))
    return res

def get_interp_points(ptA, ptB, thickness=1):
    points = []
    # Define the range for delta based on thickness
    delta_range = (-thickness // 2, thickness // 2)
    # Use Bresenham's algorithm to interpolate points between ptA and ptB
    for delta in range(delta_range[0], delta_range[1] + 1):
        points.extend(list(bresenham(ptA[0], ptA[1] + delta, ptB[0], ptB[1] + delta)))
    # Convert the list of points to a numpy array
    inter_points = np.array(points)
    return inter_points
###############--------------------##################



# Functions for transforming data

def plot_line(line_dataseries):
    try:
        # A list of lists is like a matrix, each list is a row of the matrix. Nevertheless, it's defined as a list, not like a np.array.
        for line in line_dataseries:
            x = []
            y = []

            for pt in line:
                x.append(pt[0])
                y.append(pt[1])
            
            plt.plot(x, y)
        
        # In many image coordinate systems (such as in computer graphics and image processing), the origin (0,0) is at the top-left corner, and y-coordinates increase downwards.
        # Therefore, the y-axis needs to be inverted. However, this line is commented out to keep the y-axis normal.
        plt.gca().invert_yaxis() 
        plt.xlabel('pixel')
        plt.ylabel('pixel')
        plt.title('Line')
        plt.show() 

    except Exception as e:
        print(e)
    return


def convert_points(data):
    convert = []
    #data is a list of list of dictionaries so:
    if type(data[-1][-1]) is dict:
        for line in data:
            convert.append([[pt['x'], pt['y']] for pt in line])
            #convert is a list of lists, each list contains the x and y coordinates of a point.
    else:
        for line in data:
            convert.append([{'x': pt[0],'y': pt[1]} for pt in line])

    #Reorder data: the syntax is sequence[start:stop:step].
    #convert = [line[::-1] for line in convert]
    return convert


def elimanate_duplicates(Line_dataseries, threshold=1):
    #mmdetection use the same length for all the lines in the same chart.
    def similarity_score(line1,line2,threshold):
        comparison = []
        for pt1, pt2 in zip(line1, line2):
            if (pt2[0] - threshold <= pt1[0] <= pt2[0] + threshold and
            pt1[0] - threshold <= pt2[0] <= pt1[0] + threshold and
            pt2[1] - threshold <= pt1[1] <= pt2[1] + threshold and
            pt1[1] - threshold <= pt2[1] <= pt1[1] + threshold):
                comparison.append(1)
            else:
                comparison.append(0)

        score = sum(comparison)/len(comparison)
        #print(score)
        return score
    
    unique_lines = []
    for line in Line_dataseries:
        Duplicate = False
        for unique_line in unique_lines:
            if similarity_score(line, unique_line, threshold) > 0.9:
                Duplicate = True
                break
        if not Duplicate:
            unique_lines.append(line)
        
    return unique_lines

