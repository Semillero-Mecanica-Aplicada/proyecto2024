import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def procesado(ruta,filtro,img=0):
    im = cv2.imread(ruta) 
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    umbral,im1 = cv2.threshold(im, 0, 255,  cv2.THRESH_OTSU) 
    label_im, nb_labels = sp.ndimage.label(im1)
    sizes = sp.ndimage.sum(im1, label_im, range(nb_labels + 1)) 
    mask_size = sizes < (sizes.max() - 10) 
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0 
    raiz = sp.ndimage.binary_opening(label_im, structure=np.ones((filtro,filtro)))
    label_raiz, nb_labels_raiz = sp.ndimage.label(raiz)
    sizes1 = sp.ndimage.sum(raiz, label_raiz, range(nb_labels_raiz + 1)) 
    mask_size1 = sizes1 < (sizes1.max() - 10)
    remove_pixel1 = mask_size1[label_raiz]
    label_raiz[remove_pixel1] = 0
    if label_raiz.max() > 1:  
        label_raiz_scaled = (255.0 / label_raiz.max() * label_raiz).astype(np.uint8)
    else:
        label_raiz_scaled = (label_raiz * 255).astype(np.uint8)

    if im1.max() > 1: 
        im1_scaled = (255.0 / im1.max() * im1).astype(np.uint8)
    else:
        im1_scaled = (im1 * 255).astype(np.uint8)
    if img != 0:
        return im1_scaled
    else:
        return label_raiz_scaled
    
def seleccion(ruta):
    imagen_original = procesado(ruta,3,1)
    imagen_3 = procesado(ruta,3)
    imagen_5 = procesado(ruta,5)
    area_total = 0
    for i in imagen_original:
        for j in i:
            if j != 0:
                area_total += 1
    labels3 = np.unique(imagen_3) 
    slice_x, slice_y = sp.ndimage.find_objects(imagen_3 == labels3[1])[0] 
    rec3 = imagen_3[slice_x, slice_y]
    area_3 = 0
    for i in rec3:
        for j in i:
            if j != 0:
                area_3 += 1
    labels5 = np.unique(imagen_5) 
    slice_x1, slice_y1 = sp.ndimage.find_objects(imagen_5 == labels5[1])[0] 
    rec3 = imagen_5[slice_x1, slice_y1]
    area_5 = 0
    for i in imagen_5:
        for j in i:
            if j != 0:
                area_5 += 1
    relacion_3 = (area_total- area_3)/area_total 
    relacion_5 = (area_total-area_5)/area_total
    relacion_3A = np.abs(relacion_3-40)
    relacion_5A = np.abs(relacion_5-40)
    if relacion_3A < relacion_5A:
        filtro = 3
        return imagen_3, filtro,relacion_3
    else:
        filtro = 5
        return imagen_5, filtro,relacion_5