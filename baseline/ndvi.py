import torch
import numpy as np

def compute_ndvi(image):
    """
    Calculer l'indice NDVI pour une image satellite.
    NDVI = (NIR - RED) / (NIR + RED)
    Les canaux NIR et RED sont typiquement les bandes 8 et 4 dans les images Sentinel-2.
    """
    nir = image[6]  # Canal proche infrarouge (NIR)
    red = image[2]  # Canal rouge (RED)
    ndvi = (nir - red) / (nir + red + 1e-6) 
    return ndvi

def calculate_ndvi_means(x, bid=0):
    """
    Calculer le NDVI moyen pour chaque image d'une séquence temporelle.
    
    x: données d'entrée (tensor)
    bid: index du batch (pour sélectionner une séquence spécifique)
    
    Retourne une liste de NDVI moyens pour chaque image de la séquence.
    """
    sequence = x['S2'][bid]
    T, C, H, W = sequence.shape
    
    ndvi_means = []
    
    for t in range(T):
        ndvi = compute_ndvi(sequence[t])
        ndvi_mean = torch.mean(ndvi).item() 
        ndvi_means.append(ndvi_mean)
    
    return ndvi_means


def remove_low_ndvi_images(x, ndvi_means, threshold, bid=0):
    """
    Supprimer les images dont le NDVI moyen est inférieur au seuil.

    x: données d'entrée (tensor)
    ndvi_means: liste des NDVI moyens pour chaque image
    threshold: seuil NDVI à utiliser pour supprimer les images
    bid: index du batch
    
    Retourne une nouvelle structure avec les images supprimées.
    """
    sequence = x['S2'][bid]
    T, C, H, W = sequence.shape

    valid_images = [sequence[t] for t, ndvi_mean in enumerate(ndvi_means) if ndvi_mean >= threshold]

    
    new_sequence = torch.stack(valid_images)

    new_x = x.copy() 
    new_x['S2'] = x['S2'].clone() 
    
    
    original_size = new_x['S2'][bid].shape[0]
    needed_size = original_size - new_sequence.shape[0]
    
    if needed_size >0:
        new_x['S2'][bid] = torch.cat((new_sequence, new_sequence[:needed_size]), dim=0)
    else:
        new_x['S2'][bid] = new_sequence

    return new_x
