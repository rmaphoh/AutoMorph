# Retipy - Retinal Image Processing on Python
# Copyright (C) 2018  Alejandro Valdes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Module with operations related to extracting tortuosity measures."""

import math
import numpy as np
from function_ import fractal_dimension, smoothing
from retipy import math as m
from retipy.retina import Retina, Window, detect_vessel_border
from scipy.interpolate import CubicSpline
from PIL import Image
import time
import cv2

def fractal_dimension(Z):

    assert(len(Z.shape) == 2)

    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        return len(np.where((S > 0) & (S < k*k))[0])

    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]



def vessel_density(Z):

    assert(len(Z.shape) == 2)
    vessel_total_count = np.sum(Z==1)
    pixel_total_count = Z.shape[0]*Z.shape[1]
    
    return vessel_total_count/pixel_total_count


def global_cal(retina):
    vessel_ = retina.vessel_image
    skeleton = retina.np_image
    
    if np.max(vessel_)>1:
        vessel_=vessel_/255
    if np.max(skeleton)>1:
        skeleton=skeleton/255
        
    FD_boxcounting = fractal_dimension(vessel_)
    VD = vessel_density(vessel_)
    width = np.sum(vessel_)/np.sum(skeleton)*retina.resolution
    
    return FD_boxcounting,VD,width


def Hubbard_cal(w1,w2):

    w_artery = np.sqrt(0.87*np.square(w1) + 1.01*np.square(w2) - 0.22*w1*w2 - 10.76) 
    w_vein = np.sqrt(0.72*np.square(w1)+0.91*np.square(w2)+450.05)
    
    return w_artery,w_vein

def Knudtson_cal(w1,w2):
    w_artery = 0.88*np.sqrt(np.square(w1) + np.square(w2)) 
    w_vein = 0.95*np.sqrt(np.square(w1) + np.square(w2)) 
    
    return w_artery,w_vein


def _distance_2p(x1, y1, x2, y2):
    """
    calculates the distance between two given points
    :param x1: starting x value
    :param y1: starting y value
    :param x2: ending x value
    :param y2: ending y value
    :return: the distance between [x1, y1] -> [x2, y2]
    """
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _curve_length(x, y):
    """
    calculates the length(distance) of the given curve, iterating from point to point.
    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the curve length
    """
    distance = 0
    for i in range(0, len(x) - 1):
        distance += _distance_2p(x[i], y[i], x[i + 1], y[i + 1])
    return distance


def _chord_length(x, y):
    """
    distance between starting and end point of the given curve

    :param x: the x component of the curve
    :param y: the y component of the curve
    :return: the chord length of the given curve
    """
    return _distance_2p(x[0], y[0], x[len(x) - 1], y[len(y) - 1])


def _detect_inflection_points(x, y):
    """
    This method detects the inflection points of a given curve y=f(x) by applying a convolution to
    the y values and checking for changes in the sign of this convolution, each sign change is
    interpreted as an inflection point.
    It will ignore the first and last 2 pixels.
    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the array position in x of the inflection points.
    """
    cf = np.convolve(y, [1, -1])
    inflection_points = []
    for iterator in range(2, len(x)):
        if np.sign(cf[iterator]) != np.sign(cf[iterator - 1]):
            inflection_points.append(iterator - 1)
    return inflection_points


def _curve_to_image(x, y):
    # get the maximum and minimum x and y values
    mm_values = np.empty([2, 2], dtype=np.int32)
    mm_values[0, :] = 99999999999999
    mm_values[1, :] = -99999999999999
    for i in range(0, len(x)):
        if x[i] < mm_values[0, 0]:
            mm_values[0, 0] = x[i]
        if x[i] > mm_values[1, 0]:
            mm_values[1, 0] = x[i]
        if y[i] < mm_values[0, 1]:
            mm_values[0, 1] = y[i]
        if y[i] > mm_values[1, 1]:
            mm_values[1, 1] = y[i]
    distance_x = mm_values[1, 0] - mm_values[0, 0]
    distance_y = mm_values[1, 1] - mm_values[0, 1]
    # calculate which square with side 2^n of size will contain the line
    image_dim = 2
    while image_dim < distance_x or image_dim < distance_y:
        image_dim *= 2
    image_dim *= 2
    # values to center the
    padding_x = (mm_values[1, 0] - mm_values[0, 0]) // 2
    padding_y = (mm_values[1, 1] - mm_values[0, 1]) // 2

    image_curve = np.full([image_dim, image_dim], False)

    for i in range(0, len(x)):
        x[i] = x[i] - mm_values[0, 0]
        y[i] = y[i] - mm_values[0, 1]

    for i in range(0, len(x)):
        image_curve[x[i], y[i]] = True

    return Retina(image_curve, "curve_image")


def linear_regression_tortuosity(x, y, sampling_size=6, retry=True):
    """
    This method calculates a tortuosity measure by estimating a line that start and ends with the
    first and last points of the given curve, then samples a number of pixels from the given line
    and calculates its determination coefficient, if this value is closer to 1, then the given
    curve is similar to a line.

    This method assumes that the given parameter is a sorted list.

    Returns the determination coefficient for the given curve
    :param x: the x component of the curve
    :param y: the y component of the curve
    :param sampling_size: how many pixels
    :param retry: if regression fails due to a zero division, try again by inverting x and y
    :return: the coefficient of determination of the curve.
    """
    if len(x) < 4:
        raise ValueError("Given curve must have more than 4 elements")
    try:
        min_point_x = x[0]
        min_point_y = y[0]

        slope = (y[len(y) - 1] - min_point_y)/(x[len(x) - 1] - min_point_x)

        y_intercept = min_point_y - slope*min_point_x

        sample_distance = max(round(len(x) / sampling_size), 1)

        # linear regression function
        def f_y(x1):
            return x1 * slope + y_intercept

        # calculate y_average
        y_average = 0
        item_count = 0
        for i in range(1, len(x) - 1, sample_distance):
            y_average += y[i]
            item_count += 1
        y_average /= item_count

        # calculate determination coefficient
        top_sum = 0
        bottom_sum = 0
        for i in range(1, len(x) - 1, sample_distance):
            top_sum += (f_y(x[i]) - y_average) ** 2
            bottom_sum += (y[i] - y_average) ** 2

        r_2 = top_sum / bottom_sum
    except ZeroDivisionError:
        if retry:
            #  try inverting x and y
            r_2 = linear_regression_tortuosity(y, x, retry=False)
        else:
            r_2 = 1  # mark not applicable vessels as not tortuous?
    if math.isnan(r_2):  # pragma: no cover
        r_2 = 0
    return r_2




def distance_measure_tortuosity(x, y):
    """
    Distance measure tortuosity defined in:
    William E Hart, Michael Goldbaum, Brad Côté, Paul Kube, and Mark R Nelson. Measurement and
    classification of retinal vascular tortuosity. International journal of medical informatics,
    53(2):239–252, 1999.

    :param x: the list of x points of the curve
    :param y: the list of y points of the curve
    :return: the arc-chord tortuosity measure
    """
    if len(x) < 2:
        raise ValueError("Given curve must have at least 2 elements")

    return _curve_length(x, y)/_chord_length(x, y)



def distance_inflection_count_tortuosity(x, y):
    """
    Calculates the tortuosity by using arc-chord ratio multiplied by the curve inflection count
    plus 1

    :param x: the list of x points of the curve
    :param y: the list of y points of the curve
    :return: the inflection count tortuosity
    """
    return distance_measure_tortuosity(x, y) * (len(_detect_inflection_points(x, y)) + 1), len(_detect_inflection_points(x, y))


def fractal_tortuosity(retinal_image: Retina):
    """
    Calculates the fractal dimension of the given image.
    The method used is the Minkowski-Bouligand dimension defined in
    https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension
    :param retinal_image:  a retinal image.
    :return: the fractal dimension of the given image
    """
    return fractal_dimension.fractal_dimension(retinal_image.np_image)


def fractal_tortuosity_curve(x, y):
    image = _curve_to_image(x, y)
    return fractal_dimension.fractal_dimension(image.np_image)


def tortuosity_density(x, y):
    """
    Defined in "A Novel Method for the Automatic Grading of Retinal Vessel Tortuosity" by Grisan et al.
    DOI: 10.1109/IEMBS.2003.1279902

    :param x: the x points of the curve
    :param y: the y points of the curve
    :return: tortuosity density measure
    """
    inflection_points = _detect_inflection_points(x, y)
    n = len(inflection_points)
    if not n:
        return 0
    starting_position = 0
    sum_segments = 0
    # we process the curve dividing it on its inflection points
    for in_point in inflection_points:
        segment_x = x[starting_position:in_point]
        segment_y = y[starting_position:in_point]
        chord = _chord_length(segment_x, segment_y)
        if chord:
            sum_segments += _curve_length(segment_x, segment_y) / _chord_length(segment_x, segment_y) - 1
        starting_position = in_point

    return (n - 1)/n + (1/_curve_length(x, y))*sum_segments


def squared_curvature_tortuosity(x, y):
    """
    See Measurement and classification of retinal vascular tortuosity by Hart et al.
    DOI: 10.1016/S1386-5056(98)00163-4
    :param x: the x values of the curve
    :param y: the y values of the curve
    :return: the squared curvature tortuosity of the given curve
    """
    curvatures = []
    x_values = range(1, len(x)-1)
    for i in x_values:
        x_1 = m.derivative1_centered_h1(i, x)
        x_2 = m.derivative2_centered_h1(i, x)
        y_1 = m.derivative1_centered_h1(i, y)
        y_2 = m.derivative2_centered_h1(i, y)
        curvatures.append((x_1*y_2 - x_2*y_1)/(y_1**2 + x_1**2)**1.5)
    return abs(np.trapz(curvatures, x_values))


def smooth_tortuosity_cubic(x, y):
    """
    TODO
    :param x: the list of x points of the curve
    :param y: the list of y points of the curve

    :return:
    """
    spline = CubicSpline(x, y)
    return spline(x[0])


'''
#2021/10/31 colour visualisation
def width_measurement(x, y, vessel_map):    
    
    width_list = []
    width_mask = np.zeros((vessel_map.shape))
    vessel_map = np.concatenate((vessel_map[...,np.newaxis],vessel_map[...,np.newaxis],vessel_map[...,np.newaxis]), axis=2)
    
    for i in range(1, len(x) - 1):
        
        #cv2.circle(vessel_map,(y[i],x[i]),radius=0,color=(0,0,255*(i/(len(x) - 1))),thickness=-1)
        cv2.circle(vessel_map,(y[i],x[i]),radius=0,color=(0,0,255),thickness=-1)
        #masked_vessel = vessel_map[width_mask>0]
        #print(np.unique(masked_vessel))
        #width_matrix = np.all(masked_vessel>0)
        cv2.imwrite('./intersection_test/test_mask_{}_{}.bmp'.format(x[0],y[0]),vessel_map)
    #width_list.append(width*2)
        
    return width_list
'''



def width_measurement(x, y, vessel_map):    
    
    width_list = []
    
    for i in range(0, len(x) - 1):
        width = 0
        width_matrix = 1
        width_mask = np.zeros((vessel_map.shape))
        width_cal = 0
    
        while width_matrix:
            width+=1
            cv2.circle(width_mask,(y[i],x[i]),radius=width,color=(255,255,255),thickness=-1)
            masked_vessel = vessel_map[width_mask>0]
            width_matrix = np.all(masked_vessel>0)
            
            #2021/10/31 test
            #test_case = vessel_map.copy()[...,np.newaxis]
            #test_case = np.concatenate((test_case,test_case,test_case),axis=2)
            
        #cv2.circle(test_case,(y[i],x[i]),radius=width,color=(0,0,255),thickness=-1)
        #cv2.imwrite('./intersection_test/test_{}_{}_{}.png'.format(y[i],x[i],width),test_case)
        #print(width*2)
        
        #print(np.shape(masked_vessel))
        #print(np.unique(masked_vessel))
        #print('255 is ',np.sum(masked_vessel==255))
        #print('0000 is ',np.sum(masked_vessel==0))
        #print('00000 is ',np.where(masked_vessel==0))
        
        if np.sum(masked_vessel==0)==1:
            width_cal = width*2
        elif np.sum(masked_vessel==0)==2:
            width_cal = width*2-1
        elif np.sum(masked_vessel==0)==3:
            width_cal = width*2-1
        else:
            width_cal = width*2
            
        width_list.append(width_cal)
        
    return width_list



def evaluate_window(window: Window, min_pixels_per_vessel=10, sampling_size=6, r2_threshold=0.80, store_path='/home/jupyter/Deep_rias/Results/M2/artery_vein/vein_binary_process'):  # pragma: no cover
    """
    Evaluates a Window object and sets the tortuosity values in the tag parameter.
    :param window: The window object to be evaluated
    :param min_pixels_per_vessel:
    :param sampling_size:
    :param r2_threshold:
    """
    #tags = np.empty([window.shape[0], 7])
    tags = np.empty([window.shape[0], 13])
    # preemptively switch to pytorch.
    window.mode = window.mode_pytorch
    #tft = fractal_tortuosity(window)
    tft = 0
    vessel_total_count = 0
    pixel_total_count = 0
    FD_binary,VD_binary,Average_width = 0,0,0

    for i in range(0, window.shape[0], 1):
        
        bw_window = window.windows[i, 0, :, :]
        
        vessel_total_count = np.sum(bw_window==1)
        pixel_total_count = bw_window.shape[0]*bw_window.shape[1]
        
        retina = Retina(bw_window, "window{}" + window.filename,store_path=store_path+window.filename)
        vessel_map = retina.vessel_image
        
        FD_binary,VD_binary,Average_width = global_cal(retina)
        
        vessels = detect_vessel_border(retina)
        vessel_count = 0
        vessel_count_1 = 0
        bifurcation_t = 0
        t1, t2, t3, t4, td, tfi, tcurve = 0, 0, 0, 0, 0, 0, 0
        vessel_density,average_caliber = 0, 0
        w1 = 0
        w1_list = []
        w1_list_average = []
        vessel_count_list = []

        for vessel in vessels:
            vessel_count_1 += 1

            if len(vessel[0]) > min_pixels_per_vessel:
                s1=time.time()
                vessel_count += 1
                
                s2=time.time()
                t2 += distance_measure_tortuosity(vessel[0], vessel[1])
                
                s4=time.time()
                t4 += squared_curvature_tortuosity(vessel[0], vessel[1])
                
                s5=time.time()
                td += tortuosity_density(vessel[0], vessel[1])
                
                s6=time.time()
                
                vessel_count_list.append(vessel_count)
                #tfi += fractal_tortuosity_curve(vessel[0], vessel[1])
                s7=time.time()
        
        if vessel_count > 0:
            t2 = t2/vessel_count
            t4 = t4/vessel_count
            td = td/vessel_count
    
    return FD_binary,VD_binary,Average_width, t2, t4, td
