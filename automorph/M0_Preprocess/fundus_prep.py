from cv2 import threshold
import numpy as np
import os
import cv2
from scipy.ndimage import label


def imread(file_path, c=None):
    if c is None:
        im = cv2.imread(file_path)
    else:
        im = cv2.imread(file_path, c)

    if im is None:
        print('Can not read image at filepath: {}'.format(file_path))
        raise "error"

    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) #cv2 reads images in bgr format
    return im


def imwrite(file_path, image):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, image)


def fold_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


def get_mask_BZ(img):
    if img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    threhold = np.mean(gray_img)/3-5
    #cv2.imshow('gray_img', gray_img)
    #cv2.waitKey()
    #print(threhold)
    _, mask = cv2.threshold(gray_img, max(5,threhold), 1, cv2.THRESH_BINARY)

    #cv2.imshow('bz_mask', mask*255)
    #cv2.waitKey()
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    #  cv::floodFill(Temp, Point(0, 0), Scalar(255));
    # _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, [(0, 0),(0,new_mask.shape[0])], (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _,new_mask,_,_ = cv2.floodFill(new_mask, nn_mask, (new_mask.shape[1]-1,new_mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)
    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,  20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    return mask


def _get_center_by_edge(mask):
    center=[0,0]
    x=mask.sum(axis=1)
    center[0]=np.where(x>x.max()*0.95)[0].mean()
    x=mask.sum(axis=0)
    center[1]=np.where(x>x.max()*0.95)[0].mean()
    return center


def _get_radius_by_mask_center(mask,center):
    """ apply gradient (difference between erosion and dilation) to the mask to find the border, then calculate the radius from this border. Find the 
    distance between each point in the mask and the centre then select the radius as the distance slightly larger than 99.5% of the largest distance
    """
    mask=mask.astype(np.uint8)
    ksize=max(mask.shape[1]//400*2+1,3)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))
    mask=cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
    # radius=
    index=np.where(mask>0)
    d_int=np.sqrt((index[0]-center[0])**2+(index[1]-center[1])**2)
    b_count=np.bincount(np.ceil(d_int).astype(np.int))
    if len(b_count) == 0: # if there are no unique values in the bin
        radius = 0
        return radius
    else:
        radius = np.where(b_count>b_count.max()*0.995)[0].max()
        return radius


def _get_circle_by_center_bbox(shape,center,bbox,radius):
    center_mask=np.zeros(shape=shape).astype('uint8')
    tmp_mask=np.zeros(shape=bbox[2:4])
    center_tmp=(int(center[0]),int(center[1]))
    center_mask=cv2.circle(center_mask,center_tmp[::-1],int(radius),(1),-1)
    # center_mask[bbox[0]:bbox[0]+bbox[2],bbox[1]:bbox[1]+bbox[3]]=tmp_mask
    # center_mask[bbox[0]:min(bbox[0]+bbox[2],center_mask.shape[0]),bbox[1]:min(bbox[1]+bbox[3],center_mask.shape[1])]=tmp_mask
    return center_mask

def get_largest_element(mask):
    """
    Uses structuring element in scipy ndimage label to find all connected features in an array and only keep the 
    feature with the most elements.

    Params:
    mask (np.array) binary mask
    Returns:
    largest_mask (np.array) binary mask only containing the largest connected element of the input mask
    """

    l_mask, n_labels = label(mask)

    if n_labels > 0:
        label_dict = {}
        for l in range(1,n_labels+1): 
            label_dict[l] = np.sum(l_mask == l)
        sorted_label = sorted(label_dict.items(), key= lambda x: x[1], reverse=True)
        largest_mask = l_mask == sorted_label[0][0]
        return largest_mask 
    else:
        largest_mask = mask
        return largest_mask

def get_mask(img):
    if img.ndim ==3:
        #raise 'image dim is not 3'
        g_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        #cv2.imshow('ImageWindow', g_img)
        #cv2.waitKey()
    elif img.ndim == 2:
        g_img =img.copy()
    else:
        raise 'image dim is not 1 or 3'
    h,w = g_img.shape
    shape=g_img.shape[0:2]
    #g_img = cv2.resize(g_img,(0,0),fx = 0.5,fy = 0.5)
    tg_img=cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask=get_mask_BZ(tg_img)
    tmp_mask = get_largest_element(tmp_mask)
    center=_get_center_by_edge(tmp_mask)

    #bbox=_get_bbox_by_mask(tmp_mask)
    #print(center)
    #cv2.imshow('ImageWindow', tmp_mask*255)
    #cv2.waitKey()
    radius=_get_radius_by_mask_center(tmp_mask,center)

    #resize back
    #center = [center[0]*2,center[1]*2]
    #radius = int(radius*2)
    if radius == 0: # if there is no radius then this cant work just return zeros
        return tmp_mask, (0,0,0,0), [0,0], radius
    
    # if the radius is greater than either the height or width of the image
    if ((radius +center[0]) > h) or ((center[0] - radius) <= 0):
        if center[0] < (h/2):
            radius = center[0]
        else:
            radius = h-center[0]
    elif ((radius +center[1]) > w) or ((center[1] - radius) <= 0):
        if center[1]< (w/2):
            radius = center[1]
        else:
            radius = w-center[1]

    center = [center[0], center[1]]
    radius = int(radius)
    s_h = max(0,int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h-s_h,2 * radius), min(w-s_w,2 * radius))
    tmp_mask=_get_circle_by_center_bbox(shape,center,bbox,radius)
    return tmp_mask,bbox,center,radius


def mask_image(img,mask):
    img[mask<=0,...]=0
    return img


def remove_back_area(img,bbox=None,border=None):
    image=img
    if border is None:
        border=np.array((bbox[0],bbox[0]+bbox[2],bbox[1],bbox[1]+bbox[3],img.shape[0],img.shape[1]),dtype=np.int)
    image=image[border[0]:border[1],border[2]:border[3],...]
    return image,border


def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border


def process_without_gb(img, label,radius_list,centre_list_w, centre_list_h):
    # preprocess images
    #   img : origin image
    #   tar_height: height of tar image
    # return:
    #   result_img: preprocessed image
    #   borders: remove border, supplement mask
    #   mask: mask for preprocessed image

    #return null if image is all zeros
    borders = []
    mask, bbox, center, radius = get_mask(img)
    r_img = mask_image(img, mask)
    r_img, r_border = remove_back_area(r_img,bbox=bbox)
    mask, _ = remove_back_area(mask,border=r_border)
    label, _ = remove_back_area(label,bbox=bbox)
    borders.append(r_border)
    r_img,sup_border = supplemental_black_area(r_img)
    #print(r_img.shape)
    label,sup_border = supplemental_black_area(label)
    mask,_ = supplemental_black_area(mask,border=sup_border)
    borders.append(sup_border)

    radius_list.append(radius)
    centre_list_w.append(int(center[0]))
    centre_list_h.append(int(center[1]))

    # return input image in r_img is zero dimension
    if len(r_img) == 0:
        r_img = img

    return r_img,borders,(mask*255).astype(np.uint8),label, radius_list,centre_list_w, centre_list_h

