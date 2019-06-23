import numpy as np
import cv2
from matplotlib import pyplot as plt

#bg.shape==(:,:,3),uint8
#fg.shape==(:,:,2),uint8
def collage(background,foreground,pointx,pointy):
    bg = background.copy()
    fg = foreground.copy()
    bg_at = bg[pointx:fg.shape[0]+pointx,pointy:fg.shape[1]+pointy,:]
    fg_a = fg[:,:,3]
    fg = fg[:,:,:3]
    mask = np.where(fg_a>0,255,0)
    mask = mask.astype(np.uint8)
    mask_inv = np.where(mask>0,0,255)
    mask_inv = mask_inv.astype(np.uint8)
    masked_bg = cv2.bitwise_and(bg_at,bg_at,mask=mask_inv)
    masked_fg = cv2.bitwise_and(fg,fg,mask=mask)
    dst = cv2.add(masked_bg,masked_fg)
    bg[pointx:fg.shape[0]+pointx,pointy:fg.shape[1]+pointy,:] = dst
    return bg

#locate=[[l,t,r,b],[,,,],[,,,]]
def collageNoOverlap(background,foreground,locate):
    fh,fw = foreground.shape[:2]
    bh,bw = background.shape[:2]
    for i in range(100):
        rh = np.random.randint(0,bh-fh)
        rw = np.random.randint(0,bw-fw)
        flag = False
        for lo in locate:
            if (rh <= lo[3] and rh + fh >= lo[1]) and (rw <= lo[2] and rw + fw >= lo[0]):
                flag = True
                break
        if(flag):
            continue
        
        return [rw,rh,rw+fw,rh+fh],collage(background,foreground,rh,rw)
    return [],[]
            

def createBG():
    reso = np.random.randint(4,30)
    blur = np.random.randint(1,30)
    randbg = np.random.randint(0,2,(reso,reso))
    randbg = randbg * 255
    randbg = randbg.astype(np.uint8)
    rgb = np.zeros((reso,reso,3)).astype(np.uint8)
    rgb[:,:,0] = randbg
    rgb[:,:,1] = randbg
    rgb[:,:,2] = randbg
    scaled = cv2.resize(rgb,(300,300))
    return cv2.blur(scaled,(blur,blur))

def imageRotate(img,angle):
    h,w = img.shape[:2]
    angle_rad = angle/180.0*np.pi
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot,h_rot)
    center = (w/2,h/2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center,angle,scale)
    affine_matrix = rotation_matrix.copy()
    affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
    affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2
    return cv2.warpAffine(img,affine_matrix,size_rot)

def imageScaleRandom(img):
    h,w = img.shape[:2]
    scale = np.random.rand() + 0.5
    return cv2.resize(img,(int(w*scale),int(h*scale)))

def getRandomCard():
    num = np.random.randint(1,14)
    marks = ["h","s","d","c"]
    mark = marks[np.random.randint(0,4)]
    return cv2.imread("data/cards/{0}{1}.png".format(num,mark), -1),"{0}{1}".format(num,mark)

def maskShadow(img):
    gray = np.random.randint(0,128)
    shadow = np.random.randint(0,gray,img.shape[:2]).astype(np.uint8)
    shadowrgb = np.zeros(img.shape).astype(np.uint8)
    shadowrgb[:,:,0] = shadow
    shadowrgb[:,:,1] = shadow
    shadowrgb[:,:,2] = shadow
    shadowrgb = cv2.blur(shadowrgb,(5,5))
    inv = cv2.bitwise_not(img)
    shadowed = cv2.add(shadowrgb,inv)
    return cv2.bitwise_not(shadowed)

def getChaosCard():
    card,tag = getRandomCard()
    scaled = imageScaleRandom(card)
    shadowed = scaled
    try:
        shadowed = maskShadow(scaled)
    except:
        print("error")
    rotated = imageRotate(shadowed,np.random.randint(0,360))
    return rotated,tag

def collageRandom():
    num = np.random.randint(1,4)

    bg = createBG()

    locates = []
    names = []

    name_uni = []

    for i in range(num):
        img,tag = getChaosCard()
        if tag in name_uni:
            continue
        name_uni.append(tag)
        s,bg_res = collageNoOverlap(bg,img,locates)
        if(len(s) != 0):
            names.append(tag)
            locates.append(s)
            bg = bg_res.copy()

    return bg,np.array(names),np.array(locates)
    

