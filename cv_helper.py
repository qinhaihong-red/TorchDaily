import os
import matplotlib.pyplot as plt
def get_img_path(name):
    img_folder=os.getenv('IMG_FOLDER')
    path1=os.path.join(img_folder,name)
    path2=path1+'.jpg'
    path3=path1+'.png'
    try:
        if os.path.exists(path1):
            return path1
        elif os.path.exists(path2):
            return path2
        elif os.path.exists(path3):
            return path3
    except(e):
        return None
        
    return None


def imshow_gray(img):
    plt.imshow(img,cmap='gray');