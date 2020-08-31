from img_processing import *
# import tensorflow as tf
from keras.models import load_model
import keras
import sys
import warnings
from solver import *
warnings.filterwarnings("ignore")

def load_trained_model(model_path):
    "loads trained CNN model"
    model = load_model(model_path)

    return model

def read_image(image_path):
    image = cv2.imread(image_path)

    return image
    


def digit_recognition(model,images):
    proc_i = []

    for i in images:
        ## resize and convert to rgb
        print(i.shape)
        # i = cv2.cvtColor(i, cv2.COLOR_GRAY2RGB)
        i = cv2.resize(i, size)

        ## scale image
        i = i/255.0
        proc_i.append(i)

    ## feed image arrays to model
    probs = model.predict(np.array(proc_i))
    preds = np.argmax(probs,axis=1)

    return preds


def get_cells(image):

    preprocess = cv2.resize(preprocessImage(image), (600, 600))
    print("done")
    preprocess = cv2.bitwise_not(preprocess, preprocess)
    cv2.imshow("preprocessed", preprocess)

    print("detecting sudoku.....")
    ## detect the sudoku grid from the preprocessed image
    coords = getCoords(preprocess)
    # print(preprocess.shape)
    preprocess = cv2.cvtColor(preprocess, cv2.COLOR_GRAY2BGR)
    coordsImage = preprocess.copy()
    ## mark corners of the grid
    for coord in coords:
        cv2.circle(coordsImage, (coord[0], coord[1]), 5, (255, 0, 0), -1)

    ## correct orientation of image using warping
    warpedImage = warp(preprocess, coords)

    ## draw grid lines on the image
    rects = displayGrid(warpedImage)
    ## extract cell from the detected grid
    tiles = extractGrid(warpedImage, rects)
    for i in range(len(tiles)):
        tiles[i] = cv2.resize(tiles[i],size)   
    return tiles 

## load model and image
print("loading model....")
model_path = sys.argv[1]
image_path = sys.argv[2]
size = (48, 48)

model = load_trained_model(model_path)
image = read_image(image_path)
cv2.imshow("original", image)

print("preprocessing......")
## preprocess, blurring, threshold and create mask
tiles = get_cells(image)

print("sudoku recognition....")
## digit recognition on cells

a = digit_recognition(model,np.array(tiles))
a_grid = a.reshape(9,9)
print("detectd sudoku\n", a_grid)
solve_sudoku(a_grid)
print("solved sudoku\n", a_grid)
