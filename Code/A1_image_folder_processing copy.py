#install openCV: pip install opencv-python
import os
import cv2

#########################################################Configuration
# set the size in pixels of the width and height of the output images
dst_size = 224

# set to true to convert the images to grayscale
make_grayscale = False

crop_to_square = True

# set the source and destination folders
srcFolder = 'data2/screenshot/'
dstFolder = 'data2/screenshot_processed/'
#########################################################End of configuration

#ensure that the destination folder exists
#os stands for operating system and it contains functions for interacting with the operating system
#and escpecially the file system
if not os.path.exists(dstFolder):
    os.makedirs(dstFolder)

#check if a filename is an image by seeing if it ends with .jpg or .png
def is_image(filename:str):
    extension = os.path.splitext(filename)[1]
    extension = extension.lower()
    return extension in ['.jpg', '.png', '.jpeg']

    #return any(filename.lower().endswith(extension) for extension in [".jpg", ".png", ".jpeg"])

#get a list of all the files in the srcFolder
all_src_files = os.listdir(srcFolder)

#filter the list to only include images
src_files = [file for file in all_src_files if is_image(file)]

#load each image in the srcFolder and resize it to 128x128 and save it in the dstFolder
for i, filename in enumerate(src_files):
    #build the full src file path by joining the folder name with the file name
    src_filename = os.path.join(srcFolder, filename)

    #build the full dst file path by joining the folder name with the file name
    dst_filename = os.path.join(dstFolder, filename)

    print(f'Processing {filename}...{i}/{len(src_files)}')
    #load the image using opencv
    img = cv2.imread(src_filename, cv2.IMREAD_COLOR)
   
    #get the image dimensions (height, width and number of channels)
    height, width, channels = img.shape

    #if crop_to_square is true, crop the image to a square
    if crop_to_square:
        if height > width:
            y = (height - width) // 2
            img = img[y:y+width, :]
        else:
            x = (width - height) // 2
            img = img[:, x:x+height]

    #resize the image to the desired size
    #we use different interpolation methods depending on whether we are enlarging or shrinking the image
    if dst_size> width:
        img = cv2.resize(img, (dst_size, dst_size), interpolation=cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (dst_size, dst_size), interpolation=cv2.INTER_AREA)


    #convert the image to grayscale if required
    if make_grayscale:
        #convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #save the image
    cv2.imwrite(dst_filename, img)