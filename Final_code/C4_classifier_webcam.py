import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import os
#import sys
#sys.path.append("D:/OneDrive - Harvard University/Quantatitive Aesthetics/final_project")

#websocket server
import asyncio
import websockets
import threading
import json

#use opencv to calculate the optical flow of frames from camera feed
import cv2
import numpy as np

from classifier import *
from C1_classifier_config import *



class ServerData:
    def __init__(self):
        self.to_send = {}
        self.ready_to_send = {}

serve_data = ServerData()

#------------------------------------------------------
# WebSocket
#------------------------------------------------------
#this function runs the server
# def run_server():
    # httpd = HTTPServer( (IPADDRESS, PORT), ServerHandler)
    # print('Starting httpd...')
    # httpd.serve_forever(poll_interval=0.1)
    #start socket
    # while True: 
    #     ws.Send(server_data.camera_data)
    #     time.sleep(0.1)



async def handler(websocket):
    async for message in websocket:
        #print(message)
        data = serve_data.ready_to_send
        data_json = json.dumps(data)


        reply = f"Data received as {message}!"
        await websocket.send(data_json)

def run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_server = websockets.serve(handler, "", 8001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

   
    


#start the server in a separate thread
server_thread = threading.Thread(target=run_server)
server_thread.start()






#------------------------------------------------------
# Load model and define function
#------------------------------------------------------

print(f'Using classifier model: {classifierModel}')

baseFolder = 'D:/OneDrive - Harvard University/Quantatitive Aesthetics/final_project/'
modelSaveFolder = os.path.join(baseFolder, f'models/{modelname}/')
modelSaveFile = os.path.join(modelSaveFolder, 'model_weights.pth')


image_size = 224
image_channels = 3
freeze_pretrained_parameters = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

classFolders = []
for file in os.listdir(imageFolder):
    if os.path.isdir(os.path.join(imageFolder, file)):
        classFolders.append(file) 
classNames = [os.path.splitext(x)[0] for x in classFolders]
#sort the class names so that the order is always the same
classNames.sort()


print(f'Found {len(classNames)} classes: {classNames}')
# Create the model
classifierModel = createClassifierModel(classifierModel, len(classNames), freeze_pretrained_parameters = freeze_pretrained_parameters, use_pretrained = False)
classifierModel.load_state_dict(torch.load(modelSaveFile))
classifierModel.to(device)
classifierModel.eval()


data_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



def classify(image):
        image = data_transforms(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            output = classifierModel(image)
            
            _, preds = torch.max(output, 1)
            preds = preds.cpu().numpy()
            activation_vec = output.cpu().numpy()[0]
            output = (np.exp(activation_vec) / np.sum(np.exp(activation_vec)))*100

            activation_vec = activation_vec.tolist()
            serve_data.to_send["activation_vector"] = activation_vec
            
            max_index = np.argmax(output)
            serve_data.to_send["class"] = classNames[max_index]

            #print(classNames[max_index])
            s = ""
            for i in range(len(classNames)):
                s+= f'  {classNames[i]} : {output[i].item()}'
                serve_data.to_send[classNames[i]] = output[i].item()
            print(s)





#----------------------------------------------------
#Real-time classification
#----------------------------------------------------

#open the camera
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

flow_width = 320
flow_height = 180

previous_frame = None

show_vectors = False
vector_scale = 10.0

#loop until the user presses the escape key
while True:
    #read the current frame from the camera
    ret, frame = cap.read()

    frame_width  = frame.shape[1]
    frame_height = frame.shape[0]

    #convert the current frame to grayscale
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #flip the current frame horizontally
    #current_frame = cv2.flip(current_frame, 1) 
    #resize to something smaller
    current_frame = cv2.resize(current_frame, (flow_width, flow_height), interpolation = cv2.INTER_AREA)

    if previous_frame is not None:
        #calculate the optical flow between the previous frame and the current frame
        flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        if show_vectors:
            dx = flow[:,:,0]
            dy = flow[:,:,1]

            flow_to_frame_x = frame_width/flow_width
            flow_to_frame_y = frame_height/flow_height

            average_dx = 0.0
            average_dy = 0.0

            #draw the optical flow vectors
            for y in range(0, flow_height, 4):
                point_y = int(y*flow_to_frame_y)
                for x in range(0, flow_width, 4):
                    point_x = int(x*flow_to_frame_x)

                    vec_x = int(point_x + dx[y,x]*vector_scale)
                    vec_y = int(point_y + dy[y,x]*vector_scale)

                    average_dx += dx[y,x]
                    average_dy += dy[y,x]

                    cv2.arrowedLine(frame, (point_x,point_y), (vec_x, vec_y), (0,255,255), 1, line_type=cv2.LINE_AA)


            serve_data.to_send["flow_vec"] = [average_dx, average_dy]
            cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
            cv2.imshow('preview',frame)


        else:
            #find the maximum flow in either the horizontal or vertical directio`n
            #max_flow = np.max(np.abs(flow))
            max_flow = 15

            flow = flow/(2.0*max_flow) + 0.5

            dx = flow[:,:,0]
            dy = flow[:,:,1]

            average_dx = np.mean(dx).item()
            average_dy = np.mean(dy).item()

            serve_data.to_send["flow_vec"] = [average_dx, average_dy]

            #convert to rgb with dx as red and dy as green
            flow_rgb = np.stack((np.ones_like(dx)*0.5, dy, dx), axis=2)

            flow_rgb_8bit = np.uint8(flow_rgb*255.0)
            #show the current frame in a window called 'preview'
            #set window scaling
            cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
            cv2.imshow('preview',flow_rgb_8bit)
            output_img = cv2.cvtColor(flow_rgb_8bit, cv2.COLOR_BGR2RGB)
            output_img = Image.fromarray(output_img)
            classify(output_img)


    data_to_send = serve_data.to_send
    serve_data.to_send = {}
    serve_data.ready_to_send = data_to_send
    #wait for a key to be pressed
    key = cv2.waitKey(1)
    #if the escape key is pressed, exit the loop
    if key == 27:
        break

    #set the current frame as the previous frame
    previous_frame = current_frame



#----------------------------------------------------
# main function
#----------------------------------------------------

# async def main():
#     async with websockets.serve(handler, "", 8001):
#         await asyncio.Future()  # run forever


# if __name__ == "__main__":
#     asyncio.run(main())
