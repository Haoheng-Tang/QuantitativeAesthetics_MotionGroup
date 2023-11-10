#use opencv to calculate the optical flow of frames from camera feed

import cv2
import numpy as np

#open the camera
#cap = cv2.VideoCapture(0)

#imput video
cap = cv2.VideoCapture("./videos/AmazingWaterfall.mp4")

flow_width = 852
flow_height = 480

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

            #draw the optical flow vectors
            for y in range(0, flow_height, 4):
                point_y = int(y*flow_to_frame_y)
                for x in range(0, flow_width, 4):
                    point_x = int(x*flow_to_frame_x)

                    vec_x = int(point_x + dx[y,x]*vector_scale)
                    vec_y = int(point_y + dy[y,x]*vector_scale)

                    cv2.arrowedLine(frame, (point_x,point_y), (vec_x, vec_y), (0,255,255), 1, line_type=cv2.LINE_AA)


            cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
            cv2.imshow('preview',frame)


        else:
            #find the maximum flow in either the horizontal or vertical direction
            max_flow = np.max(np.abs(flow))

            flow = flow/(2.0*max_flow) + 0.5

            dx = flow[:,:,0]
            dy = flow[:,:,1]

            #convert to rgb with dx as red and dy as green

            flow_rgb = np.stack((np.ones_like(dx)*0.5, dy, dx), axis=2)

            flow_rgb_8bit = np.uint8(flow_rgb*255.0)
            #show the current frame in a window called 'preview'
            #set window scaling
            cv2.namedWindow('preview', cv2.WINDOW_NORMAL)
            cv2.imshow('preview',flow_rgb_8bit)

    #wait for a key to be pressed
    key = cv2.waitKey(1)
    #if the escape key is pressed, exit the loop
    if key == 27:
        break

    #set the current frame as the previous frame
    previous_frame = current_frame