from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import json
import cv2
import threading


#the path of this python file to be used as the root for finding files requested by the server
#so for example if the browser requests /index.html, the server will look for the file in the same folder as this file
THIS_FOLDER = os.path.dirname(os.path.realpath(__file__))
SERVER_ROOT_FOLDER = THIS_FOLDER

#the ip address and port to listen to
IPADDRESS = 'localhost'
PORT = 8000

#this class holds the data that the server will serve. The camera loop will update this data
#and the server will serve it to the client when requested by the javascript code running in the browser
class ServerData:
    def __init__(self):
        self.some_data = {0: 'zero', 1: 'one', 'other': [3,4,5,6.5]}
        self.some_data2 = {0: 'f', 1: 'h', 'other': [3,4]}
        self.camera_data = {}

server_data = ServerData()

#this class determines the server behavior (how it responds to requests)
class ServerHandler(BaseHTTPRequestHandler):
    #this function serves files from the SERVER_ROOT_FOLDER
    def _serve_file(self, file_path):
        try:
            file_to_open = open(file_path).read()
            self.send_response(200)
            if file_path.endswith('.html'):
                self.send_header('Content-type', 'text/html')
            elif file_path.endswith('.js'):
                self.send_header('Content-type', 'text/javascript')
            elif file_path.endswith('.css'):
                self.send_header('Content-type', 'text/css')
            elif file_path.endswith('.png'):
                self.send_header('Content-type', 'image/png')
            elif file_path.endswith('.jpg'):
                self.send_header('Content-type', 'image/jpg')
        except:
            file_to_open = 'File not found'
            self.send_response(404)
        self.end_headers()
        self.wfile.write(bytes(file_to_open, 'utf-8'))

    #this function sends json data. data can be a python dictionary that will be converted to json
    def _send_json_data(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        json_str = json.dumps(data)
        self.wfile.write(bytes(json_str, 'utf-8'))
    
    #this function sends a simple 'ok' response
    def _send_ok(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(bytes('ok', 'utf-8'))

    #this function is called when the server receives a GET request
    #it determines what to do based on the path of the request. if the path is /, it serves index.html
    #if the path is /get_data, it sends the data in server_data.some_data as json etc..
    def do_GET(self):
        if self.path == '/': #serve root index.html
            self.path = 'index.html'

        if self.path == '/get_data':
            self._send_json_data(server_data.some_data)
        elif self.path == '/get_data2': 
            self._send_json_data(server_data.some_data2)
        elif self.path == '/get_camera_data': 
            self._send_json_data(server_data.camera_data)
        else: #assume file request
            if self.path.startswith('/'):
                self.path = self.path[1:]
            
            self._serve_file(os.path.join(SERVER_ROOT_FOLDER, self.path))

    def do_POST(self):
        if self.path == '/set_camera_data':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            post_data = json.loads(post_data.decode('utf-8'))
            server_data.camera_data = post_data
            self._send_ok()


#----------------------------------------------------------------------------------
#this function runs the server
def run_server():
    httpd = HTTPServer( (IPADDRESS, PORT), ServerHandler)
    print('Starting httpd...')
    httpd.serve_forever(poll_interval=0.1)
    #start socket
    # while True: 
    #     ws.Send(server_data.camera_data)
    #     time.sleep(0.1)

#start the server in a separate thread
server_thread = threading.Thread(target=run_server)
server_thread.start()
#----------------------------------------------------------------------------------



#camera setup and loop
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cv2.namedWindow("test")

current_frame = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break

    current_frame += 1

    #every 10 frames, update the server_data.camera_data
    if current_frame % 10 == 0:
        server_data.camera_data = {'frame': current_frame, 'f': [0.4, 0.5]}

    if cv2.waitKey(1) == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break


server_thread.join() #shutdown the server thread