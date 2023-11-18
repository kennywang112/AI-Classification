import pickle
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import gradio as gr
import os
import openai
import cv2
import time
import base64
import io

with open('./data/texes.txt', 'r', encoding='utf-8') as fh:
    tmp = fh.read()
    itemlist = tmp.split(',')

itemlist = str(itemlist)
itemlist

keyfile = open("./key.txt", "r")
key = keyfile.readline()

openai.api_key = key

loaded_model = load_model("./model_epoch_06.h5", compile=False)
model_config = loaded_model.get_config()

def encode_image(image_object):
    # Create a BytesIO object to hold the image data
    image_bytes = io.BytesIO()

    # Save the PIL image to the BytesIO object
    image_object.save(image_bytes, format='JPEG')  # Adjust the format as needed

    # Encode the image data in base64
    encoded_image = base64.b64encode(image_bytes.getvalue()).decode('utf-8')

    return encoded_image

def pred(img):

    # Resize the image to match the input shape of your model
    img = img.resize((256, 256))  # Adjust the size as needed
    # Convert the image to a NumPy array
    img_np = np.array(img)
    
    # Preprocess the image to match the model's input requirements
    img_np = image.img_to_array(img_np)
    
    img_for_plot = img_np / 255.0  # Normalize the image if necessary
    
    img_np = np.expand_dims(img_np, axis=0)  # Add a batch dimension
    
    # Make predictions
    predictions = loaded_model.predict(img_np)
    
    return img_for_plot, predictions

def image_mod(img):
    
    image, predictions = pred(img)

    print(predictions)
    
    result = ""
    
    if predictions[0][0] < 0.5:

        result = "fake image"
    else :
        
        result = "true image"

    return result

def chatbtn(content):
    
    print(content)
    
    return content

def display_image_from_video(video):

    capture_image = cv.VideoCapture(video) 
    # 獲取視頻中的一幀，變數分別為是否成功讀取了一幀以及捕獲的圖像幀
    ret, frame = capture_image.read()

    img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (256, 256))  # 調整大小
    img_for_plot = img / 255.0  # 正規化圖像
    
    img_np = np.expand_dims(img, axis=0)  # Add a batch dimension
    
#     # Make predictions
#     predictions = loaded_model.predict(img_np)
    
#     result = ""
    
#     if predictions[0][0] < 0.5:

#         result = "fake image"
#     else :
        
#         result = "true image"

#     return result
    return img_for_plot

def take_pic():  
    # cam_port = 0
    cam = cv2.VideoCapture(0) 
    # reading the input using the camera 
    result, image = cam.read() 
    # If image will detected without any error,  
    # show result 
    if result: 
        # showing result, it take frame name and image  
        # output 
        cv2.imshow("../images/GeeksForGeeks", image) 
        # saving image in local storage 
        # cv2.imwrite("../images/GeeksForGeeks.png", image) 
        # If keyboard interrupt occurs, destroy image  
        # window 
        # cv2.waitKey(0) 
        # cv2.destroyWindow("GeeksForGeeks") 
    else: 
        print("No image detected. Please! try again") 

def Reply(imagebox, message, chat_history):
    
    # Getting the base64 string
    base64_image = encode_image(imagebox)
    
    start_idx = 0
    result = ''
    while start_idx < len(itemlist):
        end_idx = min(start_idx + 1600, len(itemlist))
        sub_list = itemlist[start_idx:end_idx]
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ]
        )  
        for choice in response.choices:
            result += choice.message.content
        start_idx = end_idx

    with open('../output.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(result)

    output = open("../output.txt", "r", encoding="utf-8")
    read_output = output.read()
    
    chat_history.append((message, read_output))
    print(chat_history)
    
    time.sleep(10)
    
    return "", chat_history


title = """<h1 align="center">AI臉部辨識</h1>"""
textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)

with gr.Blocks(css = """.gradio-container {background-color: #3f7791}""") as demo:
    
    gr.HTML(title)
    gr.HTML('''<center><a href="https://github.com/kennywang112?tab=repositories" alt="GitHub Repo"></a></center>''')
    gr.HTML('''
        <!-- Google tag (gtag.js) -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-Y132VVZPKL"></script>
        <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());

        gtag('config', 'G-Y132VVZPKL');
        </script>
    ''')

    state = gr.State()
    
    with gr.Row():
        
        with gr.Column(scale=2):

            # for image
            imagebox = gr.Image(type="pil")
            outputs = gr.components.Text()
            img_clk = gr.Button('判斷圖像')
            img_clk.click(image_mod, inputs = [imagebox], outputs = [outputs])
            image_process_mode = gr.Radio(
                ["Crop", "Resize", "Pad", "Default"],
                value="Default",
                label="Preprocess for non-square image", visible=False)
            cur_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__"))))

            ex = gr.Examples(examples=[
                [f"{cur_dir}/images/fake_face0.jpeg", "你覺得這張圖片是真的還是假的"],
                [f"{cur_dir}/images/fake_face1.jpeg", "詳細介紹這張圖"],
                [f"{cur_dir}/images/GeeksForGeeks.png", "形容這張圖片"],
            ], inputs = [imagebox, textbox])
            
        with gr.Column(scale=2):
            
            # video
            inputs = gr.components.Video()
            outputs = gr.components.Text()
            update = gr.Button('判斷影片圖像')
            update.click(display_image_from_video, inputs = [inputs], outputs = [outputs])
            
            pic = gr.Button('拍照')
            pic.click(take_pic)
            
        with gr.Column(scale=6):
            
            chatbot = gr.Chatbot(elem_id="chatbot", label="Chatbot", height=550)
            
            with gr.Row():
                    
                with gr.Column(scale=1, min_width=50):
                    msg = textbox.render()

        # chatbox
        msg.submit(Reply, [imagebox, msg, chatbot], [msg, chatbot])