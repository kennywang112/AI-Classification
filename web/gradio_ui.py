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
import torch
import pymysql

keyfile = open("./key.txt", "r")
key = keyfile.readline()

openai.api_key = key + "i"

# loaded_model = load_model("./model_epoch_06.h5", compile=False)
loaded_model = load_model("./model.h5", compile=False)
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
    img = img.resize((224, 224))  # Adjust the size as needed
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
    
    preds = np.argmax(predictions)

    result = ""

    if preds == 1:

        result = "true image"
    else :
        
        result = "fake image"

    return result

def pred_v2(img_np):

    # img_np = cv2.resize(img_np, (256, 256))
    img_np = cv2.resize(img_np, (224, 224))
    
    img_for_plot = img_np / 255.0  # Normalize the image if necessary
    
    img_np = np.expand_dims(img_np, axis=0)  # Add a batch dimension
    
    # Make predictions
    predictions = loaded_model.predict(img_np)
    
    return img_for_plot, predictions

def image_mod_v2(img):

    image, predictions = pred_v2(img)

    preds = np.argmax(predictions)

    result = ""
    
    if preds == 1:

        result = "true image"
    else :
        
        result = "fake image"

    return result

def chatbtn(content):
    
    return content

def display_image_from_video(video):
    
    lst = []
    prediction_vote = []

    capture_image = cv2.VideoCapture(video)

    for i in range(9):

        ret, frame = capture_image.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (256, 256))  # 調整大小
        img = cv2.resize(img, (224, 224))  # 調整大小
        img_for_plot = img / 255.0  # 正規化圖像
    
        img_np = np.expand_dims(img, axis=0)  # Add a batch dimension

        lst.append(tuple([img_np[0], str(i)]))

        # Make predictions
        predictions = loaded_model.predict(img_np)

        preds = np.argmax(predictions)
        
        result = ""
        
        if preds == 1:

            prediction_vote.append(1)
        else :

            prediction_vote.append(0)

    if sum(prediction_vote) > 4:

        result = "true image"
    
    else:

        result = "fake image"

    return result, lst

def Reply(imagebox, message, chat_history):
    
    # Getting the base64 string
    base64_image = encode_image(imagebox)
    
    start_idx = 0
    result = ''

    while start_idx < 14:

        end_idx = min(start_idx + 1600, 14)

        response = openai.ChatCompletion.create(
            model = "gpt-4-vision-preview",
            messages = [
                {"role": "system", "content": "You are a chatbot"},
                {"role": "user", "content": [
                    {"type": "text", "text": message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
            max_tokens=4000,
        )  
        for choice in response.choices:
            result += choice.message.content
        start_idx = end_idx

    with open('../output.txt', 'w', encoding='utf-8') as output_file:
        output_file.write(result)

    output = open("../output.txt", "r", encoding="utf-8")
    read_output = output.read()
    
    chat_history.append((message, read_output))

    # time.sleep(10)
    
    return "", chat_history

def Store(img):
        
    img_Contents = base64.b64encode(img) # 轉成byte object
    img_Contents = str(img_Contents,'utf-8')
    conn = pymysql.connect(host='127.0.0.1',user="root", passwd="A123456",db='PIC',charset='UTF8')
    cursor = conn.cursor()# 使用該連接創建並連接
    sql="UPDATE uploadsmallrange SET id=(%s) WHERE dates=(%s)"
    val=(img_Contents,('m'))
    cursor.execute(sql,val) # 執行數據庫和命令
    conn.commit() # 提交
    cursor.close()
    conn.close()

# GAN
model = torch.hub.load("AK391/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1")
face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512,side_by_side=False
)

def inference(img):
    
    out = face2paint(model, img)
    # out = model(img)

    return out

title = """<h1 align="center">AI臉部辨識</h1>"""
textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)

ga_script = """
<script async src="https://www.googletagmanager.com/gtag/js?id=G-Y132VVZPKL"></script>
"""
ga_load = """
function() {
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-Y132VVZPKL');
}
"""

with gr.Blocks(head = ga_script) as small_block1:

    with gr.Row():
        
        with gr.Column(scale=3):

            imagebox = gr.Image(type="pil", height=300)
            outputs = gr.components.Text()
            img_clk = gr.Button('判斷圖像')
            img_clk.click(image_mod, inputs = [imagebox], outputs = [outputs])
            store_clk = gr.Button('圖像存儲')
            store_clk.click(Store, inputs = [imagebox])

            image_process_mode = gr.Radio(
                ["Crop", "Resize", "Pad", "Default"],
                value="Default",
                label="Preprocess for non-square image", visible=False)
            cur_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__"))))
            ex = gr.Examples(examples=[
                [f"{cur_dir}/images/fake_face0.jpeg", "這有什麼不尋常的地方嗎"],
                [f"{cur_dir}/images/real_face0.jpeg", "詳細形容他"],
            ], inputs = [imagebox, textbox])

        with gr.Column(scale=7):
            
            chatbot = gr.Chatbot(elem_id="chatbot", label="Chatbot", height=700)
            
            with gr.Row():
                    
                with gr.Column(scale=1, min_width=50):
                    
                    msg = textbox.render()
        
        msg.submit(Reply, [imagebox, msg, chatbot], [msg, chatbot])

    small_block1.load(None, js = ga_load)

with gr.Blocks(head = ga_script) as small_block2:

    with gr.Row():
        
        with gr.Column(scale=3):
            input = gr.components.Video(height=400)
            outputs = gr.components.Text()
            update = gr.Button('判斷影片圖像')

        with gr.Column(scale=7):
            
            gallery = gr.Gallery(
                label="Captured images", show_label=False, elem_id="gallery"
            , columns=[3], rows=[4], object_fit="contain", height=570)

        update.click(display_image_from_video, inputs = [input], outputs = [outputs, gallery])

    small_block2.load(None, js = ga_load)

with gr.Blocks(head = ga_script) as small_block3:

    with gr.Row():  

        with gr.Column(scale = 5):

            # for image
            imagebox = gr.Image(type="pil", height=500)
            img_clk = gr.Button('生成圖像')

        with gr.Column(scale = 5):

            outputs = gr.components.Image(height=400)
            res = img_clk.click(inference, inputs = [imagebox], outputs = [outputs])
            image_process_mode = gr.Radio(
                ["Crop", "Resize", "Pad", "Default"],
                value="Default",
                label="Preprocess for non-square image", visible=False)
            
            text_outputs = gr.components.Text()
            painted_img_clk = gr.Button('判斷生成圖像')

            painted_img_clk.click(image_mod_v2, inputs = [outputs], outputs = [text_outputs])
            
    with gr.Column(scale = 2):  

        cur_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__"))))

        ex = gr.Examples(examples=[
            [f"{cur_dir}/images/fake_face0.jpeg"],
            [f"{cur_dir}/images/real_face0.jpeg"],
        ], inputs = [imagebox])

    small_block3.load(None, js = ga_load)

with gr.Blocks(head = ga_script) as demo1:
    
    gr.HTML(title)

    state = gr.State()

    gr.TabbedInterface([small_block1, small_block2, small_block3], ["圖片", "影片", "GAN"])

    demo1.load(None, js = ga_load)

full_website = gr.TabbedInterface([demo1],["測試"])
