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


with open('./data/texes.txt', 'r', encoding='utf-8') as fh:
    tmp = fh.read()
    itemlist = tmp.split(',')

itemlist = str(itemlist)
itemlist

keyfile = open("./key.txt", "r")
key = keyfile.readline()

openai.api_key = key + "i"

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

    capture_image = cv2.VideoCapture(video)
    print(capture_image)
    # 獲取視頻中的一幀，變數分別為是否成功讀取了一幀以及捕獲的圖像幀
    ret, frame = capture_image.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # 調整大小
    img_for_plot = img / 255.0  # 正規化圖像
    
    img_np = np.expand_dims(img, axis=0)  # Add a batch dimension
    
    # Make predictions
    predictions = loaded_model.predict(img_np)
    
    result = ""
    
    if predictions[0][0] < 0.5:

        result = "fake image"
    else :
        
        result = "true image"

    return result
    # return img_for_plot

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
    print(chat_history)
    
    time.sleep(10)
    
    return "", chat_history

# GAN

model = torch.hub.load("AK391/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v1")
face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main', 'face2paint', 
    size=512,side_by_side=False
)

def inference(img):
    
    out = face2paint(model, img)

    return out

title = """<h1 align="center">AI臉部辨識</h1>"""
textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
# <script async src="https://www.googletagmanager.com/gtag/js?id=G-Y132VVZPKL"></script>
ga_script = '''
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-Y132VVZPKL"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());

gtag('config', 'G-Y132VVZPKL');
</script>
'''

# with gr.Blocks(head = ga_script, css = """.gradio-container {background-color: #3f7791}""") as demo1:
    
#     gr.HTML(title)
#     gr.HTML('''<center><a href="https://github.com/kennywang112?tab=repositories" alt="GitHub Repo"></a></center>''')

#     state = gr.State()
    
#     with gr.Row():
        
#         with gr.Column(scale=3):

#             # for image
#             imagebox = gr.Image(type="pil")
#             outputs = gr.components.Text()
#             img_clk = gr.Button('判斷圖像')
#             img_clk.click(image_mod, inputs = [imagebox], outputs = [outputs])
#             image_process_mode = gr.Radio(
#                 ["Crop", "Resize", "Pad", "Default"],
#                 value="Default",
#                 label="Preprocess for non-square image", visible=False)
#             cur_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__"))))

#             ex = gr.Examples(examples=[
#                 [f"{cur_dir}/images/fake_face0.jpeg", "你覺得這張圖片是真的還是假的"],
#                 [f"{cur_dir}/images/fake_face1.jpeg", "詳細介紹這張圖"],
#                 [f"{cur_dir}/images/real_face0.jpeg", "形容這張圖片"],
#             ], inputs = [imagebox, textbox])

#             pic = gr.Button('拍照')
#             pic.click(take_pic)
            
#         # with gr.Column(scale=2):
            
#         #     # video
#         #     inputs = gr.components.Video()
#         #     outputs = gr.components.Text()
#         #     update = gr.Button('判斷影片圖像')
#         #     update.click(display_image_from_video, inputs = [inputs], outputs = [outputs])
            
#         #     pic = gr.Button('拍照')
#         #     pic.click(take_pic)
            
#         with gr.Column(scale=7):
            
#             chatbot = gr.Chatbot(elem_id="chatbot", label="Chatbot", height=800)
            
#             with gr.Row():
                    
#                 with gr.Column(scale=1, min_width=50):
#                     msg = textbox.render()

#         # chatbox
#         msg.submit(Reply, [imagebox, msg, chatbot], [msg, chatbot])

with gr.Blocks() as small_block1:

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
        [f"{cur_dir}/images/real_face0.jpeg", "形容這張圖片"],
    ], inputs = [imagebox, textbox])

with gr.Blocks() as small_block2:

    inputs = gr.components.Video()
    outputs = gr.components.Text()
    update = gr.Button('判斷影片圖像')
    update.click(display_image_from_video, inputs = [inputs], outputs = [outputs])

with gr.Blocks(head = ga_script, css = """.gradio-container {background-color: #3f7791}""") as demo1:
    
    gr.HTML(title)
    gr.HTML('''<center><a href="https://github.com/kennywang112?tab=repositories" alt="GitHub Repo"></a></center>''')

    state = gr.State()
    
    with gr.Row():
        
        with gr.Column(scale=3):

            gr.TabbedInterface([small_block1, small_block2],["圖片", "影片"])

            pic = gr.Button('拍照')
            pic.click(take_pic)
            
        with gr.Column(scale=7):
            
            chatbot = gr.Chatbot(elem_id="chatbot", label="Chatbot", height=800)
            
            with gr.Row():
                    
                with gr.Column(scale=1, min_width=50):
                    msg = textbox.render()

        # chatbox
        msg.submit(Reply, [imagebox, msg, chatbot], [msg, chatbot])

css_size = """.output-image, .input-image, .image-preview {height: 600px !important} .gradio-container {background-color: #3f7791}"""
# with gr.Blocks(head = ga_script, css = """.gradio-container {background-color: #3f7791}""") as demo2:
with gr.Blocks(head = ga_script, css = css_size) as demo2:
    
    gr.HTML(title)
    gr.HTML('''<center><a href="https://github.com/kennywang112?tab=repositories" alt="GitHub Repo"></a></center>''')

    state = gr.State()
    with gr.Row():  

        with gr.Column(scale = 5):

            # for image
            imagebox = gr.Image(type="pil", height=450)
            img_clk = gr.Button('生成圖像')

        with gr.Column(scale = 5):

            outputs = gr.components.Image(height=500)
            img_clk.click(inference, inputs = [imagebox], outputs = [outputs])
            image_process_mode = gr.Radio(
                ["Crop", "Resize", "Pad", "Default"],
                value="Default",
                label="Preprocess for non-square image", visible=False)
            
    with gr.Column(scale = 2):  
        cur_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath("__file__"))))

        ex = gr.Examples(examples=[
            [f"{cur_dir}/images/fake_face0.jpeg"],
            [f"{cur_dir}/images/fake_face1.jpeg"],
            [f"{cur_dir}/images/real_face0.jpeg"],
        ], inputs = [imagebox])

full_website = gr.TabbedInterface([demo1, demo2],["測試", "GAN"])