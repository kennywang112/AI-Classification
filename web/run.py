from fastapi import FastAPI
import gradio as gr

from web.gradio_ui import full_website

app = FastAPI()

@app.get('/')
async def root():
    return 'gradio app is running at /gradio', 200

app = gr.mount_gradio_app(app, full_website, path='/gradio')