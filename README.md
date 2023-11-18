# Hackathon demo
## Goal
- A web demo of face detection, whether an image or video have been edited.
- A Convolution Neural Network model training, and use it in demo page.
- API from chatgpt to describe the Image, or video.
- Deploy on website
- Add Google Analytics

### Start in web file
run with `uvicorn web.run:app --reload`

## Deploy on website
1. Go to https://dashboard.render.com/web/new
2. Start command: `uvicorn web.run:app --host 0.0.0.0 --port 5000`
3. Add Python version in additional

## Ref
1. Website: https://www.youtube.com/watch?v=0BEBquff6rI
2. GA4: https://hackmd.io/15X08z7DSz-pSctKHLAn3Q