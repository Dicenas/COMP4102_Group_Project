from ultralytics import YOLO
"""import torch

torch.cuda.set_device(0)"""

if __name__ ==  '__main__':
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch


    #model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    results = model.train(data="data.yaml", epochs=100)  # train the model

    #metrics = model.val()  # evaluate model performance on the validation set
    #results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    #path = model.export(format="onnx")  # export the model to ONNX format