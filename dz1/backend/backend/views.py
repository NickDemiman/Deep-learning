from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import onnxruntime
import numpy as np
from PIL import Image

imageClassList = {'0': 'frog', '1': 'penguin', '2': 'turtle'}  #Сюда указать классы

def scoreImagePage(request):
    return render(request, 'scorepage.html')

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save('images/'+fileObj.name,fileObj)
    filePathName = fs.url(filePathName)
    modelName = request.POST.get('modelName')
    scorePrediction = predictImageData(modelName, '.'+filePathName)
    context = {'scorePrediction': scorePrediction}
    return render(request, 'scorepage.html', context)

def predictImageData(modelName, filePath):
    img = Image.open(filePath).convert("RGB")
    img = np.asarray(img.resize((32, 32), Image.ANTIALIAS))
    sess = onnxruntime.InferenceSession(r'G:\repos\Deep-learning\dz1\myModel.onnx') #<-Здесь требуется указать свой путь к модели
    outputOFModel = np.argmax(sess.run(None, {'input': np.asarray([img]).astype(np.float32)}))
    score = imageClassList[str(outputOFModel)]
    return score