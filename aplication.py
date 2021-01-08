import tensorflow as tf
import pygame
import pygame.freetype
import numpy
import random
import os
import math
from pathlib import Path

#Runtime Information
print("Tensorflow is running at: ",tf.__version__)
print("Eager execution: ",tf.executing_eagerly())
path = (os.path.dirname(os.path.realpath(__file__)))
print("Applicaiton running in "+path)
print("Exposed GPU's "+str(tf.config.experimental.list_physical_devices('GPU')))
print("Exposed CPU's "+str(tf.config.experimental.list_physical_devices('CPU')))#Pygame setup
print()
print("Initilizing Pygame")

#Setup AI Network

verboseagreement = 0 #Set to 0 if you don't care about the console. 2 for detailed updates
fittingRounds = 5 #Too big overfits, too small doesn't train. 5 is fine

print()
print("importing the MNIST database of handwritten digits")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() #X for properties, Y for labels
x_train, x_test = x_train / 255.0, x_test / 255.0  #Data is integer 0-255. Turning to float between 0 and 1
print("Training: "+str(x_train.shape)+" --> "+str(y_train.shape))
print("Testing: "+str(x_test.shape)+" --> "+str(y_test.shape))
print("Setup model")


#Create Model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #2d --> 1d. Flattens the 28v28 images to a long line of vectors
  tf.keras.layers.Dense(128, activation='relu'), #Manipulation Layer 
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10) #Final Layer
])

logits = model(x_train[:1]).numpy() #Get probabilities
predictions = tf.nn.softmax(logits).numpy() #Get human-readable probabilities

#Get loss from predicions
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = loss_fn(y_train[:1], predictions).numpy()

#Turn all the parameters so far setup into a usable model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print("Training",end="")
#Tell the model to minimize loss
model.fit(x_train, y_train, epochs=fittingRounds,verbose = verboseagreement) #verbose for how detailed you want
print("Testing the model")
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)



#Setup Pygame
w = 640
h = 400
textSize = 10
mainDisplay = pygame.display.set_mode((w, h), pygame.RESIZABLE)
pygame.init()
getFont = pygame.freetype.SysFont('Ubuntu Mono', textSize)
path = (os.path.dirname(os.path.realpath(__file__)))
print("Applicaiton running in "+path)
icon = (pygame.image.load(path+"\icon.png"))
pygame.display.set_icon(icon)
pygame.display.set_caption('Machine Learning '+tf.__version__)

graphSize = 1
pixelSize = 1
horrMargins = w/100
vertMargins = h/100

#Redraw static GUI elements
def update():
    print("up",end="d")
    #Background
    mainDisplay.fill((255,255,255))

    #How much space will each image take up
    estimatedWidth = (10 * graphSize) + 2 * textSize +(len(x_test[0][0]) * pixelSize)
    estimatedHeight = 2 * textSize+(len(x_test[0]) * pixelSize)

    #How many of each can we have?
    horelem = math.floor((w-(2*horrMargins)) / estimatedWidth)
    vertelem = math.floor((h-(2*vertMargins)) / estimatedHeight)

    #If we don't have enough elements to fill the screen
    if(horelem*vertelem > len(x_test)):
        if(horelem > len(x_test)):
            horelem = len(x_test)
        else:
            vertelem = math.floor(len(x_test) / horelem)
    
    for vert in range(vertelem): #Triple for loop go OOF
        for hor in range(horelem):
            #Draw Image
            for picY in range(len(x_test[0])):
                for picX in range(len(x_test[0][picY])):
                    val = 255-((x_test[vert*horelem + hor][picY][picX]) * 255)
                    pygame.draw.rect(mainDisplay,(val,val,val),(horrMargins+(hor*estimatedWidth)+picX,vertMargins+(vert*estimatedHeight)+picY,pixelSize,pixelSize))

            elementPredictions = predictions[vert*horelem + hor]
            
            #Draw graph
            for prediction in range(len(elementPredictions)):
                barX = horrMargins+(hor*estimatedWidth)+len(x_test[0][0])+prediction
                barY = vertMargins+(vert*estimatedHeight)
                height = int(elementPredictions[prediction] * len(x_test[0])) * pixelSize
                pygame.draw.rect(mainDisplay,(0,0,255),(barX,barY,1,height))
                if(prediction % 2 == 0):
                    pygame.draw.rect(mainDisplay,(0,0,0),(barX,barY-graphSize,graphSize,graphSize))
                else:
                    pygame.draw.rect(mainDisplay,(128,128,128),(barX,barY-graphSize,graphSize,graphSize))

            #Print Guess
            textColor = (0,0,0)#Black
            guess = max(elementPredictions)
            guessValue = (elementPredictions).tolist().index(guess)
            correctValue = y_test[vert*horelem + hor]
            if(correctValue != guessValue):
                textColor = (255,0,0)#Red
            text = getFont.render_to(mainDisplay, (horrMargins+(hor*estimatedWidth), vertMargins+((vert+1)*estimatedHeight)-2 * textSize), (str(correctValue)+" | "+str(guessValue)), textColor)
            text = getFont.render_to(mainDisplay, (horrMargins+(hor*estimatedWidth), vertMargins+((vert+1)*estimatedHeight)-textSize), str(round(guess*100))+"%", textColor)
    pygame.display.update()
    print("date")
    
def shutdown(error):
    print("Shutting down Pygame")
    pygame.quit()
    raise error

update()
while(True):
    try:
        #Handle Events
        for event in pygame.event.get():
            #Quit Event
            if event.type == pygame.QUIT:
                shutdown(SystemExit)
            #Resize Event
            elif event.type==pygame.VIDEORESIZE:
                w = event.dict['size'][0]
                h = event.dict['size'][1]
                screen=pygame.display.set_mode(event.dict['size'],pygame.RESIZABLE)
                update()
    except Exception as error:
        print("Error Detected")
        shutdown(error)
        
