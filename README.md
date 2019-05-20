# Your_Chatbot

This is a chatbot model that you can bring you own datasets(conversations) to train the recurrent neural networks and tweak it around to get better outputs. There is a simple GUI(based on tkinter) included so that you can chat with your Chatbot easily and immersively.

## My Testing Enviroment

OS: Windows 10 X64
Python Version: 3.5
Tensorflow-gpu Version: 1.9.0
CUDA: 9.0

## Prerequisites

Tensorflow-gpu   # pip install tensorflow-gpu
Numpy            # pip install numpy  
pandas           # pip install pandas
tkinter          # included in python distributions since python 3.1

## Datasets Preparation

I trained my version of chatbot using reddit comments which i downloaded it from the internet. They are huge files and i recommend you choose one month(one file) commemnts if you are gonna use this source, which i think its enough. 
Here is the link. [http://files.pushshift.io/reddit/comments/](http://files.pushshift.io/reddit/comments/)

Of course, you can use any datasets you want. Here is a example of what the data looks like after getting rid of the nuneccessary information. There are questions and answers saved in two files.

### Questions data preview

![Image of questions](https://github.com/zhaohehe520/Your_Chatbot/readme_pics/questions.jpg)
