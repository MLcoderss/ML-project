# Machine learning Project
## Stock Market Predictor
### Summarised Description
>This Project predicts closing value of a stock of a particular day in a market, given its value of opening, maximum, minimum and volume of a stock of that day.
* The project has been coded in Torch.
* This project implements the basic algorithms of machine learning.

### Installation of Torch
This project uses Torch , a language particularly made for Machine Learning. Torch can be installed easily on linux from its terminal. It can not installed on windows.

* Link for installation guide:- http://torch.ch/docs/getting-started.html

Other module needed can be installed by using below commands.
```sh
$ luarocks install nn
$ luarocks install itorch
$ luarocks install image
$ luarocks install optim
```
Also Dataset needed to build the project can be found [here](http://pages.swcp.com/stocks/).

### Helpful Tutorials , Books and Links
Here is the name and links of important persons and important stuffs.
  * Respected [Andrew NG] Sir and his course on [Coursera]
  * Book for [neural networks] written by [Michael Nelson]
  * Python from http://learnpython.org/
  * Git from https://try.github.io/levels/1/challenges/1
  * Torch from https://github.com/torch/torch7/wiki/Cheatsheet
  * Lua from http://tylerneylon.com/a/learn-lua/
  * Machine learning In Torch from [rnduja blog]

### Description
In total we have five main files:
  * [core_function.lua]
  * [stock_function.lua]
  * [Dataset.txt] which contains 20,232 different stocks
  * [final_output.lua]
  * [graph_output.lua]

   ##### Dataset description
   The Dataset has been cropped from other [data repository] ,which contains more than one lakh stocks of one year. Our dataset just contains 20,232 stocks of which 10,232 has been used as trainingset and remaining as validationset. We have ignored dates and tickers as they are not useful for us to calculate our desired output. One can see how we have done it in [stock_function.lua].
   ```sh
   fullset = database("Dataset.txt")	.
trainset = {
	size = 10232,
	data = fullset['datainput'][{{1,10232}}],
	label = fullset['dataoutput'][{{1,10232}}]
}
validationset = {
	size = 10000,
	data = fullset['datainput'][{{10233,20232}}],
	label = fullset['dataoutput'][{{10233,20232}}]
}
   ```
   Here 'database' is an user defined function in file [stock_function.lua] which helps us to convert [Dataset.txt] to convert into a Double Tensor so that we can apply arithmetic operations on it.
   
   ##### Description of Model
   Our model contains two hidden layer.One has 70 neurons and other one has 50 neurons.
   ```sh
    model = nn.Sequential()		
    model:add(nn.Linear(4,70))
    model:add(nn.Tanh())
    model:add(nn.Linear(70,50))
    model:add(nn.Tanh())
    model:add(nn.Linear(50,1))
    model:add(nn.Tanh())
   ```
   This model applies [Tanh](https://github.com/torch/nn/blob/master/doc/transfer.md#tanh) function. 
   Tanh is defined as f(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))

We have used MSECriterion for our loss function. Since we are not intersted in backpropagation. One can find about this criterion [here](https://github.com/torch/nn/blob/master/doc/criterion.md).
```sh
criterion = nn.MSECriterion() 
```
Since we are optimizing our model through Stochastic Gradient we have used module "optim.sgd". The tutorial link is [here][optim.sgd]
```sh
_, fs = optim.sgd(feval,x,sgd_params) 
```
The training set has been trained by "**step**" function.The code for it is simple and can be understood after going through the Turorials given above. We have used batches of **batch_size = 200** .

**eval** function gives us accuracy calculated after testing validation set with the parameters we got after training trainingset in "**step**" function.

After iterating the same datasets for **200** times or (200 epochs) we have saved the final parameters of model in other file **model.net**. This helps us to use these parameters any other time without iterating over 200 times. This file has been used in files like [final_output.lua] and [graph_output.lua].


![ScreenShot](https://github.com/MLcoderss/ML-project/raw/master/Screenshot%201.png)

#### Description of **[final_output.lua]** and **[graph_output.lua]**
  The **final_ouput.lua** , as the name itself says, gives us the closing stock value when given value of opening , highest , lowest and overall volume of a stock of a day as an input.
  
  ![ScreenShot](https://github.com/MLcoderss/ML-project/raw/master/Screenshot%202.png)
   
   
  The **graph_ouput.lua** gives us the graph between the **predicted closing value** and **actual closing value** .It also save the graph in home directory as a .png file.
  
  ![ScreenShot](https://github.com/MLcoderss/ML-project/raw/master/Screenshot%203.png)

### Contributors
Team Name = [ML-Coderss](http://mlcoderss.github.io/ML-project/)

Team Membres :
  * [Aditya Katara](https://github.com/adityakt)
  * [Palash Agarwal](https://github.com/agpalash)
  * [Piyush Bansal](https://github.com/piushbansal)
  * [Saket Harsh](https://github.com/sharsh56625)

### Credits
  * [IIT Kanpur](http://iitk.ac.in/)
  * [Pclub](https://www.google.co.in/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwiQm5PR88_NAhUFTY8KHYaoCHYQFggdMAA&url=http%3A%2F%2Fpclub.in%2F&usg=AFQjCNEobwZBgd2l9kDqJbEbuK-vvc6KkA&cad=rja)
  * Our Pclub cordii = [Vinayak Tantia](https://www.facebook.com/vinayak.tantia?fref=nf)
  * Coursera Co-founder = [Andrew Ng]

### Website
Our website is : http://mlcoderss.github.io/ML-project/


[//]: # 
   [Torch]: <https://github.com/torch/torch7/wiki/Cheatsheet>
   [Andrew NG]: <https://www.coursera.org/instructor/andrewng>
   [Coursera]: <https://www.coursera.org/learn/machine-learning>
   [neural networks]: <http://neuralnetworksanddeeplearning.com/>
   [Michael Nelson]: <http://michaelnielsen.org/>
   [rnduja blog]: <http://rnduja.github.io/2015/10/13/torch-mnist/>
   [Dataset.txt]: <https://github.com/MLcoderss/ML-project/blob/master/Dataset.txt>
   [data repository]: <https://github.com/MLcoderss/ML-project/blob/master/sp500hst.txt>
   [stock_function.lua]: <https://github.com/MLcoderss/ML-project/blob/master/stock_function.lua>
   [core_function.lua]: <https://github.com/MLcoderss/ML-project/blob/master/core_function.lua>
   [optim]: <https://github.com/torch/optim>
   [optim.sgd]: <http://torch.ch/docs/five-simple-examples.html#4-using-the-optim-package>
   [final_output.lua]: <https://github.com/MLcoderss/ML-project/blob/master/final_output.lua>
   [graph_output.lua]: <https://github.com/MLcoderss/ML-project/blob/master/graph_output.lua>
   
