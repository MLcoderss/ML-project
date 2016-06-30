# Machine Learning Project
## Stock Market Predictor
### Key Points :
  - This project predicts the closing value of a stock in a market
   - We, the students of IIT Kanpur , under the guidance of programming club coded it.
   - The project was aimed at learning the algorithms of machine learning.

Website of our project is :- http://mlcoderss.github.io/ML-project/

The project has been coded in [Torch].

## Summary of the Project
Our project aims to calculate the closing value of that day of a stock in stock market. It takes opening , high , low and  total volume of the day as an input to predict the desired ouput. It has implemented basic algorithms of machine learning.

   - accuracy achieved = 65% approx. in 150 epochs
   - It also shows the error variation on graph
   - u can also save the graph and the model parameters.

#### Our project in my eyes:-

>We are a group of four people who learned machine learning from various sites and courses.
> With the effort of everyone in the group we could give a shape to this Project.
> Now we can proudly say that we have learnes machine learnin and used our summers fruitfully.

We also thanks to coordinators of the Programming club, without them we would have not been able to do any thing. Now we also include our helpful sites and courses
we learned from:-
  * Respected [Andrew NG] Sir and his course on [Coursera]
  * Book for [neural networks] written by [Michael Nelson]
  * Python from http://learnpython.org/
  * Git from https://try.github.io/levels/1/challenges/1
  * Torch from https://github.com/torch/torch7/wiki/Cheatsheet
  * Lua from http://tylerneylon.com/a/learn-lua/
  * Machine learning In Torch from [rnduja blog]

## **Description of the Project**
We here will describe how our code works. Our main code can be found [here](https://github.com/MLcoderss/ML-project/blob/master/core_function.lua).
Our data set has been imported from :- http://pages.swcp.com/stocks/

In total we have three main file:
  * [core_function.lua]
  * [stock_function.lua]
  * [Dataset.txt] which contains 20,232 different stocks

   ##### Dataset description
   The Dataset has been cropped from other [data repository] (which contains more than one lakh stocks of one year). Our dataset just contains 20,232 stocks of which 10,232 has been used as trainingset and remaining as validationset.We have ignored dates and tickers as they are useless for us to calculate our desired output. One can see how we have done it in [stock_function.lua].
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
   
After that we will describe our model:
```sh
model = nn.Sequential()		
model:add(nn.Linear(4,70))
model:add(nn.Tanh())
model:add(nn.Linear(70,50))
model:add(nn.Tanh())
model:add(nn.Linear(50,1))
model:add(nn.Tanh())
```
Our model contains two hidden layer.One has 70 neurons and other one has 50 neurons. After trying for at least one day we finally reached on this model to minimize the error. 

Loss function we have used is MseCriterion. One can find its detail [here](https://github.com/torch/nn/blob/master/doc/criterion.md).
```sh
criterion = nn.MSECriterion() 
```
Then we have used [optim] module.Here we will be using [optim.sgd].

After that comes "step" function which just train training set. We have used stochastic graident.So we have broken our trainingset into batches of size 200.The code is very simple. Any one can understand it by seeing it.

Then we have "eval" function which gives us accuracy calculated after testing validation set with the parameters we got after training trainingset in "step" function.

After iterating the same datasets for 200 times we have saved the final parameters of model in other file model.net. This helps us to use these parameters any other time without iterating over 200 times. This file has been used in files like [final_ouput.lua] and [graph_output.lua].


Team Name = ML-Coderss
Team Membres :
  * [Aditya Katara](https://www.facebook.com/aditya.katara.9?fref=ts)
  * [Palash Agarwal](https://www.facebook.com/palash.g.agrawal?fref=ts)
  * [Piyush Bansal](https://www.facebook.com/p.bansal.98?fref=ts)
  * [Saket Harsh](https://www.facebook.com/saket.harsh1?fref=ts)

### Our deep Gratitude to
  * Our Pclub cordii = [Vinayak Tantia](https://www.facebook.com/vinayak.tantia?fref=nf)
  * Coursera Co-founder = [Andrew Ng]
  * And to our respectable PARENTS who have been Our Overall Guide whole time.




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
   [core_function.lua]: (https://github.com/MLcoderss/ML-project/blob/master/core_function.lua)
   [optim]: <https://github.com/torch/optim>
   [optim.sgd]: <http://torch.ch/docs/five-simple-examples.html#4-using-the-optim-package>
   [final_ouput.lua]: <https://github.com/MLcoderss/ML-project/blob/master/final_output.lua>
   [graph_output.lua]: <>
   
