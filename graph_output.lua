require 'nn'
require 'optim'
require 'torch'
require 'stock_function.lua'
require 'math'
require 'gnuplot'
fullset = database("Dataset.txt")

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

model1 = torch.load('model.net')


a = model1:forward(validationset.data)*70
k = torch.Tensor(a:size()[1])
for i=1,a:size()[1] do
	k[i] = a[i][1]
end
b = (validationset.label)*70
x = torch.linspace(15,60)
gnuplot.epsfigure('Hypothesis Curve.eps')
gnuplot.plot({'Comparision',k,b},{'y=x Line',x,x})
gnuplot.xlabel('Predicted')
gnuplot.ylabel('Original')

gnuplot.plotflush()



