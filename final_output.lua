require 'nn'
require 'optim'
require 'torch'
require 'stock_function.lua'
require 'math'
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
a = torch.Tensor(4)
for i=1,4 do
	a[i] = io.read()/10
end
a[4] = a[4]/1000
print (model1:forward(a)*70)
