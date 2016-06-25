require 'nn'
require 'optim'
require 'torch'
require 'stock_function.lua' 	-- user defined package
require 'math'
fullset = database("Dataset.txt")	--Dataset.txt contains the data for the stocks.It consists of,in all 20232 readings.
					--Dataset contains data,ticker value,opening price,highest price,lowest price,closing price and the 						  volume transacted throughout the day.Of these we have chosen the opening,high,low and the volume and 						  are predicting the closing value. 

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
--print (validationset.data:size()) 
model = nn.Sequential()			--The neural net consists of an input of 4 neurons, two hidden layers of 70 and 50 neurons respectively 					  and the output layer of 1 neuron.For each layer,Tanh function is applied.
model:add(nn.Linear(4,70))
model:add(nn.Tanh())
model:add(nn.Linear(70,50))
model:add(nn.Tanh())
model:add(nn.Linear(50,1))
model:add(nn.Tanh())

criterion = nn.MSECriterion()		--The model uses Mean Squared Error Criterion.

sgd_params = {
	learningRate = 0.1,
	learningRateDecay = 2 * ( 1e-4),
	weightDecay = 1e-3,
	momentum = 1e-4
}

x, dl_dx = model:getParameters()
--print (x)

step = function(batch_size)		--For training the model.
	local current_loss = 0
	local count = 0
	local shuffle = torch.randperm(trainset.size)
	batch_size = batch_size or 200
	
	for t=1,trainset.size,batch_size do
		local size = math.min(t+batch_size-1,trainset.size)-t	--This divies the data set into equal sized batches with the last batch 									  containing the leftover data
		local inputs = torch.Tensor(size,4)
		local targets = torch.Tensor(size)
		for i=1,size do
			local input = trainset.data[shuffle[i+t]]
			local target = trainset.label[shuffle[i+t]]
			inputs[i] = input
			targets[i] = target
		end
		--print (targets)
		local feval = function(x_new)
			if x~=x_new then x:copy(x_new) end
			dl_dx:zero()
			local output = model:forward(inputs)
			--print (output:size())
			--print (output)
			--print (targets)
			--print (targets:size())
			local loss =criterion:forward(output,targets)
			--print (loss)
			--print (model.output)
			--print (targets)
			--print (output:size())
			local gradoutput = criterion:backward(output,targets)	
			--print (gradouput)	
			model:backward(inputs,gradoutput)
			--print (x)
			return loss,dl_dx
		end
	
		_, fs = optim.sgd(feval,x,sgd_params)	-- _' is used because the first parameter that we get is useless.
		--print (x)
		count = count+1
		current_loss = current_loss + fs[1]
	end

	return current_loss/count		--Normalizing the current loss
end

eval = function(dataset,batch_size)
	local count = 0
	batch_size = batch_size or 200
	
	for i=1,dataset.size,batch_size do
		local size = math.min(i+batch_size-1,dataset.size)-i
		local inputs = dataset.data[{{i,i+size-1}}]
		local targets = dataset.label[{{i,i+size-1}}]
		--print (targets)
		local outputs = model:forward(inputs)
		
		--print (inputs[1])
		local output1 = torch.Tensor(outputs:size()[1])
		for i=1,outputs:size()[1] do
			output1[i] = outputs[i][1]
		end
		local indices = output1*70
		local target = targets*70
		--print (target)
		--print (indices)
		guessed_right = 0
		for k =1,size do
			if math.abs(indices[k]-target[k])<1  then	--If the difference between the hypothesized value and the actual value 									  is at most 1, it is counted.
				guessed_right = guessed_right +1
			end
		end
		--print (guessed_right)
		count = count + guessed_right
	end
	
	return count/dataset.size		--Normalizing count
end

max_iters = 200

do					-- Training the data
	local last_accuracy = 0
	local decreasing = 0
	local threshold = 1
	for i=1,max_iters do
		local loss = step()
		--print (x)
		print(string.format('Epoch: %d Current loss: %4f' , i , loss))
		local accuracy = eval(validationset)
		print(string.format('Accuracy on the validation set: %4f',accuracy))
		if accuracy < last_accuracy then 
			if decreasing > threshold then break end
			deacreasing =  decreasing + 1
		else
			decreasing = 0
		end
		last_accuracy = accuracy
	end
end



paths = require 'paths'


filename = paths.concat(paths.cwd(), 'model.net')	--Saving the model's weights and biases of the last epoch
--print (filename)
torch.save('model.net',model)


