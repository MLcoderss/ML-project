require 'nn'
require 'optim'
require 'torch'
require 'stock1.lua'
require 'math'
fullset = database("Dataset.txt")

trainset = {
	size = 10000,
	data = fullset['datainput'][{{1,10000}}],
	label = fullset['dataoutput'][{{1,10000}}]
}

validationset = {
	size = 2955,
	data = fullset['datainput'][{{10001,12955}}],
	label = fullset['dataoutput'][{{10001,12955}}]
}
--print (validationset.data:size()) 
model = nn.Sequential()
model:add(nn.Linear(4,100))
model:add(nn.Tanh())
model:add(nn.Linear(100,50))
model:add(nn.Tanh())
model:add(nn.Linear(50,1))
model:add(nn.Tanh())

criterion = nn.MSECriterion()

sgd_params = {
	learningRate = 1e-3,
	learningRateDecay =2*(1e-3),
	weightDecay = 1e-4,
	momentum = 1e-3,
}

x, dl_dx = model:getParameters()
--print (x)

step = function(batch_size)
	local current_loss = 0
	local count = 0
	local shuffle = torch.randperm(trainset.size)
	batch_size = batch_size or 200
	
	for t=1,trainset.size,batch_size do
		local size = math.min(t+batch_size-1,trainset.size)-t
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
			local loss =criterion:forward(output[{{1,output:size()[1]},1}],targets)
			--print (loss)
			--print (model.output)
			--print (targets)
			--print (output:size())
			local gradoutput = criterion:backward(output[{{1,output:size()[1]},targets}],targets)	
			--print (gradouput)	
			model:backward(inputs,gradoutput)
			--print (x)
			return loss,dl_dx
		end
	
		_, fs = optim.sgd(feval,x,sgd_params)
		--print (x)
		count = count+1
		current_loss = current_loss + fs[1]
	end

	return current_loss/count
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
			hi = indices[k]
			pi = target[k]
			ji = hi-pi
			--print (ji)
			
			if math.abs(ji) <1  then
				guessed_right = guessed_right +1
			end
		end
		--print (guessed_right)
		count = count + guessed_right
	end
	
	return count/dataset.size
end

max_iters = 200


do
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
