database = function(filename)
    datainput = {}
	dataoutput = {}
	for line in io.lines(filename) do
		date1,rating1,start1,hi1,low1,close1,vol1=unpack(line:split(","))
		local t={start1/10,hi1/10,low1/10,vol1/10000}
		table.insert(datainput,t)
		table.insert(dataoutput,{close1/70})
	end
    return {datainput = torch.Tensor(datainput),dataoutput = torch.Tensor(dataoutput)}
end

mega = function(start,stop,data)
	list = {}
	for i=start,stop do
		table.insert(list,data[i])
	end
	return list
end

