require 'torch'
require 'nn'
require 'image'
require 'cutorch'
require 'cunn'
require "optim"
require "posix"
require "network"
mlp = create_network()



input_patchsz = 16
output_patchsz = 8
sx = 1+math.floor((input_patchsz - output_patchsz) / 2)
sy = 1+math.floor((input_patchsz - output_patchsz) / 2)
ex = math.floor((input_patchsz + output_patchsz) / 2)
ey = math.floor((input_patchsz + output_patchsz) / 2)


input_nodes = input_patchsz * input_patchsz * 2
output_nodes = output_patchsz * output_patchsz


super_logger = optim.Logger('adagrad_256_BATCH.log')
super_logger:setNames{'Training Acc.','Test Acc.'}

logger = optim.Logger('adagrad_256_EPOCH.log')
logger:setNames{'Training Acc.','Test Acc.'}


print(mlp)


converter_in = nn.Sequential()
partable = nn.ParallelTable()
partable:add(nn.Reshape(input_patchsz*input_patchsz))
partable:add(nn.Reshape(input_patchsz*input_patchsz))
converter_in:add(partable)
converter_in:add(nn.JoinTable(1,1))
converter_in:cuda()
converter_out = nn.Reshape(output_patchsz*output_patchsz):cuda()


criterion = nn.MSECriterion()
criterion:cuda()
-- Load Data

params, grads = mlp:getParameters()





local function single_epoch(mbg, optimizer, optimizer_params, mlp, params, gradients, criteria, printstuff)



			local function evaluator(x)
				--if x~=params then
					x:copy(params)
					--print(x[1])
	    		--end
				gradients:zero()
				--print(input[{ {1},{1} }])
				local output = 	mlp:forward(inputs)
				local err = criteria:forward(output, target)
				local gradOutput = criteria:backward(output, target)
				local gradInput = mlp:backward(inputs, gradOutput)
				super_logger:add{err}

				return err, gradients
			end

      total_loss = 0

      for i = 1, 998 do

        image_1 = train_data_patches[i]
        image_2 = train_data_patches[i+1]
        image_3 = train_data_patches[i+2]


        --print(im_1[{ i,{},{} }])

        inputs = converter_in:forward({ image_1, image_3 })
        target = converter_out:forward(image_2[{{}, {sx, ex}, {sy, ey}}])

        global_optimizer_params = {}
        local _, errs = optim.adam(evaluator, params, global_optimizer_params)
        total_loss = total_loss + errs[1]
        --print(errs[1])
      end


			--print(errs)
  --print(total_loss)

	--print(loss.."     "..count)
	return total_loss/998
end




local converged = false


train_data_patches = torch.load("table_of_patches.t7")
for i = 1, 1000 do
  train_data_patches[i] = train_data_patches[i]:mul(2./255.):add(-1):cuda()
end


--local training_loss = single_epoch(train_data_patches, function (opfunc, params, config)  local fx, dfdx = opfunc(params) return params, {fx} end, nil, mlp, params, grads, criterion)
--local validation_loss = single_epoch(train_data_patches, function (opfunc, params, config)  local fx, dfdx = opfunc(params) return params, {fx} end, nil, mlp, params, grads, criterion)
--io.stdout:write("{")
--io.stdout:write(string.format("\n	{ %d, %.7f, %.7f }", 0, training_loss, validation_loss))


for i=1,40 do
	mlp:training()
	--print(optimizer)
	print("Epoch number  "..i)
	err = single_epoch(train_data_patches, optimizer, optimizer_params, mlp, params, grads, criterion)
	mlp:evaluate()
  print(err)
	--print("Training error: 1 epoch")
	--training_loss = single_epoch(train_data_patches, function (opfunc, params, config)  local fx, dfdx = opfunc(params) return params, {fx} end, nil, mlp, params, grads, criterion, true)
	--validation_loss = single_epoch(train_data_patches, function (opfunc, params, config)  local fx, dfdx = opfunc(params) return params, {fx} end, nil, mlp, params, grads, criterion, true)
	--training_loss = tostring(training_loss)
	--validation_loss = tostring(validation_loss)
	--logger:add{training_loss,validation_loss}
	--logger:style('+-','+-')
  --print(training_loss )
	--io.stdout:write(string.format(",\n	{ %d, %.7f, %.7f }", i, training_loss, validation_loss))
end


io.stdout:write("\n}\n")
logger:plot()

super_logger:plot()
torch.save("mlp_adagrad_256_orig.t7", mlp)
