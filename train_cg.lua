require "cunn"
require "nn"
require "image"
require "optim"
require "cutorch"
require "network"

batch_size = 1024

input_patchsz = 16
output_patchsz = 8
sx = 1+math.floor((input_patchsz - output_patchsz) / 2)
sy = 1+math.floor((input_patchsz - output_patchsz) / 2)
ex = math.floor((input_patchsz + output_patchsz) / 2)
ey = math.floor((input_patchsz + output_patchsz) / 2)
number_of_images = 2000
number_of_validation_images = 1000

input_1 = torch.load("input_1.t7")
input_2 = torch.load("input_2.t7")
output = torch.load("output.t7")

print(input_1)
print("loaded everythin")
input_1:mul(2./255.):add(-1):cuda()
input_2:mul(2./255.):add(-1):cuda()
output:mul(2./255.):add(-1):cuda()

print("All in cuda")

validation_data_patches = torch.load("validation_table_of_patches.t7")
for i = 1, number_of_validation_images do
  validation_data_patches[i] = validation_data_patches[i]:double():mul(2./255.):add(-1):cuda()
end


count = 0
mlp = create_network()
print(mlp)

convert_in = nn.Sequential()
partable = nn.ParallelTable()
partable:add(nn.Reshape(input_patchsz*input_patchsz))
partable:add(nn.Reshape(input_patchsz*input_patchsz))
convert_in:add(partable)
convert_in:add(nn.JoinTable(1,1))
convert_in:cuda()

convert_out = nn.Reshape(output_patchsz*output_patchsz):cuda()


x,dl_dx = mlp:getParameters()
criterion = nn.MSECriterion()
--print(x[2])
--print(dl_dx)
x:cuda()
dl_dx:cuda()
criterion:cuda()


local function single_epoch(mlp,criterion,input_1,input_2,output,number_of_images,batch_size,x,dl_dx)

  local function feval(x_new)
    --if x~= x_new then
      x:copy(x_new)
    --end
    --print(x[1])
    dl_dx:zero()
    pred_outputs = mlp:forward(inputs)
    --print(pred_outputs:size())
    loss = criterion:forward(pred_outputs,outputs)
    --print(loss)
    grad_outs = criterion:backward(pred_outputs,outputs)
    grad_ins = mlp:backward(inputs, grad_outs)
    return loss, dl_dx
  end

    image_1 = input_1
    image_2 = output
    image_3 = input_2


    --print(im_1[{ i,{},{} }])

    inputs = convert_in:forward({ image_1, image_3 })
    outputs = convert_out:forward(image_2[{{}, {sx, ex}, {sy, ey}}])


    count = count + 1
    optim_params = {learningRate = 0.01}
    local _,errs = optim.sgd(feval,x,optim_params)
    total_loss = total_loss + errs[1]
    --print(errs[1])

return total_loss/number_of_images

end

local function validation_epoch(mlp,criterion,train_data_patches,number_of_images,batch_size,x,dl_dx)
  validation_loss = 0
  for i = 1, number_of_images-2 do

    image_1 = train_data_patches[i]
    image_2 = train_data_patches[i+1]
    image_3 = train_data_patches[i+2]


    --print(im_1[{ i,{},{} }])

    inputs = convert_in:forward({ image_1, image_3 })
    outputs = convert_out:forward(image_2[{{}, {sx, ex}, {sy, ey}}])

    pred_outputs = mlp:forward(inputs)
    err = criterion:forward(pred_outputs,outputs)



   validation_loss = validation_loss + err
    --print(errs[1])
  end

  validation_loss = validation_loss/number_of_images

  return validation_loss
end


print("TRAINING_LOSS".."     ".."VALIDATION LOSS")


for iter = 1, 100 do
  mlp:training()
  training_loss = single_epoch(mlp,criterion,input_1,input_2,output,number_of_images,batch_size,x,dl_dx)
  mlp:evaluate()
  validation_loss = validation_epoch(mlp,criterion,validation_data_patches,number_of_validation_images,batch_size,x,dl_dx)
  print(training_loss.."    "..validation_loss)

end

torch.save("mlp.t7",mlp)
