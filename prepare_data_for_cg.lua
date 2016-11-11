--[[
  Take the input of a table 1024x16x16 (Many rows like this)
  and convert it to 2000*1024x16x16 - or Tensor of the same dim.
]]

require "nn"
require "image"
require "optim"

ran_net = nn.Sequential()
convert_input = nn.JoinTable(1,1)
ran_net:add(convert_input)

train_data = torch.load("train.t7")
patch_size = 16
number_of_images = 500

table_of_patches = {}


local function create_patches(train_data,patch_size,number_of_images)

  for num = 1, number_of_images do
    print(num)
    image_1 = train_data[num]

    height = image_1:size(1)
    width = image_1:size(2)
    number_of_patches = ((height*width)/(16*16))

    tensor_of_images = torch.zeros(number_of_patches,16,16)
    count = 0

    for i = 1,height-16,16 do
      for j = 1,width-16,16 do
        single_patch = image.crop(image_1, j,i,j+16,i+16)
        count = count+1
        tensor_of_images[{ count,{},{} }] = single_patch
      end
    end
    table.insert(table_of_patches,tensor_of_images)
  end
  return table_of_patches
end

input_1 = {}
input_2 = {}
output = {}

table_of_patches = create_patches(train_data,patch_size,number_of_images)

for i = 1, number_of_images-2 do
  table.insert(input_1,table_of_patches[i])
end
for i = 2, number_of_images-1 do
  table.insert(output,table_of_patches[i+1])
end
for i = 3, number_of_images do
  table.insert(input_2, table_of_patches[i+2])
end


--out = ran_net:forward(table_of_patches)
input_1 = ran_net:forward(input_1)
input_2 = ran_net:forward(input_2)
output = ran_net:forward(output)

--print(out:size())
--torch.save("cg.t7",out)
torch.save("input_1.t7",input_1)
torch.save("input_2.t7",input_2)
torch.save("output.t7",output)
--table_of_patches = torch.Tensor(table_of_patches)
