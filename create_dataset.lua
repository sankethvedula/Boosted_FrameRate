require "nn"
require "image"
require "optim"

data_type = "train"
no_of_frames = 4000
folder_name = "../dataset/"
table_of_frames = {}

for i = 1,no_of_frames do
  print(i)
  image_read = image.load(folder_name..i..".png",1,'byte')
  image_read = image.scale(image_read,600,600)
  table.insert(table_of_frames,image_read)
end

torch.save(data_type..".t7",table_of_frames)
