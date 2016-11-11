require "image"
require "cunn"
require "nn"

local function rip2pieces(image1,image2,mlp)

	local height = image1:size(1)
	local width  = image2:size(2)

	local recon_image = torch.zeros(height, width)

	for i = 1, height-20,1 do
		print(i)
		for j = 1, width-20,1 do
			patch_1  = image1[{ {i, i+19}, {j,j+19} }]
			patch_2  = image2[{ {i, i+19}, {j,j+19} }]

			test_patch_1 = patch_1:clone() -- Don't forget to clone()
			test_patch_2 = patch_2:clone()
			sanity_check = test_patch_2 - test_patch_1
			if torch.norm(sanity_check) == 0 then
				print("Excluded a patch")
				test_output = test_patch_1[{ {6,15},{6,15} }]:clone()
				output = test_output:clone()
			else
			output = mlp:forward(torch.cat(
					patch_1:reshape(1,400),
					patch_2:reshape(1,400),
					2):cuda())
			output = output:reshape(10,10):double()
			test_output = output:clone()
			end

			image_out = test_output:add(1):mul(255./2.):byte()
			temp =	torch.add(recon_image[{ {i+6,i+15}, {j+6,j+15} }],output)
			recon_image[{ {i+6,i+15},{j+6,j+15} }]:copy(temp)

   		patch_image_1 = test_patch_1:add(1):mul(255./2.):byte()
			patch_image_2 = test_patch_2:add(1):mul(255./2.):byte()

		end
	end


	print(recon_image:size())
	out_image = recon_image:div(100):add(1):mul(255./2.):byte()
	return out_image

end

--number_of_test = 1000
count = 0

	local mlp = torch.load("mlp_adagrad.t7")

		image1 = image.load("./Middlebury/eval-data-gray/Grove/frame10.png",1,'byte')
		image2 = image.load("./Middlebury/eval-data-gray/Grove/frame11.png",1,'byte')
		image2 = image2:reshape(image2:size(2),image2:size(3))
		image1 = image1:reshape(image1:size(2),image1:size(3))
		print(image1:size())
		--image1 = image.scale(image1,512,384)
		--image2 = image.scale(image2,512,384)
		local image1 = image1:double():mul(2./255.):add(-1)
		local image2 = image2:double():mul(2./255.):add(-1)

		out_image = rip2pieces(image1,image2,mlp)

		image.save("./1.png",image1:add(1):mul(255./2.):byte())
		image.save("./2.png",out_image)
		image.save("./3.png",image2:add(1):mul(255./2.):byte())
		count = count+2
	--rip2pieces(mlp)
